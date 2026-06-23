from dataclasses import replace
import numpy as np
import pytest
import sys
import types

from curvyzero.infra.modal import source_state_batched_observation_boundary_profile as boundary
from curvyzero.infra.modal.source_state_batched_observation_boundary_profile import (
    ACTION_COUNT,
    BOUNDARY_PARITY_MODE_EXACT,
    BOUNDARY_PARITY_MODE_TOLERANT,
    DEFAULT_BOUNDARY_GEOMETRY_DTYPE,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_DIRECT_SEMANTICS,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT,
    PLAYER_COUNT,
    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND,
    TARGET_SIZE,
    _assert_no_render_truncation_if_required,
    _assert_parity,
    _build_profile_lightzero_policy,
    _compact_hybrid_observation_profile_result,
    _DynamicJaxBatchedObservationRenderer,
    _extract_eval_action_from_plain,
    _latest_uint8_frames_from_stack,
    _LightZeroArrayCeilingCompactSearchService,
    _LightZeroCollectForwardCompactSearchService,
    _LightZeroArrayCeilingStackProbe,
    _LightZeroCollectForwardStackProbe,
    _LightZeroInitialInferenceStackProbe,
    _is_persistent_compact_render_state,
    _policy_output_row_from_plain,
    _persistent_compact_state_from_production,
    _persistent_delta_state,
    _PersistentJaxPolicyFramebufferRenderer,
    _validate_persistent_compact_render_state,
    _validate_boundary_config,
    _push_row_major_frames_into_stack,
    _row_major_render_players,
    _row_major_render_rows,
    _select_render_trail_slots,
    _trail_stats_after_owner_pack,
    _truncate_compact_trails_for_render,
    _view_major_to_row_major_frames,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderRequest,
    SourceStateBatchedRenderStateRowOverlay,
    source_state_render_state_with_row_overlays,
)
from curvyzero.training.compact_policy_row_bridge import build_compact_root_batch_v1
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_index_rows_v1_from_search_result,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_target_rows_from_search_arrays_v0,
)
from curvyzero.training.compact_policy_row_bridge import (
    materialize_compact_target_rows_from_index_rows_v1,
)
from curvyzero.training.compact_search_service import CompactSearchServiceV1
from curvyzero.training.compact_search_service import compact_search_result_v1_from_arrays
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.compact_torch_search_service import CompactTorchCompileConfig
from curvyzero.training.compact_torch_search_service import CompactTorchSearchServiceV1
from curvyzero.training.fixed_shape_batched_search_owner import (
    FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    build_source_state_multiplayer_sample_batch_v0,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    HybridBatchedStackProbeResult,
    HybridBatchedObservationProfileManager,
    HybridCompactBatch,
    HybridObservationProfileConfig,
    PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_MARKER,
    _joint_action_from_compact_search_arrays,
    _two_record_replay_chunk_from_hybrid_steps,
)


def test_view_major_to_row_major_frames_interleaves_player_views_by_row():
    batch_size = 3
    view_major = np.zeros((batch_size * 2, 1, 64, 64), dtype=np.uint8)
    view_major[:batch_size, 0, 0, 0] = np.asarray([10, 11, 12], dtype=np.uint8)
    view_major[batch_size:, 0, 0, 0] = np.asarray([20, 21, 22], dtype=np.uint8)

    row_major = _view_major_to_row_major_frames(view_major, batch_size=batch_size)

    assert row_major.shape == (batch_size, 2, 1, 64, 64)
    assert row_major.dtype == np.uint8
    np.testing.assert_array_equal(row_major[:, 0, 0, 0, 0], np.asarray([10, 11, 12]))
    np.testing.assert_array_equal(row_major[:, 1, 0, 0, 0], np.asarray([20, 21, 22]))


def test_mixed_jax_torch_modal_image_disables_jax_preallocation():
    assert boundary.MIXED_JAX_TORCH_GPU_MEMORY_ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert (
        boundary.MIXED_JAX_TORCH_GPU_MEMORY_ENV["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True"
    )


def test_view_major_to_row_major_frames_rejects_bad_shape():
    with pytest.raises(ValueError, match="view-major frames"):
        _view_major_to_row_major_frames(
            np.zeros((2, 64, 64), dtype=np.uint8),
            batch_size=1,
        )


def test_row_major_render_index_helpers_match_boundary_order():
    rows = _row_major_render_rows(np=np, batch_size=3)
    players = _row_major_render_players(np=np, batch_size=3)

    np.testing.assert_array_equal(rows, np.asarray([0, 0, 1, 1, 2, 2]))
    np.testing.assert_array_equal(players, np.asarray([0, 1, 0, 1, 0, 1]))


def _dynamic_renderer_request(
    *,
    row_indices: np.ndarray | None = None,
    controlled_players: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> SourceStateBatchedRenderRequest:
    batch_size = 2
    if row_indices is None:
        row_indices = _row_major_render_rows(np=np, batch_size=batch_size)
    if controlled_players is None:
        controlled_players = _row_major_render_players(np=np, batch_size=batch_size)
    if out is None:
        out = np.zeros((batch_size * PLAYER_COUNT, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    return SourceStateBatchedRenderRequest(
        state={},
        row_indices=row_indices,
        controlled_players=controlled_players,
        out=out,
    )


def _dynamic_renderer() -> _DynamicJaxBatchedObservationRenderer:
    return _DynamicJaxBatchedObservationRenderer(
        jax=None,
        np=np,
        config={"batch_size": 2},
        render_fn_for_slots=lambda _slots: None,
    )


def _install_fake_dynamic_render(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_render(**_kwargs):
        frames = np.zeros((2, PLAYER_COUNT, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        for row in range(2):
            for player in range(PLAYER_COUNT):
                frames[row, player, 0, 0, 0] = row * 10 + player
        timings = {
            "production_to_compact_sec": 0.0,
            "owner_ordered_pack_sec": 0.0,
            "host_to_device_sec": 0.0,
            "device_render_sec": 0.0,
            "device_to_host_sec": 0.0,
            "view_major_to_row_major_sec": 0.0,
        }
        trail_stats = {
            "render_trail_slots": 32.0,
            "active_trail_count_max": 0.0,
            "render_truncation_row_count": 0.0,
        }
        return frames, timings, trail_stats

    monkeypatch.setattr(boundary, "_render_candidate_frames_from_production_state", fake_render)


def test_dynamic_renderer_accepts_partial_row_requests(monkeypatch: pytest.MonkeyPatch):
    _install_fake_dynamic_render(monkeypatch)
    renderer = _dynamic_renderer()
    request = _dynamic_renderer_request(
        row_indices=np.asarray([0, 1], dtype=np.int64),
        controlled_players=np.asarray([0, 1], dtype=np.int64),
        out=np.zeros((2, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8),
    )

    result = renderer.render(request)

    np.testing.assert_array_equal(result.frames[:, 0, 0, 0], np.asarray([0, 11]))
    assert result.telemetry["partial_render_request"] == 1.0


def test_dynamic_renderer_accepts_requested_player_order(monkeypatch: pytest.MonkeyPatch):
    _install_fake_dynamic_render(monkeypatch)
    renderer = _dynamic_renderer()
    request = _dynamic_renderer_request(
        controlled_players=np.asarray([0, 0, 1, 1], dtype=np.int64),
    )

    result = renderer.render(request)

    np.testing.assert_array_equal(result.frames[:, 0, 0, 0], np.asarray([0, 0, 11, 11]))
    assert result.telemetry["partial_render_request"] == 1.0


def test_persistent_renderer_full_request_requires_row_major_player_order():
    renderer = _PersistentJaxPolicyFramebufferRenderer.__new__(
        _PersistentJaxPolicyFramebufferRenderer
    )
    renderer._np = np
    renderer._batch_size = 2
    rows = _row_major_render_rows(np=np, batch_size=2)
    players = _row_major_render_players(np=np, batch_size=2)
    out = np.zeros((4, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    renderer._validate_full_row_major_request(rows=rows, players=players, out=out)
    renderer._validate_full_row_major_request(
        rows=np.asarray([1, 0], dtype=np.int64),
        players=np.asarray([1, 0], dtype=np.int64),
        out=np.zeros((2, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="row-major controlled players"):
        renderer._validate_full_row_major_request(
            rows=rows,
            players=np.asarray([1, 0, 0, 1], dtype=np.int64),
            out=out,
        )


def test_persistent_renderer_applies_row_overlay_before_compact_conversion(
    monkeypatch: pytest.MonkeyPatch,
):
    renderer = _PersistentJaxPolicyFramebufferRenderer.__new__(
        _PersistentJaxPolicyFramebufferRenderer
    )
    renderer._np = np
    renderer._batch_size = 2
    renderer._config = {
        "batch_size": 2,
        "player_count": 2,
        "bonus_count": 0,
        "trail_slots": 4,
        "body_capacity": 4,
        "geometry_dtype": "float32",
    }
    base = {
        "pos": np.zeros((2, 2, 2), dtype=np.float32),
        "radius": np.ones((2, 2), dtype=np.float32),
        "alive": np.ones((2, 2), dtype=bool),
        "present": np.ones((2, 2), dtype=bool),
        "avatar_color": np.asarray([[0, 1], [0, 1]], dtype=np.int32),
        "visual_trail_pos": np.zeros((2, 4, 2), dtype=np.float32),
        "visual_trail_radius": np.ones((2, 4), dtype=np.float32),
        "visual_trail_owner": np.zeros((2, 4), dtype=np.int32),
        "visual_trail_active": np.zeros((2, 4), dtype=bool),
        "visual_trail_write_cursor": np.asarray([1, 1], dtype=np.int32),
        "visual_trail_break_before": np.zeros((2, 4), dtype=bool),
    }
    pre_reset = {key: value.copy() for key, value in base.items()}
    pre_reset["pos"][1] = 7.0
    pre_reset["alive"][1] = False
    pre_reset["visual_trail_pos"][1, :, 0] = np.asarray([1.0, 2.0, 3.0, 4.0])
    pre_reset["visual_trail_active"][1, :3] = True
    pre_reset["visual_trail_write_cursor"][1] = 3
    expected = {key: value.copy() for key, value in base.items()}
    for key, value in pre_reset.items():
        expected[key][1] = value[1]

    def fake_convert(*, np, production_state, config):
        assert config["batch_size"] == 2
        for key, expected_value in expected.items():
            np.testing.assert_array_equal(production_state[key], expected_value)
        raise RuntimeError("captured effective production state")

    monkeypatch.setattr(boundary, "_persistent_compact_state_from_production", fake_convert)

    with pytest.raises(RuntimeError, match="captured effective production state"):
        renderer.render(
            SourceStateBatchedRenderRequest(
                state=base,
                state_row_overlays=(
                    SourceStateBatchedRenderStateRowOverlay(
                        rows=np.asarray([1], dtype=np.int32),
                        state={key: value[[1]].copy() for key, value in pre_reset.items()},
                    ),
                ),
                row_indices=_row_major_render_rows(np=np, batch_size=2),
                controlled_players=_row_major_render_players(np=np, batch_size=2),
                out=np.zeros((4, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8),
            )
        )


def test_persistent_renderer_rejects_row_overlay_with_compact_passthrough_state():
    renderer = _PersistentJaxPolicyFramebufferRenderer.__new__(
        _PersistentJaxPolicyFramebufferRenderer
    )
    renderer._np = np
    renderer._batch_size = 2
    compact_state = {PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_MARKER: np.asarray(1)}

    with pytest.raises(ValueError, match="cannot include row overlays"):
        renderer.render(
            SourceStateBatchedRenderRequest(
                state=compact_state,
                state_row_overlays=(
                    SourceStateBatchedRenderStateRowOverlay(
                        rows=np.asarray([1], dtype=np.int32),
                        state={"pos": np.zeros((1, 2, 2), dtype=np.float32)},
                    ),
                ),
                row_indices=_row_major_render_rows(np=np, batch_size=2),
                controlled_players=_row_major_render_players(np=np, batch_size=2),
                out=np.zeros((4, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8),
            )
        )


def test_persistent_compact_state_trims_visual_trails_to_live_cursor_prefix():
    production_state = {
        "pos": np.zeros((2, 2, 2), dtype=np.float64),
        "radius": np.ones((2, 2), dtype=np.float64),
        "alive": np.ones((2, 2), dtype=bool),
        "present": np.ones((2, 2), dtype=bool),
        "visual_trail_pos": np.arange(2 * 8 * 2, dtype=np.float64).reshape(2, 8, 2),
        "visual_trail_radius": np.ones((2, 8), dtype=np.float64),
        "visual_trail_owner": np.tile(
            np.asarray([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.int16),
            (2, 1),
        ),
        "visual_trail_active": np.ones((2, 8), dtype=bool),
        "visual_trail_write_cursor": np.asarray([2, 5], dtype=np.int32),
        "visual_trail_break_before": np.zeros((2, 8), dtype=bool),
    }

    compact = _persistent_compact_state_from_production(
        np=np,
        production_state=production_state,
        config={
            "batch_size": 2,
            "player_count": 2,
            "trail_slots": 4,
            "body_capacity": 8,
            "bonus_count": 0,
            "controlled_player": 0,
        },
    )

    assert compact["trail_x"].shape == (2, 5)
    np.testing.assert_array_equal(compact["trail_write_cursor"], np.asarray([2, 5], dtype=np.int32))
    np.testing.assert_array_equal(
        compact["trail_active"][0],
        np.asarray([1, 1, 0, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(compact["trail_active"][1], np.ones(5, dtype=np.uint8))


def test_source_state_render_state_row_overlays_replace_only_requested_rows():
    base = {
        "pos": np.zeros((3, 2, 2), dtype=np.float32),
        "alive": np.ones((3, 2), dtype=bool),
        "visual_trail_write_cursor": np.asarray([1, 1, 1], dtype=np.int32),
    }
    pre_reset = {
        "pos": np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2),
        "alive": np.asarray([[True, True], [False, False], [True, False]], dtype=bool),
        "visual_trail_write_cursor": np.asarray([2, 7, 5], dtype=np.int32),
    }
    overlay = SourceStateBatchedRenderStateRowOverlay(
        rows=np.asarray([1], dtype=np.int32),
        state={key: value[[1]].copy() for key, value in pre_reset.items()},
    )

    effective = source_state_render_state_with_row_overlays(base, (overlay,), batch_size=3)

    expected_pos = base["pos"].copy()
    expected_pos[1] = pre_reset["pos"][1]
    np.testing.assert_array_equal(effective["pos"], expected_pos)
    np.testing.assert_array_equal(effective["alive"][1], pre_reset["alive"][1])
    assert int(effective["visual_trail_write_cursor"][1]) == 7
    np.testing.assert_array_equal(base["pos"], np.zeros((3, 2, 2), dtype=np.float32))


def test_source_state_render_state_row_overlay_rejects_bad_rows_and_shapes():
    base = {"pos": np.zeros((2, 2, 2), dtype=np.float32)}
    good_state = {"pos": np.zeros((1, 2, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="duplicate-free"):
        source_state_render_state_with_row_overlays(
            base,
            (
                SourceStateBatchedRenderStateRowOverlay(
                    rows=np.asarray([1, 1], dtype=np.int32),
                    state={"pos": np.zeros((2, 2, 2), dtype=np.float32)},
                ),
            ),
            batch_size=2,
        )
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        source_state_render_state_with_row_overlays(
            base,
            (
                SourceStateBatchedRenderStateRowOverlay(
                    rows=np.asarray([2], dtype=np.int32),
                    state=good_state,
                ),
            ),
            batch_size=2,
        )
    with pytest.raises(ValueError, match="must have shape"):
        source_state_render_state_with_row_overlays(
            base,
            (
                SourceStateBatchedRenderStateRowOverlay(
                    rows=np.asarray([1], dtype=np.int32),
                    state={"pos": np.zeros((1, 2), dtype=np.float32)},
                ),
            ),
            batch_size=2,
        )


def test_dynamic_renderer_rejects_wrong_output_shape(monkeypatch: pytest.MonkeyPatch):
    _install_fake_dynamic_render(monkeypatch)
    renderer = _dynamic_renderer()
    request = _dynamic_renderer_request(
        out=np.zeros((2, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="request.out must match requested"):
        renderer.render(request)


def test_dynamic_renderer_rejects_wrong_output_dtype(monkeypatch: pytest.MonkeyPatch):
    _install_fake_dynamic_render(monkeypatch)
    renderer = _dynamic_renderer()
    request = _dynamic_renderer_request(
        out=np.zeros((2 * PLAYER_COUNT, 1, TARGET_SIZE, TARGET_SIZE), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="request.out must be uint8"):
        renderer.render(request)


def test_validate_boundary_config_accepts_float64_geometry_dtype():
    config = _validate_boundary_config(
        np=np,
        config={"batch_size": 1, "trail_slots": 4, "geometry_dtype": "float64"},
    )

    assert config["geometry_dtype"] == "float64"
    assert config["render_config"]["geometry_dtype"] == "float64"


def test_validate_boundary_config_defaults_to_aggressive_float32_geometry():
    config = _validate_boundary_config(
        np=np,
        config={"batch_size": 1, "trail_slots": 4},
    )

    assert DEFAULT_BOUNDARY_GEOMETRY_DTYPE == "float32"
    assert config["geometry_dtype"] == "float32"
    assert config["render_config"]["geometry_dtype"] == "float32"
    assert config["parity_mode"] == BOUNDARY_PARITY_MODE_TOLERANT


def test_validate_boundary_config_uses_exact_parity_for_float64_auto_mode():
    config = _validate_boundary_config(
        np=np,
        config={"batch_size": 1, "trail_slots": 4, "geometry_dtype": "float64"},
    )

    assert config["geometry_dtype"] == "float64"
    assert config["parity_mode"] == BOUNDARY_PARITY_MODE_EXACT


def test_validate_boundary_config_allows_explicit_tolerant_float64_debug_row():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 4,
            "geometry_dtype": "float64",
            "parity_mode": "tolerant",
            "parity_max_abs_diff": 3,
            "parity_max_mismatch_fraction": 0.01,
        },
    )

    assert config["requested_parity_mode"] == BOUNDARY_PARITY_MODE_TOLERANT
    assert config["parity_mode"] == BOUNDARY_PARITY_MODE_TOLERANT
    assert config["parity_max_abs_diff"] == 3
    assert config["parity_max_mismatch_fraction"] == 0.01


def test_validate_boundary_config_propagates_terminal_max_ticks():
    config = _validate_boundary_config(
        np=np,
        config={"batch_size": 1, "trail_slots": 4, "max_ticks": 3},
    )

    assert config["max_ticks"] == 3


def test_validate_boundary_config_can_decouple_body_capacity_from_render_trail_slots():
    config = _validate_boundary_config(
        np=np,
        config={"batch_size": 1, "trail_slots": 4, "body_capacity": 16},
    )

    assert config["trail_slots"] == 4
    assert config["body_capacity"] == 16
    assert config["render_config"]["trail_slots"] == 4


def test_validate_boundary_config_accepts_dynamic_render_width():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "body_capacity": 1024,
            "dynamic_render_trail_slots": True,
            "min_render_trail_slots": 32,
        },
    )

    assert config["dynamic_render_trail_slots"] is True
    assert config["allow_render_truncation"] is False
    assert config["min_render_trail_slots"] == 32
    assert config["render_config"]["allow_render_truncation"] is False
    assert config["render_config"]["dynamic_render_trail_slots"] is True
    assert config["render_config"]["max_render_trail_slots"] == 64
    assert config["render_config"]["body_capacity"] == 1024


def test_validate_boundary_config_accepts_candidate_only_cpu_reference_skip():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "verify_steps": 0,
            "cpu_reference_interval": 0,
        },
    )

    assert config["verify_steps"] == 0
    assert config["cpu_reference_interval"] == 0


def test_validate_boundary_config_accepts_surface_stack_backend_switch():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "surface_facade_canary": True,
            "surface_stack_backend": "cpu_dirty_cache",
        },
    )

    assert config["surface_facade_canary"] is True
    assert config["surface_stack_backend"] == "cpu_dirty_cache"


def test_validate_boundary_config_accepts_direct_gray64_surface_canary():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "surface_facade_canary": True,
            "surface_stack_backend": "renderer_backed_profile",
            "render_surface": "direct_gray64",
        },
    )

    assert config["render_surface"] == "direct_gray64"
    assert config["render_config"]["render_surface"] == "direct_gray64"
    assert config["render_config"]["trail_composition"] == "priority_buffer"


def test_validate_boundary_config_accepts_surface_facade_divergence_canary():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "surface_facade_canary": True,
            "surface_facade_divergence_canary": True,
            "surface_stack_backend": "cpu_dirty_cache",
        },
    )

    assert config["surface_facade_divergence_canary"] is True


def test_validate_boundary_config_rejects_surface_divergence_without_surface_canary():
    with pytest.raises(ValueError, match="surface_facade_divergence_canary"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "surface_facade_divergence_canary": True,
            },
        )


def test_validate_boundary_config_accepts_direct_gray64_hybrid_canary():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "surface_stack_backend": "renderer_backed_profile",
            "render_surface": "direct_gray64",
        },
    )

    assert config["hybrid_observation_canary"] is True
    assert config["render_surface"] == "direct_gray64"
    assert config["render_config"]["render_surface"] == "direct_gray64"


def test_validate_boundary_config_rejects_lightzero_collect_forward_without_persistent_direct64():
    with pytest.raises(ValueError, match="observation_renderer_backend"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_lightzero_collect_forward_probe": True,
                "render_surface": "direct_gray64",
            },
        )


def test_validate_boundary_config_accepts_lightzero_collect_forward_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_lightzero_collect_forward_probe": True,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_collect_forward_probe"] is True
    assert config["render_surface"] == "direct_gray64"


def test_validate_boundary_config_accepts_lightzero_initial_inference_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_lightzero_initial_inference_probe": True,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_initial_inference_probe"] is True
    assert config["render_surface"] == "direct_gray64"


def test_validate_boundary_config_accepts_lightzero_array_ceiling_compile_spike_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE
            ),
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_array_ceiling_probe"] is True
    assert (
        config["hybrid_lightzero_array_ceiling_mode"]
        == boundary.LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE
    )


def test_validate_boundary_config_accepts_mock_search_service_ceiling_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE
            ),
            "hybrid_lightzero_mock_service_materialize_public_output": True,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_array_ceiling_probe"] is True
    assert (
        config["hybrid_lightzero_array_ceiling_mode"]
        == boundary.LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE
    )
    assert config["hybrid_lightzero_mock_service_materialize_public_output"] is True


def test_validate_boundary_config_accepts_service_tax_compact_replay_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_service_replay_proof": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_array_ceiling_probe"] is True
    assert (
        config["hybrid_lightzero_array_ceiling_mode"]
        == boundary.LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE
    )


def test_validate_boundary_config_accepts_dense_torch_compact_replay_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_service_replay_proof": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_array_ceiling_probe"] is True
    assert (
        config["hybrid_lightzero_array_ceiling_mode"]
        == boundary.LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS
    )


def test_validate_boundary_config_accepts_compact_torch_service_compact_replay_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_service_replay_proof": True,
            "hybrid_compact_torch_compile_search": False,
            "hybrid_compact_torch_model_compile_mode": "default",
            "hybrid_compact_torch_initial_inference_mode": "direct_core",
            "hybrid_compact_torch_observation_memory_format": "channels_last",
            "hybrid_compact_torch_model_memory_format": "contiguous",
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_array_ceiling_probe"] is True
    assert (
        config["hybrid_lightzero_array_ceiling_mode"]
        == boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
    )
    assert config["hybrid_compact_torch_compile_search"] is False
    assert config["hybrid_compact_torch_model_compile_mode"] == "default"
    assert config["hybrid_compact_torch_initial_inference_mode"] == "direct_core"
    assert config["hybrid_compact_torch_observation_memory_format"] == "channels_last"
    assert config["hybrid_compact_torch_model_memory_format"] == "contiguous"


def test_validate_boundary_config_rejects_model_wide_channels_last():
    with pytest.raises(ValueError, match="model_memory_format=channels_last is parked"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_torch_model_memory_format": "channels_last",
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
                ),
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_accepts_fixed_shape_search_owner_compact_replay_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_service_replay_proof": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_array_ceiling_probe"] is True
    assert (
        config["hybrid_lightzero_array_ceiling_mode"]
        == boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
    )


def test_validate_boundary_config_accepts_mctx_compact_slab_probe():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_mctx_compact_search_probe": True,
            "hybrid_mctx_num_simulations": 4,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_mctx_compact_search_probe"] is True
    assert config["hybrid_mctx_num_simulations"] == 4
    assert config["hybrid_mctx_require_gpu_backend"] is True


def test_validate_boundary_config_accepts_scripted_compact_slab_action_mode():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
            ),
            "hybrid_compact_rollout_slab_action_mode": "scripted_random",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_compact_rollout_slab_action_mode"] == "scripted_random"


def test_validate_boundary_config_accepts_compact_root_tape_gate():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_compact_root_tape_compare": True,
            "hybrid_compact_root_tape_max_records": 3,
            "hybrid_compact_root_tape_reference_label": "primary",
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
            ),
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_compact_root_tape_compare"] is True
    assert config["hybrid_compact_root_tape_max_records"] == 3
    assert config["hybrid_compact_root_tape_reference_label"] == "primary"


def test_validate_boundary_config_accepts_compact_root_tape_mctx_service():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_compact_root_tape_compare": True,
            "hybrid_compact_root_tape_compare_mctx": True,
            "hybrid_compact_root_tape_max_records": 3,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_compact_root_tape_compare"] is True
    assert config["hybrid_compact_root_tape_compare_mctx"] is True
    assert config["hybrid_mctx_require_gpu_backend"] is True


def test_validate_boundary_config_accepts_compact_root_tape_model_compile_service():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_compact_root_tape_compare": True,
            "hybrid_compact_root_tape_compare_fixed_shape_floor": False,
            "hybrid_compact_root_tape_compare_model_compile": True,
            "hybrid_compact_root_tape_model_compile_mode": "default",
            "hybrid_lightzero_consumer_root_noise_weight": 0.0,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_compact_root_tape_compare_model_compile"] is True
    assert config["hybrid_compact_root_tape_model_compile_mode"] == "default"
    assert config["hybrid_compact_root_tape_require_model_compile"] is True


def test_validate_boundary_config_accepts_compact_root_tape_direct_core_service():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_compact_root_tape_compare": True,
            "hybrid_compact_root_tape_compare_fixed_shape_floor": False,
            "hybrid_compact_root_tape_compare_direct_core": True,
            "hybrid_lightzero_consumer_root_noise_weight": 0.0,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_compact_root_tape_compare_direct_core"] is True
    assert config["hybrid_compact_torch_initial_inference_mode"] == "model_method"


def test_validate_boundary_config_rejects_compact_root_tape_without_slab():
    with pytest.raises(ValueError, match="requires hybrid_compact_rollout_slab_probe"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_root_tape_compare": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
                ),
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_compact_root_tape_resident_without_snapshot():
    with pytest.raises(ValueError, match="does not yet support resident"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_probe": True,
                "hybrid_compact_root_tape_compare": True,
                "hybrid_device_only_stack": True,
                "hybrid_resident_observation_search": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
                ),
                "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_compact_root_tape_without_second_service():
    with pytest.raises(ValueError, match="secondary service"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_probe": True,
                "hybrid_compact_root_tape_compare": True,
                "hybrid_compact_root_tape_compare_fixed_shape_floor": False,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
                ),
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_root_tape_mctx_without_root_tape():
    with pytest.raises(ValueError, match="requires hybrid_compact_root_tape_compare"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_probe": True,
                "hybrid_compact_root_tape_compare_mctx": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
                ),
                "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_root_tape_direct_core_without_root_tape():
    with pytest.raises(ValueError, match="requires hybrid_compact_root_tape_compare"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_probe": True,
                "hybrid_compact_root_tape_compare_direct_core": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
                ),
                "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_root_tape_direct_core_primary_direct_core():
    with pytest.raises(ValueError, match="model_method primary"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_probe": True,
                "hybrid_compact_root_tape_compare": True,
                "hybrid_compact_root_tape_compare_fixed_shape_floor": False,
                "hybrid_compact_root_tape_compare_direct_core": True,
                "hybrid_compact_torch_initial_inference_mode": "direct_core",
                "hybrid_lightzero_consumer_root_noise_weight": 0.0,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
                ),
                "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_root_tape_mctx_with_mctx_primary():
    with pytest.raises(ValueError, match="non-MCTX primary"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_probe": True,
                "hybrid_compact_root_tape_compare": True,
                "hybrid_compact_root_tape_compare_mctx": True,
                "hybrid_mctx_compact_search_probe": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_select_profile_function_uses_mctx_lightzero_image_for_root_tape_sidecar():
    selected = boundary._select_profile_function(
        compute=boundary.COMPUTE_H100,
        hybrid_observation_canary=True,
        hybrid_mctx_compact_search_probe=False,
        hybrid_compact_root_tape_compare_mctx=True,
        hybrid_compact_root_tape_compare_model_compile=False,
        hybrid_compact_root_tape_compare_direct_core=False,
        hybrid_mctx_lightzero_checkpoint_ref="",
        hybrid_lightzero_collect_forward_probe=False,
        hybrid_lightzero_initial_inference_probe=False,
        hybrid_lightzero_array_ceiling_probe=True,
        hybrid_lightzero_mcts_arrays_boundary_probe=False,
        profile_env_manager_canary=False,
        surface_facade_canary=False,
        include_rnd_meter=False,
    )

    assert selected is boundary.run_hybrid_observation_profile_mctx_lightzero_modal_h100


def test_select_profile_function_uses_mctx_image_for_root_tape_sidecar_without_lightzero():
    selected = boundary._select_profile_function(
        compute=boundary.COMPUTE_H100,
        hybrid_observation_canary=True,
        hybrid_mctx_compact_search_probe=False,
        hybrid_compact_root_tape_compare_mctx=True,
        hybrid_compact_root_tape_compare_model_compile=False,
        hybrid_compact_root_tape_compare_direct_core=False,
        hybrid_mctx_lightzero_checkpoint_ref="",
        hybrid_lightzero_collect_forward_probe=False,
        hybrid_lightzero_initial_inference_probe=False,
        hybrid_lightzero_array_ceiling_probe=False,
        hybrid_lightzero_mcts_arrays_boundary_probe=False,
        profile_env_manager_canary=False,
        surface_facade_canary=False,
        include_rnd_meter=False,
    )

    assert selected is boundary.run_hybrid_observation_profile_mctx_modal_h100


def test_hybrid_observation_profile_safe_exception_keeps_safety_labels(monkeypatch):
    def raise_error(_config):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(boundary, "_run_hybrid_observation_profile_impl", raise_error)

    result = boundary._run_hybrid_observation_profile_safe({"batch_size": 1})

    assert result["ok"] is False
    assert result["profile_only"] is True
    assert result["calls_train_muzero"] is False
    assert result["touches_live_runs"] is False
    assert "synthetic failure" in result["error"]


def test_validate_boundary_config_accepts_compact_muzero_learner_gate_impl():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_materialize_scalar_timestep": False,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_compact_rollout_slab_sample_gate": True,
            "hybrid_compact_rollout_slab_learner_gate": True,
            "hybrid_compact_rollout_slab_learner_gate_impl": "compact_muzero",
            "hybrid_compact_rollout_slab_learner_gate_support_scale": 3,
            "hybrid_compact_rollout_slab_learner_gate_num_unroll_steps": 2,
            "hybrid_compact_owned_loop_entrypoint": True,
            "hybrid_compact_owned_loop_policy_version_ref": "unit-owned-policy-v1",
            "hybrid_compact_owned_loop_model_version_ref": "unit-owned-model-v1",
            "hybrid_compact_owned_loop_policy_source": "unit_test_boundary_owned_loop",
            "hybrid_compact_owned_loop_capture_replay_store_state": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
            ),
            "hybrid_lightzero_array_ceiling_input_mode": "host_uint8",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_compact_rollout_slab_learner_gate_impl"] == "compact_muzero"
    assert config["hybrid_compact_rollout_slab_learner_gate_support_scale"] == 3
    assert config["hybrid_compact_rollout_slab_learner_gate_num_unroll_steps"] == 2
    assert config["hybrid_compact_owned_loop_entrypoint"] is True
    assert config["hybrid_compact_owned_loop_policy_version_ref"] == "unit-owned-policy-v1"
    assert config["hybrid_compact_owned_loop_policy_source"] == "unit_test_boundary_owned_loop"
    assert config["hybrid_compact_owned_loop_capture_replay_store_state"] is True


def test_validate_boundary_config_accepts_normal_death_mode():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "death_mode": "normal",
        },
    )

    assert config["death_mode"] == "normal"


def test_validate_boundary_config_rejects_invalid_death_mode():
    with pytest.raises(ValueError, match="death_mode"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "death_mode": "immortal",
            },
        )


def test_validate_boundary_config_accepts_hybrid_nonsearch_floor_knobs():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
            ),
            "hybrid_device_only_stack": True,
            "hybrid_refresh_observation_stack": False,
            "hybrid_native_actor_buffer": True,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_device_only_stack"] is True
    assert config["hybrid_refresh_observation_stack"] is False
    assert config["hybrid_native_actor_buffer"] is True
    assert config["hybrid_persistent_compact_render_state_buffer"] is False
    assert config["hybrid_borrow_single_actor_render_state"] is False


def test_validate_boundary_config_rejects_persistent_render_state_without_native_buffer():
    with pytest.raises(ValueError, match="requires hybrid_native_actor_buffer"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
                ),
                "hybrid_persistent_compact_render_state_buffer": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_accepts_borrowed_single_actor_render_state():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 4,
            "actor_count": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_lightzero_array_ceiling_probe": True,
            "hybrid_lightzero_array_ceiling_mode": (
                boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
            ),
            "hybrid_native_actor_buffer": True,
            "hybrid_borrow_single_actor_render_state": True,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_borrow_single_actor_render_state"] is True


def test_validate_boundary_config_rejects_borrowed_render_state_without_native_buffer():
    with pytest.raises(ValueError, match="requires hybrid_native_actor_buffer"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 4,
                "actor_count": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
                ),
                "hybrid_borrow_single_actor_render_state": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_borrowed_render_state_multi_actor():
    with pytest.raises(ValueError, match="requires actor_count=1"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 4,
                "actor_count": 2,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
                ),
                "hybrid_native_actor_buffer": True,
                "hybrid_borrow_single_actor_render_state": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_borrowed_render_state_with_persistent_buffer():
    with pytest.raises(ValueError, match="cannot be combined"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 4,
                "actor_count": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
                ),
                "hybrid_native_actor_buffer": True,
                "hybrid_persistent_compact_render_state_buffer": True,
                "hybrid_borrow_single_actor_render_state": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_borrowed_render_state_without_refresh():
    with pytest.raises(ValueError, match="requires hybrid_refresh_observation_stack"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 4,
                "actor_count": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": (
                    boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
                ),
                "hybrid_native_actor_buffer": True,
                "hybrid_refresh_observation_stack": False,
                "hybrid_borrow_single_actor_render_state": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_unknown_compact_slab_action_mode():
    with pytest.raises(ValueError, match="hybrid_compact_rollout_slab_action_mode"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_action_mode": "mystery",
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_accepts_mctx_direct_ctree_comparator_with_checkpoint():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_compact_rollout_slab_probe": True,
            "hybrid_mctx_compact_search_probe": True,
            "hybrid_mctx_lightzero_checkpoint_ref": "run/iteration_10.pth.tar",
            "hybrid_mctx_compare_direct_ctree": True,
            "hybrid_mctx_compare_direct_ctree_impl": (
                LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT
            ),
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_mctx_compare_direct_ctree"] is True
    assert config["hybrid_mctx_compare_direct_ctree_impl"] == (
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT
    )


def test_validate_boundary_config_rejects_mctx_direct_ctree_comparator_without_checkpoint():
    with pytest.raises(ValueError, match="requires hybrid_mctx_lightzero_checkpoint_ref"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_compact_rollout_slab_probe": True,
                "hybrid_mctx_compact_search_probe": True,
                "hybrid_mctx_compare_direct_ctree": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_public_output_without_mock_service():
    with pytest.raises(ValueError, match="mock_service_materialize_public_output"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": "policy_arrays",
                "hybrid_lightzero_mock_service_materialize_public_output": True,
                "render_surface": "direct_gray64",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_accepts_lightzero_mcts_arrays_boundary_contract():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_lightzero_mcts_arrays_boundary_probe": True,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["hybrid_lightzero_mcts_arrays_boundary_probe"] is True
    assert config["render_surface"] == "direct_gray64"
    assert config["hybrid_lightzero_mcts_arrays_boundary_input_mode"] == "host_uint8"


def test_validate_boundary_config_accepts_lightzero_cpu64_profile_compute():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "compute": boundary.COMPUTE_H100_CPU64,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "hybrid_lightzero_mcts_arrays_boundary_probe": True,
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert config["compute"] == boundary.COMPUTE_H100_CPU64
    assert config["hybrid_lightzero_mcts_arrays_boundary_probe"] is True


def test_validate_boundary_config_rejects_unknown_lightzero_input_modes():
    base_config = {
        "batch_size": 1,
        "trail_slots": 64,
        "hybrid_observation_canary": True,
        "render_surface": "direct_gray64",
        "observation_renderer_backend": (
            SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
        ),
    }

    with pytest.raises(ValueError, match="mcts_arrays_boundary_input_mode"):
        _validate_boundary_config(
            np=np,
            config={
                **base_config,
                "hybrid_lightzero_mcts_arrays_boundary_probe": True,
                "hybrid_lightzero_mcts_arrays_boundary_input_mode": "garbage",
            },
        )

    with pytest.raises(ValueError, match="array_ceiling_input_mode"):
        _validate_boundary_config(
            np=np,
            config={
                **base_config,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_input_mode": "garbage",
            },
        )

    with pytest.raises(ValueError, match="array_ceiling_mode"):
        _validate_boundary_config(
            np=np,
            config={
                **base_config,
                "hybrid_lightzero_array_ceiling_probe": True,
                "hybrid_lightzero_array_ceiling_mode": "garbage",
            },
        )


def test_policy_output_row_from_plain_handles_batched_mapping_outputs():
    plain = {
        "action": np.asarray([0, 2, 1], dtype=np.int64),
        "searched_value": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "visit_count_distribution": np.asarray(
            [[3.0, 1.0, 0.0], [0.0, 1.0, 3.0], [1.0, 2.0, 1.0]],
            dtype=np.float32,
        ),
    }

    row = _policy_output_row_from_plain(plain, 1)

    assert _extract_eval_action_from_plain(row) == 2
    np.testing.assert_allclose(row["visit_count_distribution"], np.asarray([0.0, 1.0, 3.0]))


def test_policy_output_row_from_plain_handles_string_keyed_rows():
    plain = {
        "0": {"action": 0, "searched_value": 0.1},
        "1": {"action": 2, "searched_value": 0.2},
    }

    row = _policy_output_row_from_plain(plain, 1)

    assert _extract_eval_action_from_plain(row) == 2
    assert row["searched_value"] == 0.2


def test_policy_output_row_from_plain_handles_list_outputs():
    plain = [{"action": 0}, {"selected_action": [1]}]

    row = _policy_output_row_from_plain(plain, 1)

    assert _extract_eval_action_from_plain(row) == 1


def test_extract_eval_action_from_plain_handles_root_wrappers_and_missing_actions():
    assert _extract_eval_action_from_plain({"0": {"selected_actions": np.asarray([2])}}) == 2

    with pytest.raises(ValueError, match="could not extract action"):
        _extract_eval_action_from_plain({"policy": {"value": 1.0}})


def test_lightzero_collect_forward_stack_probe_flattens_roots_and_decodes(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeCollectMode:
        def __init__(self):
            self.call = None

        def forward(
            self,
            obs_tensor,
            *,
            action_mask,
            temperature,
            to_play,
            epsilon,
            ready_env_id,
        ):
            self.call = {
                "obs_shape": list(obs_tensor.shape),
                "obs_min": float(obs_tensor.min()),
                "obs_max": float(obs_tensor.max()),
                "mask_shape": list(action_mask.shape),
                "temperature": float(temperature),
                "epsilon": float(epsilon),
                "to_play": list(to_play),
                "ready_env_id": np.asarray(ready_env_id).astype(int).tolist(),
            }
            return {
                row: {
                    "action": row % 3,
                    "searched_value": float(row),
                    "visit_count_distribution": [1.0, 2.0, 3.0],
                }
                for row in range(obs_tensor.shape[0])
            }

    class FakePolicy:
        def __init__(self):
            self.collect_mode = FakeCollectMode()

    policy = FakePolicy()
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.25,
        epsilon=0.05,
    )
    observation = (
        np.arange(2 * 2 * 4 * 64 * 64, dtype=np.int32).reshape(2, 2, 4, 64, 64) % 256
    ).astype(np.uint8)
    action_mask = np.ones((2, 2, 3), dtype=bool)

    result = probe.run(observation, action_mask)

    call = policy.collect_mode.call
    assert call["obs_shape"] == [4, 4, 64, 64]
    assert call["mask_shape"] == [4, 3]
    assert call["to_play"] == [-1, -1, -1, -1]
    assert call["ready_env_id"] == [0, 1, 2, 3]
    assert call["obs_min"] >= 0.0
    assert call["obs_max"] <= 1.0
    telemetry = result.telemetry
    assert telemetry["lightzero_root_count"] == 4.0
    assert telemetry["lightzero_policy_forward_calls"] == 1.0
    assert telemetry["lightzero_illegal_action_count"] == 0.0
    assert telemetry["lightzero_first_actions"] == [0, 1, 2, 0]
    assert telemetry["lightzero_consumer_policy_class"] == "fake.Policy"
    assert telemetry["lightzero_to_play_mode"] == "fixed_opponent_minus_one"
    assert telemetry["lightzero_filtered_zero_mask_root_count"] == 0.0
    assert telemetry["lightzero_consumer_model_timer_status"] == "model_missing"


def test_lightzero_collect_forward_stack_probe_times_model_calls(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModel:
        def __init__(self):
            self.initial_calls = 0
            self.recurrent_calls = 0

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return {"latent_state": obs_tensor}

        def recurrent_inference(self, latent_state, action):
            self.recurrent_calls += 1
            return {"latent_state": latent_state, "action": action}

    class FakeCollectMode:
        def __init__(self, model):
            self._model = model

        def forward(
            self,
            obs_tensor,
            *,
            action_mask,
            temperature,
            to_play,
            epsilon,
            ready_env_id,
        ):
            self._model.initial_inference(obs_tensor)
            self._model.recurrent_inference(obs_tensor, np.zeros((obs_tensor.shape[0], 1)))
            return {row: {"action": 0, "searched_value": 0.0} for row in ready_env_id}

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()
            self.collect_mode = FakeCollectMode(self._model)

    policy = FakePolicy()
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)

    result = probe.run(observation, action_mask)

    telemetry = result.telemetry
    assert telemetry["lightzero_consumer_model_timer_status"] == "installed"
    assert telemetry["lightzero_consumer_model_initial_inference_calls"] == 1.0
    assert telemetry["lightzero_consumer_model_recurrent_inference_calls"] == 1.0
    assert telemetry["lightzero_consumer_model_total_sec"] >= 0.0
    assert telemetry["lightzero_consumer_collect_forward_non_model_sec"] >= 0.0


def test_lightzero_collect_forward_stack_probe_times_mcts_search(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModel:
        def initial_inference(self, obs_tensor):
            return {"latent_state": obs_tensor}

        def recurrent_inference(self, latent_state, action):
            return {"latent_state": latent_state, "action": action}

    class FakeMCTS:
        def __init__(self, model):
            self._model = model
            self.calls = 0

        def search(self, obs_tensor):
            self.calls += 1
            self._model.recurrent_inference(obs_tensor, np.zeros((obs_tensor.shape[0], 1)))
            return None

    class FakeCollectMode:
        def __init__(self, policy):
            self._policy = policy

        def forward(
            self,
            obs_tensor,
            *,
            action_mask,
            temperature,
            to_play,
            epsilon,
            ready_env_id,
        ):
            self._policy._model.initial_inference(obs_tensor)
            self._policy._mcts_collect.search(obs_tensor)
            return {row: {"action": 0, "searched_value": 0.0} for row in ready_env_id}

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()
            self._mcts_collect = FakeMCTS(self._model)
            self.collect_mode = FakeCollectMode(self)

    policy = FakePolicy()
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)

    result = probe.run(observation, action_mask)

    telemetry = result.telemetry
    assert policy._mcts_collect.calls == 1
    assert telemetry["lightzero_consumer_mcts_timer_status"] == "installed"
    assert telemetry["lightzero_consumer_mcts_search_calls"] == 1.0
    assert telemetry["lightzero_consumer_mcts_search_sec"] >= 0.0
    assert telemetry["lightzero_consumer_mcts_search_non_model_sec"] >= 0.0
    assert telemetry["lightzero_consumer_collect_forward_outside_mcts_sec"] >= 0.0


def test_lightzero_collect_forward_stack_probe_labels_pure_policy(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeCollectMode:
        def forward(
            self,
            obs_tensor,
            *,
            action_mask,
            temperature,
            to_play,
            epsilon,
            ready_env_id,
        ):
            return {row: {"action": 0, "searched_value": 0.0} for row in ready_env_id}

    class FakePolicy:
        def __init__(self):
            self.collect_mode = FakeCollectMode()

    probe = _LightZeroCollectForwardStackProbe(
        policy=FakePolicy(),
        policy_metadata={
            "policy_class": "fake.Policy",
            "surface": {"env_variant": "test"},
            "collect_with_pure_policy": True,
        },
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)

    telemetry = probe.run(observation, action_mask).telemetry

    assert telemetry["lightzero_consumer_collect_with_pure_policy"] is True
    assert telemetry["lightzero_consumer_cpu_tree_included"] == 0.0
    assert telemetry["model_eval_count"] == 2.0


def test_lightzero_collect_forward_stack_probe_filters_zero_mask_roots(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeCollectMode:
        def __init__(self):
            self.call = None

        def forward(
            self,
            obs_tensor,
            *,
            action_mask,
            temperature,
            to_play,
            epsilon,
            ready_env_id,
        ):
            self.call = {
                "obs_shape": list(obs_tensor.shape),
                "mask": np.asarray(action_mask).copy(),
                "to_play": list(to_play),
                "ready_env_id": np.asarray(ready_env_id).astype(int).tolist(),
            }
            return {
                str(env_id): {
                    "action": int(env_id),
                    "searched_value": float(env_id),
                    "visit_count_distribution": [1.0, 0.0, 0.0],
                }
                for env_id in np.asarray(ready_env_id).astype(int).tolist()
            }

    class FakePolicy:
        def __init__(self):
            self.collect_mode = FakeCollectMode()

    policy = FakePolicy()
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
    )
    observation = np.ones((2, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((2, 2, 3), dtype=bool)
    action_mask[0, 1, :] = False
    action_mask[1, 0, :] = False

    result = probe.run(observation, action_mask)

    call = policy.collect_mode.call
    assert call["obs_shape"] == [2, 4, 64, 64]
    assert call["ready_env_id"] == [0, 1]
    assert call["to_play"] == [-1, -1]
    np.testing.assert_array_equal(call["mask"], np.ones((2, 3), dtype=np.float32))
    assert result.telemetry["lightzero_total_root_count"] == 4.0
    assert result.telemetry["lightzero_root_count"] == 2.0
    assert result.telemetry["lightzero_filtered_zero_mask_root_count"] == 2.0
    assert result.telemetry["lightzero_first_actions"] == [0, 1]


def test_lightzero_mcts_arrays_boundary_facade_returns_compact_arrays(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeCollectMode:
        def forward(
            self,
            obs_tensor,
            *,
            action_mask,
            temperature,
            to_play,
            epsilon,
            ready_env_id,
        ):
            assert list(to_play) == [-1, -1]
            assert np.asarray(ready_env_id).astype(int).tolist() == [0, 1]
            assert list(obs_tensor.shape) == [2, 4, 64, 64]
            return {
                0: {
                    "action": 1,
                    "searched_value": 0.25,
                    "predicted_value": 0.125,
                    "predicted_policy_logits": [0.1, 0.9],
                    "visit_count_distribution": [1.0, 3.0],
                },
                1: {
                    "action": 2,
                    "searched_value": -0.5,
                    "predicted_value": -0.25,
                    "predicted_policy_logits": [4.0, 5.0, 6.0],
                    "visit_count_distribution": [4.0, 1.0, 3.0],
                },
            }

    class FakePolicy:
        collect_mode = FakeCollectMode()

    probe = _LightZeroCollectForwardStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)
    action_mask[0, 0, 2] = False
    action_mask[0, 1, 0] = False

    telemetry = probe.run(observation, action_mask).telemetry

    assert probe.backend_name == "lightzero_mcts_arrays_boundary_consumer"
    assert telemetry["lightzero_mcts_arrays_boundary_enabled"] is True
    assert telemetry["lightzero_mcts_arrays_boundary_semantics"] == (
        "stock_lightzero_mcts_arrays_facade"
    )
    assert telemetry["lightzero_mcts_arrays_boundary_action_shape"] == [2]
    assert telemetry["lightzero_mcts_arrays_boundary_visit_shape"] == [2, ACTION_COUNT]
    assert telemetry["lightzero_mcts_arrays_boundary_searched_value_shape"] == [2]
    assert telemetry["lightzero_mcts_arrays_boundary_predicted_value_shape"] == [2]
    assert telemetry["lightzero_mcts_arrays_boundary_policy_logits_shape"] == [
        2,
        ACTION_COUNT,
    ]
    assert telemetry["lightzero_mcts_arrays_boundary_visit_present_count"] == 2.0
    assert telemetry["lightzero_mcts_arrays_boundary_root_value_count"] == 2.0
    assert telemetry["lightzero_mcts_arrays_boundary_compact_output_bytes"] > 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_public_output_bytes"] > 0.0
    assert telemetry["lightzero_first_actions"] == [1, 2]
    assert telemetry["lightzero_illegal_action_count"] == 0.0
    assert telemetry["model_eval_count"] == 18.0
    debug_arrays = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
    assert debug_arrays["included"] is True
    assert debug_arrays["actions"] == [1, 2]
    np.testing.assert_allclose(debug_arrays["predicted_values"], [0.125, -0.25])
    np.testing.assert_allclose(
        debug_arrays["policy_logits"],
        [[0.1, 0.9, 0.0], [4.0, 5.0, 6.0]],
    )
    np.testing.assert_allclose(
        debug_arrays["visit_distributions"][0],
        [0.25, 0.75, 0.0],
    )
    np.testing.assert_allclose(
        debug_arrays["visit_distributions"][1],
        [0.0, 0.25, 0.75],
    )


def test_lightzero_mcts_arrays_boundary_direct_ctree_returns_compact_arrays(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeTensor:
        def __init__(self, value):
            self._value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    policy_module = types.ModuleType("lzero.policy")
    policy_module.mz_network_output_unpack = lambda output: output
    policy_module.select_action = lambda visit_counts, *, temperature, deterministic: (
        int(np.argmax(np.asarray(visit_counts, dtype=np.float32))),
        0.0,
    )
    lzero_module = types.ModuleType("lzero")
    lzero_module.policy = policy_module
    monkeypatch.setitem(sys.modules, "lzero", lzero_module)
    monkeypatch.setitem(sys.modules, "lzero.policy", policy_module)

    class FakeModel:
        def __init__(self):
            self.eval_called = False
            self.last_input = None

        def eval(self):
            self.eval_called = True

        def initial_inference(self, obs_tensor):
            self.last_input = np.asarray(obs_tensor).copy()
            root_count = int(obs_tensor.shape[0])
            latent = FakeTensor(np.zeros((root_count, 2, 2), dtype=np.float32))
            reward = FakeTensor(np.zeros((root_count, 1), dtype=np.float32))
            pred_values = FakeTensor(np.asarray([[0.25], [-0.5]], dtype=np.float32))
            logits = FakeTensor(
                np.asarray(
                    [[0.1, 0.7, -0.2], [-0.4, 0.2, 0.9]],
                    dtype=np.float32,
                )
            )
            return latent, reward, pred_values, logits

        def recurrent_inference(self, latent_state, action):
            return latent_state, action

    class FakeRoots:
        def __init__(self, root_count, legal_actions):
            self.root_count = int(root_count)
            self.legal_actions = [list(actions) for actions in legal_actions]
            self.prepare_call = None

        def prepare(self, root_noise_weight, noises, reward_roots, policy_logits, to_play):
            self.prepare_call = {
                "root_noise_weight": float(root_noise_weight),
                "noise_count": len(noises),
                "reward_roots": reward_roots,
                "policy_logits": policy_logits,
                "to_play": list(to_play),
            }

        def get_distributions(self):
            return [[1.0, 3.0], [2.0, 6.0]]

        def get_values(self):
            return [0.5, -1.0]

    class FakeMCTS:
        last_roots = None

        @classmethod
        def roots(cls, root_count, legal_actions):
            cls.last_roots = FakeRoots(root_count, legal_actions)
            return cls.last_roots

        def __init__(self):
            self.search_call = None

        def search(self, roots, model, latent_state_roots, to_play):
            self.search_call = {
                "roots": roots,
                "latent_shape": list(np.asarray(latent_state_roots).shape),
                "to_play": list(to_play),
            }
            model.recurrent_inference(latent_state_roots, np.zeros((roots.root_count, 1)))

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()
            self._collect_model = self._model
            self._mcts_collect = FakeMCTS()
            self._cfg = types.SimpleNamespace(
                root_dirichlet_alpha=0.3,
                root_noise_weight=0.25,
                eps=types.SimpleNamespace(eps_greedy_exploration_in_collect=False),
            )

        def inverse_scalar_transform_handle(self, value):
            return value

    policy = FakePolicy()
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    )
    observation = np.ones((2, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.zeros((2, 2, ACTION_COUNT), dtype=bool)
    action_mask[0, 0, :2] = True
    action_mask[1, 1, 1:] = True

    telemetry = probe.run(observation, action_mask).telemetry

    roots = FakeMCTS.last_roots
    assert roots is not None
    assert probe.backend_name == "lightzero_mcts_arrays_direct_ctree_consumer"
    assert telemetry["lightzero_mcts_arrays_boundary_semantics"] == (
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_DIRECT_SEMANTICS
    )
    assert telemetry["lightzero_mcts_arrays_boundary_impl"] == (
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE
    )
    assert telemetry["lightzero_mcts_arrays_boundary_input_mode"] == "host_uint8"
    assert roots.legal_actions == [[0, 1], [1, 2]]
    assert policy._model.eval_called is True
    assert roots.prepare_call["to_play"] == [-1, -1]
    assert roots.prepare_call["noise_count"] == 2
    assert policy._mcts_collect.search_call["latent_shape"] == [2, 2, 2]
    assert policy._mcts_collect.search_call["to_play"] == [-1, -1]
    assert telemetry["lightzero_total_root_count"] == 4.0
    assert telemetry["lightzero_root_count"] == 2.0
    assert telemetry["lightzero_filtered_zero_mask_root_count"] == 2.0
    assert telemetry["lightzero_first_actions"] == [1, 2]
    assert telemetry["lightzero_rows_sample"] == [0, 1]
    assert telemetry["lightzero_players_sample"] == [0, 1]
    assert telemetry["lightzero_mcts_arrays_boundary_action_shape"] == [2]
    assert telemetry["lightzero_mcts_arrays_boundary_visit_shape"] == [2, ACTION_COUNT]
    assert telemetry["lightzero_mcts_arrays_boundary_searched_value_shape"] == [2]
    assert telemetry["lightzero_mcts_arrays_boundary_predicted_value_shape"] == [2]
    assert telemetry["lightzero_mcts_arrays_boundary_policy_logits_shape"] == [
        2,
        ACTION_COUNT,
    ]
    assert telemetry["lightzero_mcts_arrays_boundary_public_output_bytes"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_compact_output_bytes"] > 0.0
    assert telemetry["host_to_device_bytes"] == float(2 * 4 * 64 * 64)
    assert telemetry["lightzero_mcts_arrays_boundary_obs_h2d_bytes"] == float(2 * 4 * 64 * 64)
    assert telemetry["lightzero_mcts_arrays_boundary_mask_h2d_bytes"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_action_d2h_bytes"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_replay_payload_d2h_bytes"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_root_observation_copy_bytes"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_python_rows_materialized"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_rnd_materialized_rows"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_input_freshness"] == "fresh"
    assert telemetry["lightzero_mcts_arrays_boundary_model_output_d2h_sec"] >= 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_model_output_d2h_bytes"] > 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_non_model_sec"] >= 0.0
    assert telemetry["lightzero_consumer_collect_forward_non_model_sec"] == 0.0
    assert telemetry["lightzero_consumer_input_mode"] == "host_uint8"
    assert telemetry["lightzero_policy_forward_calls"] == 0.0
    assert telemetry["lightzero_illegal_action_count"] == 0.0
    assert telemetry["model_eval_count"] == 18.0

    float_policy = FakePolicy()
    float_probe = _LightZeroCollectForwardStackProbe(
        policy=float_policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        input_mode="host_float32",
    )

    float_telemetry = float_probe.run(
        np.full((2, 2, 4, 64, 64), 255, dtype=np.uint8),
        action_mask,
    ).telemetry

    assert float_telemetry["lightzero_mcts_arrays_boundary_input_mode"] == "host_float32"
    assert float_telemetry["lightzero_consumer_host_prenormalize_sec"] >= 0.0
    assert float_telemetry["host_to_device_bytes"] == float(2 * 4 * 64 * 64 * 4)
    assert (
        float_telemetry["lightzero_mcts_arrays_boundary_total_sec"]
        >= (float_telemetry["lightzero_mcts_arrays_boundary_host_prenormalize_sec"])
    )
    assert float_policy._model.last_input is not None
    np.testing.assert_allclose(float_policy._model.last_input, 1.0)

    compact_policy = FakePolicy()
    compact_probe = _LightZeroCollectForwardStackProbe(
        policy=compact_policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    )
    compact_action_mask = np.zeros((2, 2, ACTION_COUNT), dtype=bool)
    compact_action_mask[0, 0, :2] = True
    compact_action_mask[0, 1, 1:] = True
    compact_action_mask[1, 0, :] = True
    compact_action_mask[1, 1, :] = True
    compact_batch = HybridCompactBatch(
        observation=np.ones((2, 2, 4, 64, 64), dtype=np.uint8),
        action_mask=compact_action_mask,
        reward=np.zeros((2, 2), dtype=np.float32),
        final_reward_map=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, True], dtype=np.bool_),
        policy_env_id=np.asarray([0, 1, 2, 3], dtype=np.int32),
        policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
        target_reward=np.zeros((4, 1), dtype=np.float32),
        done_root=np.asarray([False, False, True, True], dtype=np.bool_),
        to_play=np.asarray([-1, -1, -1, -1], dtype=np.int64),
        active_root_mask=np.asarray([True, True, False, False], dtype=np.bool_),
        final_observation=np.zeros((2, 2, 4, 64, 64), dtype=np.uint8),
        final_observation_row_mask=np.asarray([False, True], dtype=np.bool_),
        terminal_row_mask=np.asarray([False, True], dtype=np.bool_),
        autoreset_row_mask=np.asarray([False, True], dtype=np.bool_),
        terminal_global_rows=np.asarray([1], dtype=np.int32),
        autoreset_global_rows=np.asarray([1], dtype=np.int32),
        episode_step=np.asarray([1, 1], dtype=np.int32),
        elapsed_ms=np.asarray([16.0, 16.0], dtype=np.float64),
        round_id=np.asarray([0, 0], dtype=np.int32),
        alive=np.asarray([[True, True], [True, True]], dtype=np.bool_),
        joint_action=np.asarray([[0, 1], [2, 0]], dtype=np.int16),
    )

    compact_telemetry = compact_probe.run_compact_batch(compact_batch).telemetry

    compact_roots = FakeMCTS.last_roots
    assert compact_roots is not None
    assert compact_roots.legal_actions == [[0, 1], [1, 2]]
    assert compact_roots.prepare_call["to_play"] == [-1, -1]
    assert compact_policy._mcts_collect.search_call["to_play"] == [-1, -1]
    assert compact_telemetry["compact_batch_contract"] == "compact_row_player_sidecar_v1"
    assert compact_telemetry["lightzero_total_root_count"] == 4.0
    assert compact_telemetry["lightzero_root_count"] == 2.0
    assert compact_telemetry["lightzero_filtered_zero_mask_root_count"] == 2.0
    assert compact_telemetry["lightzero_compact_batch_active_root_count"] == 2.0
    assert compact_telemetry["lightzero_compact_batch_done_root_count"] == 2.0
    assert compact_telemetry["lightzero_compact_batch_terminal_count"] == 1.0
    assert compact_telemetry["lightzero_compact_batch_autoreset_count"] == 1.0
    assert compact_telemetry["lightzero_compact_batch_final_observation_present"] is True
    assert compact_telemetry["lightzero_compact_batch_final_observation_rows"] == 1.0
    assert compact_telemetry["lightzero_first_actions"] == [1, 2]
    assert compact_telemetry["lightzero_rows_sample"] == [0, 0]
    assert compact_telemetry["lightzero_players_sample"] == [0, 1]
    assert compact_telemetry["compact_service_contract_v1_enabled"] is True
    assert compact_telemetry["compact_service_contract_v1_validation_sec"] >= 0.0
    assert compact_telemetry["compact_service_contract_v1_contract_id"] == (
        "curvyzero_compact_search_replay_service/v1"
    )
    assert compact_telemetry["compact_service_root_batch_schema_id"] == (
        "curvyzero_compact_root_batch/v1"
    )
    assert compact_telemetry["compact_service_search_result_schema_id"] == (
        "curvyzero_compact_search_result/v1"
    )
    assert compact_telemetry["compact_service_root_count"] == 4.0
    assert compact_telemetry["compact_service_active_root_count"] == 2.0
    assert compact_telemetry["compact_service_selected_action_checksum"] == 3.0
    assert compact_telemetry["compact_service_visit_policy_checksum"] == pytest.approx(2.0)
    assert compact_telemetry["compact_service_identity_checksum"] == 1.0


def test_lightzero_compact_search_service_adapter_preserves_root_identity():
    class FakeProbe:
        _arrays_boundary_impl = "unit_test_direct_ctree_adapter"
        _num_simulations = 4
        backend_name = "unit_test_fake_probe"
        semantics = "unit_test_fake_probe_semantics"

        def __init__(self) -> None:
            self._last_direct_mcts_arrays = None
            self.last_observation_shape = None
            self.last_action_mask_shape = None

        def run(self, observation, action_mask):
            self.last_observation_shape = tuple(observation.shape)
            self.last_action_mask_shape = tuple(action_mask.shape)
            flat_mask = np.asarray(action_mask, dtype=bool).reshape(-1, ACTION_COUNT)
            active_roots = np.flatnonzero(flat_mask.any(axis=1))
            selected = np.asarray([1, 2], dtype=np.int16)
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            self._last_direct_mcts_arrays = {
                "selected_action": selected,
                "visit_policy": visit_policy,
                "root_value": np.asarray([-0.5, 0.25], dtype=np.float32),
                "predicted_value": np.asarray([-0.25, 0.125], dtype=np.float32),
                "predicted_policy_logits": np.ones(
                    (active_roots.size, ACTION_COUNT),
                    dtype=np.float32,
                ),
                "search_impl": self._arrays_boundary_impl,
                "actual_search_simulations": self._num_simulations,
            }
            return HybridBatchedStackProbeResult(
                telemetry={
                    "lightzero_mcts_arrays_boundary_total_sec": 0.25,
                    "lightzero_mcts_arrays_boundary_search_sec": 0.1,
                    "lightzero_mcts_arrays_boundary_debug_arrays": {"skip": [1, 2, 3]},
                }
            )

    compact_batch = HybridCompactBatch(
        observation=np.ones((2, 2, 4, 64, 64), dtype=np.uint8),
        action_mask=np.asarray(
            [
                [[False, False, False], [False, True, True]],
                [[False, False, False], [True, True, True]],
            ],
            dtype=np.bool_,
        ),
        reward=np.zeros((2, 2), dtype=np.float32),
        final_reward_map=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        policy_env_id=np.asarray([101, 103, 107, 109], dtype=np.int32),
        policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
        target_reward=np.zeros((4, 1), dtype=np.float32),
        done_root=np.asarray([False, False, False, False], dtype=np.bool_),
        to_play=np.asarray([-1, -1, -1, -1], dtype=np.int64),
        active_root_mask=np.asarray([False, True, False, True], dtype=np.bool_),
        final_observation=np.zeros((2, 2, 4, 64, 64), dtype=np.uint8),
        final_observation_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_row_mask=np.asarray([False, False], dtype=np.bool_),
        autoreset_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.asarray([1, 1], dtype=np.int32),
        elapsed_ms=np.asarray([16.0, 16.0], dtype=np.float64),
        round_id=np.asarray([0, 0], dtype=np.int32),
        alive=np.asarray([[True, True], [True, True]], dtype=np.bool_),
        joint_action=np.asarray([[0, 1], [0, 2]], dtype=np.int16),
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_direct_ctree_adapter",
        copy_observation=False,
    )
    fake_probe = FakeProbe()
    service = _LightZeroCollectForwardCompactSearchService(fake_probe)

    assert isinstance(service, CompactSearchServiceV1)
    result = service.run(root_batch)

    assert fake_probe.last_observation_shape == (2, 2, 4, 64, 64)
    assert fake_probe.last_action_mask_shape == (2, 2, ACTION_COUNT)
    np.testing.assert_array_equal(result.root_index, np.asarray([1, 3]))
    np.testing.assert_array_equal(result.policy_env_id, np.asarray([103, 109]))
    np.testing.assert_array_equal(result.env_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(result.player, np.asarray([1, 1]))
    np.testing.assert_array_equal(result.selected_action, np.asarray([1, 2]))
    np.testing.assert_allclose(result.visit_policy.sum(axis=1), 1.0)
    assert result.metadata["compact_search_service_adapter"] is True
    assert result.metadata["profile_telemetry"]["lightzero_mcts_arrays_boundary_total_sec"] == 0.25
    assert "lightzero_mcts_arrays_boundary_debug_arrays" not in result.metadata["profile_telemetry"]


def test_lightzero_array_ceiling_compact_search_service_adapter_preserves_identity():
    class FakeArrayCeilingProbe:
        _mode = "mock_search_service"
        _num_simulations = 0
        backend_name = "unit_test_fake_array_probe"
        semantics = "unit_test_fake_array_probe_semantics"

        def __init__(self) -> None:
            self._last_compact_search_arrays = None
            self.last_observation_shape = None
            self.last_action_mask_shape = None

        def run(self, observation, action_mask):
            self.last_observation_shape = tuple(observation.shape)
            self.last_action_mask_shape = tuple(action_mask.shape)
            flat_mask = np.asarray(action_mask, dtype=bool).reshape(-1, ACTION_COUNT)
            active_roots = np.flatnonzero(flat_mask.any(axis=1))
            selected = np.asarray([1, 2], dtype=np.int16)
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            self._last_compact_search_arrays = {
                "selected_action": selected,
                "visit_policy": visit_policy,
                "root_value": np.asarray([0.125, 0.625], dtype=np.float32),
                "search_impl": self._mode,
                "actual_search_simulations": 0,
            }
            return HybridBatchedStackProbeResult(
                telemetry={
                    "lightzero_array_ceiling_total_sec": 0.5,
                    "lightzero_array_ceiling_search_update_sec": 0.25,
                }
            )

    compact_batch = HybridCompactBatch(
        observation=np.ones((2, 2, 4, 64, 64), dtype=np.uint8),
        action_mask=np.asarray(
            [
                [[False, False, False], [False, True, True]],
                [[False, False, False], [True, True, True]],
            ],
            dtype=np.bool_,
        ),
        reward=np.zeros((2, 2), dtype=np.float32),
        final_reward_map=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        policy_env_id=np.asarray([201, 203, 207, 209], dtype=np.int32),
        policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
        target_reward=np.zeros((4, 1), dtype=np.float32),
        done_root=np.asarray([False, False, False, False], dtype=np.bool_),
        to_play=np.asarray([-1, -1, -1, -1], dtype=np.int64),
        active_root_mask=np.asarray([False, True, False, True], dtype=np.bool_),
        final_observation=np.zeros((2, 2, 4, 64, 64), dtype=np.uint8),
        final_observation_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_row_mask=np.asarray([False, False], dtype=np.bool_),
        autoreset_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.asarray([1, 1], dtype=np.int32),
        elapsed_ms=np.asarray([16.0, 16.0], dtype=np.float64),
        round_id=np.asarray([0, 0], dtype=np.int32),
        alive=np.asarray([[True, True], [True, True]], dtype=np.bool_),
        joint_action=np.asarray([[0, 1], [0, 2]], dtype=np.int16),
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_array_ceiling_adapter",
        copy_observation=False,
    )
    fake_probe = FakeArrayCeilingProbe()
    service = _LightZeroArrayCeilingCompactSearchService(fake_probe)

    assert isinstance(service, CompactSearchServiceV1)
    result = service.run(root_batch)

    assert fake_probe.last_observation_shape == (2, 2, 4, 64, 64)
    assert fake_probe.last_action_mask_shape == (2, 2, ACTION_COUNT)
    np.testing.assert_array_equal(result.root_index, np.asarray([1, 3]))
    np.testing.assert_array_equal(result.policy_env_id, np.asarray([203, 209]))
    np.testing.assert_array_equal(result.env_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(result.player, np.asarray([1, 1]))
    np.testing.assert_array_equal(result.selected_action, np.asarray([1, 2]))
    np.testing.assert_allclose(result.visit_policy.sum(axis=1), 1.0)
    assert result.metadata["compact_search_service_adapter"] is True
    assert result.metadata["array_ceiling_mode"] == "mock_search_service"
    assert result.metadata["profile_telemetry"]["lightzero_array_ceiling_total_sec"] == 0.5


def test_real_direct_ctree_compact_service_drives_next_step_and_matches_rows():
    pytest.importorskip("lzero")
    pytest.importorskip("ding")
    import random

    torch = pytest.importorskip("torch")

    seed = 20260523
    num_simulations = 2
    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=False,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=8,
        max_ticks=8,
        root_noise_weight=0.0,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = 0.0
    policy._cfg.eps.eps_greedy_exploration_in_collect = False

    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=1,
            warmup_steps=0,
            seed=seed,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=_NoopCompactBatchProbe(),
    )
    step0 = manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))
    assert step0.compact_batch is not None

    random.seed(seed + num_simulations)
    np.random.seed(seed + num_simulations)
    torch.manual_seed(seed + num_simulations)
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata=policy_meta,
        num_simulations=num_simulations,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        input_mode="host_uint8",
    )
    service = _LightZeroCollectForwardCompactSearchService(probe)
    root_batch = build_compact_root_batch_v1(
        step0.compact_batch,
        search_lane="unit_test_real_direct_ctree_compact_service_closed_loop",
        copy_observation=False,
    )
    search_result = service.run(root_batch)
    search_arrays = probe._last_direct_mcts_arrays
    assert search_arrays is not None
    assert search_result.root_index.size > 0

    joint_action = _joint_action_from_compact_search_arrays(
        step0.compact_batch,
        search_arrays,
    )
    step1 = manager.step(joint_action)
    np.testing.assert_array_equal(step1.payload["joint_action"], joint_action)

    chunk = _two_record_replay_chunk_from_hybrid_steps(step0, step1)
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        step0.compact_batch,
        root_batch,
        search_result,
        record_index=0,
        next_joint_action=step1.payload["joint_action"],
        next_reward=step1.reward,
        next_done=step1.done,
        next_terminated=step1.done,
        next_truncated=np.zeros_like(step1.done, dtype=np.bool_),
        next_final_reward_map=step1.reward,
        next_final_observation_row_mask=step1.done,
        policy_source="real_direct_ctree_compact_service_closed_loop_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    immediate_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        step0.compact_batch,
        selected_action=search_result.selected_action,
        visit_policy=search_result.visit_policy,
        root_value=search_result.root_value,
        record_index=0,
        policy_source="real_direct_ctree_compact_service_closed_loop_test",
    )

    _assert_source_state_rows_equal(
        materialized_rows,
        immediate_rows,
        fields=(
            "action",
            "action_mask",
            "policy_target",
            "root_value",
            "reward",
            "final_reward",
            "done",
            "terminated",
            "truncated",
            "env_row",
            "player",
            "observation",
            "next_observation",
            "record_index",
            "next_record_index",
            "policy_row",
            "to_play",
        ),
    )
    immediate_batch = build_source_state_multiplayer_sample_batch_v0(
        immediate_rows,
        batch_size=int(immediate_rows.action.shape[0]),
        seed=seed,
    )
    materialized_batch = build_source_state_multiplayer_sample_batch_v0(
        materialized_rows,
        batch_size=int(materialized_rows.action.shape[0]),
        seed=seed,
    )
    _assert_source_state_rows_equal(
        materialized_batch,
        immediate_batch,
        fields=(
            "row_id",
            "observation",
            "action",
            "action_mask",
            "policy_target",
            "root_value",
            "reward",
            "final_reward",
            "done",
            "terminated",
            "truncated",
            "next_observation",
            "env_row",
            "player",
            "record_index",
            "next_record_index",
            "policy_row",
            "to_play",
        ),
    )


def test_compact_torch_search_service_drives_next_step_and_matches_rows():
    torch = pytest.importorskip("torch")

    class FakeOutput:
        def __init__(self, *, logits, latent) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.recurrent_calls = 0

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            batch = int(obs_tensor.shape[0])
            row = torch.arange(batch, dtype=torch.float32)
            logits = torch.stack(
                [
                    torch.zeros_like(row),
                    torch.where((row.long() % 2) == 0, torch.full_like(row, 4.0), row),
                    torch.where((row.long() % 2) == 1, torch.full_like(row, 4.0), row),
                ],
                dim=1,
            )
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits=logits, latent=latent)

        def recurrent_inference(self, latent_state, _actions):
            self.recurrent_calls += 1
            batch = int(latent_state.shape[0])
            return FakeOutput(
                logits=torch.zeros((batch, ACTION_COUNT), dtype=torch.float32),
                latent=latent_state + 1.0,
            )

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = types.SimpleNamespace(
                pb_c_base=19652,
                pb_c_init=1.25,
                discount_factor=0.997,
                root_noise_weight=0.0,
                root_dirichlet_alpha=0.3,
                value_delta_max=0.01,
            )

    seed = 20260523
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=1,
            warmup_steps=0,
            seed=seed,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=_NoopCompactBatchProbe(),
    )
    step0 = manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))
    assert step0.compact_batch is not None
    root_batch = build_compact_root_batch_v1(
        step0.compact_batch,
        search_lane="unit_test_compact_torch_search_service_closed_loop",
        copy_observation=False,
    )
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    search_result = service.run(root_batch)
    assert search_result.root_index.size > 0
    assert service._model.initial_calls == 1
    assert service._model.recurrent_calls == 1

    joint_action = np.zeros(step0.compact_batch.joint_action.shape, dtype=np.int16)
    for index, action in enumerate(search_result.selected_action):
        joint_action[int(search_result.env_row[index]), int(search_result.player[index])] = int(
            action
        )
    step1 = manager.step(joint_action)
    np.testing.assert_array_equal(step1.payload["joint_action"], joint_action)

    chunk = _two_record_replay_chunk_from_hybrid_steps(step0, step1)
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        step0.compact_batch,
        root_batch,
        search_result,
        record_index=0,
        next_joint_action=step1.payload["joint_action"],
        next_reward=step1.reward,
        next_done=step1.done,
        next_terminated=step1.done,
        next_truncated=np.zeros_like(step1.done, dtype=np.bool_),
        next_final_reward_map=step1.reward,
        next_final_observation_row_mask=step1.done,
        policy_source="compact_torch_search_service_closed_loop_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    immediate_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        step0.compact_batch,
        selected_action=search_result.selected_action,
        visit_policy=search_result.visit_policy,
        root_value=search_result.root_value,
        record_index=0,
        policy_source="compact_torch_search_service_closed_loop_test",
    )

    _assert_source_state_rows_equal(
        materialized_rows,
        immediate_rows,
        fields=(
            "action",
            "action_mask",
            "policy_target",
            "root_value",
            "reward",
            "final_reward",
            "done",
            "terminated",
            "truncated",
            "env_row",
            "player",
            "observation",
            "next_observation",
            "record_index",
            "next_record_index",
            "policy_row",
            "to_play",
        ),
    )


def test_array_ceiling_compact_torch_search_service_mode_owns_compact_service_run():
    torch = pytest.importorskip("torch")

    class FakeOutput:
        def __init__(self, *, batch: int) -> None:
            self.policy_logits = torch.tensor([[0.0, 4.0, 0.0]], dtype=torch.float32).repeat(
                batch,
                1,
            )
            self.value = torch.zeros((batch, 1), dtype=torch.float32)
            self.reward = torch.zeros((batch, 1), dtype=torch.float32)
            self.latent_state = torch.zeros((batch, 2), dtype=torch.float32)

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.recurrent_calls = 0

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(batch=int(obs_tensor.shape[0]))

        def recurrent_inference(self, latent_state, _actions):
            self.recurrent_calls += 1
            return FakeOutput(batch=int(latent_state.shape[0]))

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = types.SimpleNamespace(
                pb_c_base=19652,
                pb_c_init=1.25,
                discount_factor=0.997,
                root_noise_weight=0.0,
                root_dirichlet_alpha=0.3,
                value_delta_max=0.01,
            )

    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=1,
            actor_count=1,
            steps=1,
            warmup_steps=0,
            seed=20260523,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=_NoopCompactBatchProbe(),
    )
    step0 = manager.step(np.asarray([[0, 1]], dtype=np.int16))
    assert step0.compact_batch is not None
    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=2,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE,
    )

    telemetry = probe.run_compact_batch(step0.compact_batch).telemetry

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 2
    assert telemetry["lightzero_array_ceiling_mode"] == (
        boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE
    )
    assert telemetry["lightzero_array_ceiling_semantics"] == (
        "compact_torch_search_service_profile_not_lightzero_ctree"
    )
    assert telemetry["compact_service_contract_v1_enabled"] is True
    assert telemetry["compact_service_search_impl"] == "compact_torch_device_tree_fixed_shape_v0"
    assert telemetry["lightzero_array_ceiling_tensor_prepare_sec"] >= 0.0
    assert telemetry["lightzero_array_ceiling_initial_inference_sec"] >= 0.0
    assert telemetry["lightzero_array_ceiling_search_update_sec"] >= 0.0
    assert telemetry["lightzero_array_ceiling_readback_sec"] >= 0.0
    assert telemetry["compact_torch_search_service_initial_inference_enqueue_sec"] >= 0.0
    assert telemetry["compact_torch_search_service_initial_inference_sync_sec"] >= 0.0
    assert telemetry["compact_torch_search_service_initial_inference_cuda_event_sec"] >= 0.0
    assert (
        telemetry["compact_torch_search_service_initial_inference_cuda_event_status"] == "disabled"
    )
    assert telemetry["compact_torch_search_service_tree_recurrent_inference_enqueue_sec"] >= 0.0
    assert telemetry["compact_torch_search_service_tree_recurrent_inference_cuda_event_sec"] >= 0.0
    assert (
        telemetry["compact_torch_search_service_tree_recurrent_inference_cuda_event_status"]
        == "disabled"
    )
    assert telemetry["compact_torch_search_service_tree_sync_sec"] >= 0.0
    assert telemetry["compact_torch_search_service_tree_cuda_event_sec"] >= 0.0
    assert telemetry["compact_torch_search_service_tree_cuda_event_status"] == "disabled"
    assert telemetry["compact_torch_search_service_timing_mode"] == "host_phase_sync"
    assert telemetry["compact_torch_search_service_cuda_event_timing_enabled"] == 0.0
    assert telemetry["compact_torch_search_service_initial_sync_enabled"] == 1.0
    assert telemetry["lightzero_array_ceiling_obs_h2d_bytes"] > 0.0
    assert telemetry["lightzero_array_ceiling_mask_h2d_bytes"] > 0.0
    assert telemetry["lightzero_array_ceiling_action_d2h_bytes"] > 0.0
    assert telemetry["lightzero_array_ceiling_replay_payload_d2h_bytes"] > 0.0
    assert telemetry["lightzero_array_ceiling_root_observation_copy_bytes"] == 0.0
    assert telemetry["lightzero_array_ceiling_python_rows_materialized"] == 0.0
    assert telemetry["lightzero_array_ceiling_rnd_materialized_rows"] == 0.0
    assert telemetry["lightzero_array_ceiling_resident_obs_reused"] == 0.0
    assert telemetry["compact_torch_search_service_tree_search_includes_recurrent"] is True
    assert telemetry["lightzero_array_ceiling_root_noise_weight"] == 0.0
    assert telemetry["lightzero_array_ceiling_compile_status"] in {
        "eligible",
        "not_requested",
        "fallback_precondition",
    }
    assert probe._last_compact_search_arrays is not None
    assert probe._last_compact_service_search_result is not None
    assert probe._last_compact_service_search_result.selected_action.tolist() == [1, 1]


def test_array_ceiling_compact_torch_search_service_rejects_mislabeled_input_modes():
    class FakePolicy:
        _model = object()

    with pytest.raises(ValueError, match="hybrid_resident_observation_search"):
        _LightZeroArrayCeilingStackProbe(
            policy=FakePolicy(),
            policy_metadata={},
            num_simulations=2,
            mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE,
            input_mode=boundary.LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE,
        )


def test_array_ceiling_compact_torch_search_service_drives_compact_rollout_slab():
    torch = pytest.importorskip("torch")

    class FakeOutput:
        def __init__(self, *, batch: int) -> None:
            self.policy_logits = torch.tensor([[0.0, 4.0, 0.0]], dtype=torch.float32).repeat(
                batch,
                1,
            )
            self.value = torch.zeros((batch, 1), dtype=torch.float32)
            self.reward = torch.zeros((batch, 1), dtype=torch.float32)
            self.latent_state = torch.zeros((batch, 2), dtype=torch.float32)

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.recurrent_calls = 0

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(batch=int(obs_tensor.shape[0]))

        def recurrent_inference(self, latent_state, _actions):
            self.recurrent_calls += 1
            return FakeOutput(batch=int(latent_state.shape[0]))

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = types.SimpleNamespace(
                pb_c_base=19652,
                pb_c_init=1.25,
                discount_factor=0.997,
                root_noise_weight=0.0,
                root_dirichlet_alpha=0.3,
                value_delta_max=0.01,
            )

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=2,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=boundary._LightZeroArrayCeilingCompactSearchService(probe),
        search_lane="compact_rollout_slab:compact_torch_search_service",
        policy_source="unit_test_compact_torch_search_service_slab",
    )

    batch0 = _compact_batch_for_service_adapter_test()
    step0 = slab.step(batch0)
    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 2
    assert step0.search_result is None
    assert step0.action_step is not None
    assert step0.action_step.metadata["search_impl"] == ("compact_torch_device_tree_fixed_shape_v0")
    assert step0.telemetry["compact_rollout_slab_search_impl"] == (
        "compact_torch_device_tree_fixed_shape_v0"
    )
    assert step0.telemetry["compact_rollout_slab_two_phase_search"] is True
    assert step0.telemetry["compact_rollout_slab_action_step_only"] is True
    assert step0.telemetry["compact_rollout_slab_search_service_total_sec"] >= 0.0
    assert step0.telemetry["compact_rollout_slab_search_service_initial_output_decode_sec"] >= 0.0
    assert step0.telemetry["compact_rollout_slab_search_service_tree_root_prior_build_sec"] >= 0.0
    assert step0.telemetry["compact_rollout_slab_search_service_action_accounted_sec"] >= 0.0
    assert "compact_rollout_slab_search_service_action_residual_sec" in step0.telemetry
    assert step0.telemetry["compact_rollout_slab_search_service_core_accounted_sec"] >= 0.0
    assert step0.telemetry["compact_rollout_slab_model_sec"] >= 0.0
    assert step0.telemetry["compact_rollout_slab_replay_payload_d2h_bytes"] == 0.0
    assert step0.telemetry["compact_rollout_slab_committed_replay_payload_d2h_bytes"] == 0
    assert step0.telemetry["compact_rollout_slab_python_rows_materialized"] == 0.0
    assert (
        step0.telemetry["compact_rollout_slab_profile_telemetry"][
            "compact_torch_search_service_not_lightzero_ctree"
        ]
        is True
    )
    assert (
        step0.telemetry["compact_rollout_slab_profile_telemetry"][
            "compact_torch_search_service_two_phase_action_only"
        ]
        is True
    )

    batch1 = replace(
        batch0,
        joint_action=step0.next_joint_action.astype(np.int16, copy=True),
    )
    step1 = slab.step(batch1)
    assert step1.committed_index_rows is not None
    assert step1.telemetry["compact_rollout_slab_committed_replay_payload_flushed"] is True
    assert step1.telemetry["compact_rollout_slab_committed_replay_payload_d2h_bytes"] > 0
    assert slab.committed_index_row_count == 2


def test_array_ceiling_fixed_shape_search_owner_mode_owns_compact_service_run():
    class FakePolicy:
        _model = object()

    compact_batch = _compact_batch_for_service_adapter_test()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=7,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER,
    )

    first = probe.run_compact_batch(compact_batch).telemetry
    second = probe.run_compact_batch(compact_batch).telemetry

    assert first["lightzero_array_ceiling_mode"] == (
        boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER
    )
    assert first["lightzero_array_ceiling_semantics"] == (
        "fixed_shape_search_owner_first_legal_profile_not_mcts"
    )
    assert first["compact_service_contract_v1_enabled"] is True
    assert first["compact_service_search_impl"] == FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    assert first["fixed_shape_search_owner_service_impl"] == (FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL)
    assert first["lightzero_array_ceiling_requested_simulations"] == 7.0
    assert first["lightzero_array_ceiling_actual_search_simulations"] == 0.0
    assert first["lightzero_array_ceiling_recurrent_inference_calls"] == 0.0
    assert first["lightzero_array_ceiling_real_ctree_calls"] == 0.0
    assert first["lightzero_array_ceiling_root_observation_copy_bytes"] == 0.0
    assert first["fixed_shape_batched_search_owner_ctree_calls"] == 0
    assert first["fixed_shape_batched_search_owner_tolist_calls"] == 0
    assert first["fixed_shape_batched_search_owner_per_sim_d2h_bytes"] == 0
    assert first["fixed_shape_batched_search_owner_first_legal_policy"] is True
    assert first["fixed_shape_batched_search_owner_buffer_reused"] is False
    assert second["fixed_shape_batched_search_owner_buffer_reused"] is True
    assert first["lightzero_array_ceiling_first_actions"] == [1, 0]
    assert probe._last_compact_search_arrays is not None
    assert probe._last_compact_service_search_result is not None
    assert probe._last_compact_service_search_result.selected_action.tolist() == [1, 0]
    assert probe._last_compact_search_arrays["service_impl"] == (
        FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    )


def test_array_ceiling_fixed_shape_search_owner_rejects_mislabeled_input_modes():
    class FakePolicy:
        _model = object()

    with pytest.raises(ValueError, match="host uint8 observations directly"):
        _LightZeroArrayCeilingStackProbe(
            policy=FakePolicy(),
            policy_metadata={},
            num_simulations=2,
            mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER,
            input_mode=boundary.LIGHTZERO_ARRAY_CEILING_INPUT_MODE_RESIDENT_TORCH_REUSE,
        )


def test_array_ceiling_fixed_shape_search_owner_drives_compact_rollout_slab():
    class FakePolicy:
        _model = object()

    probe = _LightZeroArrayCeilingStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=7,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_FIXED_SHAPE_SEARCH_OWNER,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=boundary._LightZeroArrayCeilingCompactSearchService(probe),
        search_lane="compact_rollout_slab:fixed_shape_search_owner",
        policy_source="unit_test_fixed_shape_search_owner_slab",
    )

    batch0 = _compact_batch_for_service_adapter_test()
    step0 = slab.step(batch0)
    np.testing.assert_array_equal(
        step0.next_joint_action,
        np.asarray([[0, 1], [0, 0]], dtype=np.int16),
    )
    assert step0.search_result is None
    assert step0.action_step is not None
    assert step0.action_step.metadata["search_impl"] == FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    assert step0.telemetry["compact_rollout_slab_search_impl"] == (
        FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    )
    assert step0.telemetry["compact_rollout_slab_two_phase_search"] is True
    assert step0.telemetry["compact_rollout_slab_action_step_only"] is True
    assert step0.telemetry["compact_rollout_slab_search_service_total_sec"] >= 0.0
    assert step0.telemetry["compact_rollout_slab_action_d2h_bytes"] > 0
    assert step0.telemetry["compact_rollout_slab_replay_payload_d2h_bytes"] == 0
    assert step0.telemetry["compact_rollout_slab_committed_replay_payload_d2h_bytes"] == 0
    assert step0.telemetry["compact_rollout_slab_python_rows_materialized"] == 0
    assert (
        step0.telemetry["compact_rollout_slab_profile_telemetry"][
            "fixed_shape_batched_search_owner_ctree_calls"
        ]
        == 0
    )
    assert (
        step0.telemetry["compact_rollout_slab_profile_telemetry"][
            "fixed_shape_batched_search_owner_first_legal_policy"
        ]
        is True
    )

    batch1 = replace(
        batch0,
        joint_action=step0.next_joint_action.astype(np.int16, copy=True),
    )
    step1 = slab.step(batch1)
    assert step1.committed_index_rows is not None
    assert step1.committed_index_rows.action.tolist() == [1, 0]
    assert step1.telemetry["compact_rollout_slab_committed_replay_payload_flushed"] is True
    assert step1.telemetry["compact_rollout_slab_committed_replay_payload_d2h_bytes"] > 0
    assert slab.committed_index_row_count == 2


def test_direct_ctree_probe_arrays_can_validate_without_second_probe_run():
    class FakeProbe:
        backend_name = "unit_test_backend"
        semantics = "unit_test_semantics"
        _arrays_boundary_impl = "unit_test_existing_arrays"
        _num_simulations = 5

    compact_batch = _compact_batch_for_service_adapter_test()
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_existing_arrays",
        copy_observation=False,
    )
    search_arrays = {
        "selected_action": np.asarray([1, 2], dtype=np.int16),
        "visit_policy": np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "root_value": np.asarray([0.25, -0.75], dtype=np.float32),
        "predicted_value": np.asarray([0.1, -0.2], dtype=np.float32),
        "predicted_policy_logits": np.asarray(
            [
                [-0.5, 1.0, 0.0],
                [0.1, -0.1, 1.2],
            ],
            dtype=np.float32,
        ),
        "search_impl": "unit_test_existing_arrays",
        "actual_search_simulations": 5,
    }

    result = compact_search_result_v1_from_arrays(
        root_batch,
        search_arrays,
        default_search_impl=FakeProbe._arrays_boundary_impl,
        default_num_simulations=FakeProbe._num_simulations,
        metadata={
            "profile_backend": FakeProbe.backend_name,
            "profile_semantics": FakeProbe.semantics,
            "compact_search_service_adapter": False,
            "validated_from_existing_probe_arrays": True,
        },
    )

    np.testing.assert_array_equal(result.root_index, np.asarray([1, 3]))
    np.testing.assert_array_equal(result.policy_env_id, np.asarray([103, 109]))
    np.testing.assert_array_equal(result.selected_action, np.asarray([1, 2]))
    np.testing.assert_allclose(result.predicted_value, np.asarray([0.1, -0.2]))
    assert result.metadata["validated_from_existing_probe_arrays"] is True
    assert result.metadata["compact_search_service_adapter"] is False


def _compact_batch_for_service_adapter_test() -> HybridCompactBatch:
    return HybridCompactBatch(
        observation=np.ones((2, 2, 4, 64, 64), dtype=np.uint8),
        action_mask=np.asarray(
            [
                [[False, False, False], [False, True, True]],
                [[False, False, False], [True, True, True]],
            ],
            dtype=np.bool_,
        ),
        reward=np.zeros((2, 2), dtype=np.float32),
        final_reward_map=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        policy_env_id=np.asarray([101, 103, 107, 109], dtype=np.int32),
        policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
        target_reward=np.zeros((4, 1), dtype=np.float32),
        done_root=np.asarray([False, False, False, False], dtype=np.bool_),
        to_play=np.asarray([-1, -1, -1, -1], dtype=np.int64),
        active_root_mask=np.asarray([False, True, False, True], dtype=np.bool_),
        final_observation=np.zeros((2, 2, 4, 64, 64), dtype=np.uint8),
        final_observation_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_row_mask=np.asarray([False, False], dtype=np.bool_),
        autoreset_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.asarray([1, 1], dtype=np.int32),
        elapsed_ms=np.asarray([16.0, 16.0], dtype=np.float64),
        round_id=np.asarray([0, 0], dtype=np.int32),
        alive=np.asarray([[True, True], [True, True]], dtype=np.bool_),
        joint_action=np.asarray([[0, 1], [0, 2]], dtype=np.int16),
    )


def _assert_source_state_rows_equal(left, right, *, fields: tuple[str, ...]) -> None:
    for field in fields:
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


class _NoopCompactBatchProbe:
    backend_name = "noop_compact_batch_probe"
    semantics = "build_compact_batch_without_search"

    def run_compact_batch(self, _batch: HybridCompactBatch) -> HybridBatchedStackProbeResult:
        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": 0.0,
                "host_to_device_sec": 0.0,
                "normalize_sec": 0.0,
                "device_sec": 0.0,
                "readback_sec": 0.0,
            }
        )


def test_direct_ctree_compact_output_can_feed_checked_target_rows(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    policy_module = types.ModuleType("lzero.policy")
    policy_module.mz_network_output_unpack = lambda output: output
    policy_module.select_action = lambda visit_counts, *, temperature, deterministic: (
        int(np.argmax(np.asarray(visit_counts, dtype=np.float32))),
        0.0,
    )
    lzero_module = types.ModuleType("lzero")
    lzero_module.policy = policy_module
    monkeypatch.setitem(sys.modules, "lzero", lzero_module)
    monkeypatch.setitem(sys.modules, "lzero.policy", policy_module)

    class FakeTensor:
        def __init__(self, value):
            self._value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    class FakeModel:
        def eval(self):
            return None

        def initial_inference(self, obs_tensor):
            root_count = int(obs_tensor.shape[0])
            pred_values = np.linspace(0.25, -0.5, root_count, dtype=np.float32).reshape(
                root_count,
                1,
            )
            logits = np.tile(
                np.asarray([[0.1, 0.7, -0.2]], dtype=np.float32),
                (root_count, 1),
            )
            return (
                FakeTensor(np.zeros((root_count, 2, 2), dtype=np.float32)),
                FakeTensor(np.zeros((root_count, 1), dtype=np.float32)),
                FakeTensor(pred_values),
                FakeTensor(logits),
            )

        def recurrent_inference(self, latent_state, action):
            return latent_state, action

    class FakeRoots:
        def __init__(self, root_count, legal_actions):
            self.root_count = int(root_count)
            self.legal_actions = [list(actions) for actions in legal_actions]

        def prepare(self, root_noise_weight, noises, reward_roots, policy_logits, to_play):
            return None

        def get_distributions(self):
            preferred = [1, 2, 0, 1]
            counts = np.ones((self.root_count, ACTION_COUNT), dtype=np.float32)
            for row in range(self.root_count):
                action = preferred[row % len(preferred)]
                if action not in self.legal_actions[row]:
                    action = self.legal_actions[row][0]
                counts[row, action] = 8.0
            return counts

        def get_values(self):
            return np.linspace(0.5, -1.0, self.root_count, dtype=np.float32).tolist()

    class FakeMCTS:
        @classmethod
        def roots(cls, root_count, legal_actions):
            return FakeRoots(root_count, legal_actions)

        def search(self, roots, model, latent_state_roots, to_play):
            model.recurrent_inference(latent_state_roots, np.zeros((roots.root_count, 1)))

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()
            self._collect_model = self._model
            self._mcts_collect = FakeMCTS()
            self._cfg = types.SimpleNamespace(
                root_dirichlet_alpha=0.3,
                root_noise_weight=0.25,
                eps=types.SimpleNamespace(eps_greedy_exploration_in_collect=True),
            )

        def inverse_scalar_transform_handle(self, value):
            return value

    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=20260522,
        decision_source_frames=1,
        natural_bonus_spawn=False,
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(surface.reset(seed=20260522))
    joint_action = np.asarray([[1, 2], [0, 1]], dtype=np.int16)
    recorder.record(surface.step(joint_action))
    chunk = recorder.build_chunk()

    compact_batch = _compact_batch_from_replay_record_for_boundary_test(
        chunk,
        record_index=0,
        preserve_observation_dtype=True,
    )
    probe = _LightZeroCollectForwardStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        input_mode="host_float32",
    )

    telemetry = probe.run_compact_batch_with_replay_chunk(
        compact_batch,
        chunk=chunk,
        record_index=0,
        policy_source="direct_ctree_compact_output_boundary_test",
    ).telemetry
    debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
    service_chunk = probe._last_compact_service_replay_chunk
    assert service_chunk is not None
    rows = service_chunk.target_rows

    assert telemetry["lightzero_root_count"] == 4.0
    assert telemetry["lightzero_filtered_zero_mask_root_count"] == 0.0
    assert telemetry["compact_service_replay_chunk_v1_enabled"] is True
    assert telemetry["compact_service_replay_chunk_schema_id"] == (
        "curvyzero_compact_replay_chunk/v1"
    )
    assert telemetry["compact_service_replay_chunk_target_row_count"] == 4.0
    assert debug["actions"] == [1, 2, 0, 1]
    np.testing.assert_array_equal(rows.env_row, np.asarray([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(rows.player, np.asarray([0, 1, 0, 1], dtype=np.int16))
    np.testing.assert_array_equal(rows.action, np.asarray([1, 2, 0, 1], dtype=np.int16))
    np.testing.assert_allclose(
        rows.policy_target,
        np.asarray(
            [
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        rows.root_value,
        np.linspace(0.5, -1.0, 4, dtype=np.float32),
    )


def _compact_batch_from_replay_record_for_boundary_test(
    chunk,
    *,
    record_index: int,
    done_override: np.ndarray | None = None,
    preserve_observation_dtype: bool = False,
) -> HybridCompactBatch:
    observation_float = np.asarray(chunk.arrays["observation"][record_index], dtype=np.float32)
    if preserve_observation_dtype:
        observation = observation_float.copy()
    else:
        observation = np.clip(np.rint(observation_float * 255.0), 0, 255).astype(
            np.uint8,
            copy=False,
        )
    batch_size, player_count = observation.shape[:2]
    root_count = int(batch_size * player_count)
    action_mask = np.zeros((batch_size, player_count, ACTION_COUNT), dtype=np.bool_)
    policy = chunk.policy_rows[record_index]
    for policy_row, (env_row, player) in enumerate(
        zip(policy["policy_env_row"], policy["policy_player"], strict=True)
    ):
        action_mask[int(env_row), int(player)] = np.asarray(
            policy["policy_action_mask"][policy_row],
            dtype=np.bool_,
        )
    done = (
        np.asarray(chunk.arrays["done"][record_index], dtype=np.bool_)
        if done_override is None
        else np.asarray(done_override, dtype=np.bool_)
    )
    done_root = np.repeat(done, player_count)
    flat_mask = action_mask.reshape(root_count, ACTION_COUNT)
    final_observation = np.zeros_like(observation)
    terminal_global_rows = np.flatnonzero(done).astype(np.int32, copy=False)
    return HybridCompactBatch(
        observation=observation,
        action_mask=action_mask,
        reward=np.asarray(chunk.arrays["reward"][record_index], dtype=np.float32),
        final_reward_map=np.asarray(
            chunk.arrays.get("final_reward_map", chunk.arrays["reward"])[record_index],
            dtype=np.float32,
        ),
        done=done,
        policy_env_id=np.arange(root_count, dtype=np.int32),
        policy_env_row=np.repeat(np.arange(batch_size, dtype=np.int32), player_count),
        policy_player=np.tile(np.arange(player_count, dtype=np.int32), batch_size),
        target_reward=np.asarray(
            chunk.arrays["reward"][record_index],
            dtype=np.float32,
        ).reshape(root_count, 1),
        done_root=done_root,
        to_play=np.full(root_count, -1, dtype=np.int64),
        active_root_mask=np.logical_and(~done_root, flat_mask.any(axis=1)),
        final_observation=final_observation,
        final_observation_row_mask=done.copy(),
        terminal_row_mask=done.copy(),
        autoreset_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_global_rows=terminal_global_rows,
        autoreset_global_rows=np.zeros((0,), dtype=np.int32),
        episode_step=np.arange(batch_size, dtype=np.int32),
        elapsed_ms=np.zeros((batch_size,), dtype=np.float64),
        round_id=np.zeros((batch_size,), dtype=np.int32),
        alive=np.asarray(chunk.arrays["live_mask"][record_index], dtype=np.bool_),
        joint_action=np.asarray(chunk.arrays["joint_action"][record_index], dtype=np.int16),
    )


def test_lightzero_compact_batch_rejects_malformed_sidecars_before_search():
    probe = _LightZeroCollectForwardStackProbe(
        policy=object(),
        policy_metadata={},
        num_simulations=1,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    )

    def make_batch(
        *,
        action_mask: np.ndarray | None = None,
        done_root: np.ndarray | None = None,
        terminal_row_mask: np.ndarray | None = None,
    ) -> HybridCompactBatch:
        mask = (
            np.ones((2, 2, ACTION_COUNT), dtype=np.float32) if action_mask is None else action_mask
        )
        return HybridCompactBatch(
            observation=np.ones((2, 2, 4, 64, 64), dtype=np.uint8),
            action_mask=mask,
            reward=np.zeros((2, 2), dtype=np.float32),
            final_reward_map=np.zeros((2, 2), dtype=np.float32),
            done=np.asarray([False, False], dtype=np.bool_),
            policy_env_id=np.asarray([0, 1, 2, 3], dtype=np.int32),
            policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
            policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
            target_reward=np.zeros((4, 1), dtype=np.float32),
            done_root=(
                np.asarray([False, False, False, False], dtype=np.bool_)
                if done_root is None
                else done_root
            ),
            to_play=np.asarray([-1, -1, -1, -1], dtype=np.int64),
            active_root_mask=np.asarray([True, True, True, True], dtype=np.bool_),
            final_observation=None,
            final_observation_row_mask=np.asarray([False, False], dtype=np.bool_),
            terminal_row_mask=(
                np.asarray([False, False], dtype=np.bool_)
                if terminal_row_mask is None
                else terminal_row_mask
            ),
            autoreset_row_mask=np.asarray([False, False], dtype=np.bool_),
            terminal_global_rows=np.asarray([], dtype=np.int32),
            autoreset_global_rows=np.asarray([], dtype=np.int32),
            episode_step=np.asarray([1, 1], dtype=np.int32),
            elapsed_ms=np.asarray([16.0, 16.0], dtype=np.float64),
            round_id=np.asarray([0, 0], dtype=np.int32),
            alive=np.asarray([[True, True], [True, True]], dtype=np.bool_),
            joint_action=np.asarray([[0, 1], [2, 0]], dtype=np.int16),
        )

    fractional_mask = np.ones((2, 2, ACTION_COUNT), dtype=np.float32)
    fractional_mask[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="action_mask must be binary"):
        probe.run_compact_batch(make_batch(action_mask=fractional_mask))

    with pytest.raises(ValueError, match="done_root must equal repeat"):
        probe.run_compact_batch(
            make_batch(done_root=np.asarray([False, True, False, False], dtype=np.bool_))
        )

    with pytest.raises(ValueError, match="terminal_row_mask must match"):
        probe.run_compact_batch(
            make_batch(terminal_row_mask=np.asarray([False, True], dtype=np.bool_))
        )


def test_lightzero_collect_forward_probe_rejects_fractional_action_masks(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeCollectMode:
        def forward(self, *_args, **_kwargs):
            raise AssertionError("fractional masks must be rejected before LightZero")

    class FakePolicy:
        collect_mode = FakeCollectMode()

    probe = _LightZeroCollectForwardStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=np.float32)
    action_mask[0, 1, 2] = 0.5

    with pytest.raises(ValueError, match="action_mask must be binary"):
        probe.run(observation, action_mask)


def test_lightzero_mcts_arrays_boundary_direct_ctree_vectorizes_all_legal_rows(
    monkeypatch,
):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeTensor:
        def __init__(self, value):
            self._value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    policy_module = types.ModuleType("lzero.policy")
    policy_module.mz_network_output_unpack = lambda output: output

    def fail_if_slow_select_action(*_args, **_kwargs):
        raise AssertionError("all-actions-legal path should not call per-row select_action")

    policy_module.select_action = fail_if_slow_select_action
    lzero_module = types.ModuleType("lzero")
    lzero_module.policy = policy_module
    monkeypatch.setitem(sys.modules, "lzero", lzero_module)
    monkeypatch.setitem(sys.modules, "lzero.policy", policy_module)

    class FakeModel:
        def __init__(self):
            self.inputs = []

        def initial_inference(self, obs_tensor):
            self.inputs.append(np.asarray(obs_tensor).copy())
            root_count = int(obs_tensor.shape[0])
            return (
                FakeTensor(np.zeros((root_count, 1, 1), dtype=np.float32)),
                FakeTensor(np.zeros((root_count, 1), dtype=np.float32)),
                FakeTensor(np.zeros((root_count, 1), dtype=np.float32)),
                FakeTensor(np.zeros((root_count, ACTION_COUNT), dtype=np.float32)),
            )

        def recurrent_inference(self, latent_state, action):
            return latent_state, action

    class FakeRoots:
        def prepare(self, *_args, **_kwargs):
            return None

        def get_distributions(self):
            return [
                [1.0, 4.0, 2.0],
                [9.0, 3.0, 1.0],
                [2.0, 1.0, 8.0],
                [1.0, 7.0, 4.0],
            ]

        def get_values(self):
            return [0.0, 1.0, 2.0, 3.0]

    class FakeMCTS:
        @classmethod
        def roots(cls, _root_count, _legal_actions):
            return FakeRoots()

        def search(self, roots, model, latent_state_roots, to_play):
            model.recurrent_inference(latent_state_roots, np.zeros((len(to_play), 1)))

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()
            self._collect_model = self._model
            self._mcts_collect = FakeMCTS()
            self._cfg = types.SimpleNamespace(
                root_dirichlet_alpha=0.3,
                root_noise_weight=0.25,
                eps=types.SimpleNamespace(eps_greedy_exploration_in_collect=True),
            )

        def inverse_scalar_transform_handle(self, value):
            return value

    policy = FakePolicy()
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        input_mode="resident_torch_reuse",
    )
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)

    first = probe.run(np.full((2, 2, 4, 64, 64), 255, dtype=np.uint8), action_mask).telemetry
    second = probe.run(np.zeros((2, 2, 4, 64, 64), dtype=np.uint8), action_mask).telemetry

    assert first["lightzero_mcts_arrays_boundary_input_mode"] == "resident_torch_reuse"
    assert first["lightzero_mcts_arrays_boundary_input_freshness"] == "fresh_first_fill"
    assert first["lightzero_mcts_arrays_boundary_resident_reused"] == 0.0
    assert first["lightzero_mcts_arrays_boundary_resident_first_fill_sec"] >= 0.0
    assert second["lightzero_mcts_arrays_boundary_resident_reused"] == 1.0
    assert second["lightzero_mcts_arrays_boundary_input_freshness"] == ("stale_profile_ceiling")
    assert second["lightzero_mcts_arrays_boundary_total_sec"] >= 0.0
    assert second["lightzero_consumer_h2d_sec"] == 0.0
    assert second["lightzero_consumer_normalize_sec"] == 0.0
    assert second["lightzero_mcts_arrays_boundary_all_actions_legal_fast_path"] is True
    assert second["lightzero_first_actions"] == [1, 0, 2, 1]
    assert second["lightzero_mcts_arrays_boundary_visit_shape"] == [4, ACTION_COUNT]
    assert second["lightzero_illegal_action_count"] == 0.0
    np.testing.assert_allclose(policy._model.inputs[0], 1.0)
    np.testing.assert_allclose(policy._model.inputs[1], 1.0)


@pytest.mark.parametrize("num_simulations", [1, 2, 8])
def test_lightzero_mcts_arrays_boundary_real_policy_cpu_matches_stock_values_and_masks(
    num_simulations,
):
    pytest.importorskip("lzero")
    pytest.importorskip("ding")
    import random

    torch = pytest.importorskip("torch")

    seed = 20260521
    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=False,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=8,
        max_ticks=8,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = 0.0
    policy._cfg.eps.eps_greedy_exploration_in_collect = True

    observation = (
        (np.arange(2 * 2 * 4 * 64 * 64, dtype=np.uint32) % 256)
        .astype(np.uint8)
        .reshape(2, 2, 4, 64, 64)
    )
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)

    debug_rows = []
    for impl in (
        "stock_facade",
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    ):
        random.seed(seed + int(num_simulations))
        np.random.seed(seed + int(num_simulations))
        torch.manual_seed(seed + int(num_simulations))
        probe = _LightZeroCollectForwardStackProbe(
            policy=policy,
            policy_metadata=policy_meta,
            num_simulations=num_simulations,
            temperature=1.0,
            epsilon=0.0,
            arrays_boundary=True,
            arrays_boundary_impl=impl,
        )
        telemetry = probe.run(observation, action_mask).telemetry
        assert telemetry["lightzero_illegal_action_count"] == 0.0
        debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
        assert debug["included"] is True
        assert debug["root_count"] == 4
        visit_distributions = np.asarray(debug["visit_distributions"], dtype=np.float32)
        assert visit_distributions.shape == (4, ACTION_COUNT)
        np.testing.assert_allclose(visit_distributions.sum(axis=1), 1.0)
        assert np.all(visit_distributions >= 0.0)
        debug_rows.append(debug)

    stock_searched_values = np.asarray(debug_rows[0]["searched_values"], dtype=np.float32)
    for debug in debug_rows[1:]:
        np.testing.assert_allclose(
            stock_searched_values,
            np.asarray(debug["searched_values"], dtype=np.float32),
            atol=1.0e-6,
        )


def test_lightzero_mcts_arrays_boundary_real_policy_cpu_biased_logits_match_top_actions():
    pytest.importorskip("lzero")
    pytest.importorskip("ding")
    import random

    torch = pytest.importorskip("torch")

    seed = 20260521
    num_simulations = 8
    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=False,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=8,
        max_ticks=8,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = 0.0
    policy._cfg.eps.eps_greedy_exploration_in_collect = True

    model = policy._collect_model
    original_initial_inference = model.initial_inference
    original_recurrent_inference = model.recurrent_inference
    action_one_bias = torch.tensor([[0.0, 10.0, -10.0]], dtype=torch.float32)

    def _bias_policy_logits(output):
        output.policy_logits = output.policy_logits + action_one_bias.to(
            output.policy_logits.device
        )
        return output

    def _initial_inference_with_bias(*args, **kwargs):
        return _bias_policy_logits(original_initial_inference(*args, **kwargs))

    def _recurrent_inference_with_bias(*args, **kwargs):
        return _bias_policy_logits(original_recurrent_inference(*args, **kwargs))

    model.initial_inference = _initial_inference_with_bias
    model.recurrent_inference = _recurrent_inference_with_bias

    observation = (
        (np.arange(2 * 2 * 4 * 64 * 64, dtype=np.uint32) % 256)
        .astype(np.uint8)
        .reshape(2, 2, 4, 64, 64)
    )
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)

    for impl in (
        "stock_facade",
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    ):
        random.seed(seed + num_simulations)
        np.random.seed(seed + num_simulations)
        torch.manual_seed(seed + num_simulations)
        probe = _LightZeroCollectForwardStackProbe(
            policy=policy,
            policy_metadata=policy_meta,
            num_simulations=num_simulations,
            temperature=1.0,
            epsilon=0.0,
            arrays_boundary=True,
            arrays_boundary_impl=impl,
        )
        telemetry = probe.run(observation, action_mask).telemetry
        debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
        visit_distributions = np.asarray(debug["visit_distributions"], dtype=np.float32)
        assert debug["actions"] == [1, 1, 1, 1]
        np.testing.assert_array_equal(np.argmax(visit_distributions, axis=1), 1)
        assert telemetry["lightzero_illegal_action_count"] == 0.0


def test_lightzero_mcts_arrays_boundary_real_policy_cpu_single_legal_action_exact():
    pytest.importorskip("lzero")
    pytest.importorskip("ding")
    import random

    torch = pytest.importorskip("torch")

    seed = 20260521
    num_simulations = 8
    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=False,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=8,
        max_ticks=8,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = 0.0
    policy._cfg.eps.eps_greedy_exploration_in_collect = True

    observation = (
        (np.arange(2 * 2 * 4 * 64 * 64, dtype=np.uint32) % 256)
        .astype(np.uint8)
        .reshape(2, 2, 4, 64, 64)
    )
    action_mask = np.zeros((2, 2, ACTION_COUNT), dtype=bool)
    expected_actions = np.asarray([0, 1, 2, 1], dtype=np.int64)
    for flat_index, action in enumerate(expected_actions):
        row = flat_index // 2
        player = flat_index % 2
        action_mask[row, player, int(action)] = True
    expected_visits = np.zeros((4, ACTION_COUNT), dtype=np.float32)
    expected_visits[np.arange(4), expected_actions] = 1.0

    for impl in (
        "stock_facade",
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    ):
        random.seed(seed + num_simulations)
        np.random.seed(seed + num_simulations)
        torch.manual_seed(seed + num_simulations)
        probe = _LightZeroCollectForwardStackProbe(
            policy=policy,
            policy_metadata=policy_meta,
            num_simulations=num_simulations,
            temperature=1.0,
            epsilon=0.0,
            arrays_boundary=True,
            arrays_boundary_impl=impl,
        )
        telemetry = probe.run(observation, action_mask).telemetry
        debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
        assert debug["actions"] == expected_actions.astype(int).tolist()
        np.testing.assert_allclose(debug["visit_distributions"], expected_visits)
        assert telemetry["lightzero_illegal_action_count"] == 0.0


def test_lightzero_mcts_arrays_boundary_precomputed_recurrent_skips_model_recurrent():
    pytest.importorskip("lzero")
    pytest.importorskip("ding")

    seed = 20260522
    num_simulations = 4
    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=False,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=8,
        max_ticks=8,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = 0.0
    policy._cfg.eps.eps_greedy_exploration_in_collect = True

    def _fail_recurrent(*args, **kwargs):
        raise AssertionError("precomputed recurrent boundary must not call recurrent_inference")

    policy._collect_model.recurrent_inference = _fail_recurrent
    policy._model.recurrent_inference = _fail_recurrent

    observation = np.ones((2, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata=policy_meta,
        num_simulations=num_simulations,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=(
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT
        ),
    )

    telemetry = probe.run(observation, action_mask).telemetry

    assert telemetry["lightzero_illegal_action_count"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_impl"] == (
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT
    )
    assert (
        telemetry["lightzero_mcts_arrays_boundary_gpu_latent_precomputed_recurrent_enabled"] is True
    )
    assert telemetry["lightzero_consumer_model_recurrent_inference_calls"] == 0.0
    assert telemetry["lightzero_consumer_model_recurrent_inference_sec"] == 0.0
    assert telemetry["model_eval_count"] == 20.0
    assert telemetry["logical_model_eval_count"] == 20.0
    assert telemetry["actual_model_eval_count"] == 4.0
    assert telemetry["synthetic_recurrent_eval_count"] == 16.0
    assert telemetry["lightzero_consumer_logical_model_eval_count"] == 20.0
    assert telemetry["lightzero_consumer_actual_model_eval_count"] == 4.0
    assert telemetry["lightzero_consumer_synthetic_recurrent_eval_count"] == 16.0
    assert telemetry["lightzero_mcts_arrays_boundary_logical_model_eval_count"] == 20.0
    assert telemetry["lightzero_mcts_arrays_boundary_actual_model_eval_count"] == 4.0
    assert telemetry["lightzero_mcts_arrays_boundary_synthetic_recurrent_eval_count"] == 16.0
    assert telemetry["lightzero_consumer_mcts_timer_status"] == (
        "installed_gpu_latent_precomputed_recurrent"
    )
    assert telemetry["lightzero_consumer_gpu_latent_search_output_listify_sec"] >= 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_gpu_latent_search_output_listify_sec"] >= 0.0
    assert telemetry["lightzero_consumer_ctree_batch_traverse_calls"] == float(num_simulations)
    assert telemetry["lightzero_consumer_ctree_batch_backpropagate_calls"] == float(num_simulations)
    debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
    assert debug["root_count"] == 4
    np.testing.assert_allclose(
        np.asarray(debug["visit_distributions"], dtype=np.float32).sum(axis=1),
        1.0,
    )


def test_lightzero_mcts_arrays_boundary_precomputed_recurrent_respects_mixed_masks():
    pytest.importorskip("lzero")
    pytest.importorskip("ding")

    seed = 20260522
    num_simulations = 4
    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=False,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=8,
        max_ticks=8,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = 0.0
    policy._cfg.eps.eps_greedy_exploration_in_collect = True

    def _fail_recurrent(*args, **kwargs):
        raise AssertionError("precomputed recurrent boundary must not call recurrent_inference")

    policy._collect_model.recurrent_inference = _fail_recurrent
    policy._model.recurrent_inference = _fail_recurrent

    observation = (
        (np.arange(2 * 2 * 4 * 64 * 64, dtype=np.uint32) % 256)
        .astype(np.uint8)
        .reshape(2, 2, 4, 64, 64)
    )
    action_mask = np.asarray(
        [
            [[True, False, False], [False, True, True]],
            [[True, False, True], [False, False, True]],
        ],
        dtype=bool,
    )
    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata=policy_meta,
        num_simulations=num_simulations,
        temperature=1.0,
        epsilon=0.0,
        arrays_boundary=True,
        arrays_boundary_impl=(
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT
        ),
    )

    telemetry = probe.run(observation, action_mask).telemetry

    assert telemetry["lightzero_illegal_action_count"] == 0.0
    assert telemetry["lightzero_mcts_arrays_boundary_all_actions_legal_fast_path"] is False
    assert telemetry["lightzero_consumer_model_recurrent_inference_calls"] == 0.0
    assert telemetry["lightzero_consumer_actual_model_eval_count"] == 4.0
    assert telemetry["lightzero_consumer_synthetic_recurrent_eval_count"] == 16.0
    debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
    actions = np.asarray(debug["actions"], dtype=np.int64)
    flat_mask = action_mask.reshape(4, ACTION_COUNT)
    assert actions.shape == (4,)
    assert np.all(flat_mask[np.arange(4), actions])
    visits = np.asarray(debug["visit_distributions"], dtype=np.float32)
    np.testing.assert_allclose(visits.sum(axis=1), 1.0)
    np.testing.assert_allclose(visits[~flat_mask], 0.0)


def test_lightzero_mcts_arrays_boundary_real_policy_cpu_biased_logits_respect_masks():
    pytest.importorskip("lzero")
    pytest.importorskip("ding")
    import random

    torch = pytest.importorskip("torch")

    seed = 20260521
    num_simulations = 8
    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=False,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=8,
        max_ticks=8,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = 0.0
    policy._cfg.eps.eps_greedy_exploration_in_collect = True

    model = policy._collect_model
    original_initial_inference = model.initial_inference
    original_recurrent_inference = model.recurrent_inference
    action_one_bias = torch.tensor([[0.0, 10.0, -10.0]], dtype=torch.float32)

    def _bias_policy_logits(output):
        output.policy_logits = output.policy_logits + action_one_bias.to(
            output.policy_logits.device
        )
        return output

    def _initial_inference_with_bias(*args, **kwargs):
        return _bias_policy_logits(original_initial_inference(*args, **kwargs))

    def _recurrent_inference_with_bias(*args, **kwargs):
        return _bias_policy_logits(original_recurrent_inference(*args, **kwargs))

    model.initial_inference = _initial_inference_with_bias
    model.recurrent_inference = _recurrent_inference_with_bias

    observation = (
        (np.arange(2 * 2 * 4 * 64 * 64, dtype=np.uint32) % 256)
        .astype(np.uint8)
        .reshape(2, 2, 4, 64, 64)
    )
    action_mask = np.asarray(
        [
            [[True, True, True], [True, False, True]],
            [[False, True, True], [False, False, True]],
        ],
        dtype=bool,
    )
    expected_actions = np.asarray([1, 0, 1, 2], dtype=np.int64)

    for impl in (
        "stock_facade",
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
        LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    ):
        random.seed(seed + num_simulations)
        np.random.seed(seed + num_simulations)
        torch.manual_seed(seed + num_simulations)
        probe = _LightZeroCollectForwardStackProbe(
            policy=policy,
            policy_metadata=policy_meta,
            num_simulations=num_simulations,
            temperature=1.0,
            epsilon=0.0,
            arrays_boundary=True,
            arrays_boundary_impl=impl,
        )
        telemetry = probe.run(observation, action_mask).telemetry
        debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
        visit_distributions = np.asarray(debug["visit_distributions"], dtype=np.float32)
        assert debug["actions"] == expected_actions.astype(int).tolist()
        np.testing.assert_array_equal(np.argmax(visit_distributions, axis=1), expected_actions)
        assert np.all(visit_distributions[~action_mask.reshape(4, ACTION_COUNT)] == 0.0)
        assert telemetry["lightzero_illegal_action_count"] == 0.0


def test_lightzero_collect_forward_stack_probe_rejects_decoded_illegal_action(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeCollectMode:
        def forward(
            self,
            obs_tensor,
            *,
            action_mask,
            temperature,
            to_play,
            epsilon,
            ready_env_id,
        ):
            return {0: {"action": ACTION_COUNT, "searched_value": 0.0}}

    class FakePolicy:
        collect_mode = FakeCollectMode()

    probe = _LightZeroCollectForwardStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "surface": {"env_variant": "test"}},
        num_simulations=8,
        temperature=1.0,
        epsilon=0.0,
    )
    observation = np.ones((1, 1, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 1, ACTION_COUNT), dtype=bool)

    with pytest.raises(ValueError, match="decoded illegal actions"):
        probe.run(observation, action_mask)


def test_lightzero_initial_inference_stack_probe_calls_model_only(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = "float32"
            self.device = "cpu"

    class FakeOutput:
        def __init__(self, batch):
            self.policy_logits = FakeTensor((batch, 3))
            self.value = FakeTensor((batch, 1))
            self.latent_state = FakeTensor((batch, 16, 6, 6))

    class FakeModel:
        def __init__(self):
            self.call = None

        def initial_inference(self, obs_tensor):
            self.call = {
                "obs_shape": list(obs_tensor.shape),
                "obs_min": float(obs_tensor.min()),
                "obs_max": float(obs_tensor.max()),
            }
            return FakeOutput(obs_tensor.shape[0])

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroInitialInferenceStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
    )
    observation = (
        np.arange(2 * 2 * 4 * 64 * 64, dtype=np.int32).reshape(2, 2, 4, 64, 64) % 256
    ).astype(np.uint8)
    action_mask = np.ones((2, 2, 3), dtype=bool)

    result = probe.run(observation, action_mask)

    assert policy._model.call["obs_shape"] == [4, 4, 64, 64]
    assert policy._model.call["obs_min"] >= 0.0
    assert policy._model.call["obs_max"] <= 1.0
    telemetry = result.telemetry
    assert telemetry["lightzero_root_count"] == 4.0
    assert telemetry["model_eval_count"] == 4.0
    assert telemetry["lightzero_initial_inference_model_class"] == "fake.Model"
    summary = telemetry["lightzero_initial_inference_output_summary"]
    assert summary["policy_logits"]["shape"] == [4, 3]
    assert summary["latent_state"]["shape"] == [4, 16, 6, 6]


def test_lightzero_initial_inference_stack_probe_filters_zero_mask_roots(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch):
            self.policy_logits = np.zeros((batch, 3), dtype=np.float32)
            self.value = np.zeros((batch, 1), dtype=np.float32)
            self.latent_state = np.zeros((batch, 8, 4, 4), dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.call = None

        def initial_inference(self, obs_tensor):
            self.call = {
                "obs_shape": list(obs_tensor.shape),
                "first_pixels": np.asarray(obs_tensor)[:, 0, 0, 0].tolist(),
            }
            return FakeOutput(obs_tensor.shape[0])

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroInitialInferenceStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
    )
    observation = np.zeros((2, 2, 4, 64, 64), dtype=np.uint8)
    observation[0, 0, 0, 0, 0] = 11
    observation[0, 1, 0, 0, 0] = 22
    observation[1, 0, 0, 0, 0] = 33
    observation[1, 1, 0, 0, 0] = 44
    action_mask = np.ones((2, 2, 3), dtype=bool)
    action_mask[0, 1, :] = False
    action_mask[1, 0, :] = False

    result = probe.run(observation, action_mask)

    assert policy._model.call["obs_shape"] == [2, 4, 64, 64]
    np.testing.assert_allclose(
        policy._model.call["first_pixels"],
        np.asarray([11, 44], dtype=np.float32) / np.float32(255.0),
    )
    assert result.telemetry["lightzero_total_root_count"] == 4.0
    assert result.telemetry["lightzero_root_count"] == 2.0
    assert result.telemetry["lightzero_filtered_zero_mask_root_count"] == 2.0


def test_lightzero_array_ceiling_policy_arrays_is_compact_profile_only(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch):
            self.policy_logits = np.tile(
                np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
                (batch, 1),
            )
            self.value = np.arange(batch, dtype=np.float32).reshape(batch, 1)
            self.latent_state = np.zeros((batch, 4), dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.initial_calls = 0
            self.recurrent_calls = 0

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(obs_tensor.shape[0])

        def recurrent_inference(self, latent_state, action):
            self.recurrent_calls += 1
            return FakeOutput(latent_state.shape[0])

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=8,
        mode="policy_arrays",
    )
    observation = np.ones((2, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)
    action_mask[0, 0, 2] = False

    result = probe.run(observation, action_mask)
    telemetry = result.telemetry

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 0
    assert telemetry["lightzero_array_ceiling_mode"] == "policy_arrays"
    assert telemetry["lightzero_array_ceiling_semantics"] == (
        "lightzero_replacement_ceiling_not_mcts"
    )
    assert telemetry["lightzero_root_count"] == 4.0
    assert telemetry["lightzero_array_ceiling_illegal_action_count"] == 0.0
    assert telemetry["lightzero_array_ceiling_first_actions"] == [1, 2, 2, 2]
    assert telemetry["lightzero_array_ceiling_policy_shape"] == [4, ACTION_COUNT]
    assert telemetry["model_eval_count"] == 4.0


def test_lightzero_array_ceiling_mock_search_service_is_named_ceiling(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch):
            self.policy_logits = np.tile(
                np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
                (batch, 1),
            )
            self.value = np.arange(batch, dtype=np.float32).reshape(batch, 1)
            self.latent_state = np.zeros((batch, 4), dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.initial_calls = 0
            self.recurrent_calls = 0

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(obs_tensor.shape[0])

        def recurrent_inference(self, latent_state, action):
            self.recurrent_calls += 1
            return FakeOutput(latent_state.shape[0])

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=16,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE,
    )
    observation = np.ones((2, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)
    action_mask[0, 0, 2] = False

    telemetry = probe.run(observation, action_mask).telemetry

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 0
    assert telemetry["lightzero_array_ceiling_mode"] == "mock_search_service"
    assert telemetry["lightzero_array_ceiling_semantics"] == (
        "mock_search_service_compact_arrays_profile_not_mcts"
    )
    assert telemetry["mock_search_service_semantics"] == (
        "mock_search_service_compact_arrays_profile_not_mcts"
    )
    assert telemetry["mock_search_service_requested_simulations"] == 16.0
    assert telemetry["mock_search_service_actual_search_simulations"] == 0.0
    assert telemetry["mock_search_service_recurrent_inference_calls"] == 0.0
    assert telemetry["mock_search_service_real_ctree_calls"] == 0.0
    assert telemetry["mock_search_service_illegal_action_count"] == 0.0
    assert telemetry["mock_search_service_visit_shape"] == [4, ACTION_COUNT]
    assert telemetry["lightzero_array_ceiling_first_actions"] == [1, 2, 2, 2]
    assert telemetry["model_eval_count"] == 4.0
    assert telemetry["mock_search_service_public_output_count"] == 0.0
    assert telemetry["lightzero_array_ceiling_public_output_materialized"] == 0.0
    assert telemetry["lightzero_array_ceiling_actual_search_simulations"] == 0.0
    assert telemetry["lightzero_array_ceiling_compact_search_arrays_stored"] == 1.0
    assert probe._last_compact_search_arrays["search_impl"] == "mock_search_service"


def test_lightzero_array_ceiling_mock_search_service_can_price_public_output(
    monkeypatch,
):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch):
            self.policy_logits = np.tile(
                np.asarray([[0.1, 2.0, 0.3]], dtype=np.float32),
                (batch, 1),
            )
            self.value = np.arange(batch, dtype=np.float32).reshape(batch, 1)
            self.latent_state = np.zeros((batch, 4), dtype=np.float32)

    class FakeModel:
        def initial_inference(self, obs_tensor):
            return FakeOutput(obs_tensor.shape[0])

        def parameters(self):
            return iter(())

    probe = _LightZeroArrayCeilingStackProbe(
        policy=types.SimpleNamespace(_model=FakeModel()),
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=16,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE,
        materialize_public_output=True,
    )
    observation = np.ones((2, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)
    action_mask[0, 1, :] = False

    telemetry = probe.run(observation, action_mask).telemetry

    assert telemetry["lightzero_root_count"] == 3.0
    assert telemetry["mock_search_service_public_output_count"] == 3.0
    assert telemetry["mock_search_service_public_output_bytes"] > 0.0
    assert telemetry["mock_search_service_public_output_checksum"] > 0.0
    assert telemetry["lightzero_array_ceiling_public_output_materialized"] == 1.0
    assert telemetry["lightzero_array_ceiling_public_output_count"] == 3.0


def test_lightzero_array_ceiling_service_tax_probe_stores_compact_search_arrays(
    monkeypatch,
):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch, offset):
            logits = np.asarray([[0.3, 0.1, 0.2]], dtype=np.float32) + np.float32(offset)
            self.policy_logits = np.tile(logits, (batch, 1))
            self.value = np.full((batch, 1), offset, dtype=np.float32)
            self.latent_state = np.full((batch, 4), offset, dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.initial_calls = 0
            self.recurrent_calls = 0

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(obs_tensor.shape[0], 1.0)

        def recurrent_inference(self, latent_state, action):
            self.recurrent_calls += 1
            return FakeOutput(latent_state.shape[0], float(self.recurrent_calls))

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=3,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_SERVICE_TAX_PROBE,
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)
    action_mask[0, 0, 2] = False
    compact_batch = HybridCompactBatch(
        observation=observation,
        action_mask=action_mask,
        reward=np.zeros((1, 2), dtype=np.float32),
        final_reward_map=np.zeros((1, 2), dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        policy_env_id=np.asarray([11, 13], dtype=np.int32),
        policy_env_row=np.asarray([0, 0], dtype=np.int32),
        policy_player=np.asarray([0, 1], dtype=np.int32),
        target_reward=np.zeros((2, 1), dtype=np.float32),
        done_root=np.asarray([False, False], dtype=np.bool_),
        to_play=np.asarray([-1, -1], dtype=np.int64),
        active_root_mask=np.asarray([True, True], dtype=np.bool_),
        final_observation=np.zeros_like(observation),
        final_observation_row_mask=np.asarray([False], dtype=np.bool_),
        terminal_row_mask=np.asarray([False], dtype=np.bool_),
        autoreset_row_mask=np.asarray([False], dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.asarray([1], dtype=np.int32),
        elapsed_ms=np.asarray([16.0], dtype=np.float64),
        round_id=np.asarray([0], dtype=np.int32),
        alive=np.asarray([[True, True]], dtype=np.bool_),
        joint_action=np.asarray([[0, 1]], dtype=np.int16),
    )

    telemetry = probe.run_compact_batch(compact_batch).telemetry
    compact_arrays = probe._last_compact_search_arrays
    search_result = probe._last_compact_service_search_result

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 3
    assert telemetry["lightzero_array_ceiling_mode"] == "service_tax_probe"
    assert telemetry["lightzero_array_ceiling_semantics"] == (
        "service_tax_probe_compact_arrays_profile_not_mcts"
    )
    assert telemetry["lightzero_array_ceiling_requested_simulations"] == 3.0
    assert telemetry["lightzero_array_ceiling_actual_search_simulations"] == 3.0
    assert telemetry["lightzero_array_ceiling_compact_search_arrays_stored"] == 1.0
    assert telemetry["service_tax_probe_requested_simulations"] == 3.0
    assert telemetry["service_tax_probe_recurrent_inference_calls"] == 3.0
    assert telemetry["service_tax_probe_real_ctree_calls"] == 0.0
    assert telemetry["service_tax_probe_compact_search_arrays_stored"] == 1.0
    assert telemetry["compact_service_contract_v1_enabled"] is True
    assert telemetry["compact_service_root_count"] == 2.0
    assert telemetry["compact_service_active_root_count"] == 2.0
    assert telemetry["compact_service_search_result_schema_id"] == (
        "curvyzero_compact_search_result/v1"
    )
    assert telemetry["simulations"] == 3.0
    assert telemetry["model_eval_count"] == 8.0
    assert compact_arrays is not None
    assert search_result is not None
    np.testing.assert_array_equal(search_result.policy_env_id, np.asarray([11, 13]))
    assert compact_arrays["array_source"] == "array_ceiling_compact_search"
    assert compact_arrays["search_impl"] == "service_tax_probe"
    assert compact_arrays["actual_search_simulations"] == 3
    assert compact_arrays["requested_simulations"] == 3
    assert compact_arrays["selected_action"].shape == (2,)
    assert compact_arrays["visit_policy"].shape == (2, ACTION_COUNT)
    assert compact_arrays["root_value"].shape == (2,)
    np.testing.assert_allclose(
        compact_arrays["visit_policy"].sum(axis=1),
        np.ones((2,), dtype=np.float32),
    )
    assert compact_arrays["visit_policy"][0, 2] == pytest.approx(0.0)


def test_lightzero_array_ceiling_dense_torch_mcts_is_profile_only():
    torch = pytest.importorskip("torch")

    class FakeOutput:
        def __init__(self, *, batch: int, latent: torch.Tensor | None = None) -> None:
            self.policy_logits = torch.tensor(
                [[0.0, 0.25, 2.0]],
                dtype=torch.float32,
            ).repeat(batch, 1)
            self.value = torch.zeros((batch, 1), dtype=torch.float32)
            self.reward = torch.zeros((batch, 1), dtype=torch.float32)
            self.latent_state = (
                torch.zeros((batch, 2), dtype=torch.float32) if latent is None else latent.float()
            )

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.recurrent_calls = 0

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(batch=int(obs_tensor.shape[0]))

        def recurrent_inference(self, latent_state, actions):
            self.recurrent_calls += 1
            action_signal = actions.reshape(-1, 1).float() / 10.0
            return FakeOutput(
                batch=int(latent_state.shape[0]),
                latent=latent_state + torch.cat([action_signal, action_signal], dim=1),
            )

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = types.SimpleNamespace(
                pb_c_base=19652,
                pb_c_init=1.25,
                discount_factor=0.997,
                root_noise_weight=0.0,
                root_dirichlet_alpha=0.3,
                value_delta_max=0.01,
            )

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=4,
        mode="dense_torch_mcts",
    )
    observation = np.ones((2, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=bool)
    action_mask[0, 0, :] = False
    action_mask[0, 0, 1] = True

    compact_batch = HybridCompactBatch(
        observation=observation,
        action_mask=action_mask,
        reward=np.zeros((2, 2), dtype=np.float32),
        final_reward_map=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        policy_env_id=np.asarray([21, 23, 29, 31], dtype=np.int32),
        policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
        target_reward=np.zeros((4, 1), dtype=np.float32),
        done_root=np.asarray([False, False, False, False], dtype=np.bool_),
        to_play=np.asarray([-1, -1, -1, -1], dtype=np.int64),
        active_root_mask=np.asarray([True, True, True, True], dtype=np.bool_),
        final_observation=np.zeros_like(observation),
        final_observation_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_row_mask=np.asarray([False, False], dtype=np.bool_),
        autoreset_row_mask=np.asarray([False, False], dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.asarray([1, 1], dtype=np.int32),
        elapsed_ms=np.asarray([16.0, 16.0], dtype=np.float64),
        round_id=np.asarray([0, 0], dtype=np.int32),
        alive=np.asarray([[True, True], [True, True]], dtype=np.bool_),
        joint_action=np.asarray([[0, 1], [1, 2]], dtype=np.int16),
    )

    telemetry = probe.run_compact_batch(compact_batch).telemetry
    compact_arrays = probe._last_compact_search_arrays
    search_result = probe._last_compact_service_search_result

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 4
    assert telemetry["lightzero_array_ceiling_mode"] == "dense_torch_mcts"
    assert telemetry["lightzero_array_ceiling_semantics"] == (
        "dense_torch_mcts_profile_not_lightzero_ctree"
    )
    assert telemetry["lightzero_array_ceiling_illegal_action_count"] == 0.0
    assert telemetry["lightzero_array_ceiling_policy_shape"] == [4, ACTION_COUNT]
    assert telemetry["lightzero_array_ceiling_first_actions"][0] == 1
    assert telemetry["lightzero_array_ceiling_compact_search_arrays_stored"] == 1.0
    assert telemetry["compact_service_contract_v1_enabled"] is True
    assert telemetry["compact_service_active_root_count"] == 4.0
    assert telemetry["model_eval_count"] == 20.0
    assert compact_arrays is not None
    assert search_result is not None
    assert compact_arrays["array_source"] == "array_ceiling_compact_search"
    assert compact_arrays["search_impl"] == "dense_torch_mcts"
    assert compact_arrays["actual_search_simulations"] == 4
    np.testing.assert_array_equal(search_result.policy_env_id, np.asarray([21, 23, 29, 31]))


def test_lightzero_array_ceiling_dense_torch_compile_spike_cpu_falls_back_cleanly():
    torch = pytest.importorskip("torch")

    class FakeOutput:
        def __init__(self, *, batch: int, latent: torch.Tensor | None = None) -> None:
            self.policy_logits = torch.tensor(
                [[0.0, 0.25, 2.0]],
                dtype=torch.float32,
            ).repeat(batch, 1)
            self.value = torch.zeros((batch, 1), dtype=torch.float32)
            self.reward = torch.zeros((batch, 1), dtype=torch.float32)
            self.latent_state = (
                torch.zeros((batch, 2), dtype=torch.float32) if latent is None else latent.float()
            )

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.recurrent_calls = 0

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(batch=int(obs_tensor.shape[0]))

        def recurrent_inference(self, latent_state, actions):
            self.recurrent_calls += 1
            action_signal = actions.reshape(-1, 1).float() / 10.0
            return FakeOutput(
                batch=int(latent_state.shape[0]),
                latent=latent_state + torch.cat([action_signal, action_signal], dim=1),
            )

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = types.SimpleNamespace(
                pb_c_base=19652,
                pb_c_init=1.25,
                discount_factor=0.997,
                root_noise_weight=0.0,
                root_dirichlet_alpha=0.3,
                value_delta_max=0.01,
            )

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=2,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE,
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)

    telemetry = probe.run(observation, action_mask).telemetry

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 2
    assert telemetry["lightzero_array_ceiling_mode"] == (
        boundary.LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE
    )
    assert telemetry["lightzero_array_ceiling_semantics"] == (
        "dense_torch_mcts_compile_spike_profile_not_lightzero_ctree"
    )
    assert telemetry["lightzero_array_ceiling_compile_status"] == "fallback_precondition"
    assert telemetry["lightzero_array_ceiling_compile_enabled"] == 0.0
    assert telemetry["lightzero_array_ceiling_compile_attempted"] == 0.0
    assert telemetry["lightzero_array_ceiling_compile_reason"] == "requires_cuda_device"
    assert telemetry["lightzero_array_ceiling_all_actions_legal_fast_path"] == 1.0
    assert telemetry["lightzero_array_ceiling_illegal_action_count"] == 0.0
    assert telemetry["model_eval_count"] == 6.0


def test_dense_torch_compile_spike_requires_all_actions_legal_before_compile():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    def fail_compile(*_args, **_kwargs):
        raise AssertionError("compile should not be called when masks are dynamic")

    fake_torch = types.SimpleNamespace(cuda=FakeCuda, compile=fail_compile)
    probe = _LightZeroArrayCeilingStackProbe(
        policy=types.SimpleNamespace(),
        policy_metadata={},
        num_simulations=1,
        mode=boundary.LIGHTZERO_ARRAY_CEILING_MODE_DENSE_TORCH_MCTS_COMPILE_SPIKE,
    )

    _select, _backup, telemetry = probe._dense_torch_mcts_compile_helpers(
        torch=fake_torch,
        device="cuda:0",
        edge_child=None,
        edge_visit=None,
        edge_value_sum=None,
        edge_reward=None,
        edge_prior=None,
        latent_pool=None,
        node_latent_slot=None,
        next_node_index=None,
        min_value=None,
        max_value=None,
        path_node_history=None,
        path_action_history=None,
        path_active_history=None,
        flat_mask_tensor=None,
        row_index=None,
        root_latent_state=None,
        root_noise_weight=0.0,
        num_simulations=1,
        root_count=2,
        all_roots_legal_fast_path=True,
        all_actions_legal_fast_path=False,
        pb_c_base=19652.0,
        pb_c_init=1.25,
        discount_factor=0.997,
        value_delta_max=0.01,
    )

    assert telemetry["lightzero_array_ceiling_compile_status"] == "fallback_precondition"
    assert telemetry["lightzero_array_ceiling_compile_enabled"] == 0.0
    assert telemetry["lightzero_array_ceiling_compile_attempted"] == 0.0
    assert telemetry["lightzero_array_ceiling_compile_reason"] == "requires_all_actions_legal"


def test_lightzero_array_ceiling_dense_torch_mcts_backs_up_reward_and_discount():
    torch = pytest.importorskip("torch")

    class RootOutput:
        def __init__(self, *, batch: int) -> None:
            self.policy_logits = torch.tensor([[0.0, 0.25, 2.0]], dtype=torch.float32).repeat(
                batch,
                1,
            )
            self.value = torch.zeros((batch, 1), dtype=torch.float32)
            self.latent_state = torch.zeros((batch, 2), dtype=torch.float32)

    class RecurrentOutput:
        def __init__(self, *, batch: int) -> None:
            self.policy_logits = torch.zeros((batch, ACTION_COUNT), dtype=torch.float32)
            self.value = torch.full((batch, 1), 2.0, dtype=torch.float32)
            self.reward = torch.full((batch, 1), 3.0, dtype=torch.float32)
            self.latent_state = torch.zeros((batch, 2), dtype=torch.float32)

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            return RootOutput(batch=int(obs_tensor.shape[0]))

        def recurrent_inference(self, latent_state, actions):
            return RecurrentOutput(batch=int(latent_state.shape[0]))

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = types.SimpleNamespace(
                pb_c_base=19652,
                pb_c_init=1.25,
                discount_factor=0.5,
                root_noise_weight=0.0,
                root_dirichlet_alpha=0.3,
                value_delta_max=0.01,
            )

    probe = _LightZeroArrayCeilingStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=1,
        mode="dense_torch_mcts",
    )

    telemetry = probe.run(
        np.ones((1, 1, 4, 64, 64), dtype=np.uint8),
        np.ones((1, 1, ACTION_COUNT), dtype=bool),
    ).telemetry

    assert telemetry["lightzero_array_ceiling_value_checksum"] == pytest.approx(4.0)


def test_lightzero_array_ceiling_rejects_fractional_action_masks():
    class FakeModel:
        def parameters(self):
            return iter([])

        def initial_inference(self, obs_tensor):  # pragma: no cover - should fail before model
            raise AssertionError("fractional masks should fail before model inference")

    class FakePolicy:
        _model = FakeModel()

    probe = _LightZeroArrayCeilingStackProbe(
        policy=FakePolicy(),
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=1,
        mode="policy_arrays",
    )
    action_mask = np.ones((1, 1, ACTION_COUNT), dtype=np.float32)
    action_mask[0, 0, 0] = 0.5

    with pytest.raises(ValueError, match="binary action masks"):
        probe.run(
            np.ones((1, 1, 4, 64, 64), dtype=np.uint8),
            action_mask,
        )


def test_lightzero_array_ceiling_host_float32_prenormalizes(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch):
            self.policy_logits = np.tile(
                np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
                (batch, 1),
            )
            self.value = np.zeros((batch, 1), dtype=np.float32)
            self.latent_state = np.zeros((batch, 4), dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.last_obs = None

        def initial_inference(self, obs_tensor):
            self.last_obs = np.asarray(obs_tensor)
            return FakeOutput(obs_tensor.shape[0])

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={},
        num_simulations=8,
        mode="policy_arrays",
        input_mode="host_float32",
    )
    observation = np.full((1, 2, 4, 64, 64), 255, dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)

    telemetry = probe.run(observation, action_mask).telemetry

    assert telemetry["lightzero_array_ceiling_input_mode"] == "host_float32"
    assert telemetry["lightzero_array_ceiling_normalize_sec"] == 0.0
    assert telemetry["lightzero_array_ceiling_host_prenormalize_sec"] >= 0.0
    np.testing.assert_allclose(policy._model.last_obs, 1.0)


def test_lightzero_array_ceiling_resident_torch_reuse_skips_measured_h2d(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch):
            self.policy_logits = np.tile(
                np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
                (batch, 1),
            )
            self.value = np.zeros((batch, 1), dtype=np.float32)
            self.latent_state = np.zeros((batch, 4), dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.inputs = []

        def initial_inference(self, obs_tensor):
            self.inputs.append(np.asarray(obs_tensor).copy())
            return FakeOutput(obs_tensor.shape[0])

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={},
        num_simulations=8,
        mode="policy_arrays",
        input_mode="resident_torch_reuse",
    )
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)

    first = probe.run(np.full((1, 2, 4, 64, 64), 255, dtype=np.uint8), action_mask).telemetry
    second = probe.run(np.zeros((1, 2, 4, 64, 64), dtype=np.uint8), action_mask).telemetry

    assert first["lightzero_array_ceiling_resident_reused"] == 0.0
    assert first["lightzero_array_ceiling_resident_first_fill_sec"] >= 0.0
    assert second["lightzero_array_ceiling_resident_reused"] == 1.0
    assert second["lightzero_array_ceiling_h2d_sec"] == 0.0
    assert second["lightzero_array_ceiling_normalize_sec"] == 0.0
    np.testing.assert_allclose(policy._model.inputs[0], 1.0)
    np.testing.assert_allclose(policy._model.inputs[1], 1.0)


def test_lightzero_array_ceiling_recurrent_toy_calls_recurrent(monkeypatch):
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeCuda:
        @staticmethod
        def synchronize(_device=None):
            return None

    fake_torch = types.SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        no_grad=FakeNoGrad,
        cuda=FakeCuda,
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeOutput:
        def __init__(self, batch, offset):
            self.policy_logits = np.tile(
                np.asarray([[0.3, 0.1, 0.2]], dtype=np.float32),
                (batch, 1),
            ) + np.float32(offset)
            self.value = np.full((batch, 1), offset, dtype=np.float32)
            self.latent_state = np.full((batch, 4), offset, dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.initial_calls = 0
            self.recurrent_actions = []

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            return FakeOutput(obs_tensor.shape[0], 1.0)

        def recurrent_inference(self, latent_state, action):
            self.recurrent_actions.append(np.asarray(action).shape)
            return FakeOutput(latent_state.shape[0], float(len(self.recurrent_actions)))

        def parameters(self):
            return iter(())

    class FakePolicy:
        def __init__(self):
            self._model = FakeModel()

    policy = FakePolicy()
    probe = _LightZeroArrayCeilingStackProbe(
        policy=policy,
        policy_metadata={"policy_class": "fake.Policy", "model_class": "fake.Model"},
        num_simulations=3,
        mode="recurrent_toy",
    )
    observation = np.ones((1, 2, 4, 64, 64), dtype=np.uint8)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=bool)

    telemetry = probe.run(observation, action_mask).telemetry

    assert policy._model.initial_calls == 1
    assert len(policy._model.recurrent_actions) == 3
    assert telemetry["lightzero_array_ceiling_mode"] == "recurrent_toy"
    assert telemetry["lightzero_array_ceiling_recurrent_inference_calls"] == 3.0
    assert telemetry["lightzero_array_ceiling_policy_shape"] == [2, ACTION_COUNT]
    assert telemetry["lightzero_array_ceiling_value_shape"] == [2]
    assert telemetry["model_eval_count"] == 8.0


def test_validate_boundary_config_accepts_persistent_gpu_profile_backend():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "surface_facade_canary": True,
            "surface_stack_backend": "renderer_backed_profile",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
        },
    )

    assert (
        config["observation_renderer_backend"]
        == SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
    )
    assert config["render_surface"] == "direct_gray64"


def test_validate_boundary_config_passes_async_device_only_profile_to_persistent_renderer():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 64,
            "hybrid_observation_canary": True,
            "surface_stack_backend": "renderer_backed_profile",
            "render_surface": "direct_gray64",
            "observation_renderer_backend": (
                SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
            ),
            "async_device_only_profile": True,
        },
    )

    assert config["async_device_only_profile"] is True
    assert config["render_config"]["async_device_only_profile"] is True


def test_validate_boundary_config_rejects_async_device_only_without_persistent_renderer():
    with pytest.raises(ValueError, match="async_device_only_profile requires"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "hybrid_observation_canary": True,
                "surface_stack_backend": "renderer_backed_profile",
                "render_surface": "direct_gray64",
                "async_device_only_profile": True,
            },
        )


def test_validate_boundary_config_rejects_persistent_gpu_profile_without_canary():
    with pytest.raises(ValueError, match="persistent_policy_framebuffer"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_persistent_gpu_profile_block_surface():
    with pytest.raises(ValueError, match="direct_gray64"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "surface_facade_canary": True,
                "surface_stack_backend": "renderer_backed_profile",
                "observation_renderer_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_PERSISTENT_GPU_PROFILE_BACKEND
                ),
            },
        )


def test_validate_boundary_config_rejects_direct_gray64_without_surface_canary():
    with pytest.raises(ValueError, match="hybrid_observation_canary"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "render_surface": "direct_gray64",
            },
        )


def test_validate_boundary_config_rejects_direct_gray64_cpu_dirty_surface():
    with pytest.raises(ValueError, match="renderer_backed_profile"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "surface_facade_canary": True,
                "surface_stack_backend": "cpu_dirty_cache",
                "render_surface": "direct_gray64",
            },
        )


def test_validate_boundary_config_rejects_unknown_render_surface():
    with pytest.raises(ValueError, match="render_surface"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "render_surface": "mystery",
            },
        )


def test_persistent_delta_state_cold_start_packs_all_active_slots():
    compact_state = {
        "trail_write_cursor": np.asarray([3], dtype=np.int32),
        "trail_active": np.asarray([[1, 1, 0, 1]], dtype=np.uint8),
        "trail_x": np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "trail_y": np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        "trail_radius": np.asarray([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "trail_owner": np.asarray([[0, 0, 1, 1]], dtype=np.int32),
        "trail_break_before": np.asarray([[1, 0, 0, 0]], dtype=np.uint8),
        "avatar_color": np.asarray([[0, 1]], dtype=np.int32),
    }

    delta, next_cursor, stats = _persistent_delta_state(
        np=np,
        compact_state=compact_state,
        previous_cursor=None,
        previous_owner_pos=None,
        previous_owner_valid=None,
        batch_size=1,
    )

    np.testing.assert_array_equal(next_cursor, np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(delta["reset_mask"], np.asarray([1], dtype=np.uint8))
    np.testing.assert_array_equal(delta["active"], np.asarray([[1, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(delta["owner"], np.asarray([[0, 0]], dtype=np.int32))
    np.testing.assert_allclose(delta["x0"], np.asarray([[1.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(delta["x1"], np.asarray([[1.0, 2.0]], dtype=np.float32))
    assert stats["reset_row_count"] == 1
    assert stats["delta_slot_count"] == 2


def test_persistent_compact_render_state_validator_accepts_compact_passthrough():
    production_state = {
        "visual_trail_pos": np.arange(2 * 8 * 2, dtype=np.float64).reshape(2, 8, 2),
        "visual_trail_radius": np.ones((2, 8), dtype=np.float64),
        "visual_trail_owner": np.asarray(
            [[0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]],
            dtype=np.int32,
        ),
        "visual_trail_active": np.ones((2, 8), dtype=bool),
        "visual_trail_break_before": np.zeros((2, 8), dtype=bool),
        "visual_trail_write_cursor": np.asarray([3, 5], dtype=np.int32),
        "pos": np.asarray(
            [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
            dtype=np.float64,
        ),
        "radius": np.ones((2, 2), dtype=np.float64),
        "alive": np.ones((2, 2), dtype=bool),
        "present": np.ones((2, 2), dtype=bool),
        "avatar_color": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        "bonus_active": np.zeros((2, 1), dtype=bool),
        "bonus_pos": np.zeros((2, 1, 2), dtype=np.float64),
        "bonus_radius": np.zeros((2, 1), dtype=np.float64),
        "bonus_type": np.ones((2, 1), dtype=np.int32),
    }
    config = {
        "batch_size": 2,
        "player_count": 2,
        "bonus_count": 1,
        "trail_slots": 8,
        "geometry_dtype": "float32",
    }
    compact = _persistent_compact_state_from_production(
        np=np,
        production_state=production_state,
        config=config,
    )
    compact[PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_MARKER] = np.asarray(1, dtype=np.uint8)

    assert _is_persistent_compact_render_state(compact)
    validated = _validate_persistent_compact_render_state(
        np=np,
        state=compact,
        config=config,
    )

    assert validated["trail_x"].shape == (2, 5)
    assert validated["head_x"].shape == (2, 2)
    assert validated["bonus_x"].shape == (2, 1)
    np.testing.assert_array_equal(validated["trail_write_cursor"], np.asarray([3, 5]))


def test_persistent_delta_state_incremental_slot_connects_to_previous_owner_point():
    compact_state = {
        "trail_write_cursor": np.asarray([3], dtype=np.int32),
        "trail_active": np.asarray([[1, 1, 1, 0]], dtype=np.uint8),
        "trail_x": np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "trail_y": np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        "trail_radius": np.asarray([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "trail_owner": np.asarray([[0, 1, 0, 1]], dtype=np.int32),
        "trail_break_before": np.asarray([[1, 1, 0, 0]], dtype=np.uint8),
        "avatar_color": np.asarray([[0, 1]], dtype=np.int32),
    }

    delta, next_cursor, stats = _persistent_delta_state(
        np=np,
        compact_state=compact_state,
        previous_cursor=np.asarray([2], dtype=np.int32),
        previous_owner_pos=np.asarray([[[1.0, 10.0], [2.0, 20.0]]], dtype=np.float64),
        previous_owner_valid=np.asarray([[True, True]], dtype=bool),
        batch_size=1,
    )

    np.testing.assert_array_equal(next_cursor, np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(delta["reset_mask"], np.asarray([0], dtype=np.uint8))
    np.testing.assert_array_equal(delta["active"], np.asarray([[1]], dtype=np.uint8))
    np.testing.assert_array_equal(delta["owner"], np.asarray([[0]], dtype=np.int32))
    np.testing.assert_allclose(delta["x0"], np.asarray([[1.0]], dtype=np.float32))
    np.testing.assert_allclose(delta["y0"], np.asarray([[10.0]], dtype=np.float32))
    np.testing.assert_allclose(delta["x1"], np.asarray([[3.0]], dtype=np.float32))
    np.testing.assert_allclose(delta["y1"], np.asarray([[30.0]], dtype=np.float32))
    assert stats["reset_row_count"] == 0
    assert stats["delta_slot_count"] == 1


def test_persistent_delta_state_vectorized_incremental_multirow_matches_owner_state():
    compact_state = {
        "trail_write_cursor": np.asarray([3, 4], dtype=np.int32),
        "trail_active": np.ones((2, 5), dtype=np.uint8),
        "trail_x": np.asarray(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0],
                [10.0, 11.0, 12.0, 13.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "trail_y": np.asarray(
            [
                [21.0, 22.0, 23.0, 0.0, 0.0],
                [31.0, 32.0, 33.0, 34.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "trail_radius": np.ones((2, 5), dtype=np.float32),
        "trail_owner": np.asarray(
            [
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
            ],
            dtype=np.int32,
        ),
        "trail_break_before": np.zeros((2, 5), dtype=np.uint8),
        "avatar_color": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
    }

    delta, next_cursor, stats = _persistent_delta_state(
        np=np,
        compact_state=compact_state,
        previous_cursor=np.asarray([2, 2], dtype=np.int32),
        previous_owner_pos=np.asarray(
            [
                [[100.0, 101.0], [200.0, 201.0]],
                [[300.0, 301.0], [400.0, 401.0]],
            ],
            dtype=np.float64,
        ),
        previous_owner_valid=np.asarray([[True, True], [True, True]], dtype=bool),
        batch_size=2,
    )

    np.testing.assert_array_equal(next_cursor, np.asarray([3, 4], dtype=np.int32))
    np.testing.assert_array_equal(delta["reset_mask"], np.asarray([0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(delta["active"], np.asarray([[1, 0], [1, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(delta["owner"], np.asarray([[0, -1], [0, 1]], dtype=np.int32))
    np.testing.assert_allclose(
        delta["x0"], np.asarray([[100.0, 0.0], [300.0, 400.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        delta["y0"], np.asarray([[101.0, 0.0], [301.0, 401.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        delta["x1"], np.asarray([[3.0, 0.0], [12.0, 13.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        delta["y1"], np.asarray([[23.0, 0.0], [33.0, 34.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(delta["next_owner_pos"][0, 0], np.asarray([3.0, 23.0]))
    np.testing.assert_allclose(delta["next_owner_pos"][1, 1], np.asarray([13.0, 34.0]))
    assert stats["reset_row_count"] == 0
    assert stats["delta_slot_count"] == 3


def test_persistent_delta_state_vectorized_flag_false_preserves_results():
    compact_state = {
        "trail_write_cursor": np.asarray([3, 4], dtype=np.int32),
        "trail_active": np.ones((2, 5), dtype=np.uint8),
        "trail_x": np.asarray(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0],
                [10.0, 11.0, 12.0, 13.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "trail_y": np.asarray(
            [
                [21.0, 22.0, 23.0, 0.0, 0.0],
                [31.0, 32.0, 33.0, 34.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "trail_radius": np.ones((2, 5), dtype=np.float32),
        "trail_owner": np.asarray(
            [
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
            ],
            dtype=np.int32,
        ),
        "trail_break_before": np.zeros((2, 5), dtype=np.uint8),
        "avatar_color": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
    }
    kwargs = {
        "np": np,
        "compact_state": compact_state,
        "previous_cursor": np.asarray([2, 2], dtype=np.int32),
        "previous_owner_pos": np.asarray(
            [
                [[100.0, 101.0], [200.0, 201.0]],
                [[300.0, 301.0], [400.0, 401.0]],
            ],
            dtype=np.float64,
        ),
        "previous_owner_valid": np.asarray([[True, True], [True, True]], dtype=bool),
        "batch_size": 2,
    }

    fast_delta, fast_cursor, fast_stats = _persistent_delta_state(
        **kwargs,
        vectorized_fast_path=True,
    )
    slow_delta, slow_cursor, slow_stats = _persistent_delta_state(
        **kwargs,
        vectorized_fast_path=False,
    )

    assert fast_stats == slow_stats
    np.testing.assert_array_equal(fast_cursor, slow_cursor)
    for key in fast_delta:
        np.testing.assert_array_equal(fast_delta[key], slow_delta[key])


def test_persistent_delta_state_row_selective_cursor_regression_resets_only_regressed_row():
    compact_state = {
        "trail_write_cursor": np.asarray([2, 4], dtype=np.int32),
        "trail_active": np.asarray(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
            ],
            dtype=np.uint8,
        ),
        "trail_x": np.asarray(
            [
                [10.0, 11.0, 12.0, 13.0, 14.0, 0.0],
                [20.0, 21.0, 22.0, 23.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "trail_y": np.asarray(
            [
                [30.0, 31.0, 32.0, 33.0, 34.0, 0.0],
                [40.0, 41.0, 42.0, 43.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "trail_radius": np.ones((2, 6), dtype=np.float32),
        "trail_owner": np.asarray(
            [
                [0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1],
            ],
            dtype=np.int32,
        ),
        "trail_break_before": np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        ),
        "avatar_color": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
    }

    delta, next_cursor, stats = _persistent_delta_state(
        np=np,
        compact_state=compact_state,
        previous_cursor=np.asarray([5, 2], dtype=np.int32),
        previous_owner_pos=np.asarray(
            [
                [[100.0, 110.0], [120.0, 130.0]],
                [[200.0, 210.0], [220.0, 230.0]],
            ],
            dtype=np.float64,
        ),
        previous_owner_valid=np.asarray([[True, True], [True, True]], dtype=bool),
        batch_size=2,
    )

    np.testing.assert_array_equal(next_cursor, np.asarray([2, 4], dtype=np.int32))
    np.testing.assert_array_equal(delta["reset_mask"], np.asarray([1, 0], dtype=np.uint8))
    assert stats["reset_row_count"] == 1
    assert stats["delta_slot_count"] == 4

    # Row 0 regressed, so it is rebuilt from its current prefix and must not
    # connect to stale previous owner positions.
    np.testing.assert_allclose(delta["x0"][0, :2], np.asarray([10.0, 11.0], dtype=np.float32))
    np.testing.assert_allclose(delta["y0"][0, :2], np.asarray([30.0, 31.0], dtype=np.float32))
    np.testing.assert_allclose(delta["x1"][0, :2], np.asarray([10.0, 11.0], dtype=np.float32))
    np.testing.assert_allclose(delta["y1"][0, :2], np.asarray([30.0, 31.0], dtype=np.float32))

    # Row 1 did not regress, so only slots [2, 4) are appended. Slot 2
    # connects to the previous owner-0 point; slot 3 has break_before set and
    # therefore starts at its own current point.
    np.testing.assert_array_equal(delta["owner"][1, :2], np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_allclose(delta["x0"][1, :2], np.asarray([200.0, 23.0], dtype=np.float32))
    np.testing.assert_allclose(delta["y0"][1, :2], np.asarray([210.0, 43.0], dtype=np.float32))
    np.testing.assert_allclose(delta["x1"][1, :2], np.asarray([22.0, 23.0], dtype=np.float32))
    np.testing.assert_allclose(delta["y1"][1, :2], np.asarray([42.0, 43.0], dtype=np.float32))
    assert bool(delta["next_owner_valid"][0, 0])
    assert bool(delta["next_owner_valid"][0, 1])
    assert bool(delta["next_owner_valid"][1, 0])
    assert bool(delta["next_owner_valid"][1, 1])


def test_validate_boundary_config_rejects_unknown_surface_stack_backend():
    with pytest.raises(ValueError, match="surface_stack_backend"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "surface_stack_backend": "mystery",
            },
        )


def test_validate_boundary_config_rejects_cpu_reference_skip_during_parity_checks():
    with pytest.raises(ValueError, match="cpu_reference_interval"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 64,
                "verify_steps": 1,
                "cpu_reference_interval": 0,
            },
        )


def test_validate_boundary_config_rejects_dynamic_min_above_max_render_slots():
    with pytest.raises(ValueError, match="min_render_trail_slots"):
        _validate_boundary_config(
            np=np,
            config={
                "batch_size": 1,
                "trail_slots": 32,
                "dynamic_render_trail_slots": True,
                "min_render_trail_slots": 64,
            },
        )


def test_validate_boundary_config_accepts_deliberate_render_truncation_diagnostic():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 16,
            "body_capacity": 1024,
            "allow_render_truncation": True,
        },
    )

    assert config["allow_render_truncation"] is True
    assert config["render_config"]["allow_render_truncation"] is True


def test_validate_boundary_config_rejects_body_capacity_smaller_than_render_slots():
    with pytest.raises(ValueError, match="body_capacity"):
        _validate_boundary_config(
            np=np,
            config={"batch_size": 1, "trail_slots": 16, "body_capacity": 4},
        )


def test_validate_boundary_config_accepts_mock_collector_payload_flags():
    config = _validate_boundary_config(
        np=np,
        config={
            "batch_size": 1,
            "trail_slots": 4,
            "include_lightzero_payload_profile": True,
            "pickle_lightzero_payload": False,
            "include_rnd_meter": True,
            "rnd_batch_size": 2,
            "rnd_update_per_collect": 3,
            "rnd_device": "cpu",
        },
    )

    assert config["include_lightzero_payload_profile"] is True
    assert config["pickle_lightzero_payload"] is False
    assert config["include_rnd_meter"] is True
    assert config["rnd_batch_size"] == 2
    assert config["rnd_update_per_collect"] == 3
    assert config["rnd_device"] == "cpu"


def test_compact_hybrid_observation_profile_result_keeps_mapping_edges():
    result = {
        "schema_id": "s",
        "impl_id": "i",
        "ok": True,
        "profile_only": True,
        "compact_owned_loop_entrypoint_enabled": True,
        "compact_owned_loop_schema_id": "curvyzero_compact_owned_loop/v1",
        "compact_owned_loop_profile_only": True,
        "compact_owned_loop_calls_train_muzero": False,
        "compact_owned_loop_touches_live_runs": False,
        "compact_owned_loop_replay_store_owned": True,
        "compact_owned_loop_policy_version_handoff": True,
        "compact_owned_loop_policy_version_ref": "unit-policy-v1",
        "compact_owned_loop_model_version_ref": "unit-model-v1",
        "compact_owned_loop_policy_source": "unit_profile",
        "compact_owned_loop_telemetry": {
            "compact_owned_loop_sample_gate_calls": 2,
            "compact_owned_loop_sample_gate_last_sample_metadata": {
                "compact_owned_loop_policy_version_ref": "unit-policy-v1",
            },
        },
        "compact_owned_loop_replay_store_state_metadata": {
            "schema_id": "curvyzero_compact_replay_store_state/v1",
            "compact_owned_loop_replay_store_owned": True,
        },
        "compact_rollout_slab_enabled": True,
        "compact_rollout_slab_calls": 2,
        "compact_rollout_slab_total_roots": 80,
        "compact_rollout_slab_roots_per_call": 40.0,
        "compact_rollout_slab_committed_index_row_count": 40,
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_search_impl": "mock_search_service"
        },
        "compact_rollout_slab_telemetry_totals": {
            "compact_rollout_slab_search_service_total_sec": 0.25,
            "compact_rollout_slab_search_service_initial_output_decode_sec": 0.03,
            "compact_rollout_slab_search_service_tree_root_prior_build_sec": 0.04,
            "compact_rollout_slab_search_service_action_accounted_sec": 0.29,
            "compact_rollout_slab_search_service_action_residual_sec": 0.01,
            "compact_rollout_slab_search_service_core_accounted_sec": 0.20,
        },
        "compact_rollout_slab_sample_gate_enabled": True,
        "compact_rollout_slab_sample_gate_calls": 2,
        "compact_rollout_slab_sample_gate_opportunities": 3,
        "compact_rollout_slab_sample_gate_skipped_count": 1,
        "compact_rollout_slab_sample_gate_index_row_count": 40,
        "compact_rollout_slab_sample_gate_target_row_count": 40,
        "compact_rollout_slab_sample_gate_sample_row_count": 16,
        "compact_rollout_slab_sample_gate_batch_size": 16,
        "compact_rollout_slab_sample_gate_interval": 2,
        "compact_rollout_slab_sample_gate_sec": 0.05,
        "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows": 0,
        "compact_rollout_slab_sample_gate_last_telemetry": {
            "compact_rollout_slab_sample_gate_mode": "index_rows_to_target_rows_to_sample_batch",
        },
        "death_mode": "normal",
        "done_semantics_verified": True,
        "terminated_row_count": 1,
        "truncated_row_count": 0,
        "death_row_count": 1,
        "death_count_total": 1,
        "death_cause_count_by_name": {
            "none": 0,
            "wall": 0,
            "own_trail": 0,
            "opponent_trail": 1,
            "body_unknown": 0,
        },
        "normal_collision_death_causes": ["opponent_trail"],
        "normal_collision_death_hit_owner_present": True,
        "normal_collision_death_evidence_rows": [
            {
                "global_row": 3,
                "done": True,
                "terminated": True,
                "truncated": False,
                "death_count": 1,
                "death_player": [0, -1],
                "death_cause": ["opponent_trail"],
                "death_hit_owner": [1, -1],
                "winner": 1,
                "draw": False,
                "reward": [-1.0, 1.0],
                "final_reward_map": [-1.0, 1.0],
                "final_reward_map_matches_reward": True,
                "final_observation_row": True,
            }
        ],
        "terminal_final_observation_row_count": 1,
        "terminal_final_observation_before_autoreset_verified": True,
        "terminal_final_reward_map_row_count": 1,
        "terminal_final_reward_map_matches_reward_row_count": 1,
        "terminal_final_reward_map_verified": True,
        "max_ticks": 2000,
        "last_policy_env_id": list(range(40)),
        "last_policy_env_row": [idx // 2 for idx in range(40)],
        "last_policy_player": [idx % 2 for idx in range(40)],
        "last_payload_summary": {
            "global_rows": list(range(20)),
            "terminal_global_rows": [3],
            "autoreset_global_rows": [3],
        },
    }

    compact = _compact_hybrid_observation_profile_result(result)

    assert compact["last_policy_env_id_head"] == list(range(16))
    assert compact["last_policy_env_id_tail"] == list(range(24, 40))
    assert compact["last_policy_player_head"] == [idx % 2 for idx in range(16)]
    assert compact["last_payload_summary"]["global_rows_head"] == list(range(16))
    assert compact["last_payload_summary"]["global_rows_tail"] == list(range(4, 20))
    assert compact["last_payload_summary"]["terminal_global_rows"] == [3]
    assert compact["compact_owned_loop_entrypoint_enabled"] is True
    assert compact["death_mode"] == "normal"
    assert compact["terminated_row_count"] == 1
    assert compact["done_semantics_verified"] is True
    assert compact["truncated_row_count"] == 0
    assert compact["death_row_count"] == 1
    assert compact["death_count_total"] == 1
    assert compact["death_cause_count_by_name"]["opponent_trail"] == 1
    assert compact["normal_collision_death_causes"] == ["opponent_trail"]
    assert compact["normal_collision_death_hit_owner_present"] is True
    assert compact["normal_collision_death_evidence_rows"][0]["death_cause"] == ["opponent_trail"]
    assert compact["terminal_final_observation_row_count"] == 1
    assert compact["terminal_final_observation_before_autoreset_verified"] is True
    assert compact["terminal_final_reward_map_row_count"] == 1
    assert compact["terminal_final_reward_map_matches_reward_row_count"] == 1
    assert compact["terminal_final_reward_map_verified"] is True
    assert compact["compact_owned_loop_schema_id"] == "curvyzero_compact_owned_loop/v1"
    assert compact["compact_owned_loop_profile_only"] is True
    assert compact["compact_owned_loop_calls_train_muzero"] is False
    assert compact["compact_owned_loop_touches_live_runs"] is False
    assert compact["compact_owned_loop_replay_store_owned"] is True
    assert compact["compact_owned_loop_policy_version_handoff"] is True
    assert compact["compact_owned_loop_policy_version_ref"] == "unit-policy-v1"
    assert compact["compact_owned_loop_model_version_ref"] == "unit-model-v1"
    assert compact["compact_owned_loop_policy_source"] == "unit_profile"
    assert compact["compact_owned_loop_telemetry"]["compact_owned_loop_sample_gate_calls"] == 2
    assert compact["compact_owned_loop_replay_store_state_metadata"]["schema_id"] == (
        "curvyzero_compact_replay_store_state/v1"
    )
    assert compact["compact_rollout_slab_enabled"] is True
    assert compact["compact_rollout_slab_calls"] == 2
    assert compact["compact_rollout_slab_total_roots"] == 80
    assert compact["compact_rollout_slab_committed_index_row_count"] == 40
    assert (
        compact["compact_rollout_slab_last_telemetry"]["compact_rollout_slab_search_impl"]
        == "mock_search_service"
    )
    assert (
        compact["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_service_total_sec"
        ]
        == 0.25
    )
    assert (
        compact["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_service_initial_output_decode_sec"
        ]
        == 0.03
    )
    assert (
        compact["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_service_tree_root_prior_build_sec"
        ]
        == 0.04
    )
    assert (
        compact["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_service_action_accounted_sec"
        ]
        == 0.29
    )
    assert (
        compact["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_service_action_residual_sec"
        ]
        == 0.01
    )
    assert (
        compact["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_service_core_accounted_sec"
        ]
        == 0.20
    )
    assert compact["compact_rollout_slab_sample_gate_enabled"] is True
    assert compact["compact_rollout_slab_sample_gate_calls"] == 2
    assert compact["compact_rollout_slab_sample_gate_opportunities"] == 3
    assert compact["compact_rollout_slab_sample_gate_skipped_count"] == 1
    assert compact["compact_rollout_slab_sample_gate_index_row_count"] == 40
    assert compact["compact_rollout_slab_sample_gate_target_row_count"] == 40
    assert compact["compact_rollout_slab_sample_gate_sample_row_count"] == 16
    assert compact["compact_rollout_slab_sample_gate_batch_size"] == 16
    assert compact["compact_rollout_slab_sample_gate_interval"] == 2
    assert compact["compact_rollout_slab_sample_gate_sec"] == 0.05
    assert compact["compact_rollout_slab_sample_gate_mock_base_env_timestep_rows"] == 0
    assert (
        compact["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_mode"
        ]
        == "index_rows_to_target_rows_to_sample_batch"
    )
    assert compact["max_ticks"] == 2000


def test_trail_stats_after_owner_pack_reports_active_prefix_occupancy():
    compact_state = {
        "trail_write_cursor": np.asarray([0, 2, 5, 8], dtype=np.int32),
        "trail_active": np.ones((4, 16), dtype=np.uint8),
    }

    stats = _trail_stats_after_owner_pack(
        np=np,
        compact_state=compact_state,
        render_trail_slots=16,
    )

    assert stats["env_trail_slots"] == 16
    assert stats["max_render_trail_slots"] == 16
    assert stats["render_trail_slots"] == 16
    assert stats["active_trail_count_min"] == 0
    assert stats["active_trail_count_median"] == 3.5
    assert stats["active_trail_count_max"] == 8
    assert stats["active_trail_count_sum"] == 15
    assert stats["active_trail_fraction_median"] == pytest.approx(3.5 / 16)
    assert stats["active_trail_fraction_max"] == pytest.approx(8 / 16)
    assert stats["render_truncation_row_count"] == 0
    assert stats["render_truncation_max_dropped_slots"] == 0


def test_trail_stats_after_owner_pack_reports_render_truncation_risk():
    compact_state = {
        "trail_write_cursor": np.asarray([8, 16, 22], dtype=np.int32),
        "trail_active": np.ones((3, 1024), dtype=np.uint8),
    }

    stats = _trail_stats_after_owner_pack(
        np=np,
        compact_state=compact_state,
        render_trail_slots=16,
    )

    assert stats["env_trail_slots"] == 1024
    assert stats["active_trail_count_max"] == 22
    assert stats["render_truncation_row_count"] == 1
    assert stats["render_truncation_row_fraction"] == pytest.approx(1 / 3)
    assert stats["render_truncation_max_dropped_slots"] == 6


def test_truncate_compact_trails_for_render_caps_after_owner_order_pack():
    compact_state = {
        "trail_x": np.arange(8, dtype=np.float32).reshape(1, 8),
        "trail_y": np.arange(10, 18, dtype=np.float32).reshape(1, 8),
        "trail_radius": np.ones((1, 8), dtype=np.float32),
        "trail_owner": np.arange(8, dtype=np.int32).reshape(1, 8),
        "trail_active": np.ones((1, 8), dtype=np.uint8),
        "trail_break_before": np.zeros((1, 8), dtype=np.uint8),
        "trail_write_cursor": np.asarray([8], dtype=np.int32),
    }

    truncated = _truncate_compact_trails_for_render(
        np=np,
        compact_state=compact_state,
        render_trail_slots=4,
    )

    assert truncated["trail_x"].shape == (1, 4)
    np.testing.assert_array_equal(
        truncated["trail_x"], np.asarray([[0, 1, 2, 3]], dtype=np.float32)
    )
    assert int(truncated["trail_write_cursor"][0]) == 4


def test_select_render_trail_slots_uses_active_prefix_power_of_two():
    compact_state = {
        "trail_write_cursor": np.asarray([2, 17, 31], dtype=np.int32),
    }
    config = {
        "trail_slots": 1024,
        "dynamic_render_trail_slots": True,
        "min_render_trail_slots": 16,
    }

    assert _select_render_trail_slots(np=np, compact_state=compact_state, config=config) == 32


def test_select_render_trail_slots_clamps_to_max_render_slots():
    compact_state = {
        "trail_write_cursor": np.asarray([40], dtype=np.int32),
    }
    config = {
        "trail_slots": 32,
        "dynamic_render_trail_slots": True,
        "min_render_trail_slots": 16,
    }

    assert _select_render_trail_slots(np=np, compact_state=compact_state, config=config) == 32


def test_assert_no_render_truncation_if_required_rejects_lossy_rows():
    trail_stats = {
        "render_truncation_row_count": 2.0,
        "render_truncation_max_dropped_slots": 7.0,
        "render_trail_slots": 16.0,
        "active_trail_count_max": 23.0,
    }

    with pytest.raises(ValueError, match="would drop active trails"):
        _assert_no_render_truncation_if_required(
            config={"allow_render_truncation": False},
            trail_stats=trail_stats,
        )


def test_assert_no_render_truncation_if_required_allows_explicit_diagnostic_loss():
    trail_stats = {
        "render_truncation_row_count": 2.0,
        "render_truncation_max_dropped_slots": 7.0,
        "render_trail_slots": 16.0,
        "active_trail_count_max": 23.0,
    }

    _assert_no_render_truncation_if_required(
        config={"allow_render_truncation": True},
        trail_stats=trail_stats,
    )


def test_validate_boundary_config_rejects_rnd_without_payload_profile():
    with pytest.raises(ValueError, match="include_rnd_meter requires"):
        _validate_boundary_config(
            np=np,
            config={"batch_size": 1, "trail_slots": 4, "include_rnd_meter": True},
        )


def test_push_row_major_frames_into_stack_shifts_fifo_and_normalizes_last_channel():
    stacks = np.zeros((2, 2, 4, 64, 64), dtype=np.float32)
    stacks[:, :, 0, 0, 0] = 0.1
    stacks[:, :, 1, 0, 0] = 0.2
    stacks[:, :, 2, 0, 0] = 0.3
    stacks[:, :, 3, 0, 0] = 0.4
    frames = np.zeros((2, 2, 1, 64, 64), dtype=np.uint8)
    frames[:, :, 0, 0, 0] = np.asarray([[0, 255], [128, 64]], dtype=np.uint8)

    elapsed = _push_row_major_frames_into_stack(stacks, frames)

    assert elapsed >= 0.0
    np.testing.assert_allclose(stacks[:, :, 0, 0, 0], 0.2)
    np.testing.assert_allclose(stacks[:, :, 1, 0, 0], 0.3)
    np.testing.assert_allclose(stacks[:, :, 2, 0, 0], 0.4)
    expected_last = np.asarray([[0.0, 1.0], [128 / 255, 64 / 255]], dtype=np.float32)
    np.testing.assert_allclose(stacks[:, :, 3, 0, 0], expected_last)


def test_push_row_major_frames_into_stack_rejects_bad_dtype():
    stacks = np.zeros((1, 2, 4, 64, 64), dtype=np.float32)
    frames = np.zeros((1, 2, 1, 64, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="frames must be uint8"):
        _push_row_major_frames_into_stack(stacks, frames)


def test_latest_uint8_frames_from_stack_rounds_latest_channel():
    stack = np.zeros((1, 2, 4, 64, 64), dtype=np.float32)
    stack[0, 0, -1, 3, 4] = np.float32(10.0 / 255.0)
    stack[0, 1, -1, 5, 6] = np.float32(128.0 / 255.0)

    frames = _latest_uint8_frames_from_stack(np=np, stack=stack)

    assert frames.shape == (1, 2, 1, 64, 64)
    assert frames.dtype == np.uint8
    assert int(frames[0, 0, 0, 3, 4]) == 10
    assert int(frames[0, 1, 0, 5, 6]) == 128


def test_push_row_major_frames_into_stack_resets_selected_terminal_rows_only():
    stacks = np.zeros((3, 2, 4, 64, 64), dtype=np.float32)
    stacks[:, :, 0, 0, 0] = 0.1
    stacks[:, :, 1, 0, 0] = 0.2
    stacks[:, :, 2, 0, 0] = 0.3
    stacks[:, :, 3, 0, 0] = 0.4
    frames = np.zeros((3, 2, 1, 64, 64), dtype=np.uint8)
    frames[:, :, 0, 0, 0] = np.asarray(
        [[10, 20], [30, 40], [50, 60]],
        dtype=np.uint8,
    )
    row_mask = np.asarray([False, True, False], dtype=bool)

    _push_row_major_frames_into_stack(
        stacks,
        frames,
        row_mask=row_mask,
        reset_selected_rows=True,
    )

    np.testing.assert_allclose(stacks[0, :, :, 0, 0], np.asarray([[0.1, 0.2, 0.3, 0.4]] * 2))
    np.testing.assert_allclose(stacks[2, :, :, 0, 0], np.asarray([[0.1, 0.2, 0.3, 0.4]] * 2))
    np.testing.assert_allclose(stacks[1, :, :3, 0, 0], 0.0)
    np.testing.assert_allclose(stacks[1, :, 3, 0, 0], np.asarray([30 / 255, 40 / 255]))


def test_tolerant_parity_accepts_tiny_uint8_and_stack_drift():
    candidate_frames = np.zeros((1, 2, 1, 64, 64), dtype=np.uint8)
    reference_frames = candidate_frames.copy()
    reference_frames[0, 0, 0, 10, 10] = 1
    candidate_stacks = np.zeros((1, 2, 4, 64, 64), dtype=np.float32)
    reference_stacks = candidate_stacks.copy()
    reference_stacks[0, 0, 3, 10, 10] = np.float32(1.0 / 255.0)
    config = {
        "parity_mode": BOUNDARY_PARITY_MODE_TOLERANT,
        "parity_max_abs_diff": 1,
        "parity_max_mismatch_fraction": 1.0e-4,
    }

    summary = _assert_parity(
        label="tiny",
        candidate_frames=candidate_frames,
        reference_frames=reference_frames,
        candidate_stacks=candidate_stacks,
        reference_stacks=reference_stacks,
        config=config,
    )

    assert summary["raw_frames"]["exact"] is False
    assert summary["raw_frames"]["tolerated"] is True
    assert summary["raw_frames"]["sample_plane"]["plane_mismatch_count"] == 1
    assert summary["stacks"]["tolerated"] is True


def test_tolerant_parity_rejects_large_uint8_drift():
    candidate_frames = np.zeros((1, 2, 1, 64, 64), dtype=np.uint8)
    reference_frames = candidate_frames.copy()
    reference_frames[0, 0, 0, 10, 10] = 5
    candidate_stacks = np.zeros((1, 2, 4, 64, 64), dtype=np.float32)
    reference_stacks = candidate_stacks.copy()
    config = {
        "parity_mode": BOUNDARY_PARITY_MODE_TOLERANT,
        "parity_max_abs_diff": 1,
        "parity_max_mismatch_fraction": 1.0e-4,
    }

    with pytest.raises(AssertionError, match="tolerant parity failed"):
        _assert_parity(
            label="large",
            candidate_frames=candidate_frames,
            reference_frames=reference_frames,
            candidate_stacks=candidate_stacks,
            reference_stacks=reference_stacks,
            config=config,
        )


def test_exact_parity_rejects_tiny_drift():
    candidate_frames = np.zeros((1, 2, 1, 64, 64), dtype=np.uint8)
    reference_frames = candidate_frames.copy()
    reference_frames[0, 0, 0, 10, 10] = 1
    candidate_stacks = np.zeros((1, 2, 4, 64, 64), dtype=np.float32)
    reference_stacks = candidate_stacks.copy()
    config = {
        "parity_mode": BOUNDARY_PARITY_MODE_EXACT,
        "parity_max_abs_diff": 1,
        "parity_max_mismatch_fraction": 1.0e-4,
    }

    with pytest.raises(AssertionError, match=r"exact parity failed at exact\.raw_frames"):
        _assert_parity(
            label="exact",
            candidate_frames=candidate_frames,
            reference_frames=reference_frames,
            candidate_stacks=candidate_stacks,
            reference_stacks=reference_stacks,
            config=config,
        )
