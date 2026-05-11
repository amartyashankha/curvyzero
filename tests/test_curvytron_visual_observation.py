import numpy as np

from curvyzero.env.source_env import CurvyTronSourceEnv
from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_COMPARISON_TARGET,
    DEBUG_OCCUPANCY_GRAY64_FINAL_OBSERVATION_POLICY,
    DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_POLICY,
    DEBUG_OCCUPANCY_GRAY64_LABEL,
    DEBUG_OCCUPANCY_GRAY64_PIXEL_FIDELITY_BLOCKER,
    DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
    DEBUG_OCCUPANCY_GRAY64_SOURCE_CLAIM_ID,
    DEBUG_OCCUPANCY_GRAY64_SOURCE_STATE_FIELDS,
    DebugOccupancyGray64FrameStack,
    DebugOccupancyGray64Renderer,
    debug_occupancy_gray64_metadata,
    debug_occupancy_gray64_schema,
    normalize_debug_occupancy_gray64_for_lightzero,
)


def test_debug_visual_observation_shape_dtype_and_nonempty_source_positions():
    env = CurvyTronSourceEnv(random_constant=0.5)
    snapshot = env.reset(player_count=2, warmup_ms=0.0)

    frame = DebugOccupancyGray64Renderer().render(
        snapshot,
        world_bodies=env.world_bodies_snapshot(),
    )

    assert frame.shape == (1, 64, 64)
    assert frame.dtype == np.uint8
    assert int(np.count_nonzero(frame)) >= 1


def test_debug_visual_observation_is_deterministic_for_fixed_source_setup():
    env_a = CurvyTronSourceEnv(random_constant=0.5)
    env_b = CurvyTronSourceEnv(random_constant=0.5)
    snapshot_a = env_a.reset(player_count=2, warmup_ms=0.0)
    snapshot_b = env_b.reset(player_count=2, warmup_ms=0.0)

    renderer = DebugOccupancyGray64Renderer()
    frame_a = renderer.render(snapshot_a, world_bodies=env_a.world_bodies_snapshot()).copy()
    frame_b = renderer.render(snapshot_b, world_bodies=env_b.world_bodies_snapshot()).copy()

    np.testing.assert_array_equal(frame_a, frame_b)


def test_debug_visual_frame_stack_updates_to_lightzero_style_shape():
    env = CurvyTronSourceEnv(random_constant=0.5)
    snapshot = env.reset(player_count=2, warmup_ms=0.0)
    raw_frame = DebugOccupancyGray64Renderer().render(
        snapshot,
        world_bodies=env.world_bodies_snapshot(),
    )
    frame = normalize_debug_occupancy_gray64_for_lightzero(raw_frame)

    stack = DebugOccupancyGray64FrameStack().update(frame, copy=True)

    assert stack.shape == (4, 64, 64)
    assert stack.dtype == np.float32
    np.testing.assert_array_equal(stack[-1], frame[0])


def test_debug_visual_frame_stack_policy_is_fifo_and_copy_safe():
    first = np.zeros((1, 64, 64), dtype=np.float32)
    second = np.zeros((1, 64, 64), dtype=np.float32)
    first[0, 3, 5] = 0.25
    second[0, 7, 11] = 1.0

    stacker = DebugOccupancyGray64FrameStack()
    first_stack = stacker.update(first, copy=True)
    second_stack = stacker.update(second, copy=True)

    assert first_stack.shape == (4, 64, 64)
    assert second_stack.shape == (4, 64, 64)
    assert first_stack.dtype == np.float32
    assert second_stack.dtype == np.float32
    np.testing.assert_array_equal(first_stack[-1], first[0])
    np.testing.assert_array_equal(second_stack[-2], first[0])
    np.testing.assert_array_equal(second_stack[-1], second[0])
    second[0, 7, 11] = 0.0
    np.testing.assert_array_equal(second_stack[-1], stacker.stack[-1])


def test_debug_visual_schema_label_and_caveat_are_explicit():
    schema = debug_occupancy_gray64_schema()

    assert schema["label"] == DEBUG_OCCUPANCY_GRAY64_LABEL
    assert schema["observation_schema_id"] == DEBUG_OCCUPANCY_GRAY64_LABEL
    assert schema["schema_hash"] == DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH
    assert schema["shape"] == [1, 64, 64]
    assert schema["stack_shape"] == [4, 64, 64]
    assert schema["raw_renderer_dtype"] == "uint8"
    assert schema["lightzero_payload_dtype"] == "float32"
    assert schema["dtype"] == "float32"
    assert schema["range"] == [0.0, 1.0]
    assert schema["raw_value_range"] == [0, 255]
    assert schema["surface"] == "debug_visual_tensor"
    assert schema["channel_order"] == "CHW"
    assert schema["stack_axis"] == "time_as_channel"
    assert schema["truth_level"] == "debug_non_fidelity"
    assert schema["source_fidelity_level"] == "none"
    assert schema["source_claim_id"] == DEBUG_OCCUPANCY_GRAY64_SOURCE_CLAIM_ID
    assert schema["source_state_fields"] == list(DEBUG_OCCUPANCY_GRAY64_SOURCE_STATE_FIELDS)
    assert schema["comparison_target"] == DEBUG_OCCUPANCY_GRAY64_COMPARISON_TARGET
    assert schema["source_pixel_fidelity_blocker"] == DEBUG_OCCUPANCY_GRAY64_PIXEL_FIDELITY_BLOCKER
    assert schema["source_backed_observation_fidelity"] is False
    assert schema["frame_stack_owner"] == "optimizer"
    assert schema["frame_stack_policy"] == DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_POLICY
    assert schema["final_observation_policy"] == DEBUG_OCCUPANCY_GRAY64_FINAL_OBSERVATION_POLICY
    assert schema["browser_pixel_fidelity"] is False
    assert schema["uses_ale"] is False
    assert schema["ale_usage"] == "none"
    assert "debug/profiling" in schema["caveat"]
    assert "not source visual fidelity" in schema["caveat"]


def test_debug_visual_timing_metadata_has_required_claim_fields():
    metadata = debug_occupancy_gray64_metadata(includes_render_cost=True)

    expected = {
        "observation_schema_id": DEBUG_OCCUPANCY_GRAY64_LABEL,
        "truth_level": "debug_non_fidelity",
        "source_fidelity_level": "none",
        "surface": "debug_visual_tensor",
        "shape": [1, 64, 64],
        "dtype": "float32",
        "raw_renderer_dtype": "uint8",
        "range": [0.0, 1.0],
        "raw_value_range": [0, 255],
        "perspective": "global_arena_debug",
        "frame_stack_owner": "optimizer",
        "frame_stack_policy": DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_POLICY,
        "renderer_impl_id": "curvyzero_debug_occupancy_gray64_numpy/v0",
        "schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
        "observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
        "includes_render_cost": True,
        "source_claim_id": DEBUG_OCCUPANCY_GRAY64_SOURCE_CLAIM_ID,
        "source_state_fields": list(DEBUG_OCCUPANCY_GRAY64_SOURCE_STATE_FIELDS),
        "comparison_target": DEBUG_OCCUPANCY_GRAY64_COMPARISON_TARGET,
        "source_pixel_fidelity_blocker": DEBUG_OCCUPANCY_GRAY64_PIXEL_FIDELITY_BLOCKER,
        "final_observation_policy": DEBUG_OCCUPANCY_GRAY64_FINAL_OBSERVATION_POLICY,
        "browser_pixel_fidelity": False,
        "uses_ale": False,
        "ale_usage": "none",
        "source_backed_observation_fidelity": False,
    }
    for key, value in expected.items():
        assert metadata[key] == value
    assert metadata["surface"] == "debug_visual_tensor"
    assert metadata["ale_usage"] == "none"
    assert metadata["source_backed_observation_fidelity"] is False


def test_debug_visual_rejects_non_mapping_snapshot_entries():
    renderer = DebugOccupancyGray64Renderer()
    snapshot = {"game": {"size": 1000}, "avatars": ["not-a-mapping"]}

    try:
        renderer.render(snapshot)
    except ValueError as exc:
        assert "snapshot.avatars[0] must be a mapping" in str(exc)
    else:
        raise AssertionError("expected invalid avatar entry to fail")
