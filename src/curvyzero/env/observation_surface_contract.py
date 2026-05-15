"""One source of truth for CurvyTron policy observation surfaces."""

from __future__ import annotations

from typing import Any

from curvyzero.env.vector_visual_observation import (
    BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE,
    SOURCE_STATE_GRAY64_SHAPE,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    TRAIL_RENDER_MODE_BROWSER_LINES,
)


POLICY_OBSERVATION_CONTRACT_ID = "curvyzero_policy_observation_surface/v1"
POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID = (
    "curvyzero_policy_observation_controlled_player_perspective/v1"
)
POLICY_OBSERVATION_PERSPECTIVE = "controlled_player_view"
POLICY_OBSERVATION_PERSPECTIVE_OWNER = "training_env_learner_or_seat_mapping"
POLICY_OBSERVATION_PERSPECTIVE_PLAYER_AXIS = (
    "single-agent obs is the selected controlled-player view; batched obs[b,p] "
    "is player p's controlled-player view"
)

POLICY_TRAIL_RENDER_MODE = TRAIL_RENDER_MODE_BROWSER_LINES
POLICY_BONUS_RENDER_MODE = BONUS_RENDER_MODE_SIMPLE_SYMBOLS
POLICY_RENDER_SURFACE_LABEL = f"{POLICY_TRAIL_RENDER_MODE}+{POLICY_BONUS_RENDER_MODE}"
POLICY_SURFACE_SOURCE_CONTRACT_DEFAULT = "observation_surface_contract"

POLICY_FRAME_STACK_DEPTH = 4
POLICY_SINGLE_FRAME_SHAPE = SOURCE_STATE_GRAY64_SHAPE
POLICY_STACK_SHAPE = (
    POLICY_FRAME_STACK_DEPTH,
    SOURCE_STATE_GRAY64_SHAPE[1],
    SOURCE_STATE_GRAY64_SHAPE[2],
)
POLICY_SOURCE_FRAME_SIZE = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
POLICY_TARGET_FRAME_SIZE = SOURCE_STATE_GRAY64_SHAPE[1]
POLICY_RAW_DTYPE = "uint8"
POLICY_MODEL_DTYPE = "float32"
POLICY_RAW_VALUE_RANGE = (0, 255)
POLICY_MODEL_VALUE_RANGE = tuple(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE)

POLICY_DRAW_ORDER = ("background", "trails", "bonuses", "heads")
POLICY_TRAIL_GEOMETRY = "browser_lines_source_state_segments"
POLICY_BONUS_GEOMETRY = "simple_type_symbols_on_source_canvas_then_downsampled"
POLICY_BONUS_COMPOSITION = "overwrite_nonzero_symbol_pixels"
POLICY_HEAD_COMPOSITION = "heads_draw_after_bonuses"
POLICY_GRAYSCALE_METHOD = "BT.601_luma"
POLICY_DOWNSAMPLE_METHOD = "11x11_area_average_from_704_to_64"

POLICY_OBSERVATION_BACKEND_CPU = "cpu_oracle"
POLICY_OBSERVATION_BACKEND_GPU = "jax_gpu"
POLICY_OBSERVATION_BACKENDS = (
    POLICY_OBSERVATION_BACKEND_CPU,
    POLICY_OBSERVATION_BACKEND_GPU,
)
DEFAULT_POLICY_OBSERVATION_BACKEND = POLICY_OBSERVATION_BACKEND_CPU
CURRENT_RELIABLE_POLICY_OBSERVATION_BACKEND = POLICY_OBSERVATION_BACKEND_CPU
EXPERIMENTAL_SCALAR_POLICY_OBSERVATION_BACKEND = POLICY_OBSERVATION_BACKEND_GPU
TARGET_POLICY_OBSERVATION_BACKEND = CURRENT_RELIABLE_POLICY_OBSERVATION_BACKEND
POLICY_OBSERVATION_BACKEND_PRODUCTION_DIRECTION = (
    "batched_gpu_observation_backend_not_scalar_jax_gpu"
)

REFERENCE_ARTIFACT_TRAIL_RENDER_MODE = TRAIL_RENDER_MODE_BROWSER_LINES


def policy_observation_surface(
    *,
    trail_render_mode: str = POLICY_TRAIL_RENDER_MODE,
    bonus_render_mode: str = POLICY_BONUS_RENDER_MODE,
    backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND,
) -> dict[str, Any]:
    return {
        "contract_id": POLICY_OBSERVATION_CONTRACT_ID,
        "perspective_schema_id": POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
        "perspective": POLICY_OBSERVATION_PERSPECTIVE,
        "perspective_owner": POLICY_OBSERVATION_PERSPECTIVE_OWNER,
        "perspective_player_axis": POLICY_OBSERVATION_PERSPECTIVE_PLAYER_AXIS,
        "trail_render_mode": trail_render_mode,
        "bonus_render_mode": bonus_render_mode,
        "surface_label": f"{trail_render_mode}+{bonus_render_mode}",
        "stack_shape": list(POLICY_STACK_SHAPE),
        "single_frame_shape": list(POLICY_SINGLE_FRAME_SHAPE),
        "source_frame_size": POLICY_SOURCE_FRAME_SIZE,
        "target_frame_size": POLICY_TARGET_FRAME_SIZE,
        "draw_order": list(POLICY_DRAW_ORDER),
        "trail_geometry": POLICY_TRAIL_GEOMETRY,
        "bonus_geometry": POLICY_BONUS_GEOMETRY,
        "bonus_composition": POLICY_BONUS_COMPOSITION,
        "head_composition": POLICY_HEAD_COMPOSITION,
        "grayscale_method": POLICY_GRAYSCALE_METHOD,
        "downsample_method": POLICY_DOWNSAMPLE_METHOD,
        "default_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "backend": backend,
        "current_reliable_backend": CURRENT_RELIABLE_POLICY_OBSERVATION_BACKEND,
        "experimental_scalar_backend": EXPERIMENTAL_SCALAR_POLICY_OBSERVATION_BACKEND,
        "target_backend": TARGET_POLICY_OBSERVATION_BACKEND,
        "backend_production_direction": POLICY_OBSERVATION_BACKEND_PRODUCTION_DIRECTION,
        "is_target_surface": is_policy_surface(
            trail_render_mode=trail_render_mode,
            bonus_render_mode=bonus_render_mode,
        ),
    }


def is_policy_surface(*, trail_render_mode: str | None, bonus_render_mode: str | None) -> bool:
    return (
        trail_render_mode == POLICY_TRAIL_RENDER_MODE
        and bonus_render_mode == POLICY_BONUS_RENDER_MODE
    )


__all__ = [
    "CURRENT_RELIABLE_POLICY_OBSERVATION_BACKEND",
    "DEFAULT_POLICY_OBSERVATION_BACKEND",
    "EXPERIMENTAL_SCALAR_POLICY_OBSERVATION_BACKEND",
    "POLICY_BONUS_COMPOSITION",
    "POLICY_BONUS_GEOMETRY",
    "POLICY_BONUS_RENDER_MODE",
    "POLICY_DOWNSAMPLE_METHOD",
    "POLICY_DRAW_ORDER",
    "POLICY_FRAME_STACK_DEPTH",
    "POLICY_GRAYSCALE_METHOD",
    "POLICY_HEAD_COMPOSITION",
    "POLICY_MODEL_DTYPE",
    "POLICY_MODEL_VALUE_RANGE",
    "POLICY_OBSERVATION_BACKEND_CPU",
    "POLICY_OBSERVATION_BACKEND_GPU",
    "POLICY_OBSERVATION_BACKEND_PRODUCTION_DIRECTION",
    "POLICY_OBSERVATION_BACKENDS",
    "POLICY_OBSERVATION_CONTRACT_ID",
    "POLICY_OBSERVATION_PERSPECTIVE",
    "POLICY_OBSERVATION_PERSPECTIVE_OWNER",
    "POLICY_OBSERVATION_PERSPECTIVE_PLAYER_AXIS",
    "POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID",
    "POLICY_RAW_DTYPE",
    "POLICY_RAW_VALUE_RANGE",
    "POLICY_RENDER_SURFACE_LABEL",
    "POLICY_SURFACE_SOURCE_CONTRACT_DEFAULT",
    "POLICY_SINGLE_FRAME_SHAPE",
    "POLICY_SOURCE_FRAME_SIZE",
    "POLICY_STACK_SHAPE",
    "POLICY_TARGET_FRAME_SIZE",
    "POLICY_TRAIL_GEOMETRY",
    "POLICY_TRAIL_RENDER_MODE",
    "REFERENCE_ARTIFACT_TRAIL_RENDER_MODE",
    "TARGET_POLICY_OBSERVATION_BACKEND",
    "is_policy_surface",
    "policy_observation_surface",
]
