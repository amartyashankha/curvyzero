"""Environment-owned source-state visual observation contract.

This renderer is intentionally weaker than browser/canvas pixel parity. It
rasterizes the vector runtime's source-state arrays into a fixed grayscale frame
so downstream visual plumbing has a source-backed target that is not the older
debug occupancy smoke surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from pathlib import Path
from typing import Any
import struct
import time
import zlib

import numpy as np

from curvyzero.env.trainer_contract import stable_contract_hash


SOURCE_STATE_GRAY64_SCHEMA_ID = "curvyzero_source_state_gray64/v0"
SOURCE_STATE_GRAY64_RENDERER_IMPL_ID = "curvyzero_source_state_gray64_numpy/v0"
SOURCE_STATE_GRAY64_SOURCE_CLAIM_ID = "curvyzero_vector_runtime_source_state_gray64/v0"
SOURCE_STATE_GRAY64_SHAPE = (1, 64, 64)
SOURCE_STATE_GRAY64_DTYPE = "uint8"
SOURCE_STATE_GRAY64_VALUE_RANGE = (0, 255)
SOURCE_STATE_GRAY64_NORMALIZED_DTYPE = "float32"
SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE = (0.0, 1.0)
SOURCE_STATE_GRAY64_SURFACE = "source_state_visual_tensor"
SOURCE_STATE_GRAY64_TRUTH_LEVEL = "source_state_backed_non_browser_pixel"
SOURCE_STATE_GRAY64_SOURCE_FIDELITY_LEVEL = "source_vector_state_geometry_raster"
SOURCE_STATE_GRAY64_PERSPECTIVE = "global_arena_source_state"
SOURCE_STATE_GRAY64_FRAME_STACK_OWNER = "optimizer"
SOURCE_STATE_GRAY64_FRAME_STACK_POLICY = (
    "single_frame_renderer; downstream adapter may stack normalized frames FIFO"
)
SOURCE_STATE_GRAY64_COMPARISON_TARGET = "curvyzero_vector_runtime_state_arrays"
SOURCE_STATE_GRAY64_TERMINAL_POLICY = (
    "renderer consumes the supplied row exactly, including terminal rows; public envs must "
    "capture final visual observations before any autoreset mutates the source state"
)
SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY = False
SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM = "not_claimed"
SOURCE_STATE_GRAY64_PIXEL_FIDELITY_BLOCKER = (
    "no browser/canvas pixel parity artifact or source client canvas renderer is available yet"
)
SOURCE_STATE_GRAY64_SOURCE_STATE_BACKED = True
SOURCE_STATE_GRAY64_USES_ALE = False
SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID = "curvyzero_source_state_rgb_canvas_like/v0"
SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL = "source_state_backed_browser_like_non_pixel_parity"
SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE = 704
SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB = (34, 34, 34)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB = (255, 190, 40)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_OUTLINE_RGB = (255, 255, 255)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_RGB = (
    SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB,
    (78, 78, 78),  # BonusSelfSmall
    (92, 92, 92),  # BonusSelfSlow
    (106, 106, 106),  # BonusSelfFast
    (120, 120, 120),  # BonusSelfMaster
    (134, 134, 134),  # BonusEnemySlow
    (148, 148, 148),  # BonusEnemyFast
    (162, 162, 162),  # BonusEnemyBig
    (176, 176, 176),  # BonusEnemyInverse
    (190, 190, 190),  # BonusEnemyStraightAngle
    (204, 204, 204),  # BonusGameBorderless
    (218, 218, 218),  # BonusAllColor
    (232, 232, 232),  # BonusGameClear
)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_CODE_BY_NAME = {
    "BonusSelfSmall": 1,
    "BonusSelfSlow": 2,
    "BonusSelfFast": 3,
    "BonusSelfMaster": 4,
    "BonusEnemySlow": 5,
    "BonusEnemyFast": 6,
    "BonusEnemyBig": 7,
    "BonusEnemyInverse": 8,
    "BonusEnemyStraightAngle": 9,
    "BonusGameBorderless": 10,
    "BonusAllColor": 11,
    "BonusGameClear": 12,
}
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_NAME_BY_CODE = (
    None,
    "BonusSelfSmall",
    "BonusSelfSlow",
    "BonusSelfFast",
    "BonusSelfMaster",
    "BonusEnemySlow",
    "BonusEnemyFast",
    "BonusEnemyBig",
    "BonusEnemyInverse",
    "BonusEnemyStraightAngle",
    "BonusGameBorderless",
    "BonusAllColor",
    "BonusGameClear",
)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH = (
    "third_party/curvytron-reference/web/images/bonus.png"
)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_COLUMNS = 3
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_ROWS = 4
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES = (
    "BonusSelfFast",
    "BonusEnemyFast",
    "BonusSelfSlow",
    "BonusEnemySlow",
    "BonusGameBorderless",
    "BonusSelfMaster",
    "BonusEnemyBig",
    "BonusAllColor",
    "BonusEnemyInverse",
    "BonusSelfSmall",
    "BonusGameClear",
    "BonusEnemyStraightAngle",
)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_INDEX_BY_NAME = {
    name: index for index, name in enumerate(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES)
}
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_CODE_BY_SPRITE_INDEX = tuple(
    SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_CODE_BY_NAME[name]
    for name in SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES
)
SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 80, 255),
    (255, 240, 0),
)
TRAIL_RENDER_MODE_BROWSER_LINES = "browser_lines"
TRAIL_RENDER_MODE_BODY_CIRCLES_FAST = "body_circles_fast"
TRAIL_RENDER_MODES = frozenset(
    (TRAIL_RENDER_MODE_BROWSER_LINES, TRAIL_RENDER_MODE_BODY_CIRCLES_FAST)
)
TRAIL_RENDER_MODE_ORDER = (
    TRAIL_RENDER_MODE_BROWSER_LINES,
    TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
)
TRAIL_RENDER_MODE_DEFAULT = TRAIL_RENDER_MODE_BROWSER_LINES
BONUS_RENDER_MODE_BROWSER_SPRITES = "browser_sprites"
BONUS_RENDER_MODE_CIRCLES_FAST = "circles_fast"
BONUS_RENDER_MODE_SIMPLE_SYMBOLS = "simple_symbols"
BONUS_RENDER_MODES = frozenset(
    (
        BONUS_RENDER_MODE_BROWSER_SPRITES,
        BONUS_RENDER_MODE_CIRCLES_FAST,
        BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    )
)
BONUS_RENDER_MODE_ORDER = (
    BONUS_RENDER_MODE_BROWSER_SPRITES,
    BONUS_RENDER_MODE_CIRCLES_FAST,
    BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
)
BONUS_RENDER_MODE_DEFAULT = BONUS_RENDER_MODE_BROWSER_SPRITES
BONUS_SYMBOL_RENDER_MODE_DEFAULT = BONUS_RENDER_MODE_SIMPLE_SYMBOLS
BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE = (68, 148, 196)
BONUS_SYMBOL_INNER_LUMA = 212
BONUS_SYMBOL_BASE_SIZE = 7
SOURCE_STATE_RGB_BROWSER_LINES_RENDERER_IMPL_ID = (
    "curvyzero_source_state_rgb_canvas_like_browser_lines_numpy/v0"
)
SOURCE_STATE_RGB_BODY_CIRCLES_FAST_RENDERER_IMPL_ID = (
    "curvyzero_source_state_rgb_canvas_like_body_circles_fast_numpy/v0"
)
SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID = SOURCE_STATE_RGB_BROWSER_LINES_RENDERER_IMPL_ID
SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID = "curvyzero_source_state_canvas_gray64/v0"
SOURCE_STATE_CANVAS_GRAY64_BROWSER_LINES_RENDERER_IMPL_ID = (
    "curvyzero_source_state_canvas_gray64_browser_lines_downsampled_canvas_numpy/v0"
)
SOURCE_STATE_CANVAS_GRAY64_BODY_CIRCLES_FAST_RENDERER_IMPL_ID = (
    "curvyzero_source_state_canvas_gray64_body_circles_fast_downsampled_canvas_numpy/v0"
)
SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID = (
    SOURCE_STATE_CANVAS_GRAY64_BROWSER_LINES_RENDERER_IMPL_ID
)
SOURCE_STATE_CANVAS_GRAY64_SOURCE_CLAIM_ID = (
    "curvyzero_vector_runtime_source_state_canvas_gray64/v0"
)
SOURCE_STATE_CANVAS_GRAY64_SHAPE = SOURCE_STATE_GRAY64_SHAPE
SOURCE_STATE_CANVAS_GRAY64_DTYPE = SOURCE_STATE_GRAY64_DTYPE
SOURCE_STATE_CANVAS_GRAY64_VALUE_RANGE = SOURCE_STATE_GRAY64_VALUE_RANGE
SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE = SOURCE_STATE_GRAY64_NORMALIZED_DTYPE
SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE = SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE
SOURCE_STATE_CANVAS_GRAY64_SURFACE = "source_state_canvas_gray64_tensor"
SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL = SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL
SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL = (
    "source_vector_state_canvas_like_rgb_luma_area_downsampled_raster"
)
SOURCE_STATE_CANVAS_GRAY64_PERSPECTIVE = "global_browser_like_source_state"
SOURCE_STATE_CANVAS_GRAY64_COMPARISON_TARGET = "curvyzero_source_state_rgb_canvas_like/v0"
SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY = False
SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM = (
    "browser_like_source_state_renderer_not_dom_canvas"
)
SOURCE_STATE_CANVAS_GRAY64_PIXEL_FIDELITY_BLOCKER = (
    "source-state canvas-like renderer is not yet compared against real browser canvas pixels"
)
SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED = True
SOURCE_STATE_CANVAS_GRAY64_USES_ALE = False
SOURCE_STATE_GRAY64_STATE_FIELDS = (
    "state.tick",
    "state.elapsed_ms",
    "state.map_size",
    "state.present",
    "state.alive",
    "state.pos",
    "state.radius",
    "state.body_active",
    "state.body_write_cursor",
    "state.body_pos",
    "state.body_radius",
    "state.body_owner",
    "state.done",
    "state.terminated",
    "state.truncated",
    "state.terminal_reason",
)
SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS = (
    "state.bonus_active",
    "state.bonus_pos",
    "state.bonus_radius",
)
SOURCE_STATE_VISUAL_TRAIL_FIELDS = (
    "state.visual_trail_active",
    "state.visual_trail_write_cursor",
    "state.visual_trail_pos",
    "state.visual_trail_radius",
    "state.visual_trail_owner",
    "state.visual_trail_break_before",
)
_SOURCE_STATE_ARRAY_KEYS = tuple(
    field_name.removeprefix("state.") for field_name in SOURCE_STATE_GRAY64_STATE_FIELDS
)
_SOURCE_STATE_OPTIONAL_ARRAY_KEYS = tuple(
    field_name.removeprefix("state.") for field_name in SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS
)
SOURCE_STATE_GRAY64_BONUS_VALUE = 208
SOURCE_STATE_GRAY64_DEFAULT_AVATAR_RADIUS = 0.6
SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID = (
    "curvyzero_source_state_bonus64_stack4_player_perspective/v1"
)
SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE = (22, 64, 64)
SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_DTYPE = "float32"
SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_VALUE_RANGE = (0.0, 1.0)
SOURCE_STATE_BONUS64_STACK4_OCCUPANCY_CHANNELS = 4
SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL = 4
SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL = 5
SOURCE_STATE_BONUS64_STACK4_SELF_STATUS_CHANNELS = tuple(range(6, 13))
SOURCE_STATE_BONUS64_STACK4_OTHER_STATUS_CHANNELS = tuple(range(13, 20))
SOURCE_STATE_BONUS64_STACK4_GAME_BORDERLESS_CHANNEL = 20
SOURCE_STATE_BONUS64_STACK4_GAME_TTL_CHANNEL = 21
SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE = 12.0
SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS = 10_000.0
_SOURCE_BODY_VALUE_BASE = 96
_SOURCE_BODY_VALUE_STEP = 32
_SOURCE_HEAD_VALUE_BASE = 224
_SOURCE_HEAD_VALUE_STEP = 8
_SELF_BODY_VALUE = 96
_OTHER_BODY_VALUE = 128
_SELF_HEAD_VALUE = 224
_OTHER_HEAD_VALUE = 232

_SOURCE_STATE_GRAY64_RENDERER_PAYLOAD: dict[str, Any] = {
    "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
    "algorithm": (
        "rasterize active body circles, optional active bonus circles, and live "
        "present head circles from vector runtime state arrays onto a fixed "
        "64x64 global arena grid; owner-coded grayscale; nearest-pixel circle "
        "fill; no antialiasing or browser canvas effects"
    ),
    "shape": list(SOURCE_STATE_GRAY64_SHAPE),
    "dtype": SOURCE_STATE_GRAY64_DTYPE,
    "value_range": list(SOURCE_STATE_GRAY64_VALUE_RANGE),
    "state_fields": list(SOURCE_STATE_GRAY64_STATE_FIELDS),
    "optional_state_fields": list(SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS),
    "comparison_target": SOURCE_STATE_GRAY64_COMPARISON_TARGET,
}
SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH = stable_contract_hash(_SOURCE_STATE_GRAY64_RENDERER_PAYLOAD)

_SOURCE_STATE_GRAY64_SCHEMA_PAYLOAD: dict[str, Any] = {
    "schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
    "observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
    "source_claim_id": SOURCE_STATE_GRAY64_SOURCE_CLAIM_ID,
    "surface": SOURCE_STATE_GRAY64_SURFACE,
    "truth_level": SOURCE_STATE_GRAY64_TRUTH_LEVEL,
    "source_fidelity_level": SOURCE_STATE_GRAY64_SOURCE_FIDELITY_LEVEL,
    "source_state_backed": SOURCE_STATE_GRAY64_SOURCE_STATE_BACKED,
    "shape": list(SOURCE_STATE_GRAY64_SHAPE),
    "dtype": SOURCE_STATE_GRAY64_DTYPE,
    "range": list(SOURCE_STATE_GRAY64_VALUE_RANGE),
    "normalized_dtype": SOURCE_STATE_GRAY64_NORMALIZED_DTYPE,
    "normalized_range": list(SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE),
    "channel_order": "CHW",
    "pixel_semantics": {
        "0": "empty/background",
        "96..192": "active body/trail circle, owner-coded grayscale",
        str(SOURCE_STATE_GRAY64_BONUS_VALUE): "active map bonus circle",
        "224..248": "live present head circle, player-index-coded grayscale",
    },
    "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
    "renderer_impl_hash": SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH,
    "state_fields": list(SOURCE_STATE_GRAY64_STATE_FIELDS),
    "optional_state_fields": list(SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS),
    "perspective": SOURCE_STATE_GRAY64_PERSPECTIVE,
    "frame_stack_owner": SOURCE_STATE_GRAY64_FRAME_STACK_OWNER,
    "frame_stack_policy": SOURCE_STATE_GRAY64_FRAME_STACK_POLICY,
    "terminal_policy": SOURCE_STATE_GRAY64_TERMINAL_POLICY,
    "comparison_target": SOURCE_STATE_GRAY64_COMPARISON_TARGET,
    "browser_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
    "browser_pixel_fidelity_claim": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM,
    "source_pixel_fidelity_blocker": SOURCE_STATE_GRAY64_PIXEL_FIDELITY_BLOCKER,
    "uses_ale": SOURCE_STATE_GRAY64_USES_ALE,
    "ale_usage": "none",
}
SOURCE_STATE_GRAY64_SCHEMA_HASH = stable_contract_hash(_SOURCE_STATE_GRAY64_SCHEMA_PAYLOAD)


class VectorVisualObservationError(ValueError):
    """Raised when vector visual inputs do not match the source-state contract."""


@dataclass
class SourceStateGray64Renderer:
    """Render one vector runtime row into the env-owned source-state gray64 frame."""

    frame: np.ndarray | None = None
    validate_state: bool = True
    _circle_masks: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame is None:
            self.frame = np.zeros(SOURCE_STATE_GRAY64_SHAPE, dtype=np.uint8)
            return
        self.frame = _validated_frame(self.frame, name="frame")

    def render(
        self,
        state: Mapping[str, np.ndarray],
        *,
        row: int = 0,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        arrays = _validated_arrays(state) if self.validate_state else _trusted_arrays(state)
        row_index = _row_index(row, arrays["tick"].shape[0])
        frame = self.frame if out is None else _validated_frame(out, name="out")
        if frame is None:
            raise RuntimeError("renderer frame was not initialized")

        frame.fill(0)
        canvas = frame[0]
        map_size = float(arrays["map_size"][row_index])
        body_limit = _body_slot_limit(arrays, row_index)
        active_slots = np.flatnonzero(arrays["body_active"][row_index, :body_limit])
        _draw_body_circles(
            canvas,
            arrays["body_pos"][row_index, active_slots],
            arrays["body_radius"][row_index, active_slots],
            arrays["body_owner"][row_index, active_slots],
            map_size,
            mask_cache=self._circle_masks,
        )
        if "bonus_active" in arrays:
            bonus_slots = np.flatnonzero(arrays["bonus_active"][row_index])
            _draw_uniform_circles(
                canvas,
                arrays["bonus_pos"][row_index, bonus_slots],
                arrays["bonus_radius"][row_index, bonus_slots],
                map_size,
                value=SOURCE_STATE_GRAY64_BONUS_VALUE,
                mask_cache=self._circle_masks,
            )

        player_count = arrays["pos"].shape[1]
        for player in range(player_count):
            if not bool(arrays["present"][row_index, player]):
                continue
            if not bool(arrays["alive"][row_index, player]):
                continue
            _draw_world_circle(
                canvas,
                arrays["pos"][row_index, player, 0],
                arrays["pos"][row_index, player, 1],
                arrays["radius"][row_index, player],
                map_size,
                value=_head_value(player),
                mask_cache=self._circle_masks,
            )
        return frame


def render_source_state_gray64(
    state: Mapping[str, np.ndarray],
    *,
    row: int = 0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Render one vector runtime row into a raw uint8 source-state frame."""

    return SourceStateGray64Renderer(frame=out).render(state, row=row)


def _rgb_luma_uint8(color: Sequence[int] | np.ndarray) -> int:
    rgb = _rgb_triplet(color).astype(np.float32)
    value = rgb[0] * np.float32(0.299) + rgb[1] * np.float32(0.587) + rgb[2] * np.float32(0.114)
    return int(np.clip(np.rint(value), 0.0, 255.0))


def render_source_state_gray64_fast_player_perspectives(
    state: Mapping[str, np.ndarray],
    *,
    row: int = 0,
    player_count: int | None = None,
    out: np.ndarray | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    bonus_render_mode: str = BONUS_SYMBOL_RENDER_MODE_DEFAULT,
) -> np.ndarray:
    """Render an approximate player-perspective gray64 frame directly at 64x64.

    This training-oriented path keeps source-state geometry and player-relative RGB
    luma contrast, but skips the browser-like 704x704 raster and area downsample.
    It prefers dense ``visual_trail_*`` points when present and active, then falls
    back to persisted ``body_*`` points.
    """

    arrays = _trusted_arrays(state)
    row_index = _row_index(row, arrays["tick"].shape[0])
    players = int(arrays["pos"].shape[1]) if player_count is None else int(player_count)
    if players < 1:
        raise VectorVisualObservationError("player_count must be >= 1")
    frames = (
        np.empty((players, *SOURCE_STATE_GRAY64_SHAPE), dtype=np.uint8)
        if out is None
        else _validated_player_perspective_frames(out, player_count=players)
    )
    base = np.empty(SOURCE_STATE_GRAY64_SHAPE, dtype=np.uint8)
    bg = np.uint8(_rgb_luma_uint8(background_rgb))
    base.fill(bg)
    canvas = base[0]
    map_size = float(arrays["map_size"][row_index])
    used_visual_trail = False
    if _source_state_has_visual_trail_arrays(arrays):
        trail_limit = _visual_trail_slot_limit(arrays, row_index)
        active_slots = np.flatnonzero(arrays["visual_trail_active"][row_index, :trail_limit])
        if active_slots.size:
            _draw_body_circles(
                canvas,
                arrays["visual_trail_pos"][row_index, active_slots],
                arrays["visual_trail_radius"][row_index, active_slots],
                arrays["visual_trail_owner"][row_index, active_slots],
                map_size,
                mask_cache=None,
            )
            used_visual_trail = True
    if not used_visual_trail:
        body_limit = _body_slot_limit(arrays, row_index)
        active_slots = np.flatnonzero(arrays["body_active"][row_index, :body_limit])
        _draw_body_circles(
            canvas,
            arrays["body_pos"][row_index, active_slots],
            arrays["body_radius"][row_index, active_slots],
            arrays["body_owner"][row_index, active_slots],
            map_size,
            mask_cache=None,
        )

    if "bonus_active" in arrays:
        bonus_mode = _validated_bonus_render_mode(bonus_render_mode)
        if bonus_mode == BONUS_RENDER_MODE_SIMPLE_SYMBOLS:
            _draw_bonus_type_simple_symbols(canvas, arrays, row_index, map_size)
        else:
            _draw_bonus_type_luma_circles(canvas, arrays, row_index, map_size, mask_cache=None)

    source_player_count = int(arrays["pos"].shape[1])
    for player in range(source_player_count):
        if not bool(arrays["present"][row_index, player]):
            continue
        if not bool(arrays["alive"][row_index, player]):
            continue
        _draw_world_circle(
            canvas,
            arrays["pos"][row_index, player, 0],
            arrays["pos"][row_index, player, 1],
            arrays["radius"][row_index, player],
            map_size,
            value=_head_value(player),
            mask_cache=None,
        )

    for controlled in range(players):
        np.copyto(frames[controlled], base)
        _remap_source_gray64_to_player_perspective(
            frames[controlled],
            controlled_player=controlled,
            player_count=source_player_count,
        )
    return frames


def render_source_state_rgb_canvas_like(
    state: Mapping[str, np.ndarray],
    *,
    row: int = 0,
    out: np.ndarray | None = None,
    frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    trail_render_mode: str = TRAIL_RENDER_MODE_DEFAULT,
    bonus_render_mode: str = BONUS_RENDER_MODE_DEFAULT,
) -> np.ndarray:
    """Render one vector runtime row into a browser-like RGB frame.

    This is for human inspection, not model input. It uses the source-state arrays,
    the browser's dark background, player colors, visible trail bodies or browser-style
    trail lines, live heads, and active bonus sprites by default. It is
    intentionally separate from the 64x64 gray learning tensor.
    """

    mode = _validated_trail_render_mode(trail_render_mode)
    bonus_mode = _validated_bonus_render_mode(bonus_render_mode)
    arrays = _trusted_arrays(state)
    row_index = _row_index(row, arrays["tick"].shape[0])
    size = _rgb_frame_size(frame_size)
    frame = (
        np.empty((size, size, 3), dtype=np.uint8)
        if out is None
        else _validated_rgb_frame(out, frame_size=size)
    )
    frame[:, :] = _rgb_triplet(background_rgb)

    map_size = float(arrays["map_size"][row_index])
    colors = _player_rgb_values(arrays, row_index, player_rgb=player_rgb)

    if mode == TRAIL_RENDER_MODE_BROWSER_LINES:
        _render_source_state_rgb_browser_lines(
            frame,
            arrays,
            row_index,
            map_size,
            colors=colors,
        )
    else:
        _render_source_state_rgb_body_circles_fast(
            frame,
            arrays,
            row_index,
            map_size,
            colors=colors,
        )

    if "bonus_active" in arrays:
        bonus_slots = np.flatnonzero(arrays["bonus_active"][row_index])
        bonus_type = arrays.get("bonus_type")
        _draw_bonuses_rgb(
            frame,
            arrays["bonus_pos"][row_index, bonus_slots],
            arrays["bonus_radius"][row_index, bonus_slots],
            map_size,
            bonus_types=None if bonus_type is None else bonus_type[row_index, bonus_slots],
            bonus_render_mode=bonus_mode,
        )

    player_count = arrays["pos"].shape[1]
    for player in range(player_count):
        if not bool(arrays["present"][row_index, player]):
            continue
        if not bool(arrays["alive"][row_index, player]):
            continue
        _draw_world_circle_rgb(
            frame,
            arrays["pos"][row_index, player, 0],
            arrays["pos"][row_index, player, 1],
            arrays["radius"][row_index, player],
            map_size,
            color=colors[player],
        )
    return frame


def render_source_state_canvas_gray64(
    state: Mapping[str, np.ndarray],
    *,
    row: int = 0,
    out: np.ndarray | None = None,
    rgb_out: np.ndarray | None = None,
    downsample_scratch: "SourceStateGray64DownsampleScratch | None" = None,
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    trail_render_mode: str = TRAIL_RENDER_MODE_DEFAULT,
    bonus_render_mode: str = BONUS_RENDER_MODE_DEFAULT,
) -> np.ndarray:
    """Render browser-like source-state RGB, then convert it to gray64.

    This is the clean model-facing visual path: render the raw browser-like
    source canvas at its default inspection resolution, then convert/downsample
    that image to the 64x64 training tensor. It is still not a DOM/browser
    canvas pixel-parity claim.
    """

    mode = _validated_trail_render_mode(trail_render_mode)
    bonus_mode = _validated_bonus_render_mode(bonus_render_mode)
    source_frame_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    rgb = render_source_state_rgb_canvas_like(
        state,
        row=row,
        out=_compatible_rgb_out(rgb_out, frame_size=source_frame_size),
        frame_size=source_frame_size,
        player_rgb=player_rgb,
        background_rgb=background_rgb,
        trail_render_mode=mode,
        bonus_render_mode=bonus_mode,
    )
    return rgb_canvas_like_to_gray64(rgb, out=out, scratch=downsample_scratch)


def render_source_state_canvas_gray64_player_perspectives(
    state: Mapping[str, np.ndarray],
    *,
    row: int = 0,
    player_rgbs: Sequence[Sequence[Sequence[int]]],
    out: np.ndarray | None = None,
    rgb_base_out: np.ndarray | None = None,
    rgb_work_out: np.ndarray | None = None,
    trail_cache: "SourceStateBrowserLineTrailLayerCache | None" = None,
    dirty_render_cache: "SourceStateCanvasGray64DirtyRenderCache | None" = None,
    downsample_scratch: "SourceStateGray64DownsampleScratch | None" = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    trail_render_mode: str = TRAIL_RENDER_MODE_DEFAULT,
    bonus_render_mode: str = BONUS_RENDER_MODE_DEFAULT,
) -> np.ndarray:
    """Render gray64 frames for all player perspectives with one trail pass.

    The expensive browser-line trail geometry is independent of which player is
    controlled; only the self/other palette changes. For the common two-seat
    path, render trails once, remap exact RGB player colors per perspective, then
    draw perspective-specific bonuses/heads in the same order as the scalar
    renderer before downsampling.
    """

    mode = _validated_trail_render_mode(trail_render_mode)
    bonus_mode = _validated_bonus_render_mode(bonus_render_mode)
    arrays = _trusted_arrays(state)
    row_index = _row_index(row, arrays["tick"].shape[0])
    player_count = arrays["pos"].shape[1]
    palettes = tuple(player_rgbs)
    if len(palettes) != player_count:
        raise VectorVisualObservationError(
            f"player_rgbs must contain {player_count} palettes, got {len(palettes)}"
        )

    frames = (
        np.empty((player_count, *SOURCE_STATE_CANVAS_GRAY64_SHAPE), dtype=np.uint8)
        if out is None
        else _validated_player_perspective_frames(out, player_count=player_count)
    )
    if player_count != 2:
        for player in range(player_count):
            render_source_state_canvas_gray64(
                state,
                row=row_index,
                out=frames[player],
                player_rgb=palettes[player],
                background_rgb=background_rgb,
                trail_render_mode=mode,
                bonus_render_mode=bonus_mode,
                downsample_scratch=downsample_scratch,
            )
        return frames

    source_frame_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    base = (
        np.empty((source_frame_size, source_frame_size, 3), dtype=np.uint8)
        if rgb_base_out is None
        else _validated_rgb_frame(rgb_base_out, frame_size=source_frame_size)
    )
    work = (
        np.empty((source_frame_size, source_frame_size, 3), dtype=np.uint8)
        if rgb_work_out is None
        else _validated_rgb_frame(rgb_work_out, frame_size=source_frame_size)
    )
    if np.shares_memory(base, work):
        raise VectorVisualObservationError("rgb_base_out and rgb_work_out must not alias")

    base[:, :] = _rgb_triplet(background_rgb)
    map_size = float(arrays["map_size"][row_index])
    base_colors = _player_rgb_values(arrays, row_index, player_rgb=palettes[0])
    if not _rgb_palette_reuse_safe(
        base_colors,
        background_rgb=background_rgb,
        extra_reserved_colors=((120, 120, 120),),
    ):
        if dirty_render_cache is not None:
            dirty_render_cache.reset()
        for player in range(player_count):
            render_source_state_canvas_gray64(
                state,
                row=row_index,
                out=frames[player],
                player_rgb=palettes[player],
                background_rgb=background_rgb,
                trail_render_mode=mode,
                bonus_render_mode=bonus_mode,
                downsample_scratch=downsample_scratch,
        )
        return frames

    if (
        mode == TRAIL_RENDER_MODE_BROWSER_LINES
        and trail_cache is not None
        and dirty_render_cache is not None
        and dirty_render_cache.try_render(
            frames,
            arrays,
            row_index,
            map_size,
            player_rgbs=palettes,
            trail_cache=trail_cache,
            background_rgb=background_rgb,
            bonus_render_mode=bonus_mode,
        )
    ):
        return frames

    used_cache = False
    if mode == TRAIL_RENDER_MODE_BROWSER_LINES:
        if trail_cache is not None:
            used_cache = trail_cache.render_trails_rgb(
                base,
                arrays,
                row_index,
                map_size,
                colors=base_colors,
                background_rgb=background_rgb,
            )
        if not used_cache:
            _render_source_state_rgb_browser_lines(
                base,
                arrays,
                row_index,
                map_size,
                colors=base_colors,
            )
    else:
        _render_source_state_rgb_body_circles_fast(
            base,
            arrays,
            row_index,
            map_size,
            colors=base_colors,
        )

    for player, palette in enumerate(palettes[1:], start=1):
        colors = _player_rgb_values(arrays, row_index, player_rgb=palette)
        if not (
            used_cache
            and trail_cache is not None
            and trail_cache.recolor_trails_rgb(
                work,
                colors=colors,
                background_rgb=background_rgb,
            )
        ):
            work[:] = base
            _remap_rgb_palette_pixels(
                work,
                source=base,
                from_colors=base_colors,
                to_colors=colors,
            )
        _draw_source_state_rgb_bonuses(
            work,
            arrays,
            row_index,
            map_size,
            bonus_render_mode=bonus_mode,
        )
        _draw_source_state_rgb_heads(
            work,
            arrays,
            row_index,
            map_size,
            colors=colors,
        )
        rgb_canvas_like_to_gray64(
            work,
            out=frames[player],
            scratch=downsample_scratch,
        )

    _draw_source_state_rgb_bonuses(
        base,
        arrays,
        row_index,
        map_size,
        bonus_render_mode=bonus_mode,
    )
    _draw_source_state_rgb_heads(
        base,
        arrays,
        row_index,
        map_size,
        colors=base_colors,
    )
    rgb_canvas_like_to_gray64(
        base,
        out=frames[0],
        scratch=downsample_scratch,
    )
    if dirty_render_cache is not None:
        if mode == TRAIL_RENDER_MODE_BROWSER_LINES and trail_cache is not None and used_cache:
            dirty_render_cache.capture_full(
                frames,
                (base, work),
                arrays,
                row_index,
                map_size,
                player_rgbs=palettes,
                background_rgb=background_rgb,
                bonus_render_mode=bonus_mode,
            )
        else:
            dirty_render_cache.reset()
    return frames


def render_source_snapshot_rgb_canvas_like(
    snapshot: Mapping[str, Any],
    *,
    world_bodies: Sequence[Mapping[str, Any]] | None = None,
    visual_trail_points: Sequence[Mapping[str, Any]] | None = None,
    bonus_bodies: Sequence[Mapping[str, Any]] | None = None,
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None = None,
    out: np.ndarray | None = None,
    frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    default_avatar_radius: float = SOURCE_STATE_GRAY64_DEFAULT_AVATAR_RADIUS,
    trail_render_mode: str = TRAIL_RENDER_MODE_DEFAULT,
    bonus_render_mode: str = BONUS_RENDER_MODE_DEFAULT,
) -> np.ndarray:
    """Render a source-env snapshot into the browser-like RGB frame."""

    mode = _validated_trail_render_mode(trail_render_mode)
    bonus_mode = _validated_bonus_render_mode(bonus_render_mode)
    size = _rgb_frame_size(frame_size)
    frame = (
        np.empty((size, size, 3), dtype=np.uint8)
        if out is None
        else _validated_rgb_frame(out, frame_size=size)
    )
    frame[:, :] = _rgb_triplet(background_rgb)
    map_size = _source_snapshot_map_size(snapshot)
    avatars = _source_snapshot_avatars(snapshot)
    colors = _source_avatar_rgb_values(avatars, player_rgb=player_rgb)
    player_index_by_avatar_id = {
        int(avatar["id"]): index
        for index, avatar in enumerate(avatars)
        if _source_has_int(avatar.get("id"))
    }
    avatar_radius_by_id = _source_avatar_radius_by_id(avatar_body_metadata)

    body_records = _source_mapping_sequence(world_bodies, name="world_bodies")
    visual_trail_records = (
        None
        if visual_trail_points is None
        else _source_mapping_sequence(
            visual_trail_points,
            name="visual_trail_points",
        )
    )
    _render_source_snapshot_rgb_trails(
        frame,
        body_records,
        visual_trail_records,
        player_index_by_avatar_id,
        map_size,
        colors=colors,
        default_avatar_radius=default_avatar_radius,
        trail_render_mode=mode,
    )

    bonus_records = _source_mapping_sequence(bonus_bodies, name="bonus_bodies")
    if bonus_records:
        bonus_positions = np.asarray(
            [[float(bonus["x"]), float(bonus["y"])] for bonus in bonus_records],
            dtype=np.float64,
        )
        bonus_radii = np.asarray(
            [float(bonus.get("radius", 0.0)) for bonus in bonus_records],
            dtype=np.float64,
        )
        _draw_bonuses_rgb(
            frame,
            bonus_positions,
            bonus_radii,
            map_size,
            bonus_types=_source_bonus_type_codes(bonus_records),
            bonus_render_mode=bonus_mode,
        )

    for player, avatar in enumerate(avatars):
        if not bool(avatar.get("present", True)):
            continue
        if not bool(avatar.get("alive", True)):
            continue
        avatar_id = avatar.get("id")
        radius = (
            float(avatar["radius"])
            if "radius" in avatar
            else avatar_radius_by_id.get(int(avatar_id), float(default_avatar_radius))
            if _source_has_int(avatar_id)
            else float(default_avatar_radius)
        )
        _draw_world_circle_rgb(
            frame,
            avatar["x"],
            avatar["y"],
            radius,
            map_size,
            color=colors[player],
        )
    return frame


def render_source_snapshot_canvas_gray64(
    snapshot: Mapping[str, Any],
    *,
    world_bodies: Sequence[Mapping[str, Any]] | None = None,
    visual_trail_points: Sequence[Mapping[str, Any]] | None = None,
    bonus_bodies: Sequence[Mapping[str, Any]] | None = None,
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None = None,
    out: np.ndarray | None = None,
    rgb_out: np.ndarray | None = None,
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    default_avatar_radius: float = SOURCE_STATE_GRAY64_DEFAULT_AVATAR_RADIUS,
    trail_render_mode: str = TRAIL_RENDER_MODE_DEFAULT,
    bonus_render_mode: str = BONUS_RENDER_MODE_DEFAULT,
) -> np.ndarray:
    mode = _validated_trail_render_mode(trail_render_mode)
    bonus_mode = _validated_bonus_render_mode(bonus_render_mode)
    source_frame_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    rgb = render_source_snapshot_rgb_canvas_like(
        snapshot,
        world_bodies=world_bodies,
        visual_trail_points=visual_trail_points,
        bonus_bodies=bonus_bodies,
        avatar_body_metadata=avatar_body_metadata,
        out=_compatible_rgb_out(rgb_out, frame_size=source_frame_size),
        frame_size=source_frame_size,
        player_rgb=player_rgb,
        background_rgb=background_rgb,
        default_avatar_radius=default_avatar_radius,
        trail_render_mode=mode,
        bonus_render_mode=bonus_mode,
    )
    return rgb_canvas_like_to_gray64(rgb, out=out)


class SourceStateGray64DownsampleScratch:
    """Reusable scratch buffers for exact 704-to-64 RGB luma downsampling."""

    def __init__(self, source_frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE):
        source_size = _rgb_frame_size(source_frame_size)
        target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
        if source_size % target_size != 0:
            raise VectorVisualObservationError(
                "source_frame_size must be 64 or an integer multiple of 64"
            )
        self.source_frame_size = source_size
        self.gray = np.empty((source_size, source_size), dtype=np.float32)
        self.temp = np.empty_like(self.gray)
        self.downsampled = np.empty((target_size, target_size), dtype=np.float32)


class SourceStateGray64DirtyDownsampleScratch:
    """Reusable scratch buffers for exact dirty 704-to-64 RGB luma updates."""

    def __init__(self, source_frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE):
        source_size = _rgb_frame_size(source_frame_size)
        target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
        if source_size % target_size != 0:
            raise VectorVisualObservationError(
                "source_frame_size must be 64 or an integer multiple of 64"
            )
        ratio = source_size // target_size
        self.source_frame_size = source_size
        self.ratio = ratio
        self.luma = np.empty((ratio, ratio), dtype=np.float32)
        self.temp = np.empty_like(self.luma)
        self.downsampled = np.empty((1, 1), dtype=np.float32)


def rgb_canvas_like_to_gray64(
    rgb: np.ndarray,
    *,
    out: np.ndarray | None = None,
    scratch: SourceStateGray64DownsampleScratch | None = None,
) -> np.ndarray:
    """Convert an RGB canvas-like frame to CHW uint8 gray64."""

    rgb_array = np.asarray(rgb)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3 or rgb_array.shape[0] != rgb_array.shape[1]:
        raise VectorVisualObservationError(
            f"rgb must have square HWC RGB shape, got {rgb_array.shape}"
        )
    if rgb_array.dtype != np.uint8:
        raise VectorVisualObservationError(f"rgb dtype must be uint8, got {rgb_array.dtype}")
    if rgb_array.shape[0] < SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]:
        raise VectorVisualObservationError(
            "rgb frame size must be at least 64 for gray64 conversion"
        )
    frame = (
        np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
        if out is None
        else _validated_frame(out, name="out")
    )
    _rgb_canvas_like_luma_downsample64(rgb_array, out=frame[0], scratch=scratch)
    return frame


def render_source_snapshot_gray64(
    snapshot: Mapping[str, Any],
    *,
    world_bodies: Sequence[Mapping[str, Any]] | None = None,
    bonus_bodies: Sequence[Mapping[str, Any]] | None = None,
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None = None,
    out: np.ndarray | None = None,
    default_avatar_radius: float = SOURCE_STATE_GRAY64_DEFAULT_AVATAR_RADIUS,
) -> np.ndarray:
    """Render a ``CurvyTronSourceEnv`` snapshot into the raw gray64 frame.

    This is the source/oracle side of the raw-observation parity check. It uses
    the same raster rules as ``SourceStateGray64Renderer``; it is still not a
    browser/canvas pixel renderer.
    """

    frame = (
        np.zeros(SOURCE_STATE_GRAY64_SHAPE, dtype=np.uint8)
        if out is None
        else _validated_frame(out, name="out")
    )
    frame.fill(0)
    canvas = frame[0]
    map_size = _source_snapshot_map_size(snapshot)
    avatars = _source_snapshot_avatars(snapshot)
    player_index_by_avatar_id = {
        int(avatar["id"]): index
        for index, avatar in enumerate(avatars)
        if _source_has_int(avatar.get("id"))
    }
    avatar_radius_by_id = _source_avatar_radius_by_id(avatar_body_metadata)

    body_records = _source_mapping_sequence(world_bodies, name="world_bodies")
    if body_records:
        body_positions = np.asarray(
            [[float(body["x"]), float(body["y"])] for body in body_records],
            dtype=np.float64,
        )
        body_radii = np.asarray(
            [float(body.get("radius", default_avatar_radius)) for body in body_records],
            dtype=np.float64,
        )
        body_owners = np.asarray(
            [
                _source_body_owner_index(body.get("avatarId"), player_index_by_avatar_id)
                for body in body_records
            ],
            dtype=np.int16,
        )
        _draw_body_circles(
            canvas,
            body_positions,
            body_radii,
            body_owners,
            map_size,
            mask_cache=None,
        )

    bonus_records = _source_mapping_sequence(bonus_bodies, name="bonus_bodies")
    if bonus_records:
        bonus_positions = np.asarray(
            [[float(bonus["x"]), float(bonus["y"])] for bonus in bonus_records],
            dtype=np.float64,
        )
        bonus_radii = np.asarray(
            [float(bonus.get("radius", 0.0)) for bonus in bonus_records],
            dtype=np.float64,
        )
        _draw_uniform_circles(
            canvas,
            bonus_positions,
            bonus_radii,
            map_size,
            value=SOURCE_STATE_GRAY64_BONUS_VALUE,
            mask_cache=None,
        )

    for player, avatar in enumerate(avatars):
        if not bool(avatar.get("present", True)):
            continue
        if not bool(avatar.get("alive", True)):
            continue
        avatar_id = avatar.get("id")
        radius = (
            float(avatar["radius"])
            if "radius" in avatar
            else avatar_radius_by_id.get(int(avatar_id), float(default_avatar_radius))
            if _source_has_int(avatar_id)
            else float(default_avatar_radius)
        )
        _draw_world_circle(
            canvas,
            avatar["x"],
            avatar["y"],
            radius,
            map_size,
            value=_head_value(player),
            mask_cache=None,
        )
    return frame


def normalize_source_state_gray64(
    frame: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize a raw source-state frame to float32 [0, 1]."""

    frame_array = _validated_frame(frame, name="frame")
    if out is None:
        normalized = np.empty(SOURCE_STATE_GRAY64_SHAPE, dtype=np.float32)
    else:
        normalized = np.asarray(out)
        if normalized.shape != SOURCE_STATE_GRAY64_SHAPE:
            raise VectorVisualObservationError(
                f"out must have shape {SOURCE_STATE_GRAY64_SHAPE}, got {normalized.shape}"
            )
        if normalized.dtype != np.float32:
            raise VectorVisualObservationError(f"out dtype must be float32, got {normalized.dtype}")
    np.multiply(frame_array, np.float32(1.0 / 255.0), out=normalized, casting="unsafe")
    return normalized


def source_state_canvas_gray64_schema() -> dict[str, Any]:
    return {
        "schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
        "observation_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
        "source_claim_id": SOURCE_STATE_CANVAS_GRAY64_SOURCE_CLAIM_ID,
        "surface": SOURCE_STATE_CANVAS_GRAY64_SURFACE,
        "truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
        "source_fidelity_level": SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL,
        "source_state_backed": SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
        "shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
        "dtype": SOURCE_STATE_CANVAS_GRAY64_DTYPE,
        "range": list(SOURCE_STATE_CANVAS_GRAY64_VALUE_RANGE),
        "normalized_dtype": SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
        "normalized_range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
        "channel_order": "CHW",
        "pixel_semantics": (
            "area-downsampled luminance of the source-state browser-like RGB canvas"
        ),
        "rgb_source_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "rgb_renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
        "rgb_source_frame_size": SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        "downsample_target_frame_size": SOURCE_STATE_CANVAS_GRAY64_SHAPE[1],
        "downsample_method": "integer_area_average_after_luma",
        "downsample_ratio": (
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE // SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
        ),
        "renderer_impl_id": SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID,
        "state_fields": list(SOURCE_STATE_GRAY64_STATE_FIELDS),
        "optional_state_fields": [
            *SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS,
            "state.avatar_color",
            "state.body_num",
            "state.body_break_before",
            *SOURCE_STATE_VISUAL_TRAIL_FIELDS,
            "state.bonus_type",
        ],
        "perspective": SOURCE_STATE_CANVAS_GRAY64_PERSPECTIVE,
        "comparison_target": SOURCE_STATE_CANVAS_GRAY64_COMPARISON_TARGET,
        "browser_pixel_fidelity": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
        "browser_pixel_fidelity_claim": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM,
        "browser_pixel_fidelity_blocker": SOURCE_STATE_CANVAS_GRAY64_PIXEL_FIDELITY_BLOCKER,
        "default_trail_render_mode": TRAIL_RENDER_MODE_DEFAULT,
        "supported_trail_render_modes": list(TRAIL_RENDER_MODE_ORDER),
        "trail_render_mode": TRAIL_RENDER_MODE_DEFAULT,
        "trail_renderer_kind": "connected_rounded_lines",
        "trail_renderer_truth_level": "source_state_browser_style_lines_non_pixel_parity",
        "trail_renderer_is_approximation": False,
        "browser_trail_semantics": "persistent_background_canvas_round_line_caps",
        "browser_client_trail_point_caveat": (
            "browser_lines prefers optional visual_trail_* position-event points; "
            "without them it falls back to sparse persisted body points"
        ),
        "trail_raster_antialiasing": (
            "binary_numpy_coverage_at_source_resolution_then_area_downsampled"
        ),
        "default_bonus_render_mode": BONUS_RENDER_MODE_DEFAULT,
        "supported_bonus_render_modes": list(BONUS_RENDER_MODE_ORDER),
        "bonus_render_mode": BONUS_RENDER_MODE_DEFAULT,
        "bonus_renderer_kind": "source_sprite_atlas_tiles",
        "bonus_renderer_truth_level": ("source_state_browser_bonus_sprite_atlas_non_pixel_parity"),
        "bonus_renderer_is_approximation": False,
        "bonus_sprite_missing_fallback": "deterministic_type_coded_placeholder_stamp",
        "bonus_sprite_cache": "in_process_lru_stamp_cache_by_tile_index_and_pixel_size",
        "bonus_sprite_sheet": SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH,
        "bonus_sprite_grid": [
            SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_COLUMNS,
            SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_ROWS,
        ],
        "bonus_sprite_names": list(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES),
        "uses_ale": SOURCE_STATE_CANVAS_GRAY64_USES_ALE,
        "grayscale_conversion": (
            "render source RGB at 704x704; compute 0.299*r + 0.587*g + 0.114*b; "
            "average each 11x11 area block; round to uint8"
        ),
    }


SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH = stable_contract_hash(source_state_canvas_gray64_schema())


def render_source_state_bonus64_stack4_player_perspective_v1(
    state: Mapping[str, np.ndarray],
    *,
    row: int = 0,
    controlled_player: int = 0,
    occupancy_stack: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Render the bonus-aware player-perspective visual tensor.

    This is a separate v1 helper and typed-bonus parity gate. It does not
    replace the gray64 v0 geometry gate. If ``occupancy_stack`` is omitted, the
    current v0 frame is rendered, player-perspective remapped, normalized, and
    repeated into the four history channels.
    """

    arrays = _validated_arrays(state)
    row_index = _row_index(row, arrays["tick"].shape[0])
    player_count = arrays["pos"].shape[1]
    controlled = _row_index(controlled_player, player_count)
    other = 1 - controlled if player_count == 2 else _first_other_player(controlled, player_count)
    bonus_arrays = _bonus64_arrays(state, arrays)
    output = _validated_bonus64_out(out)
    output.fill(0.0)

    if occupancy_stack is None:
        raw_frame = render_source_state_gray64(state, row=row_index)
        perspective_frame = _player_perspective_gray64(raw_frame, controlled)
        output[:SOURCE_STATE_BONUS64_STACK4_OCCUPANCY_CHANNELS] = perspective_frame.astype(
            np.float32, copy=False
        ) * np.float32(1.0 / 255.0)
    else:
        output[:SOURCE_STATE_BONUS64_STACK4_OCCUPANCY_CHANNELS] = _validated_occupancy_stack(
            occupancy_stack
        )

    _draw_bonus64_bonus_planes(output, bonus_arrays, row_index)
    _fill_player_status_planes(
        output,
        bonus_arrays,
        row_index,
        player=controlled,
        start_channel=6,
    )
    _fill_player_status_planes(
        output,
        bonus_arrays,
        row_index,
        player=other,
        start_channel=13,
    )
    output[SOURCE_STATE_BONUS64_STACK4_GAME_BORDERLESS_CHANNEL].fill(
        _game_borderless_value(state, row_index)
    )
    output[SOURCE_STATE_BONUS64_STACK4_GAME_TTL_CHANNEL].fill(
        _game_bonus_ttl_value(state, row_index)
    )
    return output


def source_state_bonus64_stack4_player_perspective_v1_schema() -> dict[str, Any]:
    """Return the draft v1 bonus-aware observation contract."""

    return {
        "schema_id": SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID,
        "shape": list(SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE),
        "dtype": SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_DTYPE,
        "range": list(SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_VALUE_RANGE),
        "channel_order": "CHW",
        "status": "promoted_2p_typed_bonus_parity_gate_not_lightzero_default",
        "parity_gate": True,
        "parity_scope": (
            "2p_source_vs_vector_active_map_bonus_mask_type_and_post_catch_status_planes"
        ),
        "replaces_gray64_v0": False,
        "base_geometry_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "channels": {
            "0-3": "player-perspective occupancy history",
            "4": "active map bonus mask",
            "5": "active map bonus type code divided by 12",
            "6-12": "controlled-player status planes",
            "13-19": "other-player status planes",
            "20": "game borderless flag",
            "21": "game bonus ttl",
        },
    }


def source_state_gray64_schema() -> dict[str, Any]:
    """Return the source-state visual contract schema and stable hash."""

    return {
        **_SOURCE_STATE_GRAY64_SCHEMA_PAYLOAD,
        "label": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "schema_hash": SOURCE_STATE_GRAY64_SCHEMA_HASH,
        "observation_schema_hash": SOURCE_STATE_GRAY64_SCHEMA_HASH,
        "required_metadata_fields": [
            "observation_schema_id",
            "observation_schema_hash",
            "renderer_impl_id",
            "renderer_impl_hash",
            "shape",
            "dtype",
            "range",
            "state_fields",
            "perspective",
            "terminal_policy",
            "comparison_target",
            "browser_pixel_fidelity",
            "uses_ale",
        ],
    }


def source_state_gray64_metadata(*, includes_render_cost: bool) -> dict[str, Any]:
    """Return metadata for records that carry the env-owned source-state frame."""

    return {
        "observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "observation_schema_hash": SOURCE_STATE_GRAY64_SCHEMA_HASH,
        "schema_hash": SOURCE_STATE_GRAY64_SCHEMA_HASH,
        "source_claim_id": SOURCE_STATE_GRAY64_SOURCE_CLAIM_ID,
        "surface": SOURCE_STATE_GRAY64_SURFACE,
        "truth_level": SOURCE_STATE_GRAY64_TRUTH_LEVEL,
        "source_fidelity_level": SOURCE_STATE_GRAY64_SOURCE_FIDELITY_LEVEL,
        "source_state_backed": SOURCE_STATE_GRAY64_SOURCE_STATE_BACKED,
        "shape": list(SOURCE_STATE_GRAY64_SHAPE),
        "dtype": SOURCE_STATE_GRAY64_DTYPE,
        "range": list(SOURCE_STATE_GRAY64_VALUE_RANGE),
        "normalized_dtype": SOURCE_STATE_GRAY64_NORMALIZED_DTYPE,
        "normalized_range": list(SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE),
        "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
        "renderer_impl_hash": SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH,
        "state_fields": list(SOURCE_STATE_GRAY64_STATE_FIELDS),
        "optional_state_fields": list(SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS),
        "perspective": SOURCE_STATE_GRAY64_PERSPECTIVE,
        "frame_stack_owner": SOURCE_STATE_GRAY64_FRAME_STACK_OWNER,
        "frame_stack_policy": SOURCE_STATE_GRAY64_FRAME_STACK_POLICY,
        "terminal_policy": SOURCE_STATE_GRAY64_TERMINAL_POLICY,
        "comparison_target": SOURCE_STATE_GRAY64_COMPARISON_TARGET,
        "browser_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
        "browser_pixel_fidelity_claim": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM,
        "source_pixel_fidelity_blocker": SOURCE_STATE_GRAY64_PIXEL_FIDELITY_BLOCKER,
        "uses_ale": SOURCE_STATE_GRAY64_USES_ALE,
        "ale_usage": "none",
        "includes_render_cost": bool(includes_render_cost),
    }


def _validated_arrays(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    tick = _integer_array(state, "tick", ndim=1)
    batch_size = tick.shape[0]
    map_size = _numeric_array(state, "map_size", shape=(batch_size,))
    if bool((~np.isfinite(map_size)).any()) or bool((map_size <= 0.0).any()):
        raise VectorVisualObservationError("state['map_size'] must be finite and positive")
    elapsed_ms = _numeric_array(state, "elapsed_ms", shape=(batch_size,))
    if bool((~np.isfinite(elapsed_ms)).any()):
        raise VectorVisualObservationError("state['elapsed_ms'] must contain finite values")

    pos = _numeric_array(state, "pos", ndim=3)
    if pos.shape[0] != batch_size or pos.shape[2] != 2:
        raise VectorVisualObservationError("state['pos'] must have shape [B,P,2]")
    player_count = pos.shape[1]
    player_shape = (batch_size, player_count)
    present = _bool_array(state, "present", shape=player_shape)
    alive = _bool_array(state, "alive", shape=player_shape)
    radius = _numeric_array(state, "radius", shape=player_shape)
    if bool((~np.isfinite(pos)).any()) or bool((~np.isfinite(radius)).any()):
        raise VectorVisualObservationError("position and radius arrays must contain finite values")
    if bool((radius < 0.0).any()):
        raise VectorVisualObservationError("state['radius'] values must be non-negative")

    body_active = _bool_array(state, "body_active", ndim=2)
    if body_active.shape[0] != batch_size:
        raise VectorVisualObservationError("state['body_active'] must have shape [B,C]")
    body_capacity = body_active.shape[1]
    body_shape = (batch_size, body_capacity)
    body_pos = _numeric_array(state, "body_pos", shape=(batch_size, body_capacity, 2))
    body_radius = _numeric_array(state, "body_radius", shape=body_shape)
    body_owner = _integer_array(state, "body_owner", shape=body_shape)
    body_num = _integer_array(state, "body_num", shape=body_shape) if "body_num" in state else None
    body_break_before = (
        _bool_array(state, "body_break_before", shape=body_shape)
        if "body_break_before" in state
        else None
    )
    body_write_cursor = _integer_array(state, "body_write_cursor", shape=(batch_size,))
    if bool(((body_write_cursor < 0) | (body_write_cursor > body_capacity)).any()):
        raise VectorVisualObservationError(
            "state['body_write_cursor'] values must be in [0, body_capacity]"
        )
    if bool((~np.isfinite(body_pos)).any()) or bool((~np.isfinite(body_radius)).any()):
        raise VectorVisualObservationError(
            "body position and radius arrays must contain finite values"
        )
    if bool((body_radius < 0.0).any()):
        raise VectorVisualObservationError("state['body_radius'] values must be non-negative")

    done = _bool_array(state, "done", shape=(batch_size,))
    terminated = _bool_array(state, "terminated", shape=(batch_size,))
    truncated = _bool_array(state, "truncated", shape=(batch_size,))
    terminal_reason = _integer_array(state, "terminal_reason", shape=(batch_size,))

    arrays = {
        "tick": tick,
        "elapsed_ms": elapsed_ms,
        "map_size": map_size,
        "pos": pos,
        "present": present,
        "alive": alive,
        "radius": radius,
        "body_active": body_active,
        "body_pos": body_pos,
        "body_radius": body_radius,
        "body_owner": body_owner,
        "body_write_cursor": body_write_cursor,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "terminal_reason": terminal_reason,
    }
    if body_num is not None:
        arrays["body_num"] = body_num
    if body_break_before is not None:
        arrays["body_break_before"] = body_break_before
    arrays.update(_optional_visual_trail_arrays(state, batch_size=batch_size))
    arrays.update(_optional_bonus_arrays(state, batch_size=batch_size))
    return arrays


def _trusted_arrays(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(state[name]) for name in _SOURCE_STATE_ARRAY_KEYS}
    for name in _SOURCE_STATE_OPTIONAL_ARRAY_KEYS:
        if name in state:
            arrays[name] = np.asarray(state[name])
    if "avatar_color" in state:
        arrays["avatar_color"] = np.asarray(state["avatar_color"])
    if "body_num" in state:
        arrays["body_num"] = np.asarray(state["body_num"])
    if "body_break_before" in state:
        arrays["body_break_before"] = np.asarray(state["body_break_before"])
    for field_name in SOURCE_STATE_VISUAL_TRAIL_FIELDS:
        name = field_name.removeprefix("state.")
        if name in state:
            arrays[name] = np.asarray(state[name])
    if "bonus_type" in state:
        arrays["bonus_type"] = np.asarray(state["bonus_type"])
    return arrays


def _optional_visual_trail_arrays(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    names = tuple(
        field_name.removeprefix("state.") for field_name in SOURCE_STATE_VISUAL_TRAIL_FIELDS
    )
    present = [name in state for name in names]
    if not any(present):
        return {}
    if not all(present):
        missing = [name for name, is_present in zip(names, present, strict=True) if not is_present]
        raise VectorVisualObservationError(
            f"optional visual trail arrays must be supplied together; missing {missing}"
        )
    active = _bool_array(state, "visual_trail_active", ndim=2)
    if active.shape[0] != batch_size:
        raise VectorVisualObservationError("state['visual_trail_active'] must have shape [B,C]")
    capacity = active.shape[1]
    shape = (batch_size, capacity)
    write_cursor = _integer_array(state, "visual_trail_write_cursor", shape=(batch_size,))
    pos = _numeric_array(state, "visual_trail_pos", shape=(batch_size, capacity, 2))
    radius = _numeric_array(state, "visual_trail_radius", shape=shape)
    owner = _integer_array(state, "visual_trail_owner", shape=shape)
    break_before = _bool_array(state, "visual_trail_break_before", shape=shape)
    if bool(((write_cursor < 0) | (write_cursor > capacity)).any()):
        raise VectorVisualObservationError(
            "state['visual_trail_write_cursor'] values must be in [0, visual capacity]"
        )
    if bool((~np.isfinite(pos)).any()) or bool((~np.isfinite(radius)).any()):
        raise VectorVisualObservationError(
            "visual trail position and radius arrays must contain finite values"
        )
    if bool((radius < 0.0).any()):
        raise VectorVisualObservationError(
            "state['visual_trail_radius'] values must be non-negative"
        )
    return {
        "visual_trail_active": active,
        "visual_trail_write_cursor": write_cursor,
        "visual_trail_pos": pos,
        "visual_trail_radius": radius,
        "visual_trail_owner": owner,
        "visual_trail_break_before": break_before,
    }


def _optional_bonus_arrays(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    present = [name in state for name in _SOURCE_STATE_OPTIONAL_ARRAY_KEYS]
    if not any(present):
        return {}
    if not all(present):
        missing = [
            name
            for name, is_present in zip(_SOURCE_STATE_OPTIONAL_ARRAY_KEYS, present, strict=True)
            if not is_present
        ]
        raise VectorVisualObservationError(
            f"optional bonus visual arrays must be supplied together; missing {missing}"
        )
    bonus_active = _bool_array(state, "bonus_active", ndim=2)
    if bonus_active.shape[0] != batch_size:
        raise VectorVisualObservationError("state['bonus_active'] must have shape [B,C]")
    bonus_capacity = bonus_active.shape[1]
    bonus_pos = _numeric_array(state, "bonus_pos", shape=(batch_size, bonus_capacity, 2))
    bonus_radius = _numeric_array(state, "bonus_radius", shape=(batch_size, bonus_capacity))
    bonus_type = (
        _integer_array(state, "bonus_type", shape=(batch_size, bonus_capacity))
        if "bonus_type" in state
        else None
    )
    if bool((~np.isfinite(bonus_pos)).any()) or bool((~np.isfinite(bonus_radius)).any()):
        raise VectorVisualObservationError(
            "bonus position and radius arrays must contain finite values"
        )
    if bool((bonus_radius < 0.0).any()):
        raise VectorVisualObservationError("state['bonus_radius'] values must be non-negative")
    arrays: dict[str, np.ndarray] = {
        "bonus_active": bonus_active,
        "bonus_pos": bonus_pos,
        "bonus_radius": bonus_radius,
    }
    if bonus_type is not None:
        arrays["bonus_type"] = bonus_type
    return arrays


def _state_array(state: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in state:
        raise VectorVisualObservationError(f"state is missing required array {name!r}")
    return np.asarray(state[name])


def _numeric_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    array = _state_array(state, name)
    if not np.issubdtype(array.dtype, np.number):
        raise VectorVisualObservationError(f"state[{name!r}] must be numeric")
    _check_shape(name, array, shape=shape, ndim=ndim)
    return array


def _integer_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    array = _state_array(state, name)
    if not np.issubdtype(array.dtype, np.integer):
        raise VectorVisualObservationError(f"state[{name!r}] must be integer")
    _check_shape(name, array, shape=shape, ndim=ndim)
    return array


def _bool_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    array = _state_array(state, name)
    if array.dtype != np.bool_:
        raise VectorVisualObservationError(f"state[{name!r}] must be bool")
    _check_shape(name, array, shape=shape, ndim=ndim)
    return array


def _check_shape(
    name: str,
    array: np.ndarray,
    *,
    shape: tuple[int, ...] | None,
    ndim: int | None,
) -> None:
    if shape is not None and array.shape != shape:
        raise VectorVisualObservationError(f"state[{name!r}] must have shape {shape}")
    if ndim is not None and array.ndim != ndim:
        raise VectorVisualObservationError(f"state[{name!r}] must have ndim {ndim}")


def _row_index(row: int, batch_size: int) -> int:
    try:
        row_index = int(row)
    except (TypeError, ValueError) as exc:
        raise VectorVisualObservationError("row must be an integer") from exc
    if row_index < 0 or row_index >= batch_size:
        raise VectorVisualObservationError(f"row must be in [0, {batch_size}), got {row_index}")
    return row_index


def _validated_frame(frame: np.ndarray, *, name: str) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.shape != SOURCE_STATE_GRAY64_SHAPE:
        raise VectorVisualObservationError(
            f"{name} must have shape {SOURCE_STATE_GRAY64_SHAPE}, got {frame_array.shape}"
        )
    if frame_array.dtype != np.uint8:
        raise VectorVisualObservationError(f"{name} dtype must be uint8, got {frame_array.dtype}")
    return frame_array


def _validated_bonus64_out(out: np.ndarray | None) -> np.ndarray:
    if out is None:
        return np.zeros(SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE, dtype=np.float32)
    array = np.asarray(out)
    if array.shape != SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE:
        raise VectorVisualObservationError(
            "out must have shape "
            f"{SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE}, got {array.shape}"
        )
    if array.dtype != np.float32:
        raise VectorVisualObservationError(f"out dtype must be float32, got {array.dtype}")
    return array


def _validated_occupancy_stack(stack: np.ndarray) -> np.ndarray:
    array = np.asarray(stack)
    expected = (
        SOURCE_STATE_BONUS64_STACK4_OCCUPANCY_CHANNELS,
        SOURCE_STATE_GRAY64_SHAPE[1],
        SOURCE_STATE_GRAY64_SHAPE[2],
    )
    if array.shape != expected:
        raise VectorVisualObservationError(
            f"occupancy_stack must have shape {expected}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise VectorVisualObservationError("occupancy_stack must be numeric")
    if bool((~np.isfinite(array)).any()):
        raise VectorVisualObservationError("occupancy_stack must contain finite values")
    return np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)


def _bonus64_arrays(
    state: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = dict(arrays)
    for name in (
        "bonus_type",
        "bonus_id",
        "base_radius",
        "speed",
        "base_speed",
        "inverse",
        "invincible",
        "printing",
        "angular_velocity_per_ms",
        "base_angular_velocity_per_ms",
        "bonus_stack_count",
        "bonus_stack_duration_ms",
    ):
        if name in state:
            result[name] = np.asarray(state[name])
    return result


def _player_perspective_gray64(frame: np.ndarray, controlled_player: int) -> np.ndarray:
    frame_array = _validated_frame(frame, name="frame")
    mapping = np.arange(256, dtype=np.uint8)
    for source_player in range(2):
        body_value = _SOURCE_BODY_VALUE_BASE + source_player * _SOURCE_BODY_VALUE_STEP
        head_value = _SOURCE_HEAD_VALUE_BASE + source_player * _SOURCE_HEAD_VALUE_STEP
        mapping[body_value] = (
            _SELF_BODY_VALUE if source_player == controlled_player else _OTHER_BODY_VALUE
        )
        mapping[head_value] = (
            _SELF_HEAD_VALUE if source_player == controlled_player else _OTHER_HEAD_VALUE
        )
    return np.take(mapping, frame_array)


def _draw_bonus64_bonus_planes(
    output: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
) -> None:
    if "bonus_active" not in arrays:
        return
    bonus_type = arrays.get("bonus_type")
    bonus_id = arrays.get("bonus_id")
    map_size = float(arrays["map_size"][row])
    winners = np.full((64, 64), -1, dtype=np.int32)
    bonus_slots = np.flatnonzero(arrays["bonus_active"][row])
    for slot in bonus_slots:
        x = float(arrays["bonus_pos"][row, slot, 0])
        y = float(arrays["bonus_pos"][row, slot, 1])
        radius = float(arrays["bonus_radius"][row, slot])
        mask_info = _world_circle_patch_mask(x, y, radius, map_size)
        if mask_info is None:
            continue
        y_slice, x_slice, mask = mask_info
        slot_id = int(bonus_id[row, slot]) if bonus_id is not None else int(slot) + 1
        patch_winners = winners[y_slice, x_slice]
        replace = mask & (slot_id >= patch_winners)
        if not bool(replace.any()):
            continue
        patch_winners[replace] = slot_id
        output[SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL, y_slice, x_slice][replace] = 1.0
        code = int(bonus_type[row, slot]) if bonus_type is not None else 0
        output[SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL, y_slice, x_slice][replace] = (
            np.float32(np.clip(code, 0, int(SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE)))
            / np.float32(SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE)
        )


def _world_circle_patch_mask(
    x: float,
    y: float,
    radius: float,
    map_size: float,
) -> tuple[slice, slice, np.ndarray] | None:
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(radius):
        return None
    if x + radius <= 0.0 or x - radius >= map_size:
        return None
    if y + radius <= 0.0 or y - radius >= map_size:
        return None
    radius_px = int(max(0, np.ceil((radius / map_size) * 64.0)))
    px = int(np.clip(np.rint((x / map_size) * 63.0), 0, 63))
    py = int(np.clip(np.rint((y / map_size) * 63.0), 0, 63))
    if radius_px == 0:
        return slice(py, py + 1), slice(px, px + 1), np.ones((1, 1), dtype=bool)
    x0 = max(0, px - radius_px)
    x1 = min(63, px + radius_px)
    y0 = max(0, py - radius_px)
    y1 = min(63, py + radius_px)
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius_px**2
    return slice(y0, y1 + 1), slice(x0, x1 + 1), mask


def _rgb_frame_size(value: int) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise VectorVisualObservationError("frame_size must be an integer") from exc
    if size < 64:
        raise VectorVisualObservationError("frame_size must be at least 64")
    return size


def _validated_rgb_frame(out: np.ndarray, *, frame_size: int) -> np.ndarray:
    array = np.asarray(out)
    expected = (frame_size, frame_size, 3)
    if array.shape != expected:
        raise VectorVisualObservationError(f"out must have shape {expected}, got {array.shape}")
    if array.dtype != np.uint8:
        raise VectorVisualObservationError(f"out dtype must be uint8, got {array.dtype}")
    return array


def _compatible_rgb_out(out: np.ndarray | None, *, frame_size: int) -> np.ndarray | None:
    if out is None:
        return None
    array = np.asarray(out)
    if array.shape == (frame_size, frame_size, 3):
        return _validated_rgb_frame(array, frame_size=frame_size)
    if array.shape == (64, 64, 3) and frame_size != 64:
        if array.dtype != np.uint8:
            raise VectorVisualObservationError(f"rgb_out dtype must be uint8, got {array.dtype}")
        return None
    return _validated_rgb_frame(array, frame_size=frame_size)


def _rgb_canvas_like_luma_downsample64(
    rgb_array: np.ndarray,
    *,
    out: np.ndarray | None = None,
    scratch: SourceStateGray64DownsampleScratch | None = None,
) -> np.ndarray:
    target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
    source_size = int(rgb_array.shape[0])
    if source_size % target_size != 0:
        raise VectorVisualObservationError(
            "rgb frame size must be 64 or an integer multiple of 64 for gray64 conversion"
        )
    ratio = source_size // target_size
    if out is not None:
        out_array = np.asarray(out)
        expected = (target_size, target_size)
        if out_array.shape != expected:
            raise VectorVisualObservationError(f"out must have shape {expected}, got {out_array.shape}")
        if out_array.dtype != np.uint8:
            raise VectorVisualObservationError(f"out dtype must be uint8, got {out_array.dtype}")
    else:
        out_array = None
    if scratch is not None:
        if scratch.source_frame_size != source_size:
            raise VectorVisualObservationError(
                "downsample scratch source_frame_size does not match RGB frame"
            )
        np.multiply(rgb_array[:, :, 0], np.float32(0.299), out=scratch.gray)
        np.multiply(rgb_array[:, :, 1], np.float32(0.587), out=scratch.temp)
        np.add(scratch.gray, scratch.temp, out=scratch.gray)
        np.multiply(rgb_array[:, :, 2], np.float32(0.114), out=scratch.temp)
        np.add(scratch.gray, scratch.temp, out=scratch.gray)
        blocks = scratch.gray.reshape(target_size, ratio, target_size, ratio)
        np.mean(blocks, axis=(1, 3), dtype=np.float32, out=scratch.downsampled)
        np.rint(scratch.downsampled, out=scratch.downsampled)
        np.clip(scratch.downsampled, 0.0, 255.0, out=scratch.downsampled)
        if out_array is None:
            return scratch.downsampled.astype(np.uint8)
        np.copyto(out_array, scratch.downsampled, casting="unsafe")
        return out_array
    gray = (
        rgb_array[:, :, 0].astype(np.float32) * np.float32(0.299)
        + rgb_array[:, :, 1].astype(np.float32) * np.float32(0.587)
        + rgb_array[:, :, 2].astype(np.float32) * np.float32(0.114)
    )
    downsampled = gray.reshape(target_size, ratio, target_size, ratio).mean(
        axis=(1, 3),
        dtype=np.float32,
    )
    np.rint(downsampled, out=downsampled)
    np.clip(downsampled, 0.0, 255.0, out=downsampled)
    if out_array is None:
        return downsampled.astype(np.uint8)
    np.copyto(out_array, downsampled, casting="unsafe")
    return out_array


def _rgb_canvas_like_luma_downsample64_dirty(
    rgb_array: np.ndarray,
    *,
    out: np.ndarray,
    dirty_blocks: np.ndarray,
    scratch: SourceStateGray64DirtyDownsampleScratch | None = None,
) -> int:
    target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
    source_size = int(rgb_array.shape[0])
    if source_size % target_size != 0:
        raise VectorVisualObservationError(
            "rgb frame size must be 64 or an integer multiple of 64 for gray64 conversion"
        )
    ratio = source_size // target_size
    out_array = np.asarray(out)
    if out_array.shape != (target_size, target_size):
        raise VectorVisualObservationError(f"out must have shape {(target_size, target_size)}, got {out_array.shape}")
    if out_array.dtype != np.uint8:
        raise VectorVisualObservationError(f"out dtype must be uint8, got {out_array.dtype}")
    blocks = _validated_dirty_block_mask(dirty_blocks)
    if scratch is None:
        scratch = SourceStateGray64DirtyDownsampleScratch(source_size)
    if scratch.source_frame_size != source_size:
        raise VectorVisualObservationError(
            "dirty downsample scratch source_frame_size does not match RGB frame"
        )
    dirty_y, dirty_x = np.nonzero(blocks)
    for block_y, block_x in zip(dirty_y, dirty_x, strict=True):
        y0 = int(block_y) * ratio
        x0 = int(block_x) * ratio
        patch = rgb_array[y0 : y0 + ratio, x0 : x0 + ratio]
        np.multiply(patch[:, :, 0], np.float32(0.299), out=scratch.luma)
        np.multiply(patch[:, :, 1], np.float32(0.587), out=scratch.temp)
        np.add(scratch.luma, scratch.temp, out=scratch.luma)
        np.multiply(patch[:, :, 2], np.float32(0.114), out=scratch.temp)
        np.add(scratch.luma, scratch.temp, out=scratch.luma)
        downsample_blocks = scratch.luma.reshape(1, ratio, 1, ratio)
        np.mean(downsample_blocks, axis=(1, 3), dtype=np.float32, out=scratch.downsampled)
        np.rint(scratch.downsampled, out=scratch.downsampled)
        np.clip(scratch.downsampled, 0.0, 255.0, out=scratch.downsampled)
        np.copyto(
            out_array[int(block_y) : int(block_y) + 1, int(block_x) : int(block_x) + 1],
            scratch.downsampled,
            casting="unsafe",
        )
    return int(dirty_y.size)


def _rgb_triplet(value: Sequence[int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.int16)
    if array.shape != (3,):
        raise VectorVisualObservationError("RGB values must have shape [3]")
    if bool(((array < 0) | (array > 255)).any()):
        raise VectorVisualObservationError("RGB values must be in [0, 255]")
    return array.astype(np.uint8)


def _player_rgb_values(
    arrays: Mapping[str, np.ndarray],
    row: int,
    *,
    player_rgb: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    player_count = arrays["pos"].shape[1]
    raw_colors = (
        SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB
        if player_rgb is None
        else tuple(tuple(color) for color in player_rgb)
    )
    colors = np.zeros((player_count, 3), dtype=np.uint8)
    if "avatar_color" in arrays:
        color_indices = np.asarray(arrays["avatar_color"][row, :player_count], dtype=np.int64)
    else:
        color_indices = np.arange(player_count, dtype=np.int64)
    for player, color_index in enumerate(color_indices):
        colors[player] = _rgb_triplet(raw_colors[int(color_index) % len(raw_colors)])
    return colors


@dataclass(slots=True)
class SourceStateBrowserLineTrailLayerCacheStats:
    render_calls: int = 0
    cache_hits: int = 0
    rebuilds: int = 0
    incremental_updates: int = 0
    fallback_full_renders: int = 0
    cursor_regression_rebuilds: int = 0
    prefix_mutation_rebuilds: int = 0
    unsupported_full_renders: int = 0
    mask_recolors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "render_calls": self.render_calls,
            "cache_hits": self.cache_hits,
            "rebuilds": self.rebuilds,
            "incremental_updates": self.incremental_updates,
            "fallback_full_renders": self.fallback_full_renders,
            "cursor_regression_rebuilds": self.cursor_regression_rebuilds,
            "prefix_mutation_rebuilds": self.prefix_mutation_rebuilds,
            "unsupported_full_renders": self.unsupported_full_renders,
            "mask_recolors": self.mask_recolors,
        }


@dataclass(slots=True)
class _SourceStateTrailRecords:
    slots: np.ndarray
    positions: np.ndarray
    radii: np.ndarray
    owners: np.ndarray
    break_before: np.ndarray


@dataclass(slots=True)
class _SourceStateOwnerTrailLayer:
    owner: int
    rgb: np.ndarray
    mask: np.ndarray
    slots: np.ndarray
    positions: np.ndarray
    radii: np.ndarray
    break_before: np.ndarray

    @classmethod
    def empty(
        cls,
        *,
        owner: int,
        frame_size: int,
        sentinel: np.ndarray,
    ) -> "_SourceStateOwnerTrailLayer":
        rgb = np.empty((frame_size, frame_size, 3), dtype=np.uint8)
        rgb[:, :] = sentinel
        return cls(
            owner=int(owner),
            rgb=rgb,
            mask=np.zeros((frame_size, frame_size), dtype=bool),
            slots=np.empty(0, dtype=np.int64),
            positions=np.empty((0, 2), dtype=np.float64),
            radii=np.empty(0, dtype=np.float64),
            break_before=np.empty(0, dtype=bool),
        )


class SourceStateBrowserLineTrailLayerCache:
    """Append-only visual-trail cache for exact browser-line trail layers."""

    _sentinel_rgb = (1, 2, 3)

    def __init__(
        self,
        *,
        frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        min_active_slots: int = 256,
    ) -> None:
        self.frame_size = _rgb_frame_size(frame_size)
        self.min_active_slots = max(0, int(min_active_slots))
        self.sentinel = _rgb_triplet(self._sentinel_rgb)
        self.layers: dict[int, _SourceStateOwnerTrailLayer] = {}
        self.last_cursor: int | None = None
        self.last_prefix: _SourceStateTrailRecords | None = None
        self.last_map_size: float | None = None
        self.last_colors_key: tuple[tuple[int, int, int], ...] | None = None
        self.stats = SourceStateBrowserLineTrailLayerCacheStats()

    def reset(self) -> None:
        self.layers.clear()
        self.last_cursor = None
        self.last_prefix = None
        self.last_map_size = None
        self.last_colors_key = None

    def render_trails_rgb(
        self,
        frame: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        row: int,
        map_size: float,
        *,
        colors: np.ndarray,
        background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    ) -> bool:
        """Render only browser-line trails into ``frame`` if the cache can handle the row."""

        output = _validated_rgb_frame(frame, frame_size=self.frame_size)
        if not self.update_trail_layers(arrays, row, map_size, colors=colors):
            return False

        output[:, :] = _rgb_triplet(background_rgb)
        self._composite_layers(output)
        return True

    def update_trail_layers(
        self,
        arrays: Mapping[str, np.ndarray],
        row: int,
        map_size: float,
        *,
        colors: np.ndarray,
    ) -> bool:
        """Update cached browser-line layers without compositing a full RGB frame."""

        self.stats.render_calls += 1
        row_index = _row_index(row, arrays["tick"].shape[0])
        if not self._is_supported_visual_state(arrays, row_index, colors=colors):
            self.stats.unsupported_full_renders += 1
            self.stats.fallback_full_renders += 1
            self.reset()
            return False

        records = _source_state_visual_trail_records(arrays, row_index)
        cursor = int(arrays["visual_trail_write_cursor"][row_index])
        colors_key = _source_state_colors_key(colors)
        rebuild_reason = self._rebuild_reason(
            records=records,
            cursor=cursor,
            map_size=map_size,
            colors_key=colors_key,
        )
        if rebuild_reason is not None:
            if rebuild_reason == "cursor_regression":
                self.stats.cursor_regression_rebuilds += 1
            elif rebuild_reason == "prefix_mutation":
                self.stats.prefix_mutation_rebuilds += 1
            self._rebuild(records, map_size=map_size, colors=colors)
            self.stats.rebuilds += 1
        else:
            changed = self._append(records, map_size=map_size, colors=colors)
            if changed:
                self.stats.incremental_updates += 1
            else:
                self.stats.cache_hits += 1

        self.last_cursor = cursor
        self.last_prefix = _source_state_copy_trail_records(records)
        self.last_map_size = float(map_size)
        self.last_colors_key = colors_key
        return True

    def _is_supported_visual_state(
        self,
        arrays: Mapping[str, np.ndarray],
        row: int,
        *,
        colors: np.ndarray,
    ) -> bool:
        required = (
            "visual_trail_active",
            "visual_trail_write_cursor",
            "visual_trail_pos",
            "visual_trail_radius",
            "visual_trail_owner",
            "visual_trail_break_before",
        )
        if any(name not in arrays for name in required):
            return False
        active = arrays["visual_trail_active"]
        cursor = int(arrays["visual_trail_write_cursor"][row])
        if active.ndim != 2 or cursor < 0 or cursor > active.shape[1]:
            return False
        active_slots = int(np.count_nonzero(active[row, :cursor]))
        if active_slots <= 0 or active_slots < self.min_active_slots:
            return False
        reserved = tuple(int(value) for value in self.sentinel)
        owner_colors = [tuple(int(value) for value in _body_owner_rgb(-1, colors))]
        owner_colors.extend(tuple(int(value) for value in color) for color in colors)
        return reserved not in owner_colors

    def _rebuild_reason(
        self,
        *,
        records: _SourceStateTrailRecords,
        cursor: int,
        map_size: float,
        colors_key: tuple[tuple[int, int, int], ...],
    ) -> str | None:
        if self.last_cursor is None or self.last_prefix is None:
            return "cold_start"
        if cursor < self.last_cursor:
            return "cursor_regression"
        if self.last_map_size is None or not np.isclose(
            map_size,
            self.last_map_size,
            rtol=0.0,
            atol=0.0,
        ):
            return "map_size_or_geometry_change"
        if colors_key != self.last_colors_key:
            return "palette_change"
        old = self.last_prefix
        old_count = int(old.slots.size)
        if records.slots.size < old_count:
            return "prefix_mutation"
        if not (
            np.array_equal(records.slots[:old_count], old.slots)
            and np.array_equal(records.positions[:old_count], old.positions)
            and np.array_equal(records.radii[:old_count], old.radii)
            and np.array_equal(records.owners[:old_count], old.owners)
            and np.array_equal(records.break_before[:old_count], old.break_before)
        ):
            return "prefix_mutation"
        return None

    def _rebuild(
        self,
        records: _SourceStateTrailRecords,
        *,
        map_size: float,
        colors: np.ndarray,
    ) -> None:
        self.layers.clear()
        for owner in _browser_line_owner_draw_order(records.owners):
            owner_records = _source_state_owner_trail_records(records, owner)
            layer = _SourceStateOwnerTrailLayer.empty(
                owner=owner,
                frame_size=self.frame_size,
                sentinel=self.sentinel,
            )
            if owner_records.slots.size:
                _draw_ordered_browser_line_path_rgb(
                    layer.rgb,
                    owner_records.positions,
                    owner_records.radii,
                    map_size,
                    color=_body_owner_rgb(owner, colors),
                    break_before=owner_records.break_before,
                )
                _source_state_refresh_layer_mask(layer, sentinel=self.sentinel)
            _source_state_replace_layer_records(layer, owner_records)
            self.layers[int(owner)] = layer

    def _append(
        self,
        records: _SourceStateTrailRecords,
        *,
        map_size: float,
        colors: np.ndarray,
    ) -> bool:
        changed = False
        current_owners = set(int(owner) for owner in np.unique(records.owners))
        for owner in current_owners:
            owner_records = _source_state_owner_trail_records(records, owner)
            layer = self.layers.get(owner)
            if layer is None:
                layer = _SourceStateOwnerTrailLayer.empty(
                    owner=owner,
                    frame_size=self.frame_size,
                    sentinel=self.sentinel,
                )
                self.layers[owner] = layer
            old_count = int(layer.slots.size)
            if owner_records.slots.size == old_count:
                continue
            if owner_records.slots.size < old_count:
                self._rebuild(records, map_size=map_size, colors=colors)
                self.stats.rebuilds += 1
                return True
            if not _source_state_owner_prefix_matches(layer, owner_records, old_count):
                self._rebuild(records, map_size=map_size, colors=colors)
                self.stats.prefix_mutation_rebuilds += 1
                self.stats.rebuilds += 1
                return True
            draw_start = _source_state_incremental_draw_start(owner_records, old_count)
            _draw_ordered_browser_line_path_rgb(
                layer.rgb,
                owner_records.positions[draw_start:],
                owner_records.radii[draw_start:],
                map_size,
                color=_body_owner_rgb(owner, colors),
                break_before=owner_records.break_before[draw_start:],
            )
            _source_state_refresh_layer_mask_for_points(
                layer,
                owner_records.positions[draw_start:],
                owner_records.radii[draw_start:],
                map_size=map_size,
                sentinel=self.sentinel,
            )
            _source_state_replace_layer_records(layer, owner_records)
            changed = True

        stale_owners = set(self.layers) - current_owners
        if stale_owners:
            self._rebuild(records, map_size=map_size, colors=colors)
            self.stats.prefix_mutation_rebuilds += 1
            self.stats.rebuilds += 1
            return True
        return changed

    def _composite_layers(self, frame: np.ndarray) -> None:
        if not self.layers:
            return
        owners = np.asarray(tuple(self.layers), dtype=np.int64)
        for owner in _browser_line_owner_draw_order(owners):
            layer = self.layers[int(owner)]
            if bool(layer.mask.any()):
                frame[layer.mask] = layer.rgb[layer.mask]

    def recolor_trails_rgb(
        self,
        frame: np.ndarray,
        *,
        colors: np.ndarray,
        background_rgb: Sequence[int] | None = None,
    ) -> bool:
        """Recolor cached flat trail layers in draw order without scanning RGB pixels."""

        output = _validated_rgb_frame(frame, frame_size=self.frame_size)
        if not self.layers:
            return False
        if background_rgb is not None:
            output[:, :] = _rgb_triplet(background_rgb)
        owners = np.asarray(tuple(self.layers), dtype=np.int64)
        for owner in _browser_line_owner_draw_order(owners):
            layer = self.layers[int(owner)]
            if bool(layer.mask.any()):
                output[layer.mask] = _body_owner_rgb(owner, colors)
        self.stats.mask_recolors += 1
        return True

    def compose_trail_blocks_rgb(
        self,
        frame: np.ndarray,
        dirty_blocks: np.ndarray,
        *,
        colors: np.ndarray,
        background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    ) -> int:
        """Recompose cached trail layers only inside dirty 64x64 blocks."""

        output = _validated_rgb_frame(frame, frame_size=self.frame_size)
        blocks = _validated_dirty_block_mask(dirty_blocks)
        ratio = self.frame_size // SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
        if ratio < 1 or self.frame_size % SOURCE_STATE_CANVAS_GRAY64_SHAPE[1] != 0:
            raise VectorVisualObservationError("frame size must divide into 64 dirty blocks")
        dirty_y, dirty_x = np.nonzero(blocks)
        if dirty_y.size == 0:
            return 0
        background = _rgb_triplet(background_rgb)
        owners = np.asarray(tuple(self.layers), dtype=np.int64)
        draw_order = _browser_line_owner_draw_order(owners) if owners.size else ()
        for block_y, block_x in zip(dirty_y, dirty_x, strict=True):
            y0 = int(block_y) * ratio
            x0 = int(block_x) * ratio
            y1 = y0 + ratio
            x1 = x0 + ratio
            output[y0:y1, x0:x1] = background
            for owner in draw_order:
                layer = self.layers[int(owner)]
                mask = layer.mask[y0:y1, x0:x1]
                if bool(mask.any()):
                    output[y0:y1, x0:x1][mask] = _body_owner_rgb(owner, colors)
        return int(dirty_y.size)


@dataclass(slots=True)
class SourceStateCanvasGray64DirtyRenderStats:
    attempts: int = 0
    hits: int = 0
    cold_starts: int = 0
    fallbacks: int = 0
    dirty_blocks_total: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    timing_sec: dict[str, float] = field(default_factory=dict)
    timing_calls: dict[str, int] = field(default_factory=dict)

    def record_fallback_reason(self, reason: str) -> None:
        key = str(reason)
        self.fallback_reasons[key] = int(self.fallback_reasons.get(key, 0)) + 1

    def record_fallback(self, reason: str) -> None:
        self.fallbacks += 1
        self.record_fallback_reason(reason)

    def record_timing(self, key: str, seconds: float) -> None:
        name = str(key)
        self.timing_sec[name] = float(self.timing_sec.get(name, 0.0) + float(seconds))
        self.timing_calls[name] = int(self.timing_calls.get(name, 0)) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "hits": self.hits,
            "cold_starts": self.cold_starts,
            "fallbacks": self.fallbacks,
            "dirty_blocks_total": self.dirty_blocks_total,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "timing_sec": dict(sorted(self.timing_sec.items())),
            "timing_calls": dict(sorted(self.timing_calls.items())),
        }


@dataclass(slots=True)
class _SourceStateBonusSnapshot:
    slots: np.ndarray
    types: np.ndarray
    ids: np.ndarray
    positions: np.ndarray
    radii: np.ndarray


class SourceStateCanvasGray64DirtyRenderCache:
    """Exact dirty-block cache for two-player source-state gray64 rendering.

    The cache never changes observation meaning. It starts from a full rendered
    RGB/gray baseline, then only recomposes and redownsamples changed 11x11
    source blocks when the visual trail is append-only.
    """

    def __init__(
        self,
        *,
        player_count: int = 2,
        frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        max_dirty_blocks: int = 1024,
        profile_timing: bool = False,
    ) -> None:
        self.player_count = int(player_count)
        self.frame_size = _rgb_frame_size(frame_size)
        self.max_dirty_blocks = max(1, int(max_dirty_blocks))
        self.profile_timing = bool(profile_timing)
        self.rgb_frames = np.empty(
            (self.player_count, self.frame_size, self.frame_size, 3),
            dtype=np.uint8,
        )
        self.gray_frames = np.empty(
            (self.player_count, *SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            dtype=np.uint8,
        )
        self.dirty_blocks = np.zeros(
            (SOURCE_STATE_CANVAS_GRAY64_SHAPE[1], SOURCE_STATE_CANVAS_GRAY64_SHAPE[2]),
            dtype=bool,
        )
        self.downsample_scratch = SourceStateGray64DirtyDownsampleScratch(self.frame_size)
        self.initialized = False
        self.previous_cursor: int | None = None
        self.previous_map_size: float | None = None
        self.previous_palette_key: tuple[tuple[tuple[int, int, int], ...], ...] | None = None
        self.previous_background_key: tuple[int, int, int] | None = None
        self.previous_bonus_render_mode: str | None = None
        self.previous_head_positions = np.zeros((self.player_count, 2), dtype=np.float64)
        self.previous_head_radii = np.zeros(self.player_count, dtype=np.float64)
        self.previous_head_visible = np.zeros(self.player_count, dtype=bool)
        self.previous_bonus_snapshot = _SourceStateBonusSnapshot(
            slots=np.empty(0, dtype=np.int64),
            types=np.empty(0, dtype=np.int64),
            ids=np.empty(0, dtype=np.int64),
            positions=np.empty((0, 2), dtype=np.float64),
            radii=np.empty(0, dtype=np.float64),
        )
        self.stats = SourceStateCanvasGray64DirtyRenderStats()

    def reset(self) -> None:
        self.initialized = False
        self.previous_cursor = None
        self.previous_map_size = None
        self.previous_palette_key = None
        self.previous_background_key = None
        self.previous_bonus_render_mode = None

    def _phase_start(self) -> float:
        return time.perf_counter() if self.profile_timing else 0.0

    def _phase_end(self, key: str, started: float) -> None:
        if self.profile_timing:
            self.stats.record_timing(key, time.perf_counter() - started)

    def _record_cold_start(self) -> bool:
        self.stats.cold_starts += 1
        if self.profile_timing:
            self.stats.record_fallback_reason("cold_start")
        return False

    def _fallback(self, reason: str, *, reset: bool = False) -> bool:
        if reset:
            self.reset()
        if self.profile_timing:
            self.stats.record_fallback(reason)
        else:
            self.stats.fallbacks += 1
        return False

    def capture_full(
        self,
        gray_frames: np.ndarray,
        rgb_frames: Sequence[np.ndarray],
        arrays: Mapping[str, np.ndarray],
        row: int,
        map_size: float,
        *,
        player_rgbs: Sequence[Sequence[Sequence[int]]],
        background_rgb: Sequence[int],
        bonus_render_mode: str,
    ) -> None:
        started = self._phase_start()
        if len(rgb_frames) != self.player_count:
            self.reset()
            return
        gray = _validated_player_perspective_frames(gray_frames, player_count=self.player_count)
        np.copyto(self.gray_frames, gray)
        for player, frame in enumerate(rgb_frames):
            np.copyto(
                self.rgb_frames[player],
                _validated_rgb_frame(frame, frame_size=self.frame_size),
            )
        self._capture_metadata(
            arrays,
            row,
            map_size,
            player_rgbs=player_rgbs,
            background_rgb=background_rgb,
            bonus_render_mode=bonus_render_mode,
        )
        self.initialized = True
        self._phase_end("capture_full_sec", started)

    def try_render(
        self,
        out_frames: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        row: int,
        map_size: float,
        *,
        player_rgbs: Sequence[Sequence[Sequence[int]]],
        trail_cache: SourceStateBrowserLineTrailLayerCache,
        background_rgb: Sequence[int],
        bonus_render_mode: str,
    ) -> bool:
        self.stats.attempts += 1
        total_started = self._phase_start()
        out = _validated_player_perspective_frames(out_frames, player_count=self.player_count)
        if not self.initialized:
            return self._record_cold_start()
        if len(player_rgbs) != self.player_count:
            return self._fallback("player_count_mismatch", reset=True)
        if self.frame_size % SOURCE_STATE_CANVAS_GRAY64_SHAPE[1] != 0:
            return self._fallback("frame_size_not_divisible", reset=True)
        if not _source_state_has_visual_trail_arrays(arrays):
            return self._fallback("missing_visual_trail_arrays", reset=True)
        if self.previous_map_size is None or not np.isclose(
            map_size,
            self.previous_map_size,
            rtol=0.0,
            atol=0.0,
        ):
            return self._fallback("map_size_changed")

        row_index = _row_index(row, arrays["tick"].shape[0])
        current_cursor = int(arrays["visual_trail_write_cursor"][row_index])
        if self.previous_cursor is None or current_cursor < self.previous_cursor:
            return self._fallback("cursor_regressed")
        palette_key = _source_state_player_palettes_key(arrays, row_index, player_rgbs)
        if palette_key != self.previous_palette_key:
            return self._fallback("palette_changed")
        background_key = tuple(int(value) for value in _rgb_triplet(background_rgb))
        if background_key != self.previous_background_key:
            return self._fallback("background_changed")
        if str(bonus_render_mode) != self.previous_bonus_render_mode:
            return self._fallback("bonus_mode_changed")

        base_colors = _player_rgb_values(arrays, row_index, player_rgb=player_rgbs[0])
        started = self._phase_start()
        records = _source_state_visual_trail_records(arrays, row_index)
        rebuild_reason = trail_cache._rebuild_reason(
            records=records,
            cursor=current_cursor,
            map_size=map_size,
            colors_key=_source_state_colors_key(base_colors),
        )
        self._phase_end("rebuild_reason_sec", started)
        if rebuild_reason is not None:
            return self._fallback(f"trail_cache_rebuild:{rebuild_reason}")

        started = self._phase_start()
        dirty_blocks = _source_state_dirty_blocks_for_append(
            arrays,
            row_index,
            previous_cursor=self.previous_cursor,
            previous_head_positions=self.previous_head_positions,
            previous_head_radii=self.previous_head_radii,
            previous_head_visible=self.previous_head_visible,
            previous_bonus_snapshot=self.previous_bonus_snapshot,
            bonus_render_mode=bonus_render_mode,
            out=self.dirty_blocks,
        )
        self._phase_end("dirty_blocks_sec", started)
        started = self._phase_start()
        dirty_count = int(np.count_nonzero(dirty_blocks))
        self._phase_end("dirty_count_sec", started)
        if dirty_count <= 0:
            started = self._phase_start()
            np.copyto(out, self.gray_frames)
            self._phase_end("copy_out_sec", started)
            started = self._phase_start()
            self._capture_metadata(
                arrays,
                row_index,
                map_size,
                player_rgbs=player_rgbs,
                background_rgb=background_rgb,
                bonus_render_mode=bonus_render_mode,
            )
            self._phase_end("capture_metadata_sec", started)
            self.stats.hits += 1
            self._phase_end("total_hit_sec", total_started)
            return True
        if dirty_count > self.max_dirty_blocks:
            return self._fallback("too_many_dirty_blocks")
        started = self._phase_start()
        if not trail_cache.update_trail_layers(arrays, row_index, map_size, colors=base_colors):
            self._phase_end("trail_layer_update_sec", started)
            return self._fallback("trail_layer_update_failed")
        self._phase_end("trail_layer_update_sec", started)

        for player, palette in enumerate(player_rgbs):
            colors = _player_rgb_values(arrays, row_index, player_rgb=palette)
            started = self._phase_start()
            trail_cache.compose_trail_blocks_rgb(
                self.rgb_frames[player],
                dirty_blocks,
                colors=colors,
                background_rgb=background_rgb,
            )
            self._phase_end("compose_trails_sec", started)
            started = self._phase_start()
            _draw_source_state_rgb_bonuses(
                self.rgb_frames[player],
                arrays,
                row_index,
                map_size,
                bonus_render_mode=bonus_render_mode,
                dirty_blocks=dirty_blocks,
            )
            self._phase_end("draw_bonuses_sec", started)
            started = self._phase_start()
            _draw_source_state_rgb_heads(
                self.rgb_frames[player],
                arrays,
                row_index,
                map_size,
                colors=colors,
            )
            self._phase_end("draw_heads_sec", started)
            started = self._phase_start()
            _rgb_canvas_like_luma_downsample64_dirty(
                self.rgb_frames[player],
                out=self.gray_frames[player, 0],
                dirty_blocks=dirty_blocks,
                scratch=self.downsample_scratch,
            )
            self._phase_end("dirty_downsample_sec", started)
        started = self._phase_start()
        np.copyto(out, self.gray_frames)
        self._phase_end("copy_out_sec", started)
        started = self._phase_start()
        self._capture_metadata(
            arrays,
            row_index,
            map_size,
            player_rgbs=player_rgbs,
            background_rgb=background_rgb,
            bonus_render_mode=bonus_render_mode,
        )
        self._phase_end("capture_metadata_sec", started)
        self.stats.hits += 1
        self.stats.dirty_blocks_total += dirty_count
        self._phase_end("total_hit_sec", total_started)
        return True

    def _capture_metadata(
        self,
        arrays: Mapping[str, np.ndarray],
        row: int,
        map_size: float,
        *,
        player_rgbs: Sequence[Sequence[Sequence[int]]],
        background_rgb: Sequence[int],
        bonus_render_mode: str,
    ) -> None:
        row_index = _row_index(row, arrays["tick"].shape[0])
        self.previous_cursor = (
            int(arrays["visual_trail_write_cursor"][row_index])
            if "visual_trail_write_cursor" in arrays
            else None
        )
        self.previous_map_size = float(map_size)
        self.previous_palette_key = _source_state_player_palettes_key(
            arrays,
            row_index,
            player_rgbs,
        )
        self.previous_background_key = tuple(int(value) for value in _rgb_triplet(background_rgb))
        self.previous_bonus_render_mode = str(bonus_render_mode)
        count = min(self.player_count, int(arrays["pos"].shape[1]))
        self.previous_head_positions[:count] = arrays["pos"][row_index, :count]
        self.previous_head_radii[:count] = arrays["radius"][row_index, :count]
        self.previous_head_visible[:count] = (
            arrays["present"][row_index, :count] & arrays["alive"][row_index, :count]
        )
        if count < self.player_count:
            self.previous_head_visible[count:] = False
        self.previous_bonus_snapshot = _source_state_bonus_snapshot(arrays, row_index)


def _source_state_has_visual_trail_arrays(arrays: Mapping[str, np.ndarray]) -> bool:
    return all(
        name in arrays
        for name in (
            "visual_trail_active",
            "visual_trail_write_cursor",
            "visual_trail_pos",
            "visual_trail_radius",
            "visual_trail_owner",
            "visual_trail_break_before",
        )
    )


def _source_state_player_palettes_key(
    arrays: Mapping[str, np.ndarray],
    row: int,
    player_rgbs: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return tuple(
        _source_state_colors_key(_player_rgb_values(arrays, row, player_rgb=palette))
        for palette in player_rgbs
    )


def _source_state_bonus_snapshot(
    arrays: Mapping[str, np.ndarray],
    row: int,
) -> _SourceStateBonusSnapshot:
    if "bonus_active" not in arrays:
        return _SourceStateBonusSnapshot(
            slots=np.empty(0, dtype=np.int64),
            types=np.empty(0, dtype=np.int64),
            ids=np.empty(0, dtype=np.int64),
            positions=np.empty((0, 2), dtype=np.float64),
            radii=np.empty(0, dtype=np.float64),
        )
    slots = np.flatnonzero(arrays["bonus_active"][row])
    if slots.size == 0:
        return _SourceStateBonusSnapshot(
            slots=np.empty(0, dtype=np.int64),
            types=np.empty(0, dtype=np.int64),
            ids=np.empty(0, dtype=np.int64),
            positions=np.empty((0, 2), dtype=np.float64),
            radii=np.empty(0, dtype=np.float64),
        )
    bonus_types = arrays.get("bonus_type")
    bonus_ids = arrays.get("bonus_id")
    return _SourceStateBonusSnapshot(
        slots=slots.astype(np.int64, copy=True),
        types=(
            bonus_types[row, slots].astype(np.int64, copy=True)
            if bonus_types is not None
            else np.full(slots.shape, -1, dtype=np.int64)
        ),
        ids=(
            bonus_ids[row, slots].astype(np.int64, copy=True)
            if bonus_ids is not None
            else slots.astype(np.int64, copy=True)
        ),
        positions=arrays["bonus_pos"][row, slots].astype(np.float64, copy=True),
        radii=arrays["bonus_radius"][row, slots].astype(np.float64, copy=True),
    )


def _source_state_dirty_blocks_for_append(
    arrays: Mapping[str, np.ndarray],
    row: int,
    *,
    previous_cursor: int,
    previous_head_positions: np.ndarray,
    previous_head_radii: np.ndarray,
    previous_head_visible: np.ndarray,
    previous_bonus_snapshot: _SourceStateBonusSnapshot,
    bonus_render_mode: str,
    out: np.ndarray,
) -> np.ndarray:
    blocks = _validated_dirty_block_mask(out)
    blocks[:, :] = False
    map_size = float(arrays["map_size"][row])
    current_cursor = int(arrays["visual_trail_write_cursor"][row])
    for slot in range(int(previous_cursor), current_cursor):
        if not bool(arrays["visual_trail_active"][row, slot]):
            continue
        owner = int(arrays["visual_trail_owner"][row, slot])
        pos = arrays["visual_trail_pos"][row, slot].astype(np.float64, copy=False)
        radius = float(arrays["visual_trail_radius"][row, slot])
        previous = _source_state_previous_owner_visual_trail_position(
            arrays,
            row=row,
            owner=owner,
            before_cursor=slot,
        )
        if previous is None or bool(arrays["visual_trail_break_before"][row, slot]):
            _source_state_mark_world_cap_dirty_blocks(
                blocks,
                pos=pos,
                radius=radius,
                map_size=map_size,
            )
        else:
            _source_state_mark_world_segment_dirty_blocks(
                blocks,
                start=previous,
                end=pos,
                radius=radius,
                map_size=map_size,
            )

    player_count = int(arrays["pos"].shape[1])
    for player in range(player_count):
        if player < previous_head_visible.size and bool(previous_head_visible[player]):
            _source_state_mark_world_circle_dirty_blocks(
                blocks,
                pos=previous_head_positions[player],
                radius=float(previous_head_radii[player]),
                map_size=map_size,
            )
        if bool(arrays["present"][row, player]) and bool(arrays["alive"][row, player]):
            _source_state_mark_world_circle_dirty_blocks(
                blocks,
                pos=arrays["pos"][row, player],
                radius=float(arrays["radius"][row, player]),
                map_size=map_size,
            )

    current_bonus_snapshot = _source_state_bonus_snapshot(arrays, row)
    if not _source_state_bonus_snapshots_equal(previous_bonus_snapshot, current_bonus_snapshot):
        _source_state_mark_bonus_snapshot_dirty_blocks(
            blocks,
            previous_bonus_snapshot,
            map_size=map_size,
            bonus_render_mode=bonus_render_mode,
        )
        _source_state_mark_bonus_snapshot_dirty_blocks(
            blocks,
            current_bonus_snapshot,
            map_size=map_size,
            bonus_render_mode=bonus_render_mode,
        )
    _source_state_expand_dirty_blocks_for_current_bonuses(
        blocks,
        current_bonus_snapshot,
        map_size=map_size,
        bonus_render_mode=bonus_render_mode,
    )
    return blocks


def _source_state_bonus_snapshots_equal(
    left: _SourceStateBonusSnapshot,
    right: _SourceStateBonusSnapshot,
) -> bool:
    return bool(
        np.array_equal(left.slots, right.slots)
        and np.array_equal(left.types, right.types)
        and np.array_equal(left.ids, right.ids)
        and np.array_equal(left.positions, right.positions)
        and np.array_equal(left.radii, right.radii)
    )


def _source_state_previous_owner_visual_trail_position(
    arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    owner: int,
    before_cursor: int,
) -> np.ndarray | None:
    if before_cursor <= 0:
        return None
    active = arrays["visual_trail_active"][row, :before_cursor]
    owners = arrays["visual_trail_owner"][row, :before_cursor]
    slots = np.flatnonzero(active & (owners == int(owner)))
    if slots.size == 0:
        return None
    return arrays["visual_trail_pos"][row, int(slots[-1])].astype(np.float64, copy=True)


def _source_state_mark_bonus_snapshot_dirty_blocks(
    blocks: np.ndarray,
    snapshot: _SourceStateBonusSnapshot,
    *,
    map_size: float,
    bonus_render_mode: str,
) -> None:
    for pos, radius in zip(snapshot.positions, snapshot.radii, strict=True):
        block_slice = _source_state_bonus_dirty_block_slice(
            pos,
            float(radius),
            map_size=map_size,
            bonus_render_mode=bonus_render_mode,
        )
        if block_slice is None:
            continue
        y_slice, x_slice = block_slice
        blocks[y_slice, x_slice] = True


def _source_state_expand_dirty_blocks_for_current_bonuses(
    blocks: np.ndarray,
    snapshot: _SourceStateBonusSnapshot,
    *,
    map_size: float,
    bonus_render_mode: str,
) -> None:
    slices = [
        block_slice
        for pos, radius in zip(snapshot.positions, snapshot.radii, strict=True)
        if (
            block_slice := _source_state_bonus_dirty_block_slice(
                pos,
                float(radius),
                map_size=map_size,
                bonus_render_mode=bonus_render_mode,
            )
        )
        is not None
    ]
    while True:
        changed = False
        for y_slice, x_slice in slices:
            view = blocks[y_slice, x_slice]
            if bool(np.any(view)) and not bool(np.all(view)):
                view[:, :] = True
                changed = True
        if not changed:
            return


def _source_state_bonus_intersects_dirty_blocks(
    pos: np.ndarray,
    radius: float,
    *,
    map_size: float,
    bonus_render_mode: str,
    dirty_blocks: np.ndarray,
) -> bool:
    block_slice = _source_state_bonus_dirty_block_slice(
        pos,
        radius,
        map_size=map_size,
        bonus_render_mode=bonus_render_mode,
    )
    if block_slice is None:
        return False
    y_slice, x_slice = block_slice
    return bool(np.any(dirty_blocks[y_slice, x_slice]))


def _source_state_bonus_dirty_block_slice(
    pos: np.ndarray,
    radius: float,
    *,
    map_size: float,
    bonus_render_mode: str,
) -> tuple[slice, slice] | None:
    if bonus_render_mode == BONUS_RENDER_MODE_BROWSER_SPRITES:
        return _source_state_world_sprite_dirty_block_slice(
            pos=pos,
            radius=float(radius),
            map_size=map_size,
        )
    outline_radius = map_size * 2.0 / float(SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE)
    return _source_state_world_circle_dirty_block_slice(
        pos=pos,
        radius=float(radius) + outline_radius,
        map_size=map_size,
    )


def _source_state_world_circle_dirty_block_slice(
    *,
    pos: np.ndarray,
    radius: float,
    map_size: float,
) -> tuple[slice, slice] | None:
    if not np.isfinite(pos).all() or not np.isfinite(radius) or map_size <= 0.0:
        return None
    size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    radius_px = int(max(0, np.ceil((float(radius) / float(map_size)) * float(size))))
    px = int(np.clip(np.rint((float(pos[0]) / float(map_size)) * float(size - 1)), 0, size - 1))
    py = int(np.clip(np.rint((float(pos[1]) / float(map_size)) * float(size - 1)), 0, size - 1))
    return _source_state_pixel_bbox_dirty_block_slice(
        min_x=float(px - radius_px),
        min_y=float(py - radius_px),
        max_x=float(px + radius_px),
        max_y=float(py + radius_px),
    )


def _source_state_world_sprite_dirty_block_slice(
    *,
    pos: np.ndarray,
    radius: float,
    map_size: float,
) -> tuple[slice, slice] | None:
    if not np.isfinite(pos).all() or not np.isfinite(radius) or radius <= 0.0 or map_size <= 0.0:
        return None
    size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    scale = float(size) / float(map_size)
    dst_x = _canvas_round((float(pos[0]) - float(radius)) * scale)
    dst_y = _canvas_round((float(pos[1]) - float(radius)) * scale)
    dst_size = _canvas_round((float(radius) * 2.0) * scale)
    if dst_size <= 0:
        return None
    return _source_state_pixel_bbox_dirty_block_slice(
        min_x=float(dst_x),
        min_y=float(dst_y),
        max_x=float(dst_x + dst_size - 1),
        max_y=float(dst_y + dst_size - 1),
    )


def _source_state_pixel_bbox_dirty_block_slice(
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> tuple[slice, slice] | None:
    if not all(np.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
        return None
    source_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
    ratio = source_size // target_size
    px0 = max(0, int(np.floor(min_x - 2.0)))
    py0 = max(0, int(np.floor(min_y - 2.0)))
    px1 = min(source_size - 1, int(np.ceil(max_x + 2.0)))
    py1 = min(source_size - 1, int(np.ceil(max_y + 2.0)))
    if px0 > px1 or py0 > py1:
        return None
    return (
        slice(py0 // ratio, (py1 // ratio) + 1),
        slice(px0 // ratio, (px1 // ratio) + 1),
    )


def _source_state_mark_world_cap_dirty_blocks(
    blocks: np.ndarray,
    *,
    pos: np.ndarray,
    radius: float,
    map_size: float,
) -> None:
    if not np.isfinite(pos).all() or not np.isfinite(radius) or map_size <= 0.0:
        return
    scale = float(SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE) / float(map_size)
    center = pos.astype(np.float64, copy=False) * scale
    radius_px = float(radius) * scale
    _source_state_mark_pixel_bbox_dirty_blocks(
        blocks,
        min_x=float(center[0] - radius_px - 1.0),
        min_y=float(center[1] - radius_px - 1.0),
        max_x=float(center[0] + radius_px + 1.0),
        max_y=float(center[1] + radius_px + 1.0),
    )


def _source_state_mark_world_segment_dirty_blocks(
    blocks: np.ndarray,
    *,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    map_size: float,
) -> None:
    if (
        not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or not np.isfinite(radius)
        or map_size <= 0.0
    ):
        return
    scale = float(SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE) / float(map_size)
    start_px = start.astype(np.float64, copy=False) * scale
    end_px = end.astype(np.float64, copy=False) * scale
    radius_px = float(radius) * scale
    _source_state_mark_pixel_bbox_dirty_blocks(
        blocks,
        min_x=float(min(start_px[0], end_px[0]) - radius_px - 1.0),
        min_y=float(min(start_px[1], end_px[1]) - radius_px - 1.0),
        max_x=float(max(start_px[0], end_px[0]) + radius_px + 1.0),
        max_y=float(max(start_px[1], end_px[1]) + radius_px + 1.0),
    )


def _source_state_mark_world_circle_dirty_blocks(
    blocks: np.ndarray,
    *,
    pos: np.ndarray,
    radius: float,
    map_size: float,
) -> None:
    if not np.isfinite(pos).all() or not np.isfinite(radius) or map_size <= 0.0:
        return
    size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    radius_px = int(max(0, np.ceil((float(radius) / float(map_size)) * float(size))))
    px = int(np.clip(np.rint((float(pos[0]) / float(map_size)) * float(size - 1)), 0, size - 1))
    py = int(np.clip(np.rint((float(pos[1]) / float(map_size)) * float(size - 1)), 0, size - 1))
    _source_state_mark_pixel_bbox_dirty_blocks(
        blocks,
        min_x=float(px - radius_px),
        min_y=float(py - radius_px),
        max_x=float(px + radius_px),
        max_y=float(py + radius_px),
    )


def _source_state_mark_world_sprite_dirty_blocks(
    blocks: np.ndarray,
    *,
    pos: np.ndarray,
    radius: float,
    map_size: float,
) -> None:
    if not np.isfinite(pos).all() or not np.isfinite(radius) or radius <= 0.0 or map_size <= 0.0:
        return
    size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    scale = float(size) / float(map_size)
    dst_x = _canvas_round((float(pos[0]) - float(radius)) * scale)
    dst_y = _canvas_round((float(pos[1]) - float(radius)) * scale)
    dst_size = _canvas_round((float(radius) * 2.0) * scale)
    if dst_size <= 0:
        return
    _source_state_mark_pixel_bbox_dirty_blocks(
        blocks,
        min_x=float(dst_x),
        min_y=float(dst_y),
        max_x=float(dst_x + dst_size - 1),
        max_y=float(dst_y + dst_size - 1),
    )


def _source_state_mark_pixel_bbox_dirty_blocks(
    blocks: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> None:
    if not all(np.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
        return
    source_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
    ratio = source_size // target_size
    px0 = max(0, int(np.floor(min_x - 2.0)))
    py0 = max(0, int(np.floor(min_y - 2.0)))
    px1 = min(source_size - 1, int(np.ceil(max_x + 2.0)))
    py1 = min(source_size - 1, int(np.ceil(max_y + 2.0)))
    if px0 > px1 or py0 > py1:
        return
    blocks[py0 // ratio : (py1 // ratio) + 1, px0 // ratio : (px1 // ratio) + 1] = True


def _source_state_visual_trail_records(
    arrays: Mapping[str, np.ndarray],
    row: int,
) -> _SourceStateTrailRecords:
    capacity = arrays["visual_trail_active"].shape[1]
    cursor = int(np.clip(int(arrays["visual_trail_write_cursor"][row]), 0, capacity))
    slots = np.flatnonzero(arrays["visual_trail_active"][row, :cursor]).astype(np.int64)
    if slots.size == 0:
        return _source_state_empty_trail_records()
    positions = arrays["visual_trail_pos"][row, slots].astype(np.float64, copy=False)
    radii = arrays["visual_trail_radius"][row, slots].astype(np.float64, copy=False)
    owners = arrays["visual_trail_owner"][row, slots].astype(np.int64, copy=False)
    breaks = arrays["visual_trail_break_before"][row, slots].astype(bool, copy=False)
    finite = (
        np.isfinite(positions[:, 0])
        & np.isfinite(positions[:, 1])
        & np.isfinite(radii)
        & (radii >= 0.0)
    )
    if not bool(finite.any()):
        return _source_state_empty_trail_records()
    return _SourceStateTrailRecords(
        slots=slots[finite].astype(np.int64, copy=True),
        positions=positions[finite].astype(np.float64, copy=True),
        radii=radii[finite].astype(np.float64, copy=True),
        owners=owners[finite].astype(np.int64, copy=True),
        break_before=breaks[finite].astype(bool, copy=True),
    )


def _source_state_empty_trail_records() -> _SourceStateTrailRecords:
    return _SourceStateTrailRecords(
        slots=np.empty(0, dtype=np.int64),
        positions=np.empty((0, 2), dtype=np.float64),
        radii=np.empty(0, dtype=np.float64),
        owners=np.empty(0, dtype=np.int64),
        break_before=np.empty(0, dtype=bool),
    )


def _source_state_owner_trail_records(
    records: _SourceStateTrailRecords,
    owner: int,
) -> _SourceStateTrailRecords:
    owner_mask = records.owners == int(owner)
    slots = records.slots[owner_mask]
    order = _browser_line_body_order(slots, body_nums=None)
    return _SourceStateTrailRecords(
        slots=slots[order].astype(np.int64, copy=True),
        positions=records.positions[owner_mask][order].astype(np.float64, copy=True),
        radii=records.radii[owner_mask][order].astype(np.float64, copy=True),
        owners=records.owners[owner_mask][order].astype(np.int64, copy=True),
        break_before=records.break_before[owner_mask][order].astype(bool, copy=True),
    )


def _source_state_copy_trail_records(
    records: _SourceStateTrailRecords,
) -> _SourceStateTrailRecords:
    return _SourceStateTrailRecords(
        slots=records.slots.copy(),
        positions=records.positions.copy(),
        radii=records.radii.copy(),
        owners=records.owners.copy(),
        break_before=records.break_before.copy(),
    )


def _source_state_replace_layer_records(
    layer: _SourceStateOwnerTrailLayer,
    records: _SourceStateTrailRecords,
) -> None:
    layer.slots = records.slots.copy()
    layer.positions = records.positions.copy()
    layer.radii = records.radii.copy()
    layer.break_before = records.break_before.copy()


def _source_state_owner_prefix_matches(
    layer: _SourceStateOwnerTrailLayer,
    records: _SourceStateTrailRecords,
    old_count: int,
) -> bool:
    return bool(
        np.array_equal(records.slots[:old_count], layer.slots)
        and np.array_equal(records.positions[:old_count], layer.positions)
        and np.array_equal(records.radii[:old_count], layer.radii)
        and np.array_equal(records.break_before[:old_count], layer.break_before)
    )


def _source_state_incremental_draw_start(
    records: _SourceStateTrailRecords,
    old_count: int,
) -> int:
    if old_count <= 0:
        return 0
    if old_count >= records.slots.size:
        return old_count
    first_new = old_count
    connects_to_previous = (
        not bool(records.break_before[first_new])
        and np.isclose(
            records.radii[first_new],
            records.radii[first_new - 1],
            rtol=0.0,
            atol=1e-9,
        )
    )
    return first_new - 1 if connects_to_previous else first_new


def _source_state_refresh_layer_mask(
    layer: _SourceStateOwnerTrailLayer,
    *,
    sentinel: np.ndarray,
) -> None:
    layer.mask[:, :] = np.any(layer.rgb != sentinel, axis=2)


def _source_state_refresh_layer_mask_for_points(
    layer: _SourceStateOwnerTrailLayer,
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    map_size: float,
    sentinel: np.ndarray,
) -> None:
    if positions.size == 0 or radii.size == 0 or map_size <= 0.0:
        return
    finite = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 1]) & np.isfinite(radii)
    if not bool(finite.any()):
        return
    size = int(layer.rgb.shape[0])
    scale = float(size) / float(map_size)
    xy = positions[finite].astype(np.float64, copy=False) * scale
    radius_px = float(np.max(np.maximum(radii[finite], 0.0))) * scale
    x0 = max(0, int(np.floor(float(np.min(xy[:, 0])) - radius_px - 2.0)))
    x1 = min(size - 1, int(np.ceil(float(np.max(xy[:, 0])) + radius_px + 2.0)))
    y0 = max(0, int(np.floor(float(np.min(xy[:, 1])) - radius_px - 2.0)))
    y1 = min(size - 1, int(np.ceil(float(np.max(xy[:, 1])) + radius_px + 2.0)))
    if x0 > x1 or y0 > y1:
        return
    layer.mask[y0 : y1 + 1, x0 : x1 + 1] = np.any(
        layer.rgb[y0 : y1 + 1, x0 : x1 + 1] != sentinel,
        axis=2,
    )


def _source_state_colors_key(colors: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(int(channel) for channel in color) for color in colors)


def _validated_player_perspective_frames(
    out: np.ndarray,
    *,
    player_count: int,
) -> np.ndarray:
    array = np.asarray(out)
    expected = (int(player_count), *SOURCE_STATE_CANVAS_GRAY64_SHAPE)
    if array.shape != expected:
        raise VectorVisualObservationError(f"out must have shape {expected}, got {array.shape}")
    if array.dtype != np.uint8:
        raise VectorVisualObservationError(f"out dtype must be uint8, got {array.dtype}")
    return array


def _remap_source_gray64_to_player_perspective(
    frame: np.ndarray,
    *,
    controlled_player: int,
    player_count: int,
) -> None:
    canvas = _validated_frame(frame, name="frame")[0]
    source = canvas.copy()
    controlled = int(controlled_player)
    for player in range(int(player_count)):
        body_value = _body_value(player)
        head_value = _head_value(player)
        target_body = _SELF_BODY_VALUE if player == controlled else _OTHER_BODY_VALUE
        canvas[source == body_value] = np.uint8(target_body)
        canvas[source == head_value] = np.uint8(target_body)


def _validated_dirty_block_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    expected = (SOURCE_STATE_CANVAS_GRAY64_SHAPE[1], SOURCE_STATE_CANVAS_GRAY64_SHAPE[2])
    if array.shape != expected:
        raise VectorVisualObservationError(f"dirty_blocks must have shape {expected}, got {array.shape}")
    if array.dtype == np.bool_:
        return array
    return array.astype(bool, copy=False)


def _remap_rgb_palette_pixels(
    target: np.ndarray,
    *,
    source: np.ndarray,
    from_colors: np.ndarray,
    to_colors: np.ndarray,
) -> None:
    if from_colors.shape != to_colors.shape:
        raise VectorVisualObservationError(
            f"palette shapes differ: {from_colors.shape} vs {to_colors.shape}"
        )
    for from_rgb, to_rgb in zip(from_colors, to_colors, strict=True):
        from_triplet = _rgb_triplet(from_rgb)
        to_triplet = _rgb_triplet(to_rgb)
        if bool(np.array_equal(from_triplet, to_triplet)):
            continue
        mask = np.all(source == from_triplet, axis=2)
        if bool(mask.any()):
            target[mask] = to_triplet


def _rgb_palette_reuse_safe(
    colors: np.ndarray,
    *,
    background_rgb: Sequence[int],
    extra_reserved_colors: Sequence[Sequence[int]] = (),
) -> bool:
    triplets = [_rgb_triplet(color) for color in colors]
    reserved = [_rgb_triplet(background_rgb), *[_rgb_triplet(color) for color in extra_reserved_colors]]
    seen: set[tuple[int, int, int]] = set()
    for triplet in triplets:
        key = tuple(int(value) for value in triplet)
        if key in seen:
            return False
        seen.add(key)
        for reserved_triplet in reserved:
            if bool(np.array_equal(triplet, reserved_triplet)):
                return False
    return True


def _draw_source_state_rgb_bonuses(
    frame: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
    map_size: float,
    *,
    bonus_render_mode: str,
    dirty_blocks: np.ndarray | None = None,
) -> None:
    if "bonus_active" not in arrays:
        return
    bonus_slots = np.flatnonzero(arrays["bonus_active"][row])
    if dirty_blocks is not None and bonus_slots.size:
        dirty_mask = _validated_dirty_block_mask(dirty_blocks)
        bonus_slots = np.asarray(
            [
                slot
                for slot in bonus_slots
                if _source_state_bonus_intersects_dirty_blocks(
                    arrays["bonus_pos"][row, slot],
                    float(arrays["bonus_radius"][row, slot]),
                    map_size=map_size,
                    bonus_render_mode=bonus_render_mode,
                    dirty_blocks=dirty_mask,
                )
            ],
            dtype=np.int64,
        )
    if bonus_slots.size == 0:
        return
    bonus_type = arrays.get("bonus_type")
    _draw_bonuses_rgb(
        frame,
        arrays["bonus_pos"][row, bonus_slots],
        arrays["bonus_radius"][row, bonus_slots],
        map_size,
        bonus_types=None if bonus_type is None else bonus_type[row, bonus_slots],
        bonus_render_mode=bonus_render_mode,
    )


def _draw_source_state_rgb_heads(
    frame: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
    map_size: float,
    *,
    colors: np.ndarray,
) -> None:
    player_count = arrays["pos"].shape[1]
    for player in range(player_count):
        if not bool(arrays["present"][row, player]):
            continue
        if not bool(arrays["alive"][row, player]):
            continue
        _draw_world_circle_rgb(
            frame,
            arrays["pos"][row, player, 0],
            arrays["pos"][row, player, 1],
            arrays["radius"][row, player],
            map_size,
            color=colors[player],
        )


def _source_avatar_rgb_values(
    avatars: Sequence[Mapping[str, Any]],
    *,
    player_rgb: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    raw_colors = (
        SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB
        if player_rgb is None
        else tuple(tuple(color) for color in player_rgb)
    )
    colors = np.zeros((len(avatars), 3), dtype=np.uint8)
    for player, avatar in enumerate(avatars):
        color = str(avatar.get("color", ""))
        if color:
            colors[player] = _hex_color_to_rgb(color, fallback=raw_colors[player % len(raw_colors)])
        else:
            colors[player] = _rgb_triplet(raw_colors[player % len(raw_colors)])
    return colors


def _hex_color_to_rgb(value: str, *, fallback: Sequence[int]) -> np.ndarray:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        return _rgb_triplet(fallback)
    try:
        return _rgb_triplet((int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)))
    except ValueError:
        return _rgb_triplet(fallback)


def _fill_player_status_planes(
    output: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
    *,
    player: int,
    start_channel: int,
) -> None:
    output[start_channel + 0].fill(
        _ratio_plane_value(arrays, "radius", "base_radius", row=row, player=player, scale=4.0)
    )
    output[start_channel + 1].fill(
        _ratio_plane_value(arrays, "speed", "base_speed", row=row, player=player, scale=2.0)
    )
    output[start_channel + 2].fill(_bool_player_value(arrays, "inverse", row, player))
    output[start_channel + 3].fill(_turn_override_value(arrays, row=row, player=player))
    output[start_channel + 4].fill(_bool_player_value(arrays, "invincible", row, player))
    output[start_channel + 5].fill(_bool_player_value(arrays, "printing", row, player))
    output[start_channel + 6].fill(_player_bonus_ttl_value(arrays, row=row, player=player))


def _ratio_plane_value(
    arrays: Mapping[str, np.ndarray],
    value_name: str,
    base_name: str,
    *,
    row: int,
    player: int,
    scale: float,
) -> float:
    value_array = arrays.get(value_name)
    if value_array is None:
        return 0.0
    value = float(value_array[row, player])
    base_array = arrays.get(base_name)
    if base_array is None:
        return 0.0
    base = float(base_array[row, player])
    if not np.isfinite(value) or not np.isfinite(base) or base <= 0.0:
        return 0.0
    return float(np.clip(value / base, 0.0, scale) / scale)


def _bool_player_value(
    arrays: Mapping[str, np.ndarray],
    name: str,
    row: int,
    player: int,
) -> float:
    array = arrays.get(name)
    if array is None:
        return 0.0
    return 1.0 if bool(array[row, player]) else 0.0


def _turn_override_value(
    arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
) -> float:
    value_array = arrays.get("angular_velocity_per_ms")
    base_array = arrays.get("base_angular_velocity_per_ms")
    if value_array is None or base_array is None:
        return 0.0
    return (
        1.0
        if not np.isclose(float(value_array[row, player]), float(base_array[row, player]))
        else 0.0
    )


def _player_bonus_ttl_value(
    arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
) -> float:
    count_array = arrays.get("bonus_stack_count")
    duration_array = arrays.get("bonus_stack_duration_ms")
    if count_array is None or duration_array is None:
        return 0.0
    count = int(count_array[row, player])
    if count <= 0:
        return 0.0
    durations = np.asarray(duration_array[row, player, :count], dtype=np.float64)
    if durations.size == 0:
        return 0.0
    return float(
        np.clip(np.nanmax(durations), 0.0, SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS)
        / SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS
    )


def _game_borderless_value(state: Mapping[str, np.ndarray], row: int) -> float:
    stack_count = state.get("bonus_game_stack_count")
    stack_borderless = state.get("bonus_game_stack_borderless")
    if stack_count is not None and stack_borderless is not None:
        count = int(np.asarray(stack_count)[row])
        if count > 0 and bool(np.asarray(stack_borderless)[row, :count].any()):
            return 1.0
    borderless = state.get("borderless")
    if borderless is None:
        return 0.0
    return 1.0 if bool(np.asarray(borderless)[row]) else 0.0


def _game_bonus_ttl_value(state: Mapping[str, np.ndarray], row: int) -> float:
    count_array = state.get("bonus_game_stack_count")
    duration_array = state.get("bonus_game_stack_duration_ms")
    if count_array is None or duration_array is None:
        return 0.0
    count = int(np.asarray(count_array)[row])
    if count <= 0:
        return 0.0
    durations = np.asarray(duration_array)[row, :count].astype(np.float64, copy=False)
    if durations.size == 0:
        return 0.0
    return float(
        np.clip(np.nanmax(durations), 0.0, SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS)
        / SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS
    )


def _first_other_player(controlled: int, player_count: int) -> int:
    for player in range(player_count):
        if player != controlled:
            return player
    return controlled


def _validated_trail_render_mode(value: str) -> str:
    mode = str(value)
    if mode not in TRAIL_RENDER_MODES:
        supported = ", ".join(TRAIL_RENDER_MODE_ORDER)
        raise VectorVisualObservationError(
            f"trail_render_mode must be one of [{supported}], got {value!r}"
        )
    return mode


def _validated_bonus_render_mode(value: str) -> str:
    mode = str(value)
    if mode not in BONUS_RENDER_MODES:
        supported = ", ".join(BONUS_RENDER_MODE_ORDER)
        raise VectorVisualObservationError(
            f"bonus_render_mode must be one of [{supported}], got {value!r}"
        )
    return mode


def _trail_renderer_kind(mode: str) -> str:
    if mode == TRAIL_RENDER_MODE_BROWSER_LINES:
        return "connected_rounded_lines"
    return "circle_per_body"


def _trail_renderer_truth_level(mode: str) -> str:
    if mode == TRAIL_RENDER_MODE_BROWSER_LINES:
        return "source_state_browser_style_lines_non_pixel_parity"
    return "source_state_fast_body_circle_approximation"


def _trail_renderer_is_approximation(mode: str) -> bool:
    return mode == TRAIL_RENDER_MODE_BODY_CIRCLES_FAST


def _rgb_renderer_impl_id(mode: str) -> str:
    if mode == TRAIL_RENDER_MODE_BROWSER_LINES:
        return SOURCE_STATE_RGB_BROWSER_LINES_RENDERER_IMPL_ID
    return SOURCE_STATE_RGB_BODY_CIRCLES_FAST_RENDERER_IMPL_ID


def _canvas_gray64_renderer_impl_id(mode: str) -> str:
    if mode == TRAIL_RENDER_MODE_BROWSER_LINES:
        return SOURCE_STATE_CANVAS_GRAY64_BROWSER_LINES_RENDERER_IMPL_ID
    return SOURCE_STATE_CANVAS_GRAY64_BODY_CIRCLES_FAST_RENDERER_IMPL_ID


def _body_slot_limit(arrays: Mapping[str, np.ndarray], row: int) -> int:
    capacity = arrays["body_active"].shape[1]
    cursor = int(arrays["body_write_cursor"][row])
    return int(np.clip(cursor, 0, capacity))


def _visual_trail_slot_limit(arrays: Mapping[str, np.ndarray], row: int) -> int:
    capacity = arrays["visual_trail_active"].shape[1]
    cursor = int(arrays["visual_trail_write_cursor"][row])
    return int(np.clip(cursor, 0, capacity))


def _body_value(owner: Any) -> int:
    owner_index = int(owner)
    if owner_index < 0:
        return 80
    return int(min(192, 96 + owner_index * 32))


def _head_value(player: int) -> int:
    return int(min(255, 224 + int(player) * 8))


def _render_source_state_rgb_body_circles_fast(
    frame: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
    map_size: float,
    *,
    colors: np.ndarray,
) -> None:
    mask_cache: dict[int, np.ndarray] = {}
    body_limit = _body_slot_limit(arrays, row)
    active_slots = np.flatnonzero(arrays["body_active"][row, :body_limit])
    _draw_body_circles_rgb(
        frame,
        arrays["body_pos"][row, active_slots],
        arrays["body_radius"][row, active_slots],
        arrays["body_owner"][row, active_slots],
        map_size,
        colors=colors,
        mask_cache=mask_cache,
    )


def _render_source_state_rgb_browser_lines(
    frame: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
    map_size: float,
    *,
    colors: np.ndarray,
) -> None:
    if "visual_trail_active" in arrays:
        trail_limit = _visual_trail_slot_limit(arrays, row)
        active_slots = np.flatnonzero(arrays["visual_trail_active"][row, :trail_limit])
        if active_slots.size:
            _draw_browser_line_trails_rgb(
                frame,
                arrays["visual_trail_pos"][row, active_slots],
                arrays["visual_trail_radius"][row, active_slots],
                arrays["visual_trail_owner"][row, active_slots],
                map_size,
                colors=colors,
                slots=active_slots,
                body_nums=None,
                break_before=arrays["visual_trail_break_before"][row, active_slots],
            )
            return

    body_limit = _body_slot_limit(arrays, row)
    active_slots = np.flatnonzero(arrays["body_active"][row, :body_limit])
    body_num = arrays.get("body_num")
    body_break_before = arrays.get("body_break_before")
    _draw_browser_line_trails_rgb(
        frame,
        arrays["body_pos"][row, active_slots],
        arrays["body_radius"][row, active_slots],
        arrays["body_owner"][row, active_slots],
        map_size,
        colors=colors,
        slots=active_slots,
        body_nums=None if body_num is None else body_num[row, active_slots],
        break_before=(None if body_break_before is None else body_break_before[row, active_slots]),
    )


def _render_source_snapshot_rgb_trails(
    frame: np.ndarray,
    body_records: Sequence[Mapping[str, Any]],
    visual_trail_records: Sequence[Mapping[str, Any]] | None,
    player_index_by_avatar_id: Mapping[int, int],
    map_size: float,
    *,
    colors: np.ndarray,
    default_avatar_radius: float,
    trail_render_mode: str,
) -> None:
    if (
        trail_render_mode == TRAIL_RENDER_MODE_BROWSER_LINES
        and visual_trail_records is not None
        and len(visual_trail_records) > 0
    ):
        _render_source_snapshot_visual_trail_records(
            frame,
            visual_trail_records,
            player_index_by_avatar_id,
            map_size,
            colors=colors,
            default_avatar_radius=default_avatar_radius,
        )
        return

    if not body_records:
        return
    body_positions = np.asarray(
        [[float(body["x"]), float(body["y"])] for body in body_records],
        dtype=np.float64,
    )
    body_radii = np.asarray(
        [float(body.get("radius", default_avatar_radius)) for body in body_records],
        dtype=np.float64,
    )
    body_owners = np.asarray(
        [
            _source_body_owner_index(body.get("avatarId"), player_index_by_avatar_id)
            for body in body_records
        ],
        dtype=np.int16,
    )
    if trail_render_mode == TRAIL_RENDER_MODE_BODY_CIRCLES_FAST:
        _draw_body_circles_rgb(
            frame,
            body_positions,
            body_radii,
            body_owners,
            map_size,
            colors=colors,
        )
        return
    body_nums = _source_body_nums(body_records)
    break_before = _source_body_break_before(body_records)
    _draw_browser_line_trails_rgb(
        frame,
        body_positions,
        body_radii,
        body_owners,
        map_size,
        colors=colors,
        slots=np.arange(len(body_records), dtype=np.int32),
        body_nums=body_nums,
        break_before=break_before,
    )


def _render_source_snapshot_visual_trail_records(
    frame: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    player_index_by_avatar_id: Mapping[int, int],
    map_size: float,
    *,
    colors: np.ndarray,
    default_avatar_radius: float,
) -> None:
    if not records:
        return
    positions = np.asarray(
        [[float(point["x"]), float(point["y"])] for point in records],
        dtype=np.float64,
    )
    radii = np.asarray(
        [float(point.get("radius", default_avatar_radius)) for point in records],
        dtype=np.float64,
    )
    owners = np.asarray(
        [
            _source_body_owner_index(point.get("avatarId"), player_index_by_avatar_id)
            for point in records
        ],
        dtype=np.int16,
    )
    break_before = _source_body_break_before(records)
    _draw_browser_line_trails_rgb(
        frame,
        positions,
        radii,
        owners,
        map_size,
        colors=colors,
        slots=np.arange(len(records), dtype=np.int32),
        body_nums=None,
        break_before=break_before,
    )


def _source_body_nums(body_records: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    values: list[int] = []
    for body in body_records:
        if "num" not in body:
            return None
        try:
            values.append(int(body["num"]))
        except (TypeError, ValueError):
            return None
    return np.asarray(values, dtype=np.int32)


def _source_body_break_before(
    body_records: Sequence[Mapping[str, Any]],
) -> np.ndarray | None:
    values: list[bool] = []
    for body in body_records:
        if "breakBefore" in body:
            values.append(bool(body["breakBefore"]))
        elif "body_break_before" in body:
            values.append(bool(body["body_break_before"]))
        else:
            return None
    return np.asarray(values, dtype=bool)


def _source_bonus_type_codes(
    bonus_records: Sequence[Mapping[str, Any]],
) -> np.ndarray | None:
    values: list[int] = []
    for bonus in bonus_records:
        bonus_type = bonus.get("type")
        if bonus_type is None:
            return None
        try:
            values.append(int(bonus_type))
            continue
        except (TypeError, ValueError):
            pass
        code = SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_CODE_BY_NAME.get(str(bonus_type))
        if code is None:
            return None
        values.append(code)
    return np.asarray(values, dtype=np.int16)


def _draw_browser_line_trails_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    owners: np.ndarray,
    map_size: float,
    *,
    colors: np.ndarray,
    slots: np.ndarray,
    body_nums: np.ndarray | None,
    break_before: np.ndarray | None,
) -> None:
    if positions.size == 0:
        return
    x = positions[:, 0].astype(np.float64, copy=False)
    y = positions[:, 1].astype(np.float64, copy=False)
    radius = radii.astype(np.float64, copy=False)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(radius) & (radius >= 0.0)
    if not bool(finite.any()):
        return
    positions = positions[finite].astype(np.float64, copy=False)
    radii = radius[finite]
    owners = owners[finite].astype(np.int64, copy=False)
    slots = slots[finite].astype(np.int64, copy=False)
    if body_nums is not None:
        body_nums = body_nums[finite].astype(np.int64, copy=False)
    if break_before is not None:
        break_before = break_before[finite].astype(bool, copy=False)

    for owner in _browser_line_owner_draw_order(owners):
        owner_mask = owners == owner
        owner_positions = positions[owner_mask]
        owner_radii = radii[owner_mask]
        owner_slots = slots[owner_mask]
        owner_body_nums = None if body_nums is None else body_nums[owner_mask]
        order = _browser_line_body_order(owner_slots, owner_body_nums)
        color = _body_owner_rgb(owner, colors)
        ordered_breaks = None if break_before is None else break_before[owner_mask][order]
        _draw_ordered_browser_line_path_rgb(
            canvas,
            owner_positions[order],
            owner_radii[order],
            map_size,
            color=color,
            break_before=ordered_breaks,
        )


def _browser_line_owner_draw_order(owners: np.ndarray) -> tuple[int, ...]:
    unique = tuple(int(owner) for owner in np.unique(owners))
    invalid = tuple(sorted(owner for owner in unique if owner < 0))
    players = tuple(sorted((owner for owner in unique if owner >= 0), reverse=True))
    return (*invalid, *players)


def _browser_line_body_order(slots: np.ndarray, body_nums: np.ndarray | None) -> np.ndarray:
    if body_nums is None:
        return np.argsort(slots, kind="stable")
    return np.lexsort((slots, body_nums))


def _body_owner_rgb(owner: int, colors: np.ndarray) -> np.ndarray | tuple[int, int, int]:
    return (120, 120, 120) if owner < 0 else colors[owner % len(colors)]


def _draw_ordered_browser_line_path_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_size: float,
    *,
    color: Sequence[int] | np.ndarray,
    break_before: np.ndarray | None,
) -> None:
    if positions.size == 0:
        return
    start = 0
    for index in range(1, positions.shape[0]):
        radius_changed = not np.isclose(radii[index], radii[start], rtol=0.0, atol=1e-9)
        segment_break = break_before is not None and bool(break_before[index])
        if segment_break or radius_changed:
            _draw_rounded_world_polyline_rgb(
                canvas,
                positions[start:index],
                float(radii[start]),
                map_size,
                color=color,
            )
            start = index
    _draw_rounded_world_polyline_rgb(
        canvas,
        positions[start:],
        float(radii[start]),
        map_size,
        color=color,
    )


def _draw_rounded_world_polyline_rgb(
    canvas: np.ndarray,
    points: np.ndarray,
    radius: float,
    map_size: float,
    *,
    color: Sequence[int] | np.ndarray,
) -> None:
    if points.size == 0 or not np.isfinite(radius) or radius < 0.0:
        return
    size = int(canvas.shape[0])
    scale = float(size) / float(map_size)
    xy = points.astype(np.float64, copy=False) * scale
    radius_px = float(radius) * scale
    rgb = _rgb_triplet(color)
    if xy.shape[0] == 1:
        _draw_rounded_pixel_cap_rgb(canvas, xy[0, 0], xy[0, 1], radius_px, rgb)
        return
    for start, end in zip(xy[:-1], xy[1:], strict=True):
        _draw_rounded_pixel_segment_rgb(
            canvas,
            float(start[0]),
            float(start[1]),
            float(end[0]),
            float(end[1]),
            radius_px,
            rgb,
        )


def _draw_rounded_pixel_cap_rgb(
    canvas: np.ndarray,
    x: float,
    y: float,
    radius_px: float,
    rgb: np.ndarray,
) -> None:
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(radius_px):
        return
    size = int(canvas.shape[0])
    if x + radius_px < 0.0 or x - radius_px > float(size - 1):
        return
    if y + radius_px < 0.0 or y - radius_px > float(size - 1):
        return
    x0 = max(0, int(np.floor(x - radius_px - 1.0)))
    x1 = min(size - 1, int(np.ceil(x + radius_px + 1.0)))
    y0 = max(0, int(np.floor(y - radius_px - 1.0)))
    y1 = min(size - 1, int(np.ceil(y + radius_px + 1.0)))
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx.astype(np.float64) - x) ** 2 + (yy.astype(np.float64) - y) ** 2 <= (
        radius_px * radius_px
    )
    if bool(mask.any()):
        canvas[y0 : y1 + 1, x0 : x1 + 1][mask] = rgb


def _draw_rounded_pixel_segment_rgb(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius_px: float,
    rgb: np.ndarray,
) -> None:
    if not all(np.isfinite(value) for value in (x0, y0, x1, y1, radius_px)):
        return
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        _draw_rounded_pixel_cap_rgb(canvas, x0, y0, radius_px, rgb)
        return
    size = int(canvas.shape[0])
    min_x = min(x0, x1) - radius_px
    max_x = max(x0, x1) + radius_px
    min_y = min(y0, y1) - radius_px
    max_y = max(y0, y1) + radius_px
    if max_x < 0.0 or min_x > float(size - 1) or max_y < 0.0 or min_y > float(size - 1):
        return
    px0 = max(0, int(np.floor(min_x - 1.0)))
    px1 = min(size - 1, int(np.ceil(max_x + 1.0)))
    py0 = max(0, int(np.floor(min_y - 1.0)))
    py1 = min(size - 1, int(np.ceil(max_y + 1.0)))
    yy, xx = np.ogrid[py0 : py1 + 1, px0 : px1 + 1]
    rel_x = xx.astype(np.float64) - x0
    rel_y = yy.astype(np.float64) - y0
    projection = np.clip((rel_x * dx + rel_y * dy) / length_sq, 0.0, 1.0)
    closest_x = x0 + projection * dx
    closest_y = y0 + projection * dy
    distance_sq = (xx.astype(np.float64) - closest_x) ** 2 + (
        yy.astype(np.float64) - closest_y
    ) ** 2
    mask = distance_sq <= radius_px * radius_px
    if bool(mask.any()):
        canvas[py0 : py1 + 1, px0 : px1 + 1][mask] = rgb


def _draw_world_circle(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    radius_value: Any,
    map_size: float,
    *,
    value: int,
    mask_cache: dict[int, np.ndarray] | None = None,
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
    mask = _circle_mask(radius_px, mask_cache)
    mask_x0 = x0 - (px - radius_px)
    mask_y0 = y0 - (py - radius_px)
    mask_view = mask[
        mask_y0 : mask_y0 + (y1 - y0 + 1),
        mask_x0 : mask_x0 + (x1 - x0 + 1),
    ]
    view = canvas[y0 : y1 + 1, x0 : x1 + 1]
    np.maximum(view, np.uint8(value), out=view, where=mask_view)


def _draw_body_circles(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    owners: np.ndarray,
    map_size: float,
    *,
    mask_cache: dict[int, np.ndarray] | None,
) -> None:
    if positions.size == 0:
        return
    x = positions[:, 0].astype(np.float64, copy=False)
    y = positions[:, 1].astype(np.float64, copy=False)
    radius = radii.astype(np.float64, copy=False)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(radius)
    intersects = (
        (x + radius > 0.0) & (x - radius < map_size) & (y + radius > 0.0) & (y - radius < map_size)
    )
    finite &= intersects
    if not bool(finite.any()):
        return
    x = x[finite]
    y = y[finite]
    radius = radius[finite]
    owner_values = _body_values(owners[finite])
    radius_px = np.maximum(0, np.ceil((radius / map_size) * 64.0)).astype(np.int32)
    px = np.clip(np.rint((x / map_size) * 63.0), 0, 63).astype(np.int16)
    py = np.clip(np.rint((y / map_size) * 63.0), 0, 63).astype(np.int16)

    zero_mask = radius_px == 0
    if bool(zero_mask.any()):
        np.maximum.at(canvas, (py[zero_mask], px[zero_mask]), owner_values[zero_mask])

    one_mask = radius_px == 1
    if bool(one_mask.any()):
        _draw_radius_one_circles(canvas, px[one_mask], py[one_mask], owner_values[one_mask])

    other_radii = np.flatnonzero(~zero_mask & ~one_mask)
    for index in other_radii:
        _draw_world_circle(
            canvas,
            x[index],
            y[index],
            radius[index],
            map_size,
            value=int(owner_values[index]),
            mask_cache=mask_cache,
        )


def _draw_uniform_circles(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_size: float,
    *,
    value: int,
    mask_cache: dict[int, np.ndarray] | None,
) -> None:
    for position, radius in zip(positions, radii, strict=True):
        _draw_world_circle(
            canvas,
            position[0],
            position[1],
            radius,
            map_size,
            value=value,
            mask_cache=mask_cache,
        )


def _draw_bonus_type_luma_circles(
    canvas: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
    map_size: float,
    *,
    mask_cache: dict[int, np.ndarray] | None,
) -> None:
    bonus_slots = np.flatnonzero(arrays["bonus_active"][row])
    bonus_types = arrays.get("bonus_type")
    for slot in bonus_slots:
        bonus_type = None if bonus_types is None else int(bonus_types[row, slot])
        _draw_world_circle(
            canvas,
            arrays["bonus_pos"][row, slot, 0],
            arrays["bonus_pos"][row, slot, 1],
            arrays["bonus_radius"][row, slot],
            map_size,
            value=_rgb_luma_uint8(_bonus_type_rgb(bonus_type)),
            mask_cache=mask_cache,
        )


def _draw_bonus_type_simple_symbols(
    canvas: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    row: int,
    map_size: float,
) -> None:
    bonus_slots = np.flatnonzero(arrays["bonus_active"][row])
    bonus_types = arrays.get("bonus_type")
    for slot in bonus_slots:
        bonus_type = None if bonus_types is None else int(bonus_types[row, slot])
        _draw_world_simple_bonus_symbol(
            canvas,
            arrays["bonus_pos"][row, slot, 0],
            arrays["bonus_pos"][row, slot, 1],
            arrays["bonus_radius"][row, slot],
            map_size,
            bonus_type=bonus_type,
        )


def _draw_world_simple_bonus_symbol(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    radius_value: Any,
    map_size: float,
    *,
    bonus_type: int | None,
) -> None:
    x = float(x_value)
    y = float(y_value)
    radius = float(radius_value)
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(radius):
        return
    if radius <= 0.0:
        return
    if x + radius <= 0.0 or x - radius >= map_size:
        return
    if y + radius <= 0.0 or y - radius >= map_size:
        return

    size = int(canvas.shape[0])
    radius_px = int(max(3, np.ceil((radius / map_size) * float(size))))
    px = int(np.clip(np.rint((x / map_size) * float(size - 1)), 0, size - 1))
    py = int(np.clip(np.rint((y / map_size) * float(size - 1)), 0, size - 1))
    dst_size = int(radius_px * 2 + 1)
    stamp = _simple_bonus_symbol_stamp(bonus_type, dst_size)

    x0 = max(0, px - radius_px)
    x1 = min(size, px + radius_px + 1)
    y0 = max(0, py - radius_px)
    y1 = min(size, py + radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        return

    stamp_x0 = x0 - (px - radius_px)
    stamp_y0 = y0 - (py - radius_px)
    patch = stamp[
        stamp_y0 : stamp_y0 + (y1 - y0),
        stamp_x0 : stamp_x0 + (x1 - x0),
    ]
    mask = patch > 0
    if bool(mask.any()):
        canvas[y0:y1, x0:x1][mask] = patch[mask]


@lru_cache(maxsize=512)
def _simple_bonus_symbol_stamp(bonus_type: int | None, dst_size: int) -> np.ndarray:
    size = int(dst_size)
    if size <= 0:
        raise VectorVisualObservationError("simple bonus symbol stamp size must be positive")
    base = _simple_bonus_symbol_base(bonus_type)
    if size == BONUS_SYMBOL_BASE_SIZE:
        return base.copy()
    indices = np.rint(
        np.linspace(0, BONUS_SYMBOL_BASE_SIZE - 1, size, dtype=np.float64)
    ).astype(np.int16)
    return base[indices][:, indices].copy()


@lru_cache(maxsize=16)
def _simple_bonus_symbol_base(bonus_type: int | None) -> np.ndarray:
    code = _bonus_type_code_or_default(bonus_type)
    symbol_index = code - 1
    outer_index = symbol_index // 4
    inner_index = symbol_index % 4

    size = BONUS_SYMBOL_BASE_SIZE
    center = size // 2
    yy, xx = np.ogrid[:size, :size]
    dx = np.abs(xx - center)
    dy = np.abs(yy - center)
    if outer_index == 0:
        outer_mask = (dx * dx + dy * dy) <= center * center
    elif outer_index == 1:
        outer_mask = (dx + dy) <= center + 1
    else:
        outer_mask = np.ones((size, size), dtype=bool)

    stamp = np.zeros((size, size), dtype=np.uint8)
    stamp[outer_mask] = np.uint8(BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE[outer_index])
    inner = np.zeros((size, size), dtype=bool)
    if inner_index == 0:
        inner[center, 1 : size - 1] = True
        inner[1 : size - 1, center] = True
    elif inner_index == 1:
        for offset in range(1, size - 1):
            inner[offset, offset] = True
            inner[offset, size - 1 - offset] = True
    elif inner_index == 2:
        inner[center - 1 : center + 2, 1 : size - 1] = True
    else:
        inner[1 : size - 1, center - 1 : center + 2] = True
    stamp[outer_mask & inner] = np.uint8(BONUS_SYMBOL_INNER_LUMA)
    return stamp


def _bonus_type_code_or_default(bonus_type: int | None) -> int:
    if bonus_type is None:
        return 1
    try:
        code = int(bonus_type)
    except (TypeError, ValueError):
        return 1
    if code < 1 or code >= len(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_NAME_BY_CODE):
        return 1
    return code


def _draw_world_circle_rgb(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    radius_value: Any,
    map_size: float,
    *,
    color: Sequence[int] | np.ndarray,
    mask_cache: dict[int, np.ndarray] | None = None,
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

    size = int(canvas.shape[0])
    radius_px = int(max(0, np.ceil((radius / map_size) * float(size))))
    px = int(np.clip(np.rint((x / map_size) * float(size - 1)), 0, size - 1))
    py = int(np.clip(np.rint((y / map_size) * float(size - 1)), 0, size - 1))
    rgb = _rgb_triplet(color)
    if radius_px == 0:
        canvas[py, px] = rgb
        return

    x0 = max(0, px - radius_px)
    x1 = min(size - 1, px + radius_px)
    y0 = max(0, py - radius_px)
    y1 = min(size - 1, py + radius_px)
    mask = _circle_mask(radius_px, mask_cache)
    mask_x0 = x0 - (px - radius_px)
    mask_y0 = y0 - (py - radius_px)
    mask_view = mask[
        mask_y0 : mask_y0 + (y1 - y0 + 1),
        mask_x0 : mask_x0 + (x1 - x0 + 1),
    ]
    view = canvas[y0 : y1 + 1, x0 : x1 + 1]
    view[mask_view] = rgb


def _draw_body_circles_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    owners: np.ndarray,
    map_size: float,
    *,
    colors: np.ndarray,
    mask_cache: dict[int, np.ndarray] | None = None,
) -> None:
    for position, radius, owner in zip(positions, radii, owners, strict=True):
        owner_index = int(owner)
        color = (120, 120, 120) if owner_index < 0 else colors[owner_index % len(colors)]
        _draw_world_circle_rgb(
            canvas,
            position[0],
            position[1],
            radius,
            map_size,
            color=color,
            mask_cache=mask_cache,
        )


def _draw_bonuses_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_size: float,
    *,
    bonus_types: np.ndarray | None,
    bonus_render_mode: str,
) -> None:
    if bonus_render_mode == BONUS_RENDER_MODE_CIRCLES_FAST:
        _draw_bonus_circles_rgb(
            canvas,
            positions,
            radii,
            map_size,
            bonus_types=bonus_types,
        )
        return
    if bonus_render_mode == BONUS_RENDER_MODE_SIMPLE_SYMBOLS:
        _draw_bonus_simple_symbols_rgb(
            canvas,
            positions,
            radii,
            map_size,
            bonus_types=bonus_types,
        )
        return
    _draw_bonus_sprites_rgb(
        canvas,
        positions,
        radii,
        map_size,
        bonus_types=bonus_types,
    )


def _draw_bonus_simple_symbols_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_size: float,
    *,
    bonus_types: np.ndarray | None = None,
) -> None:
    if positions.size == 0:
        return
    if bonus_types is None:
        bonus_types_iter = (None for _ in range(len(positions)))
    else:
        bonus_types_iter = (int(value) for value in bonus_types)
    scratch = np.zeros(canvas.shape[:2], dtype=np.uint8)
    for position, radius, bonus_type in zip(positions, radii, bonus_types_iter, strict=True):
        scratch.fill(0)
        _draw_world_simple_bonus_symbol(
            scratch,
            position[0],
            position[1],
            radius,
            map_size,
            bonus_type=bonus_type,
        )
        mask = scratch > 0
        if bool(mask.any()):
            canvas[mask] = scratch[mask][:, None]


def _draw_bonus_sprites_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_size: float,
    *,
    bonus_types: np.ndarray | None = None,
) -> None:
    if positions.size == 0:
        return
    if bonus_types is None:
        bonus_types_iter = (None for _ in range(len(positions)))
    else:
        bonus_types_iter = (int(value) for value in bonus_types)
    for position, radius, bonus_type in zip(positions, radii, bonus_types_iter, strict=True):
        sprite_index = _bonus_sprite_index(bonus_type)
        if sprite_index is None:
            _draw_bonus_circles_rgb(
                canvas,
                np.asarray([position], dtype=np.float64),
                np.asarray([radius], dtype=np.float64),
                map_size,
                bonus_types=None,
            )
            continue
        _draw_world_sprite_rgb(
            canvas,
            position[0],
            position[1],
            radius,
            map_size,
            sprite_index=sprite_index,
        )


def _draw_world_sprite_rgb(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    radius_value: Any,
    map_size: float,
    *,
    sprite_index: int,
) -> None:
    x = float(x_value)
    y = float(y_value)
    radius = float(radius_value)
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(radius):
        return
    if radius <= 0.0:
        return
    if x + radius <= 0.0 or x - radius >= map_size:
        return
    if y + radius <= 0.0 or y - radius >= map_size:
        return

    size = int(canvas.shape[0])
    scale = float(size) / float(map_size)
    dst_x = _canvas_round((x - radius) * scale)
    dst_y = _canvas_round((y - radius) * scale)
    dst_size = _canvas_round((radius * 2.0) * scale)
    if dst_size <= 0:
        return

    x0 = max(0, dst_x)
    y0 = max(0, dst_y)
    x1 = min(size, dst_x + dst_size)
    y1 = min(size, dst_y + dst_size)
    if x0 >= x1 or y0 >= y1:
        return

    stamp_rgb, stamp_alpha = _source_bonus_sprite_stamp(sprite_index, dst_size)
    local_x0 = x0 - dst_x
    local_y0 = y0 - dst_y
    local_x1 = local_x0 + (x1 - x0)
    local_y1 = local_y0 + (y1 - y0)
    sprite_patch = stamp_rgb[local_y0:local_y1, local_x0:local_x1]
    alpha = stamp_alpha[local_y0:local_y1, local_x0:local_x1]
    if not bool((alpha > 0.0).any()):
        return

    view = canvas[y0:y1, x0:x1]
    rgb = sprite_patch.astype(np.float32)
    blended = rgb * alpha + view.astype(np.float32) * (np.float32(1.0) - alpha)
    np.rint(blended, out=blended)
    np.clip(blended, 0.0, 255.0, out=blended)
    view[:] = blended.astype(np.uint8)


def _draw_bonus_circles_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_size: float,
    *,
    bonus_types: np.ndarray | None = None,
) -> None:
    outline_radius = map_size * 2.0 / float(canvas.shape[0])
    if bonus_types is None:
        bonus_types_iter = (None for _ in range(len(positions)))
    else:
        bonus_types_iter = (int(value) for value in bonus_types)
    for position, radius, bonus_type in zip(positions, radii, bonus_types_iter, strict=True):
        _draw_world_circle_rgb(
            canvas,
            position[0],
            position[1],
            float(radius) + outline_radius,
            map_size,
            color=SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_OUTLINE_RGB,
        )
        _draw_world_circle_rgb(
            canvas,
            position[0],
            position[1],
            radius,
            map_size,
            color=_bonus_type_rgb(bonus_type),
        )


def _bonus_type_rgb(bonus_type: int | None) -> tuple[int, int, int]:
    if bonus_type is None:
        return SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB
    if bonus_type < 0 or bonus_type >= len(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_RGB):
        return SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB
    return SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_RGB[bonus_type]


def _bonus_sprite_index(bonus_type: int | None) -> int | None:
    if bonus_type is None:
        return None
    if bonus_type < 0 or bonus_type >= len(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_NAME_BY_CODE):
        return None
    name = SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_NAME_BY_CODE[bonus_type]
    if name is None:
        return None
    return SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_INDEX_BY_NAME.get(name)


def _canvas_round(value: float) -> int:
    return int(value + 0.5)


@lru_cache(maxsize=1)
def _source_bonus_sprite_tiles() -> np.ndarray:
    path = _source_bonus_sprite_sheet_path()
    image = _load_rgba_png(path)
    rows = SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_ROWS
    columns = SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_COLUMNS
    if image.shape[0] % rows != 0 or image.shape[1] % columns != 0:
        raise VectorVisualObservationError(
            f"bonus sprite sheet dimensions must divide evenly by {columns} columns and {rows} rows"
        )
    tile_height = image.shape[0] // rows
    tile_width = image.shape[1] // columns
    tiles = []
    for index in range(rows * columns):
        row = index // columns
        column = index % columns
        tiles.append(
            image[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ].copy()
        )
    return np.stack(tiles, axis=0)


@lru_cache(maxsize=256)
def _source_bonus_sprite_stamp(sprite_index: int, dst_size: int) -> tuple[np.ndarray, np.ndarray]:
    if dst_size <= 0:
        raise VectorVisualObservationError("sprite stamp size must be positive")
    if (
        sprite_index < 0
        or sprite_index >= len(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES)
    ):
        raise VectorVisualObservationError(f"bonus sprite index out of range: {sprite_index}")
    try:
        tiles = _source_bonus_sprite_tiles()
    except VectorVisualObservationError:
        return _source_bonus_placeholder_stamp(sprite_index, dst_size)
    tile = tiles[sprite_index]
    tile_height, tile_width = tile.shape[:2]
    local_x = np.arange(dst_size, dtype=np.int32)
    local_y = np.arange(dst_size, dtype=np.int32)
    src_x = np.clip((local_x * tile_width) // dst_size, 0, tile_width - 1)
    src_y = np.clip((local_y * tile_height) // dst_size, 0, tile_height - 1)
    stamp = tile[src_y[:, None], src_x[None, :]]
    stamp_rgb = stamp[:, :, :3].copy()
    stamp_alpha = stamp[:, :, 3:4].astype(np.float32, copy=True) * np.float32(1.0 / 255.0)
    return stamp_rgb, stamp_alpha


@lru_cache(maxsize=256)
def _source_bonus_placeholder_stamp(
    sprite_index: int,
    dst_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        sprite_index < 0
        or sprite_index >= len(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_CODE_BY_SPRITE_INDEX)
    ):
        raise VectorVisualObservationError(f"bonus sprite index out of range: {sprite_index}")
    type_code = SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_TYPE_CODE_BY_SPRITE_INDEX[sprite_index]
    fill = np.asarray(_bonus_type_rgb(type_code), dtype=np.uint8)
    stamp_rgb = np.empty((dst_size, dst_size, 3), dtype=np.uint8)
    stamp_rgb[:, :] = fill
    yy, xx = np.ogrid[:dst_size, :dst_size]
    center = (float(dst_size) - 1.0) * 0.5
    radius = max(0.0, center)
    distance_sq = (xx.astype(np.float64) - center) ** 2 + (
        yy.astype(np.float64) - center
    ) ** 2
    circle = distance_sq <= radius * radius
    border_width = max(1.0, float(dst_size) * 0.08)
    inner_radius = max(0.0, radius - border_width)
    border = circle & (distance_sq >= inner_radius * inner_radius)
    stamp_rgb[border] = SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_OUTLINE_RGB
    if dst_size >= 5:
        stripe_period = 3 + (sprite_index % 4)
        stripe = circle & (((xx + yy + sprite_index) % stripe_period) == 0)
        stamp_rgb[stripe] = np.minimum(
            stamp_rgb[stripe].astype(np.int16) + np.int16(36),
            np.int16(255),
        ).astype(np.uint8)
    stamp_alpha = circle.astype(np.float32)[:, :, None]
    return stamp_rgb, stamp_alpha


def _source_bonus_sprite_sheet_path() -> Path:
    relative_path = Path(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH)
    candidates = (
        Path(__file__).resolve().parents[3] / relative_path,
        Path.cwd() / relative_path,
        Path("/repo") / relative_path,
        Path("/") / relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _load_rgba_png(path: Path) -> np.ndarray:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VectorVisualObservationError(f"could not read bonus sprite sheet: {path}") from exc
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise VectorVisualObservationError(f"bonus sprite sheet is not a PNG: {path}")

    offset = 8
    width = height = bit_depth = color_type = interlace = None
    idat_parts: list[bytes] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise VectorVisualObservationError("bonus sprite sheet PNG is truncated")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + length
        if chunk_data_end + 4 > len(data):
            raise VectorVisualObservationError("bonus sprite sheet PNG chunk is truncated")
        chunk_data = data[chunk_data_start:chunk_data_end]
        offset = chunk_data_end + 4
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = (
                _parse_png_ihdr(chunk_data)
            )
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if (
        width is None
        or height is None
        or bit_depth != 8
        or color_type != 6
        or interlace != 0
        or not idat_parts
    ):
        raise VectorVisualObservationError(
            "bonus sprite sheet must be an 8-bit non-interlaced RGBA PNG"
        )
    return _decode_rgba_png_scanlines(width, height, b"".join(idat_parts))


def _parse_png_ihdr(chunk_data: bytes) -> tuple[int, int, int, int, int, int, int]:
    if len(chunk_data) != 13:
        raise VectorVisualObservationError("bonus sprite sheet has invalid IHDR")
    return struct.unpack(">IIBBBBB", chunk_data)


def _decode_rgba_png_scanlines(width: int, height: int, compressed: bytes) -> np.ndarray:
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise VectorVisualObservationError("bonus sprite sheet IDAT data is invalid") from exc
    bytes_per_pixel = 4
    stride = width * bytes_per_pixel
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise VectorVisualObservationError("bonus sprite sheet PNG payload has invalid size")

    rows = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        scanline = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=offset).copy()
        offset += stride
        _unfilter_png_scanline(scanline, previous, filter_type, bytes_per_pixel)
        rows[row] = scanline
        previous = scanline
    return rows.reshape(height, width, bytes_per_pixel)


def _unfilter_png_scanline(
    scanline: np.ndarray,
    previous: np.ndarray,
    filter_type: int,
    bytes_per_pixel: int,
) -> None:
    if filter_type == 0:
        return
    if filter_type == 1:
        for index in range(bytes_per_pixel, scanline.size):
            scanline[index] = (int(scanline[index]) + int(scanline[index - bytes_per_pixel])) & 0xFF
        return
    if filter_type == 2:
        for index in range(scanline.size):
            scanline[index] = (int(scanline[index]) + int(previous[index])) & 0xFF
        return
    if filter_type == 3:
        for index in range(scanline.size):
            left = int(scanline[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
            up = int(previous[index])
            scanline[index] = (int(scanline[index]) + ((left + up) // 2)) & 0xFF
        return
    if filter_type == 4:
        for index in range(scanline.size):
            left = int(scanline[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
            up = int(previous[index])
            up_left = int(previous[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
            scanline[index] = (
                int(scanline[index]) + _png_paeth_predictor(left, up, up_left)
            ) & 0xFF
        return
    raise VectorVisualObservationError(f"unsupported PNG filter type {filter_type}")


def _png_paeth_predictor(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _body_values(owners: np.ndarray) -> np.ndarray:
    owner_index = owners.astype(np.int16, copy=False)
    values = np.where(owner_index < 0, 80, np.minimum(192, 96 + owner_index * 32))
    return values.astype(np.uint8, copy=False)


def _draw_radius_one_circles(
    canvas: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    values: np.ndarray,
) -> None:
    count = int(px.size)
    xs = np.empty(count * 5, dtype=np.int16)
    ys = np.empty(count * 5, dtype=np.int16)
    vals = np.empty(count * 5, dtype=np.uint8)
    xs[0::5] = px
    ys[0::5] = py
    xs[1::5] = px - 1
    ys[1::5] = py
    xs[2::5] = px + 1
    ys[2::5] = py
    xs[3::5] = px
    ys[3::5] = py - 1
    xs[4::5] = px
    ys[4::5] = py + 1
    vals.reshape(count, 5)[:] = values[:, None]
    in_bounds = (xs >= 0) & (xs < 64) & (ys >= 0) & (ys < 64)
    np.maximum.at(canvas, (ys[in_bounds], xs[in_bounds]), vals[in_bounds])


def _circle_mask(radius_px: int, mask_cache: dict[int, np.ndarray] | None) -> np.ndarray:
    if mask_cache is not None:
        cached = mask_cache.get(radius_px)
        if cached is not None:
            return cached
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    mask = (xx * xx + yy * yy) <= radius_px * radius_px
    if mask_cache is not None:
        mask_cache[radius_px] = mask
    return mask


def _source_snapshot_map_size(snapshot: Mapping[str, Any]) -> float:
    game = snapshot.get("game")
    if not isinstance(game, Mapping):
        raise VectorVisualObservationError("source snapshot must contain game mapping")
    try:
        map_size = float(game["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VectorVisualObservationError("source snapshot game.size must be numeric") from exc
    if not np.isfinite(map_size) or map_size <= 0.0:
        raise VectorVisualObservationError("source snapshot game.size must be finite and positive")
    return map_size


def _source_snapshot_avatars(snapshot: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    return _source_mapping_sequence(snapshot.get("avatars"), name="snapshot.avatars")


def _source_mapping_sequence(
    value: Sequence[Mapping[str, Any]] | Any | None,
    *,
    name: str,
) -> Sequence[Mapping[str, Any]]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VectorVisualObservationError(f"{name} must be a sequence of mappings")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise VectorVisualObservationError(f"{name}[{index}] must be a mapping")
    return value


def _source_has_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _source_avatar_radius_by_id(
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None,
) -> dict[int, float]:
    radii: dict[int, float] = {}
    for item in _source_mapping_sequence(avatar_body_metadata, name="avatar_body_metadata"):
        avatar_id = item.get("id")
        if not _source_has_int(avatar_id):
            continue
        try:
            radii[int(avatar_id)] = float(item["radius"])
        except (KeyError, TypeError, ValueError):
            continue
    return radii


def _source_body_owner_index(
    avatar_id: Any,
    player_index_by_avatar_id: Mapping[int, int],
) -> int:
    if avatar_id is None:
        return -1
    try:
        return int(player_index_by_avatar_id[int(avatar_id)])
    except (KeyError, TypeError, ValueError):
        return -1


__all__ = [
    "BONUS_RENDER_MODE_BROWSER_SPRITES",
    "BONUS_RENDER_MODE_CIRCLES_FAST",
    "BONUS_RENDER_MODE_SIMPLE_SYMBOLS",
    "BONUS_SYMBOL_BASE_SIZE",
    "BONUS_SYMBOL_INNER_LUMA",
    "BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE",
    "BONUS_SYMBOL_RENDER_MODE_DEFAULT",
    "BONUS_RENDER_MODE_DEFAULT",
    "BONUS_RENDER_MODE_ORDER",
    "BONUS_RENDER_MODES",
    "SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY",
    "SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM",
    "SOURCE_STATE_CANVAS_GRAY64_COMPARISON_TARGET",
    "SOURCE_STATE_CANVAS_GRAY64_DTYPE",
    "SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE",
    "SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE",
    "SOURCE_STATE_CANVAS_GRAY64_PERSPECTIVE",
    "SOURCE_STATE_CANVAS_GRAY64_PIXEL_FIDELITY_BLOCKER",
    "SOURCE_STATE_CANVAS_GRAY64_BODY_CIRCLES_FAST_RENDERER_IMPL_ID",
    "SOURCE_STATE_CANVAS_GRAY64_BROWSER_LINES_RENDERER_IMPL_ID",
    "SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID",
    "SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH",
    "SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID",
    "SOURCE_STATE_CANVAS_GRAY64_SHAPE",
    "SOURCE_STATE_CANVAS_GRAY64_SOURCE_CLAIM_ID",
    "SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL",
    "SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED",
    "SOURCE_STATE_CANVAS_GRAY64_SURFACE",
    "SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL",
    "SOURCE_STATE_CANVAS_GRAY64_USES_ALE",
    "SOURCE_STATE_CANVAS_GRAY64_VALUE_RANGE",
    "SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL",
    "SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL",
    "SOURCE_STATE_BONUS64_STACK4_GAME_BORDERLESS_CHANNEL",
    "SOURCE_STATE_BONUS64_STACK4_GAME_TTL_CHANNEL",
    "SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE",
    "SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS",
    "SOURCE_STATE_BONUS64_STACK4_OCCUPANCY_CHANNELS",
    "SOURCE_STATE_BONUS64_STACK4_OTHER_STATUS_CHANNELS",
    "SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_DTYPE",
    "SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID",
    "SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE",
    "SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_VALUE_RANGE",
    "SOURCE_STATE_BONUS64_STACK4_SELF_STATUS_CHANNELS",
    "SOURCE_STATE_GRAY64_BONUS_VALUE",
    "SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY",
    "SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM",
    "SOURCE_STATE_GRAY64_COMPARISON_TARGET",
    "SOURCE_STATE_GRAY64_DTYPE",
    "SOURCE_STATE_GRAY64_FRAME_STACK_OWNER",
    "SOURCE_STATE_GRAY64_FRAME_STACK_POLICY",
    "SOURCE_STATE_GRAY64_NORMALIZED_DTYPE",
    "SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE",
    "SOURCE_STATE_GRAY64_PERSPECTIVE",
    "SOURCE_STATE_GRAY64_PIXEL_FIDELITY_BLOCKER",
    "SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH",
    "SOURCE_STATE_GRAY64_RENDERER_IMPL_ID",
    "SOURCE_STATE_GRAY64_SCHEMA_HASH",
    "SOURCE_STATE_GRAY64_SCHEMA_ID",
    "SOURCE_STATE_GRAY64_SHAPE",
    "SOURCE_STATE_GRAY64_SOURCE_CLAIM_ID",
    "SOURCE_STATE_GRAY64_SOURCE_FIDELITY_LEVEL",
    "SOURCE_STATE_GRAY64_SOURCE_STATE_BACKED",
    "SOURCE_STATE_GRAY64_STATE_FIELDS",
    "SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS",
    "SOURCE_STATE_GRAY64_SURFACE",
    "SOURCE_STATE_GRAY64_TERMINAL_POLICY",
    "SOURCE_STATE_GRAY64_TRUTH_LEVEL",
    "SOURCE_STATE_GRAY64_USES_ALE",
    "SOURCE_STATE_GRAY64_VALUE_RANGE",
    "SOURCE_STATE_RGB_BODY_CIRCLES_FAST_RENDERER_IMPL_ID",
    "SOURCE_STATE_RGB_BROWSER_LINES_RENDERER_IMPL_ID",
    "SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES",
    "SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH",
    "SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID",
    "SourceStateBrowserLineTrailLayerCache",
    "SourceStateBrowserLineTrailLayerCacheStats",
    "SourceStateCanvasGray64DirtyRenderCache",
    "SourceStateCanvasGray64DirtyRenderStats",
    "SourceStateGray64DirtyDownsampleScratch",
    "SourceStateGray64DownsampleScratch",
    "TRAIL_RENDER_MODE_BODY_CIRCLES_FAST",
    "TRAIL_RENDER_MODE_BROWSER_LINES",
    "TRAIL_RENDER_MODE_DEFAULT",
    "TRAIL_RENDER_MODE_ORDER",
    "TRAIL_RENDER_MODES",
    "SourceStateGray64Renderer",
    "VectorVisualObservationError",
    "normalize_source_state_gray64",
    "render_source_state_canvas_gray64",
    "render_source_state_canvas_gray64_player_perspectives",
    "render_source_state_bonus64_stack4_player_perspective_v1",
    "render_source_state_rgb_canvas_like",
    "render_source_snapshot_canvas_gray64",
    "render_source_snapshot_gray64",
    "render_source_snapshot_rgb_canvas_like",
    "render_source_state_gray64",
    "render_source_state_gray64_fast_player_perspectives",
    "rgb_canvas_like_to_gray64",
    "source_state_bonus64_stack4_player_perspective_v1_schema",
    "source_state_canvas_gray64_schema",
    "source_state_gray64_metadata",
    "source_state_gray64_schema",
]
