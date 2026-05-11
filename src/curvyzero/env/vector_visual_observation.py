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
from typing import Any

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
SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID = (
    "curvyzero_source_state_rgb_canvas_like_numpy/v0"
)
SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL = "source_state_backed_browser_like_non_pixel_parity"
SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE = 704
SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB = (34, 34, 34)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB = (255, 190, 40)
SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_OUTLINE_RGB = (255, 255, 255)
SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 80, 255),
    (255, 240, 0),
)
SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID = "curvyzero_source_state_canvas_gray64/v0"
SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID = (
    "curvyzero_source_state_rgb_canvas_like_to_gray64_numpy/v0"
)
SOURCE_STATE_CANVAS_GRAY64_SOURCE_CLAIM_ID = (
    "curvyzero_vector_runtime_source_state_canvas_gray64/v0"
)
SOURCE_STATE_CANVAS_GRAY64_SHAPE = SOURCE_STATE_GRAY64_SHAPE
SOURCE_STATE_CANVAS_GRAY64_DTYPE = SOURCE_STATE_GRAY64_DTYPE
SOURCE_STATE_CANVAS_GRAY64_VALUE_RANGE = SOURCE_STATE_GRAY64_VALUE_RANGE
SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE = SOURCE_STATE_GRAY64_NORMALIZED_DTYPE
SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE = (
    SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE
)
SOURCE_STATE_CANVAS_GRAY64_SURFACE = "source_state_canvas_gray64_tensor"
SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL = SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL
SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL = (
    "source_vector_state_canvas_like_rgb_luma_raster"
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
_SOURCE_STATE_ARRAY_KEYS = tuple(
    field_name.removeprefix("state.") for field_name in SOURCE_STATE_GRAY64_STATE_FIELDS
)
_SOURCE_STATE_OPTIONAL_ARRAY_KEYS = tuple(
    field_name.removeprefix("state.")
    for field_name in SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS
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
SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH = stable_contract_hash(
    _SOURCE_STATE_GRAY64_RENDERER_PAYLOAD
)

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


def render_source_state_rgb_canvas_like(
    state: Mapping[str, np.ndarray],
    *,
    row: int = 0,
    out: np.ndarray | None = None,
    frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
) -> np.ndarray:
    """Render one vector runtime row into a browser-like RGB frame.

    This is for human inspection, not model input. It uses the source-state arrays,
    the browser's dark background, player colors, visible trail bodies, live heads,
    and active bonus circles. It is intentionally separate from the 64x64 gray
    learning tensor.
    """

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

    body_limit = _body_slot_limit(arrays, row_index)
    active_slots = np.flatnonzero(arrays["body_active"][row_index, :body_limit])
    _draw_body_circles_rgb(
        frame,
        arrays["body_pos"][row_index, active_slots],
        arrays["body_radius"][row_index, active_slots],
        arrays["body_owner"][row_index, active_slots],
        map_size,
        colors=colors,
    )

    if "bonus_active" in arrays:
        bonus_slots = np.flatnonzero(arrays["bonus_active"][row_index])
        _draw_bonus_circles_rgb(
            frame,
            arrays["bonus_pos"][row_index, bonus_slots],
            arrays["bonus_radius"][row_index, bonus_slots],
            map_size,
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
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
) -> np.ndarray:
    """Render browser-like source-state RGB, then convert it to gray64.

    This is the clean model-facing visual path: one source-state, browser-like
    RGB image at 64x64, converted with fixed luminance weights. It is still not
    a DOM/browser canvas pixel-parity claim.
    """

    rgb = render_source_state_rgb_canvas_like(
        state,
        row=row,
        out=rgb_out,
        frame_size=SOURCE_STATE_CANVAS_GRAY64_SHAPE[1],
        player_rgb=player_rgb,
        background_rgb=background_rgb,
    )
    return rgb_canvas_like_to_gray64(rgb, out=out)


def render_source_snapshot_rgb_canvas_like(
    snapshot: Mapping[str, Any],
    *,
    world_bodies: Sequence[Mapping[str, Any]] | None = None,
    bonus_bodies: Sequence[Mapping[str, Any]] | None = None,
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None = None,
    out: np.ndarray | None = None,
    frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    default_avatar_radius: float = SOURCE_STATE_GRAY64_DEFAULT_AVATAR_RADIUS,
) -> np.ndarray:
    """Render a source-env snapshot into the browser-like RGB frame."""

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
        _draw_body_circles_rgb(
            frame,
            body_positions,
            body_radii,
            body_owners,
            map_size,
            colors=colors,
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
        _draw_bonus_circles_rgb(frame, bonus_positions, bonus_radii, map_size)

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
    bonus_bodies: Sequence[Mapping[str, Any]] | None = None,
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None = None,
    out: np.ndarray | None = None,
    rgb_out: np.ndarray | None = None,
    player_rgb: Sequence[Sequence[int]] | None = None,
    background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    default_avatar_radius: float = SOURCE_STATE_GRAY64_DEFAULT_AVATAR_RADIUS,
) -> np.ndarray:
    rgb = render_source_snapshot_rgb_canvas_like(
        snapshot,
        world_bodies=world_bodies,
        bonus_bodies=bonus_bodies,
        avatar_body_metadata=avatar_body_metadata,
        out=rgb_out,
        frame_size=SOURCE_STATE_CANVAS_GRAY64_SHAPE[1],
        player_rgb=player_rgb,
        background_rgb=background_rgb,
        default_avatar_radius=default_avatar_radius,
    )
    return rgb_canvas_like_to_gray64(rgb, out=out)


def rgb_canvas_like_to_gray64(
    rgb: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert an RGB canvas-like frame to CHW uint8 gray64."""

    rgb_array = np.asarray(rgb)
    if rgb_array.shape != (64, 64, 3):
        raise VectorVisualObservationError(
            f"rgb must have shape (64, 64, 3), got {rgb_array.shape}"
        )
    if rgb_array.dtype != np.uint8:
        raise VectorVisualObservationError(f"rgb dtype must be uint8, got {rgb_array.dtype}")
    frame = (
        np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
        if out is None
        else _validated_frame(out, name="out")
    )
    gray = (
        rgb_array[:, :, 0].astype(np.float32) * np.float32(0.299)
        + rgb_array[:, :, 1].astype(np.float32) * np.float32(0.587)
        + rgb_array[:, :, 2].astype(np.float32) * np.float32(0.114)
    )
    np.rint(gray, out=gray)
    np.clip(gray, 0.0, 255.0, out=gray)
    frame[0] = gray.astype(np.uint8)
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
            "luminance of source-state browser-like RGB frame rendered at 64x64"
        ),
        "rgb_source_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "rgb_renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
        "renderer_impl_id": SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID,
        "state_fields": list(SOURCE_STATE_GRAY64_STATE_FIELDS),
        "optional_state_fields": [
            *SOURCE_STATE_GRAY64_OPTIONAL_STATE_FIELDS,
            "state.avatar_color",
        ],
        "perspective": SOURCE_STATE_CANVAS_GRAY64_PERSPECTIVE,
        "comparison_target": SOURCE_STATE_CANVAS_GRAY64_COMPARISON_TARGET,
        "browser_pixel_fidelity": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
        "browser_pixel_fidelity_claim": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM,
        "browser_pixel_fidelity_blocker": SOURCE_STATE_CANVAS_GRAY64_PIXEL_FIDELITY_BLOCKER,
        "uses_ale": SOURCE_STATE_CANVAS_GRAY64_USES_ALE,
        "grayscale_conversion": "round(0.299*r + 0.587*g + 0.114*b)",
    }


SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH = stable_contract_hash(
    source_state_canvas_gray64_schema()
)


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
    other = (
        1 - controlled
        if player_count == 2
        else _first_other_player(controlled, player_count)
    )
    bonus_arrays = _bonus64_arrays(state, arrays)
    output = _validated_bonus64_out(out)
    output.fill(0.0)

    if occupancy_stack is None:
        raw_frame = render_source_state_gray64(state, row=row_index)
        perspective_frame = _player_perspective_gray64(raw_frame, controlled)
        output[:SOURCE_STATE_BONUS64_STACK4_OCCUPANCY_CHANNELS] = (
            perspective_frame.astype(np.float32, copy=False) * np.float32(1.0 / 255.0)
        )
    else:
        output[:SOURCE_STATE_BONUS64_STACK4_OCCUPANCY_CHANNELS] = (
            _validated_occupancy_stack(occupancy_stack)
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
    body_write_cursor = _integer_array(state, "body_write_cursor", shape=(batch_size,))
    if bool(((body_write_cursor < 0) | (body_write_cursor > body_capacity)).any()):
        raise VectorVisualObservationError(
            "state['body_write_cursor'] values must be in [0, body_capacity]"
        )
    if bool((~np.isfinite(body_pos)).any()) or bool((~np.isfinite(body_radius)).any()):
        raise VectorVisualObservationError("body position and radius arrays must contain finite values")
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
    arrays.update(_optional_bonus_arrays(state, batch_size=batch_size))
    return arrays


def _trusted_arrays(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(state[name]) for name in _SOURCE_STATE_ARRAY_KEYS}
    for name in _SOURCE_STATE_OPTIONAL_ARRAY_KEYS:
        if name in state:
            arrays[name] = np.asarray(state[name])
    if "avatar_color" in state:
        arrays["avatar_color"] = np.asarray(state["avatar_color"])
    return arrays


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
            "optional bonus visual arrays must be supplied together; "
            f"missing {missing}"
        )
    bonus_active = _bool_array(state, "bonus_active", ndim=2)
    if bonus_active.shape[0] != batch_size:
        raise VectorVisualObservationError("state['bonus_active'] must have shape [B,C]")
    bonus_capacity = bonus_active.shape[1]
    bonus_pos = _numeric_array(state, "bonus_pos", shape=(batch_size, bonus_capacity, 2))
    bonus_radius = _numeric_array(state, "bonus_radius", shape=(batch_size, bonus_capacity))
    if bool((~np.isfinite(bonus_pos)).any()) or bool((~np.isfinite(bonus_radius)).any()):
        raise VectorVisualObservationError("bonus position and radius arrays must contain finite values")
    if bool((bonus_radius < 0.0).any()):
        raise VectorVisualObservationError("state['bonus_radius'] values must be non-negative")
    return {
        "bonus_active": bonus_active,
        "bonus_pos": bonus_pos,
        "bonus_radius": bonus_radius,
    }


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
        raise VectorVisualObservationError(
            f"row must be in [0, {batch_size}), got {row_index}"
        )
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
        return _rgb_triplet(
            (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
        )
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
    output[start_channel + 3].fill(
        _turn_override_value(arrays, row=row, player=player)
    )
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
    return 1.0 if not np.isclose(float(value_array[row, player]), float(base_array[row, player])) else 0.0


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
    return float(np.clip(np.nanmax(durations), 0.0, SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS) / SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS)


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
    return float(np.clip(np.nanmax(durations), 0.0, SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS) / SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS)


def _first_other_player(controlled: int, player_count: int) -> int:
    for player in range(player_count):
        if player != controlled:
            return player
    return controlled


def _body_slot_limit(arrays: Mapping[str, np.ndarray], row: int) -> int:
    capacity = arrays["body_active"].shape[1]
    cursor = int(arrays["body_write_cursor"][row])
    return int(np.clip(cursor, 0, capacity))


def _body_value(owner: Any) -> int:
    owner_index = int(owner)
    if owner_index < 0:
        return 80
    return int(min(192, 96 + owner_index * 32))


def _head_value(player: int) -> int:
    return int(min(255, 224 + int(player) * 8))


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
        (x + radius > 0.0)
        & (x - radius < map_size)
        & (y + radius > 0.0)
        & (y - radius < map_size)
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


def _draw_world_circle_rgb(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    radius_value: Any,
    map_size: float,
    *,
    color: Sequence[int] | np.ndarray,
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
    mask = _circle_mask(radius_px, mask_cache=None)
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
        )


def _draw_bonus_circles_rgb(
    canvas: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_size: float,
) -> None:
    outline_radius = map_size * 2.0 / float(canvas.shape[0])
    for position, radius in zip(positions, radii, strict=True):
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
            color=SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB,
        )


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
    "SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY",
    "SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY_CLAIM",
    "SOURCE_STATE_CANVAS_GRAY64_COMPARISON_TARGET",
    "SOURCE_STATE_CANVAS_GRAY64_DTYPE",
    "SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE",
    "SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE",
    "SOURCE_STATE_CANVAS_GRAY64_PERSPECTIVE",
    "SOURCE_STATE_CANVAS_GRAY64_PIXEL_FIDELITY_BLOCKER",
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
    "SourceStateGray64Renderer",
    "VectorVisualObservationError",
    "normalize_source_state_gray64",
    "render_source_state_canvas_gray64",
    "render_source_state_bonus64_stack4_player_perspective_v1",
    "render_source_state_rgb_canvas_like",
    "render_source_snapshot_canvas_gray64",
    "render_source_snapshot_gray64",
    "render_source_snapshot_rgb_canvas_like",
    "render_source_state_gray64",
    "rgb_canvas_like_to_gray64",
    "source_state_bonus64_stack4_player_perspective_v1_schema",
    "source_state_canvas_gray64_schema",
    "source_state_gray64_metadata",
    "source_state_gray64_schema",
]
