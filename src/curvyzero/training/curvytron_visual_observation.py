"""Debug visual observation smoke surface for project-owned CurvyTron states.

This module is intentionally small and explicitly non-fidelity. It renders a
CurvyTron state/snapshot into a coarse grayscale occupancy frame and names the
truth metadata downstream Optimizer-owned visual plumbing must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from curvyzero.env.config import CurvyTronConfig
from curvyzero.env.state import EnvState


DEBUG_OCCUPANCY_GRAY64_LABEL = "curvyzero_debug_occupancy_gray64/v0"
DEBUG_OCCUPANCY_GRAY64_SCHEMA_ID = DEBUG_OCCUPANCY_GRAY64_LABEL
DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL = (
    "curvyzero_debug_occupancy_gray64_player_aware/v1"
)
DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_ID = DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL
DEBUG_OCCUPANCY_GRAY64_SHAPE = (1, 64, 64)
DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE = (4, 64, 64)
DEBUG_OCCUPANCY_GRAY64_RENDER_DTYPE = "uint8"
DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE = "float32"
DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID = "curvyzero_debug_occupancy_gray64_numpy/v0"
DEBUG_OCCUPANCY_GRAY64_SURFACE = "debug_visual_tensor"
DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL = "debug_non_fidelity"
DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL = "none"
DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE = (0.0, 1.0)
DEBUG_OCCUPANCY_GRAY64_RAW_VALUE_RANGE = (0, 255)
DEBUG_OCCUPANCY_GRAY64_PERSPECTIVE = "global_arena_debug"
DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_OWNER = "optimizer"
DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_POLICY = (
    "single_frame_renderer; downstream adapter may stack normalized frames FIFO"
)
DEBUG_OCCUPANCY_GRAY64_SOURCE_CLAIM_ID = "curvyzero_source_state_debug_occupancy_gray64/v0"
DEBUG_OCCUPANCY_GRAY64_SOURCE_STATE_FIELDS = (
    "snapshot.game.size",
    "snapshot.avatars[].x",
    "snapshot.avatars[].y",
    "snapshot.avatars[].alive",
    "world_bodies_snapshot()[].x",
    "world_bodies_snapshot()[].y",
)
DEBUG_OCCUPANCY_GRAY64_COMPARISON_TARGET = "source_state_coordinates_only"
DEBUG_OCCUPANCY_GRAY64_PIXEL_FIDELITY_BLOCKER = (
    "no source-faithful browser/canvas renderer or pixel parity artifact is available yet"
)
DEBUG_OCCUPANCY_GRAY64_FINAL_OBSERVATION_POLICY = (
    "adapter copies the terminal normalized frame into info.final_observation when done"
)
DEBUG_OCCUPANCY_GRAY64_RESET_SOURCE = (
    "CurvyTronSourceEnv snapshot or project EnvState supplied by caller"
)
DEBUG_OCCUPANCY_GRAY64_CAVEAT = (
    "debug/profiling occupancy view only; marks source world body centers and "
    "avatar positions, not source visual fidelity or exact circle geometry"
)
DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY = False
DEBUG_OCCUPANCY_GRAY64_BROWSER_PIXEL_FIDELITY = False
DEBUG_OCCUPANCY_GRAY64_USES_ALE = False
DEBUG_OCCUPANCY_GRAY64_WORLD_BODY_VALUE = 160
DEBUG_OCCUPANCY_GRAY64_OTHER_AVATAR_VALUE = 220
DEBUG_OCCUPANCY_GRAY64_CONTROLLED_AVATAR_VALUE = 255

_DEBUG_OCCUPANCY_GRAY64_SCHEMA_PAYLOAD: dict[str, Any] = {
    "schema_id": DEBUG_OCCUPANCY_GRAY64_SCHEMA_ID,
    "observation_schema_id": DEBUG_OCCUPANCY_GRAY64_SCHEMA_ID,
    "shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
    "stack_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
    "raw_renderer_dtype": DEBUG_OCCUPANCY_GRAY64_RENDER_DTYPE,
    "dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
    "lightzero_payload_dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
    "range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
    "raw_value_range": list(DEBUG_OCCUPANCY_GRAY64_RAW_VALUE_RANGE),
    "surface": DEBUG_OCCUPANCY_GRAY64_SURFACE,
    "channel_order": "CHW",
    "stack_axis": "time_as_channel",
    "pixel_semantics": {
        "0": "empty/background",
        "160": "source world body center debug mark",
        "255": "live avatar position debug mark",
    },
    "renderer_kind": "debug_occupancy",
    "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
    "truth_level": DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
    "source_fidelity_level": DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
    "source_claim_id": DEBUG_OCCUPANCY_GRAY64_SOURCE_CLAIM_ID,
    "source_state_fields": list(DEBUG_OCCUPANCY_GRAY64_SOURCE_STATE_FIELDS),
    "comparison_target": DEBUG_OCCUPANCY_GRAY64_COMPARISON_TARGET,
    "source_pixel_fidelity_blocker": DEBUG_OCCUPANCY_GRAY64_PIXEL_FIDELITY_BLOCKER,
    "source_backed_observation_fidelity": DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
    "source_visual_fidelity": DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
    "browser_pixel_fidelity": DEBUG_OCCUPANCY_GRAY64_BROWSER_PIXEL_FIDELITY,
    "uses_ale": DEBUG_OCCUPANCY_GRAY64_USES_ALE,
    "ale_usage": "none",
    "perspective": DEBUG_OCCUPANCY_GRAY64_PERSPECTIVE,
    "frame_stack_owner": DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_OWNER,
    "frame_stack_policy": DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_POLICY,
    "reset_source": DEBUG_OCCUPANCY_GRAY64_RESET_SOURCE,
    "final_observation_policy": DEBUG_OCCUPANCY_GRAY64_FINAL_OBSERVATION_POLICY,
    "caveat": DEBUG_OCCUPANCY_GRAY64_CAVEAT,
}
DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        _DEBUG_OCCUPANCY_GRAY64_SCHEMA_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()[:16]
_DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_PAYLOAD: dict[str, Any] = {
    **_DEBUG_OCCUPANCY_GRAY64_SCHEMA_PAYLOAD,
    "schema_id": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_ID,
    "observation_schema_id": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_ID,
    "perspective": "controlled_player_arena_debug",
    "player_aware": True,
    "controlled_player_id_field": "controlled_player_id",
    "legacy_anonymous_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_SCHEMA_ID,
    "pixel_semantics": {
        "0": "empty/background",
        str(DEBUG_OCCUPANCY_GRAY64_WORLD_BODY_VALUE): (
            "source world body center debug mark"
        ),
        str(DEBUG_OCCUPANCY_GRAY64_OTHER_AVATAR_VALUE): (
            "other live avatar position debug mark"
        ),
        str(DEBUG_OCCUPANCY_GRAY64_CONTROLLED_AVATAR_VALUE): (
            "controlled live avatar position debug mark"
        ),
    },
}
DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        _DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()[:16]


def debug_occupancy_gray64_schema() -> dict[str, Any]:
    schema = dict(_DEBUG_OCCUPANCY_GRAY64_SCHEMA_PAYLOAD)
    schema.update(
        {
            "label": DEBUG_OCCUPANCY_GRAY64_LABEL,
            "schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
            "source_backing": (
                "CurvyTronEnv EnvState occupancy/positions or CurvyTronSourceEnv "
                "snapshot/world_bodies_snapshot"
            ),
            "required_timing_metadata_fields": [
                "observation_schema_id",
                "truth_level",
                "source_fidelity_level",
                "shape",
                "dtype",
                "range",
                "perspective",
                "frame_stack_owner",
                "frame_stack_policy",
                "renderer_impl_id",
                "includes_render_cost",
                "source_claim_id",
                "comparison_target",
                "final_observation_policy",
            ],
        }
    )
    return schema


def debug_occupancy_gray64_metadata(*, includes_render_cost: bool) -> dict[str, Any]:
    """Return claim metadata for timing/debug records that use this visual schema."""

    return {
        "observation_schema_id": DEBUG_OCCUPANCY_GRAY64_SCHEMA_ID,
        "truth_level": DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
        "source_fidelity_level": DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
        "surface": DEBUG_OCCUPANCY_GRAY64_SURFACE,
        "shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
        "dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
        "raw_renderer_dtype": DEBUG_OCCUPANCY_GRAY64_RENDER_DTYPE,
        "range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
        "raw_value_range": list(DEBUG_OCCUPANCY_GRAY64_RAW_VALUE_RANGE),
        "perspective": DEBUG_OCCUPANCY_GRAY64_PERSPECTIVE,
        "frame_stack_owner": DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_OWNER,
        "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
        "schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
        "observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
        "includes_render_cost": bool(includes_render_cost),
        "source_claim_id": DEBUG_OCCUPANCY_GRAY64_SOURCE_CLAIM_ID,
        "source_state_fields": list(DEBUG_OCCUPANCY_GRAY64_SOURCE_STATE_FIELDS),
        "comparison_target": DEBUG_OCCUPANCY_GRAY64_COMPARISON_TARGET,
        "source_pixel_fidelity_blocker": DEBUG_OCCUPANCY_GRAY64_PIXEL_FIDELITY_BLOCKER,
        "browser_pixel_fidelity": DEBUG_OCCUPANCY_GRAY64_BROWSER_PIXEL_FIDELITY,
        "frame_stack_policy": DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_POLICY,
        "reset_source": DEBUG_OCCUPANCY_GRAY64_RESET_SOURCE,
        "final_observation_policy": DEBUG_OCCUPANCY_GRAY64_FINAL_OBSERVATION_POLICY,
        "uses_ale": DEBUG_OCCUPANCY_GRAY64_USES_ALE,
        "ale_usage": "none",
        "source_backed_observation_fidelity": DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
    }


@dataclass
class DebugOccupancyGray64Renderer:
    """Reusable source snapshot -> coarse grayscale frame renderer."""

    frame: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.frame is None:
            self.frame = np.zeros(DEBUG_OCCUPANCY_GRAY64_SHAPE, dtype=np.uint8)
        else:
            frame = np.asarray(self.frame)
            if frame.shape != DEBUG_OCCUPANCY_GRAY64_SHAPE:
                raise ValueError(
                    f"frame must have shape {DEBUG_OCCUPANCY_GRAY64_SHAPE}, got {frame.shape}"
                )
            if frame.dtype != np.uint8:
                raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
            self.frame = frame

    def render(
        self,
        snapshot: Mapping[str, Any],
        *,
        world_bodies: Sequence[Mapping[str, Any]] | None = None,
        controlled_player_id: str | int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        frame = self.frame if out is None else np.asarray(out)
        if frame is None:
            raise RuntimeError("renderer frame was not initialized")
        if frame.shape != DEBUG_OCCUPANCY_GRAY64_SHAPE:
            raise ValueError(f"out must have shape {DEBUG_OCCUPANCY_GRAY64_SHAPE}, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"out dtype must be uint8, got {frame.dtype}")

        frame.fill(0)
        size = _source_size(snapshot)
        canvas = frame[0]
        if world_bodies is not None:
            _mark_world_points(
                canvas,
                _mapping_sequence(world_bodies, name="world_bodies"),
                size,
                value=DEBUG_OCCUPANCY_GRAY64_WORLD_BODY_VALUE,
            )

        live_avatars = [avatar for avatar in _avatars(snapshot) if bool(avatar.get("alive", True))]
        controlled_avatar_id = _controlled_avatar_id(controlled_player_id)
        if controlled_avatar_id is None:
            _mark_world_points(
                canvas,
                live_avatars,
                size,
                value=DEBUG_OCCUPANCY_GRAY64_CONTROLLED_AVATAR_VALUE,
            )
        else:
            other_avatars: list[Mapping[str, Any]] = []
            controlled_avatars: list[Mapping[str, Any]] = []
            for avatar in live_avatars:
                if _avatar_id(avatar) == controlled_avatar_id:
                    controlled_avatars.append(avatar)
                else:
                    other_avatars.append(avatar)
            _mark_world_points(
                canvas,
                other_avatars,
                size,
                value=DEBUG_OCCUPANCY_GRAY64_OTHER_AVATAR_VALUE,
            )
            _mark_world_points(
                canvas,
                controlled_avatars,
                size,
                value=DEBUG_OCCUPANCY_GRAY64_CONTROLLED_AVATAR_VALUE,
            )
        return frame


@dataclass
class DebugOccupancyGray64FrameStack:
    """Four-frame shape helper; Optimizer owns production stack timing/plumbing."""

    stack: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.stack is None:
            self.stack = np.zeros(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE, dtype=np.float32)
        else:
            stack = np.asarray(self.stack)
            if stack.shape != DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE:
                raise ValueError(
                    f"stack must have shape {DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE}, got {stack.shape}"
                )
            if stack.dtype != np.float32:
                raise ValueError(f"stack dtype must be float32, got {stack.dtype}")
            self.stack = stack

    def update(self, frame: np.ndarray, *, copy: bool = True) -> np.ndarray:
        if self.stack is None:
            raise RuntimeError("frame stack was not initialized")
        frame_array = np.asarray(frame)
        if frame_array.shape != DEBUG_OCCUPANCY_GRAY64_SHAPE:
            raise ValueError(
                f"frame must have shape {DEBUG_OCCUPANCY_GRAY64_SHAPE}, got {frame_array.shape}"
            )
        if frame_array.dtype != np.float32:
            raise ValueError(f"frame dtype must be float32, got {frame_array.dtype}")

        self.stack[:-1] = self.stack[1:]
        self.stack[-1] = frame_array[0]
        return self.stack.copy() if copy else self.stack


def render_debug_occupancy_gray64(
    snapshot: Mapping[str, Any],
    *,
    world_bodies: Sequence[Mapping[str, Any]] | None = None,
    controlled_player_id: str | int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    return DebugOccupancyGray64Renderer(frame=out).render(
        snapshot,
        world_bodies=world_bodies,
        controlled_player_id=controlled_player_id,
    )


def render_debug_occupancy_gray64_from_env_state(
    state: EnvState,
    config: CurvyTronConfig,
    *,
    controlled_player_id: str | int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Render toy/project ``EnvState`` occupancy and live heads as debug pixels.

    This is intentionally not a source canvas renderer. It marks occupied grid
    cells and live avatar positions only, giving LightZero visual plumbing a
    small non-empty frame without changing the scalar/ray trainer contract.
    """

    frame = _validated_frame(out)
    frame.fill(0)
    canvas = frame[0]
    occupancy = np.asarray(state.occupancy)
    expected_occupancy_shape = (int(config.height), int(config.width))
    if occupancy.shape != expected_occupancy_shape:
        raise ValueError(
            f"state.occupancy must have shape {expected_occupancy_shape}, got {occupancy.shape}"
        )
    grid_height, grid_width = occupancy.shape
    occupied_y, occupied_x = np.nonzero(occupancy)
    for cell_y, cell_x in zip(occupied_y, occupied_x, strict=True):
        _mark_grid_cell(canvas, int(cell_x), int(cell_y), grid_width, grid_height, value=160)

    positions = np.asarray(state.positions)
    alive = np.asarray(state.alive, dtype=np.bool_)
    controlled_player_index = _controlled_player_index(controlled_player_id)
    if positions.shape != (int(config.players), 2):
        raise ValueError(
            f"state.positions must have shape {(int(config.players), 2)}, got {positions.shape}"
        )
    if alive.shape != (int(config.players),):
        raise ValueError(f"state.alive must have shape {(int(config.players),)}, got {alive.shape}")
    for player_index, (position, is_alive) in enumerate(zip(positions, alive, strict=True)):
        if bool(is_alive):
            value = (
                DEBUG_OCCUPANCY_GRAY64_CONTROLLED_AVATAR_VALUE
                if controlled_player_index is None or player_index == controlled_player_index
                else DEBUG_OCCUPANCY_GRAY64_OTHER_AVATAR_VALUE
            )
            _mark_rect_world_point(
                canvas,
                position[0],
                position[1],
                float(config.width),
                float(config.height),
                value=value,
            )
    return frame


def normalize_debug_occupancy_gray64_for_lightzero(
    frame: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return a float32 ``[0, 1]`` frame for LightZero env payloads."""

    frame_array = np.asarray(frame)
    if frame_array.shape != DEBUG_OCCUPANCY_GRAY64_SHAPE:
        raise ValueError(
            f"frame must have shape {DEBUG_OCCUPANCY_GRAY64_SHAPE}, got {frame_array.shape}"
        )
    if frame_array.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, got {frame_array.dtype}")
    if out is None:
        normalized = np.empty(DEBUG_OCCUPANCY_GRAY64_SHAPE, dtype=np.float32)
    else:
        normalized = np.asarray(out)
        if normalized.shape != DEBUG_OCCUPANCY_GRAY64_SHAPE:
            raise ValueError(
                f"out must have shape {DEBUG_OCCUPANCY_GRAY64_SHAPE}, got {normalized.shape}"
            )
        if normalized.dtype != np.float32:
            raise ValueError(f"out dtype must be float32, got {normalized.dtype}")
    np.multiply(frame_array, np.float32(1.0 / 255.0), out=normalized, casting="unsafe")
    return normalized


def _validated_frame(out: np.ndarray | None) -> np.ndarray:
    frame = np.zeros(DEBUG_OCCUPANCY_GRAY64_SHAPE, dtype=np.uint8) if out is None else np.asarray(out)
    if frame.shape != DEBUG_OCCUPANCY_GRAY64_SHAPE:
        raise ValueError(f"out must have shape {DEBUG_OCCUPANCY_GRAY64_SHAPE}, got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"out dtype must be uint8, got {frame.dtype}")
    return frame


def _source_size(snapshot: Mapping[str, Any]) -> float:
    game = snapshot.get("game")
    if not isinstance(game, Mapping):
        raise ValueError("snapshot must contain game mapping")
    size = float(game.get("size", 0.0))
    if size <= 0.0:
        raise ValueError(f"snapshot game.size must be positive, got {size}")
    return size


def _avatars(snapshot: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    avatars = snapshot.get("avatars")
    return _mapping_sequence(avatars, name="snapshot.avatars")


def _mapping_sequence(value: Any, *, name: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of mappings")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{name}[{index}] must be a mapping")
    return value


def _controlled_avatar_id(controlled_player_id: str | int | None) -> int | None:
    if controlled_player_id is None:
        return None
    if isinstance(controlled_player_id, str) and controlled_player_id.startswith("player_"):
        suffix = controlled_player_id.removeprefix("player_")
        try:
            return int(suffix) + 1
        except ValueError:
            return None
    try:
        return int(controlled_player_id)
    except (TypeError, ValueError):
        return None


def _controlled_player_index(controlled_player_id: str | int | None) -> int | None:
    if controlled_player_id is None:
        return None
    if isinstance(controlled_player_id, str) and controlled_player_id.startswith("player_"):
        suffix = controlled_player_id.removeprefix("player_")
        try:
            return int(suffix)
        except ValueError:
            return None
    try:
        return int(controlled_player_id)
    except (TypeError, ValueError):
        return None


def _avatar_id(avatar: Mapping[str, Any]) -> int | None:
    try:
        return int(avatar.get("id"))
    except (TypeError, ValueError):
        return None


def _mark_world_point(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    world_size: float,
    *,
    value: int,
) -> None:
    try:
        x = float(x_value)
        y = float(y_value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(x) or not np.isfinite(y):
        return
    px = int(np.clip(round((x / world_size) * 63.0), 0, 63))
    py = int(np.clip(round((y / world_size) * 63.0), 0, 63))
    canvas[py, px] = max(int(canvas[py, px]), int(value))


def _mark_world_points(
    canvas: np.ndarray,
    points: Sequence[Mapping[str, Any]],
    world_size: float,
    *,
    value: int,
) -> None:
    if not points:
        return
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        try:
            x = float(point.get("x"))
            y = float(point.get("y"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return

    x_array = np.asarray(xs, dtype=np.float64)
    y_array = np.asarray(ys, dtype=np.float64)
    px = np.clip(np.rint((x_array / world_size) * 63.0), 0, 63).astype(np.intp)
    py = np.clip(np.rint((y_array / world_size) * 63.0), 0, 63).astype(np.intp)
    np.maximum.at(canvas, (py, px), int(value))


def _mark_grid_cell(
    canvas: np.ndarray,
    cell_x: int,
    cell_y: int,
    grid_width: int,
    grid_height: int,
    *,
    value: int,
) -> None:
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("grid dimensions must be positive")
    px = int(np.clip(((cell_x + 0.5) / grid_width) * 64.0, 0, 63))
    py = int(np.clip(((cell_y + 0.5) / grid_height) * 64.0, 0, 63))
    canvas[py, px] = max(int(canvas[py, px]), int(value))


def _mark_rect_world_point(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    world_width: float,
    world_height: float,
    *,
    value: int,
) -> None:
    if world_width <= 0.0 or world_height <= 0.0:
        raise ValueError("world dimensions must be positive")
    try:
        x = float(x_value)
        y = float(y_value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(x) or not np.isfinite(y):
        return
    px = int(np.clip((x / world_width) * 64.0, 0, 63))
    py = int(np.clip((y / world_height) * 64.0, 0, 63))
    canvas[py, px] = max(int(canvas[py, px]), int(value))


__all__ = [
    "DEBUG_OCCUPANCY_GRAY64_CAVEAT",
    "DEBUG_OCCUPANCY_GRAY64_BROWSER_PIXEL_FIDELITY",
    "DEBUG_OCCUPANCY_GRAY64_COMPARISON_TARGET",
    "DEBUG_OCCUPANCY_GRAY64_FINAL_OBSERVATION_POLICY",
    "DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_POLICY",
    "DEBUG_OCCUPANCY_GRAY64_LABEL",
    "DEBUG_OCCUPANCY_GRAY64_PIXEL_FIDELITY_BLOCKER",
    "DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL",
    "DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH",
    "DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_ID",
    "DEBUG_OCCUPANCY_GRAY64_RESET_SOURCE",
    "DEBUG_OCCUPANCY_GRAY64_SOURCE_CLAIM_ID",
    "DEBUG_OCCUPANCY_GRAY64_SOURCE_STATE_FIELDS",
    "DEBUG_OCCUPANCY_GRAY64_FRAME_STACK_OWNER",
    "DEBUG_OCCUPANCY_GRAY64_PERSPECTIVE",
    "DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID",
    "DEBUG_OCCUPANCY_GRAY64_RENDER_DTYPE",
    "DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE",
    "DEBUG_OCCUPANCY_GRAY64_RAW_VALUE_RANGE",
    "DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH",
    "DEBUG_OCCUPANCY_GRAY64_SCHEMA_ID",
    "DEBUG_OCCUPANCY_GRAY64_SHAPE",
    "DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE",
    "DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY",
    "DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL",
    "DEBUG_OCCUPANCY_GRAY64_SURFACE",
    "DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL",
    "DEBUG_OCCUPANCY_GRAY64_USES_ALE",
    "DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE",
    "DebugOccupancyGray64FrameStack",
    "DebugOccupancyGray64Renderer",
    "debug_occupancy_gray64_metadata",
    "debug_occupancy_gray64_schema",
    "normalize_debug_occupancy_gray64_for_lightzero",
    "render_debug_occupancy_gray64",
    "render_debug_occupancy_gray64_from_env_state",
]
