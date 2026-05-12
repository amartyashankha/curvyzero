"""Prototype per-owner incremental trail-layer benchmark for browser_lines.

This is deliberately a render-only experiment. It does not modify production
renderer modules; instead it imports the current renderer and selected private
helpers so the prototype can answer one narrow question: can a persistent
per-owner trail layer match the current full browser_lines renderer byte-for-byte
while avoiding long trail redraws on append-only visual_trail growth?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.vector_visual_observation import (  # noqa: E402
    BONUS_RENDER_MODE_DEFAULT,
    SOURCE_STATE_CANVAS_GRAY64_SHAPE,
    SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    TRAIL_RENDER_MODE_BROWSER_LINES,
    _body_owner_rgb,
    _browser_line_body_order,
    _browser_line_owner_draw_order,
    _compatible_rgb_out,
    _draw_ordered_browser_line_path_rgb,
    _draw_source_state_rgb_bonuses,
    _draw_source_state_rgb_heads,
    _player_rgb_values,
    _rgb_frame_size,
    _rgb_triplet,
    _row_index,
    _trusted_arrays,
    _validated_rgb_frame,
    render_source_state_canvas_gray64,
    render_source_state_rgb_canvas_like,
    rgb_canvas_like_to_gray64,
)


REPORT_SCHEMA_ID = "curvyzero_incremental_trail_layer_prototype_report/v0"
LAYER_SENTINEL_RGB = (1, 2, 3)


@dataclass
class TrailLayerCacheStats:
    render_calls: int = 0
    cache_hits: int = 0
    rebuilds: int = 0
    incremental_updates: int = 0
    fallback_full_renders: int = 0
    cursor_regression_rebuilds: int = 0
    unsupported_full_renders: int = 0
    prefix_mutation_rebuilds: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "render_calls": self.render_calls,
            "cache_hits": self.cache_hits,
            "rebuilds": self.rebuilds,
            "incremental_updates": self.incremental_updates,
            "fallback_full_renders": self.fallback_full_renders,
            "cursor_regression_rebuilds": self.cursor_regression_rebuilds,
            "unsupported_full_renders": self.unsupported_full_renders,
            "prefix_mutation_rebuilds": self.prefix_mutation_rebuilds,
        }


@dataclass
class _OwnerTrailLayer:
    owner: int
    rgb: np.ndarray
    mask: np.ndarray
    slots: np.ndarray
    positions: np.ndarray
    radii: np.ndarray
    break_before: np.ndarray

    @classmethod
    def empty(cls, *, owner: int, frame_size: int, sentinel: np.ndarray) -> "_OwnerTrailLayer":
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


@dataclass
class _TrailRecords:
    slots: np.ndarray
    positions: np.ndarray
    radii: np.ndarray
    owners: np.ndarray
    break_before: np.ndarray


class IncrementalTrailLayerRenderer:
    """Prototype browser_lines renderer with persistent per-owner trail layers."""

    def __init__(self, *, frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE) -> None:
        self.frame_size = _rgb_frame_size(frame_size)
        self.sentinel = _rgb_triplet(LAYER_SENTINEL_RGB)
        self.layers: dict[int, _OwnerTrailLayer] = {}
        self.last_cursor: int | None = None
        self.last_prefix: _TrailRecords | None = None
        self.last_map_size: float | None = None
        self.last_colors_key: tuple[tuple[int, int, int], ...] | None = None
        self.stats = TrailLayerCacheStats()

    def reset(self) -> None:
        self.layers.clear()
        self.last_cursor = None
        self.last_prefix = None
        self.last_map_size = None
        self.last_colors_key = None

    def render_rgb(
        self,
        state: Mapping[str, np.ndarray],
        *,
        row: int = 0,
        out: np.ndarray | None = None,
        player_rgb: Sequence[Sequence[int]] | None = None,
        background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
        bonus_render_mode: str = BONUS_RENDER_MODE_DEFAULT,
    ) -> np.ndarray:
        """Render RGB via cached trail layers, falling back to full renderer when unsafe."""

        self.stats.render_calls += 1
        frame = (
            np.empty((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            if out is None
            else _validated_rgb_frame(out, frame_size=self.frame_size)
        )
        arrays = _trusted_arrays(state)
        row_index = _row_index(row, arrays["tick"].shape[0])
        map_size = float(arrays["map_size"][row_index])
        colors = _player_rgb_values(arrays, row_index, player_rgb=player_rgb)
        colors_key = _colors_key(colors)
        background = _rgb_triplet(background_rgb)

        if not self._is_supported_visual_state(arrays, row_index, colors=colors):
            self.stats.unsupported_full_renders += 1
            self.stats.fallback_full_renders += 1
            self.reset()
            return render_source_state_rgb_canvas_like(
                state,
                row=row_index,
                out=frame,
                frame_size=self.frame_size,
                player_rgb=player_rgb,
                background_rgb=background_rgb,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
                bonus_render_mode=bonus_render_mode,
            )

        records = _visual_trail_records(arrays, row_index)
        cursor = int(arrays["visual_trail_write_cursor"][row_index])
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
            assert self.last_prefix is not None
            changed = self._append(records, map_size=map_size, colors=colors)
            if changed:
                self.stats.incremental_updates += 1
            else:
                self.stats.cache_hits += 1

        self.last_cursor = cursor
        self.last_prefix = _copy_records(records)
        self.last_map_size = map_size
        self.last_colors_key = colors_key

        frame[:, :] = background
        self._composite_layers(frame)
        _draw_source_state_rgb_bonuses(
            frame,
            arrays,
            row_index,
            map_size,
            bonus_render_mode=bonus_render_mode,
        )
        _draw_source_state_rgb_heads(frame, arrays, row_index, map_size, colors=colors)
        return frame

    def render_canvas_gray64(
        self,
        state: Mapping[str, np.ndarray],
        *,
        row: int = 0,
        out: np.ndarray | None = None,
        rgb_out: np.ndarray | None = None,
        player_rgb: Sequence[Sequence[int]] | None = None,
        background_rgb: Sequence[int] = SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
        bonus_render_mode: str = BONUS_RENDER_MODE_DEFAULT,
    ) -> np.ndarray:
        rgb = self.render_rgb(
            state,
            row=row,
            out=_compatible_rgb_out(rgb_out, frame_size=self.frame_size),
            player_rgb=player_rgb,
            background_rgb=background_rgb,
            bonus_render_mode=bonus_render_mode,
        )
        return rgb_canvas_like_to_gray64(rgb, out=out)

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
        if not bool(np.any(active[row, :cursor])):
            return False
        reserved = tuple(int(value) for value in self.sentinel)
        owner_colors = [tuple(int(value) for value in _body_owner_rgb(-1, colors))]
        owner_colors.extend(tuple(int(value) for value in color) for color in colors)
        return reserved not in owner_colors

    def _rebuild_reason(
        self,
        *,
        records: _TrailRecords,
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

    def _rebuild(self, records: _TrailRecords, *, map_size: float, colors: np.ndarray) -> None:
        self.layers.clear()
        for owner in _browser_line_owner_draw_order(records.owners):
            owner_records = _owner_records(records, owner)
            layer = _OwnerTrailLayer.empty(
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
                _refresh_layer_mask(layer, sentinel=self.sentinel)
            _replace_layer_records(layer, owner_records)
            self.layers[int(owner)] = layer

    def _append(self, records: _TrailRecords, *, map_size: float, colors: np.ndarray) -> bool:
        changed = False
        current_owners = set(int(owner) for owner in np.unique(records.owners))
        for owner in current_owners:
            owner_records = _owner_records(records, owner)
            layer = self.layers.get(owner)
            if layer is None:
                layer = _OwnerTrailLayer.empty(
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
                return True
            if not _owner_prefix_matches(layer, owner_records, old_count):
                self._rebuild(records, map_size=map_size, colors=colors)
                self.stats.prefix_mutation_rebuilds += 1
                self.stats.rebuilds += 1
                return True
            draw_start = _incremental_draw_start(owner_records, old_count)
            _draw_ordered_browser_line_path_rgb(
                layer.rgb,
                owner_records.positions[draw_start:],
                owner_records.radii[draw_start:],
                map_size,
                color=_body_owner_rgb(owner, colors),
                break_before=owner_records.break_before[draw_start:],
            )
            _refresh_layer_mask(layer, sentinel=self.sentinel)
            _replace_layer_records(layer, owner_records)
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


def _visual_trail_records(arrays: Mapping[str, np.ndarray], row: int) -> _TrailRecords:
    capacity = arrays["visual_trail_active"].shape[1]
    cursor = int(np.clip(int(arrays["visual_trail_write_cursor"][row]), 0, capacity))
    slots = np.flatnonzero(arrays["visual_trail_active"][row, :cursor]).astype(np.int64)
    if slots.size == 0:
        return _TrailRecords(
            slots=slots,
            positions=np.empty((0, 2), dtype=np.float64),
            radii=np.empty(0, dtype=np.float64),
            owners=np.empty(0, dtype=np.int64),
            break_before=np.empty(0, dtype=bool),
        )
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
        return _TrailRecords(
            slots=np.empty(0, dtype=np.int64),
            positions=np.empty((0, 2), dtype=np.float64),
            radii=np.empty(0, dtype=np.float64),
            owners=np.empty(0, dtype=np.int64),
            break_before=np.empty(0, dtype=bool),
        )
    return _TrailRecords(
        slots=slots[finite].astype(np.int64, copy=True),
        positions=positions[finite].astype(np.float64, copy=True),
        radii=radii[finite].astype(np.float64, copy=True),
        owners=owners[finite].astype(np.int64, copy=True),
        break_before=breaks[finite].astype(bool, copy=True),
    )


def _owner_records(records: _TrailRecords, owner: int) -> _TrailRecords:
    owner_mask = records.owners == int(owner)
    slots = records.slots[owner_mask]
    order = _browser_line_body_order(slots, body_nums=None)
    return _TrailRecords(
        slots=slots[order].astype(np.int64, copy=True),
        positions=records.positions[owner_mask][order].astype(np.float64, copy=True),
        radii=records.radii[owner_mask][order].astype(np.float64, copy=True),
        owners=records.owners[owner_mask][order].astype(np.int64, copy=True),
        break_before=records.break_before[owner_mask][order].astype(bool, copy=True),
    )


def _copy_records(records: _TrailRecords) -> _TrailRecords:
    return _TrailRecords(
        slots=records.slots.copy(),
        positions=records.positions.copy(),
        radii=records.radii.copy(),
        owners=records.owners.copy(),
        break_before=records.break_before.copy(),
    )


def _replace_layer_records(layer: _OwnerTrailLayer, records: _TrailRecords) -> None:
    layer.slots = records.slots.copy()
    layer.positions = records.positions.copy()
    layer.radii = records.radii.copy()
    layer.break_before = records.break_before.copy()


def _owner_prefix_matches(
    layer: _OwnerTrailLayer,
    records: _TrailRecords,
    old_count: int,
) -> bool:
    return bool(
        np.array_equal(records.slots[:old_count], layer.slots)
        and np.array_equal(records.positions[:old_count], layer.positions)
        and np.array_equal(records.radii[:old_count], layer.radii)
        and np.array_equal(records.break_before[:old_count], layer.break_before)
    )


def _incremental_draw_start(records: _TrailRecords, old_count: int) -> int:
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


def _refresh_layer_mask(layer: _OwnerTrailLayer, *, sentinel: np.ndarray) -> None:
    layer.mask[:, :] = np.any(layer.rgb != sentinel, axis=2)


def _colors_key(colors: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(int(channel) for channel in color) for color in colors)


def make_visual_trail_state(
    *,
    points: Sequence[tuple[float, float]],
    owners: Sequence[int],
    radii: Sequence[float],
    break_before: Sequence[bool] | None = None,
    cursor: int | None = None,
    player_count: int = 2,
) -> dict[str, np.ndarray]:
    if not (len(points) == len(owners) == len(radii)):
        raise ValueError("points, owners, and radii must have the same length")
    if break_before is not None and len(break_before) != len(points):
        raise ValueError("break_before must match points length")
    capacity = max(1, len(points))
    player_count = max(1, int(player_count))
    state: dict[str, np.ndarray] = {
        "tick": np.asarray([0], dtype=np.int32),
        "elapsed_ms": np.asarray([0.0], dtype=np.float64),
        "map_size": np.asarray([64.0], dtype=np.float64),
        "present": np.zeros((1, player_count), dtype=bool),
        "alive": np.zeros((1, player_count), dtype=bool),
        "pos": np.zeros((1, player_count, 2), dtype=np.float64),
        "radius": np.full((1, player_count), 0.8, dtype=np.float64),
        "avatar_color": np.tile(np.arange(player_count, dtype=np.int16), (1, 1)),
        "body_active": np.zeros((1, capacity), dtype=bool),
        "body_pos": np.zeros((1, capacity, 2), dtype=np.float64),
        "body_radius": np.zeros((1, capacity), dtype=np.float64),
        "body_owner": np.full((1, capacity), -1, dtype=np.int16),
        "body_write_cursor": np.asarray([0], dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "terminated": np.asarray([False], dtype=bool),
        "truncated": np.asarray([False], dtype=bool),
        "terminal_reason": np.asarray([0], dtype=np.int16),
        "visual_trail_active": np.zeros((1, capacity), dtype=bool),
        "visual_trail_pos": np.zeros((1, capacity, 2), dtype=np.float64),
        "visual_trail_radius": np.zeros((1, capacity), dtype=np.float64),
        "visual_trail_owner": np.full((1, capacity), -1, dtype=np.int16),
        "visual_trail_break_before": np.zeros((1, capacity), dtype=bool),
        "visual_trail_write_cursor": np.asarray(
            [len(points) if cursor is None else int(cursor)],
            dtype=np.int32,
        ),
    }
    if points:
        count = len(points)
        state["visual_trail_active"][0, :count] = True
        state["visual_trail_pos"][0, :count] = np.asarray(points, dtype=np.float64)
        state["visual_trail_radius"][0, :count] = np.asarray(radii, dtype=np.float64)
        state["visual_trail_owner"][0, :count] = np.asarray(owners, dtype=np.int16)
        if break_before is not None:
            state["visual_trail_break_before"][0, :count] = np.asarray(
                break_before,
                dtype=bool,
            )
    return state


def parity_case_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "owner_overlap",
            "points": [(16.0, 32.0), (48.0, 32.0), (32.0, 16.0), (32.0, 48.0)],
            "owners": [1, 1, 0, 0],
            "radii": [1.1, 1.1, 1.1, 1.1],
            "break_before": [False, False, False, False],
            "cursors": [4],
        },
        {
            "name": "break_before",
            "points": [(16.0, 20.0), (48.0, 20.0), (16.0, 44.0), (48.0, 44.0)],
            "owners": [0, 0, 0, 0],
            "radii": [0.9, 0.9, 0.9, 0.9],
            "break_before": [False, True, True, False],
            "cursors": [4],
        },
        {
            "name": "radius_changes",
            "points": [(12.0, 32.0), (26.0, 32.0), (42.0, 32.0), (56.0, 32.0)],
            "owners": [0, 0, 0, 0],
            "radii": [0.5, 1.4, 1.4, 0.7],
            "break_before": [False, False, False, False],
            "cursors": [4],
        },
        {
            "name": "cursor_append_growth",
            "points": [
                (8.0, 8.0),
                (20.0, 12.0),
                (32.0, 16.0),
                (44.0, 20.0),
                (56.0, 24.0),
                (56.0, 40.0),
                (44.0, 44.0),
                (32.0, 48.0),
            ],
            "owners": [0, 0, 0, 0, 1, 1, 1, 1],
            "radii": [0.8, 0.8, 1.2, 1.2, 0.7, 0.7, 0.7, 1.1],
            "break_before": [False, False, False, False, True, False, False, False],
            "cursors": [2, 3, 4, 5, 6, 7, 8],
        },
    ]


def run_parity_cases() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in parity_case_specs():
        state = make_visual_trail_state(
            points=spec["points"],
            owners=spec["owners"],
            radii=spec["radii"],
            break_before=spec["break_before"],
            cursor=spec["cursors"][0],
        )
        renderer = IncrementalTrailLayerRenderer()
        rgb_out = np.empty(
            (
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                3,
            ),
            dtype=np.uint8,
        )
        cached_rgb_out = np.empty_like(rgb_out)
        gray_out = np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
        cached_gray_out = np.empty_like(gray_out)
        rgb_equal = True
        gray64_equal = True
        first_mismatch: dict[str, Any] | None = None
        for frame_index, cursor in enumerate(spec["cursors"]):
            state["visual_trail_write_cursor"][0] = int(cursor)
            full_rgb = render_source_state_rgb_canvas_like(
                state,
                out=rgb_out,
                frame_size=SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            )
            cached_rgb = renderer.render_rgb(state, out=cached_rgb_out)
            full_gray = render_source_state_canvas_gray64(
                state,
                out=gray_out,
                rgb_out=rgb_out,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            )
            cached_gray = renderer.render_canvas_gray64(
                state,
                out=cached_gray_out,
                rgb_out=cached_rgb_out,
            )
            frame_rgb_equal = bool(np.array_equal(full_rgb, cached_rgb))
            frame_gray_equal = bool(np.array_equal(full_gray, cached_gray))
            rgb_equal = rgb_equal and frame_rgb_equal
            gray64_equal = gray64_equal and frame_gray_equal
            if first_mismatch is None and (not frame_rgb_equal or not frame_gray_equal):
                first_mismatch = {
                    "frame_index": frame_index,
                    "cursor": int(cursor),
                    "rgb_equal": frame_rgb_equal,
                    "gray64_equal": frame_gray_equal,
                    "full_rgb_hash": _array_digest(full_rgb),
                    "cached_rgb_hash": _array_digest(cached_rgb),
                    "full_gray64_hash": _array_digest(full_gray),
                    "cached_gray64_hash": _array_digest(cached_gray),
                }
        results.append(
            {
                "name": spec["name"],
                "frames": len(spec["cursors"]),
                "rgb_equal": rgb_equal,
                "gray64_equal": gray64_equal,
                "parity": rgb_equal and gray64_equal,
                "first_mismatch": first_mismatch,
                "cache_stats": renderer.stats.as_dict(),
            }
        )
    return results


def make_benchmark_state(*, trail_length: int, append_step: int) -> dict[str, np.ndarray]:
    points: list[tuple[float, float]] = []
    owners: list[int] = []
    radii: list[float] = []
    breaks: list[bool] = []
    count = max(1, int(trail_length))
    for index in range(count):
        owner = index % 2
        owner_progress = index // 2
        angle = owner_progress * 0.085 + owner * np.pi
        ring = 18.0 + 4.0 * np.sin(owner_progress * 0.013)
        x = 32.0 + ring * np.cos(angle)
        y = 32.0 + ring * np.sin(angle)
        points.append((float(np.clip(x, 2.0, 62.0)), float(np.clip(y, 2.0, 62.0))))
        owners.append(owner)
        radii.append(0.7 + 0.2 * ((owner_progress // 64) % 3))
        breaks.append(owner_progress % max(8, append_step * 4) == 0)
    return make_visual_trail_state(
        points=points,
        owners=owners,
        radii=radii,
        break_before=breaks,
        cursor=min(count, max(1, append_step)),
        player_count=2,
    )


def run_benchmark(
    *,
    trail_lengths: Sequence[int],
    append_step: int,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for trail_length in trail_lengths:
        state = make_benchmark_state(trail_length=int(trail_length), append_step=append_step)
        cells.append(
            _run_benchmark_cell(
                state=state,
                trail_length=int(trail_length),
                append_step=append_step,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
            )
        )
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "plan_id": "incremental_visual_trail_layer_v0",
        "prototype": {
            "renderer": "IncrementalTrailLayerRenderer",
            "trail_mode": TRAIL_RENDER_MODE_BROWSER_LINES,
            "visual_trail_arrays_required": True,
            "frame_size": SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
            "fallback_policy": "rebuild on cursor regression/prefix mutation; full render on unsupported state",
        },
        "config": {
            "trail_lengths": [int(value) for value in trail_lengths],
            "append_step": int(append_step),
            "iterations": int(iterations),
            "warmup_iterations": int(warmup_iterations),
        },
        "parity_cases": run_parity_cases(),
        "cells": cells,
    }


def _run_benchmark_cell(
    *,
    state: dict[str, np.ndarray],
    trail_length: int,
    append_step: int,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    cursors = list(range(max(1, append_step), trail_length + 1, append_step))
    if cursors[-1] != trail_length:
        cursors.append(trail_length)
    for _ in range(warmup_iterations):
        _time_sequence(
            state=state,
            cursors=cursors,
            compare_parity=False,
        )
    result = _time_sequence(
        state=state,
        cursors=cursors,
        compare_parity=True,
        repeat=iterations,
    )
    full_sec = result["full_sec"]
    cached_sec = result["cached_sec"]
    frames = len(cursors) * iterations
    return {
        "label": f"visual_trail_append_L{trail_length}_step{append_step}",
        "trail_length": int(trail_length),
        "append_step": int(append_step),
        "frames": int(frames),
        "final_cursor": int(cursors[-1]),
        "parity": result["parity"],
        "first_mismatch": result["first_mismatch"],
        "timing_sec": {
            "full": full_sec,
            "cached": cached_sec,
        },
        "timing_us_per_frame": {
            "full": _unit_us(full_sec, frames),
            "cached": _unit_us(cached_sec, frames),
        },
        "speedup": float(full_sec / cached_sec) if cached_sec > 0.0 else 0.0,
        "cache_stats": result["cache_stats"],
        "hashes": result["hashes"],
    }


def _time_sequence(
    *,
    state: dict[str, np.ndarray],
    cursors: Sequence[int],
    compare_parity: bool,
    repeat: int = 1,
) -> dict[str, Any]:
    renderer = IncrementalTrailLayerRenderer()
    rgb_full = np.empty(
        (
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
            3,
        ),
        dtype=np.uint8,
    )
    rgb_cached = np.empty_like(rgb_full)
    gray_full = np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
    gray_cached = np.empty_like(gray_full)
    full_sec = 0.0
    cached_sec = 0.0
    parity = True
    first_mismatch: dict[str, Any] | None = None
    last_full_hash: str | None = None
    last_cached_hash: str | None = None
    for iteration in range(int(repeat)):
        renderer.reset()
        for frame_index, cursor in enumerate(cursors):
            state["tick"][0] += np.int32(1)
            state["elapsed_ms"][0] += np.float64(1000.0 / 60.0)
            state["visual_trail_write_cursor"][0] = int(cursor)

            full_started = time.perf_counter()
            full = render_source_state_canvas_gray64(
                state,
                out=gray_full,
                rgb_out=rgb_full,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            )
            full_sec += time.perf_counter() - full_started

            cached_started = time.perf_counter()
            cached = renderer.render_canvas_gray64(
                state,
                out=gray_cached,
                rgb_out=rgb_cached,
            )
            cached_sec += time.perf_counter() - cached_started

            if compare_parity:
                frame_equal = bool(np.array_equal(full, cached))
                parity = parity and frame_equal
                if not frame_equal and first_mismatch is None:
                    first_mismatch = {
                        "iteration": int(iteration),
                        "frame_index": int(frame_index),
                        "cursor": int(cursor),
                        "full_gray64_hash": _array_digest(full),
                        "cached_gray64_hash": _array_digest(cached),
                    }
            last_full_hash = _array_digest(full)
            last_cached_hash = _array_digest(cached)
    return {
        "full_sec": full_sec,
        "cached_sec": cached_sec,
        "parity": parity,
        "first_mismatch": first_mismatch,
        "cache_stats": renderer.stats.as_dict(),
        "hashes": {
            "last_full_gray64": last_full_hash,
            "last_cached_gray64": last_cached_hash,
        },
    }


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


def _unit_us(seconds: float, count: int) -> float:
    return float((seconds / count) * 1_000_000.0) if count > 0 else 0.0


def _parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _positive_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-lengths", type=_parse_int_csv, default=[256, 1024, 4096])
    parser.add_argument("--append-step", type=_positive_arg, default=16)
    parser.add_argument("--iterations", type=_positive_arg, default=2)
    parser.add_argument("--warmup-iterations", type=_nonnegative_arg, default=1)
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(
        trail_lengths=args.trail_lengths,
        append_step=int(args.append_step),
        iterations=int(args.iterations),
        warmup_iterations=int(args.warmup_iterations),
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_plain(report)
    return 0 if _report_has_parity(report) else 1


def _report_has_parity(report: Mapping[str, Any]) -> bool:
    return bool(
        all(case["parity"] for case in report["parity_cases"])
        and all(cell["parity"] for cell in report["cells"])
    )


def _print_plain(report: Mapping[str, Any]) -> None:
    print(f"{report['schema_id']} plan={report['plan_id']}")
    print("parity cases:")
    for case in report["parity_cases"]:
        stats = case["cache_stats"]
        print(
            " ".join(
                (
                    f"  {case['name']}:",
                    f"parity={case['parity']}",
                    f"frames={case['frames']}",
                    f"rebuilds={stats['rebuilds']}",
                    f"increments={stats['incremental_updates']}",
                    f"fallbacks={stats['fallback_full_renders']}",
                )
            )
        )
    print("bench cells:")
    for cell in report["cells"]:
        timing = cell["timing_us_per_frame"]
        stats = cell["cache_stats"]
        print(
            " ".join(
                (
                    f"  {cell['label']}:",
                    f"parity={cell['parity']}",
                    f"full_us={timing['full']:.2f}",
                    f"cached_us={timing['cached']:.2f}",
                    f"speedup={cell['speedup']:.2f}x",
                    f"rebuilds={stats['rebuilds']}",
                    f"increments={stats['incremental_updates']}",
                    f"fallbacks={stats['fallback_full_renders']}",
                )
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
