"""Prototype dirty-block gray64 updates for CurvyTron canvas rendering.

This is not production renderer code. It uses the current full RGB renderer as
the oracle, computes exact dirty 64x64 blocks from RGB diffs, and measures the
best-case value of redownsampling only those blocks. It also measures a
conservative geometry-bounds mask for appended visual_trail segments and moving
heads; production integration would need to harden that path and fall back on
unsupported state changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_render_lane_microbench import make_synthetic_source_state  # noqa: E402
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_CANVAS_GRAY64_SHAPE,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    TRAIL_RENDER_MODE_BROWSER_LINES,
    SourceStateGray64DownsampleScratch,
    render_source_state_canvas_gray64,
    rgb_canvas_like_to_gray64,
)
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (  # noqa: E402
    player_perspective_rgb_palette,
)


REPORT_SCHEMA_ID = "curvyzero_dirty_block_render_prototype/v0"
PLAN_ID = "dirty_block_gray64_update_v0"
FILES_CHANGED = ("scripts/prototype_dirty_block_render_bench.py",)
TARGET_SIZE = int(SOURCE_STATE_CANVAS_GRAY64_SHAPE[1])
SOURCE_SIZE = int(SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE)
RATIO = SOURCE_SIZE // TARGET_SIZE
LUMA_R = np.float32(0.299)
LUMA_G = np.float32(0.587)
LUMA_B = np.float32(0.114)


class _DirtyDownsampleScratch:
    """Reusable 11x11 work buffers for exact dirty-cell updates."""

    def __init__(self, ratio: int = RATIO) -> None:
        self.ratio = int(ratio)
        self.luma = np.empty((self.ratio, self.ratio), dtype=np.float32)
        self.temp = np.empty_like(self.luma)
        self.down_one = np.empty((1, 1), dtype=np.float32)


def _positive(value: int, *, name: str) -> int:
    checked = int(value)
    if checked < 1:
        raise ValueError(f"{name} must be >= 1")
    return checked


def _nonnegative(value: int, *, name: str) -> int:
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be >= 0")
    return checked


def _dirty_blocks_from_rgb_diff(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    changed_pixels = np.any(previous != current, axis=2)
    return changed_pixels.reshape(TARGET_SIZE, RATIO, TARGET_SIZE, RATIO).any(axis=(1, 3))


def _mark_pixel_bbox_blocks(
    blocks: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    pad_px: float = 2.0,
) -> None:
    px0 = max(0, int(np.floor(min_x - pad_px)))
    py0 = max(0, int(np.floor(min_y - pad_px)))
    px1 = min(SOURCE_SIZE - 1, int(np.ceil(max_x + pad_px)))
    py1 = min(SOURCE_SIZE - 1, int(np.ceil(max_y + pad_px)))
    if px0 > px1 or py0 > py1:
        return
    blocks[py0 // RATIO : (py1 // RATIO) + 1, px0 // RATIO : (px1 // RATIO) + 1] = True


def _mark_world_circle_blocks(
    blocks: np.ndarray,
    *,
    pos: np.ndarray,
    radius: float,
    map_size: float,
) -> None:
    if not np.isfinite(pos).all() or not np.isfinite(radius):
        return
    center = pos.astype(np.float64, copy=False) * (SOURCE_SIZE / map_size)
    radius_px = float(radius) * (SOURCE_SIZE / map_size)
    _mark_pixel_bbox_blocks(
        blocks,
        min_x=float(center[0] - radius_px),
        min_y=float(center[1] - radius_px),
        max_x=float(center[0] + radius_px),
        max_y=float(center[1] + radius_px),
    )


def _mark_world_segment_blocks(
    blocks: np.ndarray,
    *,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    map_size: float,
) -> None:
    if not np.isfinite(start).all() or not np.isfinite(end).all() or not np.isfinite(radius):
        return
    scale = SOURCE_SIZE / map_size
    start_px = start.astype(np.float64, copy=False) * scale
    end_px = end.astype(np.float64, copy=False) * scale
    radius_px = float(radius) * scale
    _mark_pixel_bbox_blocks(
        blocks,
        min_x=float(min(start_px[0], end_px[0]) - radius_px),
        min_y=float(min(start_px[1], end_px[1]) - radius_px),
        max_x=float(max(start_px[0], end_px[0]) + radius_px),
        max_y=float(max(start_px[1], end_px[1]) + radius_px),
    )


def _previous_owner_position(
    state: dict[str, np.ndarray],
    *,
    row: int,
    owner: int,
    before_cursor: int,
) -> np.ndarray | None:
    if before_cursor <= 0:
        return None
    active = state["visual_trail_active"][row, :before_cursor]
    owners = state["visual_trail_owner"][row, :before_cursor]
    slots = np.flatnonzero(active & (owners == owner))
    if slots.size == 0:
        return None
    return np.asarray(state["visual_trail_pos"][row, slots[-1]], dtype=np.float64)


def _geometry_dirty_blocks(
    state: dict[str, np.ndarray],
    *,
    row: int,
    previous_cursor: int,
    previous_head_positions: np.ndarray,
) -> np.ndarray:
    blocks = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
    map_size = float(state["map_size"][row])
    player_count = int(state["pos"].shape[1])
    current_cursor = int(state["visual_trail_write_cursor"][row])
    for slot in range(previous_cursor, current_cursor):
        if not bool(state["visual_trail_active"][row, slot]):
            continue
        owner = int(state["visual_trail_owner"][row, slot])
        pos = np.asarray(state["visual_trail_pos"][row, slot], dtype=np.float64)
        radius = float(state["visual_trail_radius"][row, slot])
        previous = _previous_owner_position(
            state,
            row=row,
            owner=owner,
            before_cursor=slot,
        )
        if previous is None or bool(state["visual_trail_break_before"][row, slot]):
            _mark_world_circle_blocks(blocks, pos=pos, radius=radius, map_size=map_size)
        else:
            _mark_world_segment_blocks(
                blocks,
                start=previous,
                end=pos,
                radius=radius,
                map_size=map_size,
            )
    for player in range(player_count):
        radius = float(state["radius"][row, player])
        _mark_world_circle_blocks(
            blocks,
            pos=np.asarray(previous_head_positions[player], dtype=np.float64),
            radius=radius,
            map_size=map_size,
        )
        _mark_world_circle_blocks(
            blocks,
            pos=np.asarray(state["pos"][row, player], dtype=np.float64),
            radius=radius,
            map_size=map_size,
        )
    return blocks


def _redownsample_dirty_blocks(
    rgb: np.ndarray,
    out_gray_chw: np.ndarray,
    dirty_blocks: np.ndarray,
    scratch: _DirtyDownsampleScratch,
) -> int:
    """Update ``out_gray_chw`` exactly for the dirty 11x11 source blocks."""

    dirty_y, dirty_x = np.nonzero(dirty_blocks)
    gray = out_gray_chw[0]
    for block_y, block_x in zip(dirty_y, dirty_x, strict=True):
        y0 = int(block_y) * scratch.ratio
        x0 = int(block_x) * scratch.ratio
        patch = rgb[y0 : y0 + scratch.ratio, x0 : x0 + scratch.ratio]
        np.multiply(patch[:, :, 0], LUMA_R, out=scratch.luma)
        np.multiply(patch[:, :, 1], LUMA_G, out=scratch.temp)
        np.add(scratch.luma, scratch.temp, out=scratch.luma)
        np.multiply(patch[:, :, 2], LUMA_B, out=scratch.temp)
        np.add(scratch.luma, scratch.temp, out=scratch.luma)
        blocks = scratch.luma.reshape(1, scratch.ratio, 1, scratch.ratio)
        np.mean(blocks, axis=(1, 3), dtype=np.float32, out=scratch.down_one)
        np.rint(scratch.down_one, out=scratch.down_one)
        np.clip(
            scratch.down_one,
            np.float32(0.0),
            np.float32(255.0),
            out=scratch.down_one,
        )
        np.copyto(
            gray[int(block_y) : int(block_y) + 1, int(block_x) : int(block_x) + 1],
            scratch.down_one,
            casting="unsafe",
        )
    return int(dirty_y.size)


def _append_visual_trail_points(
    state: dict[str, np.ndarray],
    *,
    step: int,
    rng: np.random.Generator,
) -> None:
    batch_size = int(state["visual_trail_active"].shape[0])
    capacity = int(state["visual_trail_active"].shape[1])
    player_count = int(state["pos"].shape[1])
    map_size = float(state["map_size"][0])
    for row in range(batch_size):
        cursor = int(state["visual_trail_write_cursor"][row])
        for player in range(player_count):
            if cursor >= capacity:
                break
            previous_slots = np.flatnonzero(
                state["visual_trail_active"][row, :cursor]
                & (state["visual_trail_owner"][row, :cursor] == player)
            )
            if previous_slots.size:
                previous = state["visual_trail_pos"][row, previous_slots[-1]]
                break_before = False
            else:
                angle = (2.0 * np.pi * player / player_count) + row * 0.17
                previous = np.array(
                    [
                        32.0 + 12.0 * np.cos(angle),
                        32.0 + 12.0 * np.sin(angle),
                    ],
                    dtype=np.float64,
                )
                break_before = True
            angle = (
                0.33 * float(step)
                + 0.19 * float(row)
                + (2.0 * np.pi * player / player_count)
            )
            delta = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            delta *= rng.uniform(0.45, 0.9)
            position = np.clip(previous + delta, 1.0, map_size - 1.0)
            state["visual_trail_active"][row, cursor] = True
            state["visual_trail_pos"][row, cursor] = position
            state["visual_trail_radius"][row, cursor] = state["radius"][row, player]
            state["visual_trail_owner"][row, cursor] = player
            state["visual_trail_break_before"][row, cursor] = break_before
            state["pos"][row, player] = position
            cursor += 1
        state["visual_trail_write_cursor"][row] = cursor
        state["body_write_cursor"][row] = cursor


def _initial_state(
    *,
    batch_size: int,
    player_count: int,
    initial_trail_length: int,
    append_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    capacity = initial_trail_length + append_steps * player_count
    state = make_synthetic_source_state(
        batch_size=batch_size,
        player_count=player_count,
        trail_length=capacity,
        bonus_count=0,
        trail_source="visual",
        seed=seed,
    )
    if initial_trail_length < capacity:
        state["body_active"][:, initial_trail_length:] = False
        state["visual_trail_active"][:, initial_trail_length:] = False
        state["body_owner"][:, initial_trail_length:] = -1
        state["visual_trail_owner"][:, initial_trail_length:] = -1
    state["body_write_cursor"][:] = initial_trail_length
    state["visual_trail_write_cursor"][:] = initial_trail_length
    return state


def _render_oracle_gray64(
    state: dict[str, np.ndarray],
    *,
    row: int,
    player: int,
    rgb_out: np.ndarray,
    gray_out: np.ndarray,
) -> np.ndarray:
    palette = player_perspective_rgb_palette(
        state,
        row=row,
        controlled_player=player,
        player_count=int(state["pos"].shape[1]),
    )
    return render_source_state_canvas_gray64(
        state,
        row=row,
        out=gray_out,
        rgb_out=rgb_out,
        player_rgb=palette,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )


def run_probe(
    *,
    batch_size: int,
    player_count: int,
    initial_trail_length: int,
    append_steps: int,
    measured_steps: int,
    seed: int,
) -> dict[str, Any]:
    batch = _positive(batch_size, name="batch_size")
    players = _positive(player_count, name="player_count")
    initial = _nonnegative(initial_trail_length, name="initial_trail_length")
    warm_append = _nonnegative(append_steps, name="append_steps")
    measured = _positive(measured_steps, name="measured_steps")
    rng = np.random.default_rng(int(seed))
    state = _initial_state(
        batch_size=batch,
        player_count=players,
        initial_trail_length=initial,
        append_steps=warm_append + measured,
        seed=seed,
    )

    frame_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    previous_rgb = np.empty((batch, players, frame_size, frame_size, 3), dtype=np.uint8)
    previous_gray = np.empty((batch, players, *SOURCE_STATE_CANVAS_GRAY64_SHAPE), dtype=np.uint8)
    current_rgb = np.empty((frame_size, frame_size, 3), dtype=np.uint8)
    full_gray = np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
    scratch = SourceStateGray64DownsampleScratch(frame_size)
    dirty_scratch = _DirtyDownsampleScratch()

    for row in range(batch):
        for player in range(players):
            _render_oracle_gray64(
                state,
                row=row,
                player=player,
                rgb_out=previous_rgb[row, player],
                gray_out=previous_gray[row, player],
            )

    for step in range(warm_append):
        _append_visual_trail_points(state, step=step, rng=rng)
        for row in range(batch):
            for player in range(players):
                _render_oracle_gray64(
                    state,
                    row=row,
                    player=player,
                    rgb_out=previous_rgb[row, player],
                    gray_out=previous_gray[row, player],
                )

    dirty_block_counts: list[int] = []
    geometry_block_counts: list[int] = []
    missed_geometry_block_counts: list[int] = []
    extra_geometry_block_counts: list[int] = []
    parity_failures = 0
    geometry_parity_failures = 0
    max_abs_diff = 0
    geometry_max_abs_diff = 0
    render_oracle_sec = 0.0
    dirty_mask_sec = 0.0
    geometry_mask_sec = 0.0
    full_downsample_sec = 0.0
    dirty_downsample_sec = 0.0
    geometry_downsample_sec = 0.0
    update_count = 0

    for step in range(measured):
        previous_cursors = state["visual_trail_write_cursor"].copy()
        previous_heads = state["pos"].copy()
        geometry_gray = previous_gray.copy()
        _append_visual_trail_points(state, step=warm_append + step, rng=rng)
        for row in range(batch):
            for player in range(players):
                started = time.perf_counter()
                _render_oracle_gray64(
                    state,
                    row=row,
                    player=player,
                    rgb_out=current_rgb,
                    gray_out=full_gray,
                )
                render_oracle_sec += time.perf_counter() - started

                started = time.perf_counter()
                dirty_blocks = _dirty_blocks_from_rgb_diff(previous_rgb[row, player], current_rgb)
                dirty_mask_sec += time.perf_counter() - started

                started = time.perf_counter()
                geometry_blocks = _geometry_dirty_blocks(
                    state,
                    row=row,
                    previous_cursor=int(previous_cursors[row]),
                    previous_head_positions=previous_heads[row],
                )
                geometry_mask_sec += time.perf_counter() - started

                started = time.perf_counter()
                rgb_canvas_like_to_gray64(current_rgb, out=full_gray, scratch=scratch)
                full_downsample_sec += time.perf_counter() - started

                started = time.perf_counter()
                dirty_count = _redownsample_dirty_blocks(
                    current_rgb,
                    previous_gray[row, player],
                    dirty_blocks,
                    dirty_scratch,
                )
                dirty_downsample_sec += time.perf_counter() - started

                started = time.perf_counter()
                geometry_dirty_count = _redownsample_dirty_blocks(
                    current_rgb,
                    geometry_gray[row, player],
                    geometry_blocks,
                    dirty_scratch,
                )
                geometry_downsample_sec += time.perf_counter() - started

                if not np.array_equal(previous_gray[row, player], full_gray):
                    parity_failures += 1
                    diff = np.abs(
                        previous_gray[row, player].astype(np.int16)
                        - full_gray.astype(np.int16)
                    )
                    max_abs_diff = max(max_abs_diff, int(diff.max()))
                if not np.array_equal(geometry_gray[row, player], full_gray):
                    geometry_parity_failures += 1
                    diff = np.abs(
                        geometry_gray[row, player].astype(np.int16)
                        - full_gray.astype(np.int16)
                    )
                    geometry_max_abs_diff = max(geometry_max_abs_diff, int(diff.max()))
                previous_rgb[row, player] = current_rgb
                dirty_block_counts.append(dirty_count)
                geometry_block_counts.append(geometry_dirty_count)
                missed_geometry_block_counts.append(
                    int(np.count_nonzero(dirty_blocks & ~geometry_blocks))
                )
                extra_geometry_block_counts.append(
                    int(np.count_nonzero(geometry_blocks & ~dirty_blocks))
                )
                update_count += 1

    dirty_counts = np.asarray(dirty_block_counts, dtype=np.float64)
    geometry_counts = np.asarray(geometry_block_counts, dtype=np.float64)
    missed_counts = np.asarray(missed_geometry_block_counts, dtype=np.float64)
    extra_counts = np.asarray(extra_geometry_block_counts, dtype=np.float64)
    total_blocks = 64 * 64
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "covered_cases": {
            "player_count": players,
            "bonus_count": 0,
            "trail_render_mode": TRAIL_RENDER_MODE_BROWSER_LINES,
            "oracle": "render_source_state_canvas_gray64",
            "dirty_blocks_source": "rgb_diff_oracle_exact_measurement",
            "geometry_dirty_blocks_source": (
                "conservative bbox from appended visual_trail segments and "
                "old/new live heads"
            ),
            "redownsample": "exact 11x11 block luma mean/rint update",
            "unsupported": [
                "first frame/reset without previous gray64",
                "active bonus sprites",
                "exact raster dirty-mask derivation",
                "reset/wrap/prefix mutation fallback",
                "production cache integration",
            ],
        },
        "config": {
            "batch_size": batch,
            "player_count": players,
            "initial_trail_length": initial,
            "warm_append_steps": warm_append,
            "measured_steps": measured,
            "updates": update_count,
            "seed": int(seed),
        },
        "dirty_blocks": {
            "total_blocks_per_frame": total_blocks,
            "mean": float(dirty_counts.mean()) if dirty_counts.size else 0.0,
            "median": float(np.median(dirty_counts)) if dirty_counts.size else 0.0,
            "max": int(dirty_counts.max()) if dirty_counts.size else 0,
            "p90": float(np.percentile(dirty_counts, 90)) if dirty_counts.size else 0.0,
            "mean_fraction": float(dirty_counts.mean() / total_blocks)
            if dirty_counts.size
            else 0.0,
            "max_fraction": float(dirty_counts.max() / total_blocks)
            if dirty_counts.size
            else 0.0,
        },
        "geometry_dirty_blocks": {
            "mean": float(geometry_counts.mean()) if geometry_counts.size else 0.0,
            "median": float(np.median(geometry_counts)) if geometry_counts.size else 0.0,
            "max": int(geometry_counts.max()) if geometry_counts.size else 0,
            "p90": float(np.percentile(geometry_counts, 90)) if geometry_counts.size else 0.0,
            "mean_fraction": float(geometry_counts.mean() / total_blocks)
            if geometry_counts.size
            else 0.0,
            "missed_rgb_diff_blocks_total": int(missed_counts.sum()) if missed_counts.size else 0,
            "missed_rgb_diff_blocks_max": int(missed_counts.max()) if missed_counts.size else 0,
            "extra_blocks_mean": float(extra_counts.mean()) if extra_counts.size else 0.0,
            "extra_blocks_max": int(extra_counts.max()) if extra_counts.size else 0,
        },
        "timing_sec": {
            "render_oracle_full_gray64": render_oracle_sec,
            "dirty_mask_from_rgb_diff": dirty_mask_sec,
            "geometry_dirty_mask": geometry_mask_sec,
            "full_downsample": full_downsample_sec,
            "dirty_block_redownsample": dirty_downsample_sec,
            "geometry_block_redownsample": geometry_downsample_sec,
        },
        "timing_us_per_update": {
            "render_oracle_full_gray64": render_oracle_sec / update_count * 1e6,
            "dirty_mask_from_rgb_diff": dirty_mask_sec / update_count * 1e6,
            "geometry_dirty_mask": geometry_mask_sec / update_count * 1e6,
            "full_downsample": full_downsample_sec / update_count * 1e6,
            "dirty_block_redownsample": dirty_downsample_sec / update_count * 1e6,
            "geometry_block_redownsample": geometry_downsample_sec / update_count * 1e6,
        },
        "speedup": {
            "dirty_redownsample_vs_full_downsample": (
                full_downsample_sec / dirty_downsample_sec
                if dirty_downsample_sec > 0.0
                else None
            ),
            "geometry_redownsample_vs_full_downsample": (
                full_downsample_sec / geometry_downsample_sec
                if geometry_downsample_sec > 0.0
                else None
            ),
        },
        "parity": {
            "failures": int(parity_failures),
            "max_abs_diff": int(max_abs_diff),
            "geometry_failures": int(geometry_parity_failures),
            "geometry_max_abs_diff": int(geometry_max_abs_diff),
        },
        "files_changed": [path for path in FILES_CHANGED if (REPO_ROOT / path).exists()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--player-count", type=int, default=2)
    parser.add_argument("--initial-trail-length", type=int, default=1024)
    parser.add_argument("--warm-append-steps", type=int, default=8)
    parser.add_argument("--measured-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_probe(
        batch_size=args.batch_size,
        player_count=args.player_count,
        initial_trail_length=args.initial_trail_length,
        append_steps=args.warm_append_steps,
        measured_steps=args.measured_steps,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(
        "dirty-block prototype "
        f"B{args.batch_size}/P{args.player_count}/L{args.initial_trail_length} "
        f"steps={args.measured_steps}"
    )
    print(json.dumps(report["dirty_blocks"], indent=2, sort_keys=True))
    print(json.dumps(report["geometry_dirty_blocks"], indent=2, sort_keys=True))
    print(json.dumps(report["timing_us_per_update"], indent=2, sort_keys=True))
    print(json.dumps(report["speedup"], indent=2, sort_keys=True))
    print(json.dumps(report["parity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
