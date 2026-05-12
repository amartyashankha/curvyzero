"""Prototype P=2 safe-palette perspective remap/copy strategies.

This is a render-lane scratch benchmark, not production code. It compares each
prototype against ``render_source_state_canvas_gray64_player_perspectives`` for
byte equality, then times the same synthetic P=2 safe-palette workloads at
L0/L1024/L4096 by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

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
    _draw_source_state_rgb_bonuses,
    _draw_source_state_rgb_heads,
    _player_rgb_values,
    _remap_rgb_palette_pixels,
    _render_source_state_rgb_body_circles_fast,
    _render_source_state_rgb_browser_lines,
    _rgb_palette_reuse_safe,
    _rgb_triplet,
    _row_index,
    _trusted_arrays,
    render_source_state_canvas_gray64_player_perspectives,
    rgb_canvas_like_to_gray64,
)
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (  # noqa: E402
    player_perspective_rgb_palette,
)


REPORT_SCHEMA_ID = "curvyzero_perspective_remap_prototype_bench/v0"
INVALID_TRAIL_RGB = (120, 120, 120)
DEFAULT_TRAIL_LENGTHS = (0, 1024, 4096)


@dataclass(frozen=True)
class Variant:
    name: str
    exact_when: str
    fallback_when: str
    run: Callable[["BenchState", "Scratch"], "VariantRun"]


@dataclass(frozen=True)
class VariantRun:
    frames: np.ndarray
    status: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class BenchState:
    state: dict[str, np.ndarray]
    row: int
    palettes: tuple[tuple[tuple[int, int, int], ...], ...]
    trail_render_mode: str
    bonus_render_mode: str


@dataclass(frozen=True)
class RenderContext:
    arrays: dict[str, np.ndarray]
    row: int
    palettes: tuple[tuple[tuple[int, int, int], ...], ...]
    base_colors: np.ndarray
    map_size: float
    trail_render_mode: str
    bonus_render_mode: str
    player_count: int


class Scratch:
    def __init__(self) -> None:
        size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
        self.frames = np.empty((2, *SOURCE_STATE_CANVAS_GRAY64_SHAPE), dtype=np.uint8)
        self.rgb_base = np.empty((size, size, 3), dtype=np.uint8)
        self.rgb_work = np.empty_like(self.rgb_base)
        self.luma_work = np.empty((size, size), dtype=np.uint8)
        self.packed = np.empty((size, size), dtype=np.uint32)
        self.mask = np.empty((size, size), dtype=bool)
        self.downsample = np.empty((64, 64), dtype=np.float32)
        self.identity = np.arange(256, dtype=np.uint8)
        self.lut = self.identity.copy()
        self.lut_rgb = np.empty((256, 3), dtype=np.uint8)


VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="production_helper",
        exact_when="production reference helper",
        fallback_when="n/a",
        run=lambda bench, scratch: _run_production(bench, scratch, status="exact"),
    ),
    Variant(
        name="rgb_mask_copy_once",
        exact_when=(
            "P=2 palette-reuse-safe RGB colors; remapped player copies base, "
            "identity player mutates base after remap"
        ),
        fallback_when="falls back to production helper when P!=2 or palette reuse is unsafe",
        run=lambda bench, scratch: _run_rgb_mask_copy_once(bench, scratch),
    ),
    Variant(
        name="packed_u32_copy_once",
        exact_when=(
            "P=2 palette-reuse-safe RGB colors; RGB24 is packed into uint32 keys "
            "for exact color masks, then identity player mutates base"
        ),
        fallback_when="falls back to production helper when P!=2 or palette reuse is unsafe",
        run=lambda bench, scratch: _run_packed_u32_copy_once(bench, scratch),
    ),
    Variant(
        name="gray_ch0_mask_copy_once",
        exact_when=(
            "P=2 safe grayscale palettes with unique first-channel player values; "
            "channel-0 masks are equivalent to full RGB masks"
        ),
        fallback_when="falls back unless safe grayscale/channel-0 uniqueness holds",
        run=lambda bench, scratch: _run_gray_ch0_mask_copy_once(bench, scratch),
    ),
    Variant(
        name="gray_ch0_lut_rgb",
        exact_when=(
            "P=2 safe grayscale palettes; first-channel LUT expands directly back "
            "to RGB before normal bonus/head draw and production downsample"
        ),
        fallback_when="falls back unless safe grayscale/channel-0 uniqueness holds",
        run=lambda bench, scratch: _run_gray_ch0_lut_rgb(bench, scratch),
    ),
    Variant(
        name="gray_luma_lut_no_bonus",
        exact_when=(
            "P=2 safe grayscale palettes and no active bonuses; first-channel LUT, "
            "direct luma head draw, and 11x11 area downsample are byte-exact"
        ),
        fallback_when="falls back when bonuses are active or grayscale/channel-0 safety fails",
        run=lambda bench, scratch: _run_gray_luma_lut_no_bonus(bench, scratch),
    ),
)


def make_synthetic_source_state(*, trail_length: int, seed: int) -> dict[str, np.ndarray]:
    """Build one P=2 source-state row with deterministic visible trails."""

    trail = _nonnegative_int(trail_length, "trail_length")
    rng = np.random.default_rng(int(seed))
    players = 2
    state: dict[str, np.ndarray] = {
        "tick": np.zeros(1, dtype=np.int32),
        "elapsed_ms": np.zeros(1, dtype=np.float64),
        "map_size": np.full(1, 64.0, dtype=np.float64),
        "present": np.ones((1, players), dtype=bool),
        "alive": np.ones((1, players), dtype=bool),
        "pos": np.zeros((1, players, 2), dtype=np.float64),
        "radius": np.full((1, players), 1.0, dtype=np.float64),
        "avatar_color": np.asarray([[0, 1]], dtype=np.int16),
        "body_active": np.zeros((1, trail), dtype=bool),
        "body_pos": np.zeros((1, trail, 2), dtype=np.float64),
        "body_radius": np.full((1, trail), 0.8, dtype=np.float64),
        "body_owner": np.full((1, trail), -1, dtype=np.int16),
        "body_num": np.full((1, trail), -1, dtype=np.int32),
        "body_write_cursor": np.full(1, trail, dtype=np.int32),
        "done": np.zeros(1, dtype=bool),
        "terminated": np.zeros(1, dtype=bool),
        "truncated": np.zeros(1, dtype=bool),
        "terminal_reason": np.zeros(1, dtype=np.int16),
    }

    if trail:
        owner_counts = (trail // 2, trail - (trail // 2))
        slot = 0
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        for player, count in enumerate(owner_counts):
            if count <= 0:
                continue
            progress = np.linspace(0.0, 1.0, count, dtype=np.float64)
            player_phase = phase + player * np.pi
            swirl = player_phase + progress * (4.0 * np.pi + player * 0.4)
            center_x = 32.0 + (8.0 if player == 0 else -8.0)
            center_y = 32.0
            x = center_x + 11.0 * np.cos(swirl) + 5.0 * (progress - 0.5)
            y = center_y + 11.0 * np.sin(swirl) - 4.0 * (progress - 0.5)
            stop = slot + count
            state["body_active"][0, slot:stop] = True
            state["body_pos"][0, slot:stop, 0] = np.clip(x, 2.0, 62.0)
            state["body_pos"][0, slot:stop, 1] = np.clip(y, 2.0, 62.0)
            state["body_radius"][0, slot:stop] = 0.75 + player * 0.1
            state["body_owner"][0, slot:stop] = player
            state["body_num"][0, slot:stop] = np.arange(count, dtype=np.int32)
            state["pos"][0, player] = state["body_pos"][0, stop - 1]
            slot = stop
    else:
        state["pos"][0, 0] = (22.0, 32.0)
        state["pos"][0, 1] = (42.0, 32.0)

    return state


def run_probe(
    *,
    trail_lengths: Sequence[int] = DEFAULT_TRAIL_LENGTHS,
    iterations: int = 3,
    warmup_iterations: int = 1,
    seed: int = 20260512,
) -> dict[str, Any]:
    checked_iterations = _positive_int(iterations, "iterations")
    checked_warmups = _nonnegative_int(warmup_iterations, "warmup_iterations")
    checked_lengths = tuple(_nonnegative_int(value, "trail_length") for value in trail_lengths)
    scratch = Scratch()

    workloads = []
    for index, trail_length in enumerate(checked_lengths):
        state = make_synthetic_source_state(
            trail_length=trail_length,
            seed=int(seed) + index,
        )
        palettes = tuple(
            player_perspective_rgb_palette(
                state,
                row=0,
                controlled_player=player,
                player_count=2,
            )
            for player in range(2)
        )
        bench = BenchState(
            state=state,
            row=0,
            palettes=palettes,
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            bonus_render_mode=BONUS_RENDER_MODE_DEFAULT,
        )
        workloads.append(
            _run_workload(
                bench,
                scratch,
                trail_length=trail_length,
                iterations=checked_iterations,
                warmup_iterations=checked_warmups,
            )
        )

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "config": {
            "player_count": 2,
            "safe_palette": "self/other grayscale RGB from player_perspective_rgb_palette",
            "trail_render_mode": TRAIL_RENDER_MODE_BROWSER_LINES,
            "trail_lengths": list(checked_lengths),
            "iterations": checked_iterations,
            "warmup_iterations": checked_warmups,
            "seed": int(seed),
        },
        "variants": [
            {
                "name": variant.name,
                "exact_when": variant.exact_when,
                "fallback_when": variant.fallback_when,
            }
            for variant in VARIANTS
        ],
        "workloads": workloads,
        "summary": _best_summary(workloads),
        "files_changed": ["scripts/prototype_perspective_remap_bench.py"],
    }


def _run_workload(
    bench: BenchState,
    scratch: Scratch,
    *,
    trail_length: int,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    reference = _run_production(bench, scratch, status="exact").frames.copy()
    variant_reports = []
    baseline_us = 0.0

    for variant in VARIANTS:
        first = variant.run(bench, scratch)
        parity = bool(np.array_equal(first.frames, reference))
        max_abs_diff = 0 if parity else int(
            np.max(
                np.abs(
                    first.frames.astype(np.int16, copy=False)
                    - reference.astype(np.int16, copy=False)
                )
            )
        )
        for _ in range(warmup_iterations):
            variant.run(bench, scratch)
        started = time.perf_counter()
        last = first
        for _ in range(iterations):
            last = variant.run(bench, scratch)
        elapsed = time.perf_counter() - started
        us_per_call = (elapsed / float(iterations)) * 1_000_000.0
        if variant.name == "production_helper":
            baseline_us = us_per_call
        variant_reports.append(
            {
                "name": variant.name,
                "status": first.status,
                "fallback_reason": first.fallback_reason,
                "parity": parity,
                "max_abs_diff": max_abs_diff,
                "us_per_call": us_per_call,
                "us_per_policy_row": us_per_call / 2.0,
                "speedup_vs_production": 1.0,
                "last_hash": _array_digest(last.frames),
            }
        )

    if baseline_us > 0.0:
        for report in variant_reports:
            report["speedup_vs_production"] = baseline_us / float(report["us_per_call"])

    return {
        "trail_length": int(trail_length),
        "reference_hash": _array_digest(reference),
        "active_bodies": int(np.count_nonzero(bench.state["body_active"])),
        "variants": variant_reports,
        "best_exact": _best_variant(variant_reports),
    }


def _run_production(bench: BenchState, scratch: Scratch, *, status: str) -> VariantRun:
    frames = render_source_state_canvas_gray64_player_perspectives(
        bench.state,
        row=bench.row,
        player_rgbs=bench.palettes,
        out=scratch.frames,
        rgb_base_out=scratch.rgb_base,
        rgb_work_out=scratch.rgb_work,
        trail_render_mode=bench.trail_render_mode,
        bonus_render_mode=bench.bonus_render_mode,
    )
    return VariantRun(frames=frames, status=status)


def _fallback_production(bench: BenchState, scratch: Scratch, reason: str) -> VariantRun:
    frames = _run_production(bench, scratch, status="fallback").frames
    return VariantRun(frames=frames, status="fallback", fallback_reason=reason)


def _run_rgb_mask_copy_once(bench: BenchState, scratch: Scratch) -> VariantRun:
    ctx = _context(bench)
    if not _reuse_safe_p2(ctx):
        return _fallback_production(bench, scratch, "P=2 palette reuse safety failed")
    _render_trail_base(ctx, scratch.rgb_base)
    _render_remapped_rgb_player(ctx, scratch, player=1, remap="rgb_mask")
    _render_identity_rgb_player(ctx, scratch, player=0)
    return VariantRun(frames=scratch.frames, status="exact")


def _run_packed_u32_copy_once(bench: BenchState, scratch: Scratch) -> VariantRun:
    ctx = _context(bench)
    if not _reuse_safe_p2(ctx):
        return _fallback_production(bench, scratch, "P=2 palette reuse safety failed")
    _render_trail_base(ctx, scratch.rgb_base)
    _pack_rgb24_to_u32(scratch.rgb_base, scratch.packed)
    _render_remapped_rgb_player(ctx, scratch, player=1, remap="packed_u32")
    _render_identity_rgb_player(ctx, scratch, player=0)
    return VariantRun(frames=scratch.frames, status="exact")


def _run_gray_ch0_mask_copy_once(bench: BenchState, scratch: Scratch) -> VariantRun:
    ctx = _context(bench)
    if not _gray_ch0_safe_p2(ctx):
        return _fallback_production(bench, scratch, "safe grayscale/channel-0 condition failed")
    _render_trail_base(ctx, scratch.rgb_base)
    _render_remapped_rgb_player(ctx, scratch, player=1, remap="gray_ch0_mask")
    _render_identity_rgb_player(ctx, scratch, player=0)
    return VariantRun(frames=scratch.frames, status="exact")


def _run_gray_ch0_lut_rgb(bench: BenchState, scratch: Scratch) -> VariantRun:
    ctx = _context(bench)
    if not _gray_ch0_safe_p2(ctx):
        return _fallback_production(bench, scratch, "safe grayscale/channel-0 condition failed")
    _render_trail_base(ctx, scratch.rgb_base)

    colors = _player_rgb_values(ctx.arrays, ctx.row, player_rgb=ctx.palettes[1])
    _fill_gray_rgb_lut(scratch, ctx.base_colors, colors)
    np.take(scratch.lut_rgb, scratch.rgb_base[:, :, 0], axis=0, out=scratch.rgb_work)
    _draw_source_state_rgb_bonuses(
        scratch.rgb_work,
        ctx.arrays,
        ctx.row,
        ctx.map_size,
        bonus_render_mode=ctx.bonus_render_mode,
    )
    _draw_source_state_rgb_heads(
        scratch.rgb_work,
        ctx.arrays,
        ctx.row,
        ctx.map_size,
        colors=colors,
    )
    rgb_canvas_like_to_gray64(scratch.rgb_work, out=scratch.frames[1])

    _render_identity_rgb_player(ctx, scratch, player=0)
    return VariantRun(frames=scratch.frames, status="exact")


def _run_gray_luma_lut_no_bonus(bench: BenchState, scratch: Scratch) -> VariantRun:
    ctx = _context(bench)
    if not _gray_ch0_safe_p2(ctx):
        return _fallback_production(bench, scratch, "safe grayscale/channel-0 condition failed")
    if _has_active_bonus(ctx):
        return _fallback_production(bench, scratch, "active bonuses require RGB sprite draw")
    _render_trail_base(ctx, scratch.rgb_base)

    colors = _player_rgb_values(ctx.arrays, ctx.row, player_rgb=ctx.palettes[1])
    _fill_gray_lut(scratch, ctx.base_colors, colors)
    np.take(scratch.lut, scratch.rgb_base[:, :, 0], out=scratch.luma_work)
    _draw_heads_luma(scratch.luma_work, ctx, colors=colors)
    _luma_canvas_to_gray64(scratch.luma_work, out=scratch.frames[1], scratch=scratch)

    luma_base = scratch.rgb_base[:, :, 0]
    _draw_heads_luma(luma_base, ctx, colors=ctx.base_colors)
    _luma_canvas_to_gray64(luma_base, out=scratch.frames[0], scratch=scratch)
    return VariantRun(frames=scratch.frames, status="exact")


def _context(bench: BenchState) -> RenderContext:
    arrays = _trusted_arrays(bench.state)
    row = _row_index(bench.row, arrays["tick"].shape[0])
    player_count = int(arrays["pos"].shape[1])
    base_colors = _player_rgb_values(arrays, row, player_rgb=bench.palettes[0])
    return RenderContext(
        arrays=arrays,
        row=row,
        palettes=bench.palettes,
        base_colors=base_colors,
        map_size=float(arrays["map_size"][row]),
        trail_render_mode=bench.trail_render_mode,
        bonus_render_mode=bench.bonus_render_mode,
        player_count=player_count,
    )


def _reuse_safe_p2(ctx: RenderContext) -> bool:
    return (
        ctx.player_count == 2
        and len(ctx.palettes) == 2
        and _rgb_palette_reuse_safe(
            ctx.base_colors,
            background_rgb=SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
            extra_reserved_colors=(INVALID_TRAIL_RGB,),
        )
    )


def _gray_ch0_safe_p2(ctx: RenderContext) -> bool:
    if not _reuse_safe_p2(ctx):
        return False
    reserved = (
        _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB),
        _rgb_triplet(INVALID_TRAIL_RGB),
    )
    all_colors: list[np.ndarray] = [*reserved, *[np.asarray(color) for color in ctx.base_colors]]
    for palette in ctx.palettes:
        colors = _player_rgb_values(ctx.arrays, ctx.row, player_rgb=palette)
        all_colors.extend(np.asarray(color) for color in colors)
    if not all(_is_grayscale_rgb(color) for color in all_colors):
        return False
    base_values = [int(color[0]) for color in ctx.base_colors]
    reserved_values = {int(color[0]) for color in reserved}
    return len(set(base_values)) == len(base_values) and not (set(base_values) & reserved_values)


def _render_trail_base(ctx: RenderContext, base: np.ndarray) -> None:
    base[:, :] = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)
    if ctx.trail_render_mode == TRAIL_RENDER_MODE_BROWSER_LINES:
        _render_source_state_rgb_browser_lines(
            base,
            ctx.arrays,
            ctx.row,
            ctx.map_size,
            colors=ctx.base_colors,
        )
    else:
        _render_source_state_rgb_body_circles_fast(
            base,
            ctx.arrays,
            ctx.row,
            ctx.map_size,
            colors=ctx.base_colors,
        )


def _render_remapped_rgb_player(
    ctx: RenderContext,
    scratch: Scratch,
    *,
    player: int,
    remap: str,
) -> None:
    colors = _player_rgb_values(ctx.arrays, ctx.row, player_rgb=ctx.palettes[player])
    scratch.rgb_work[:] = scratch.rgb_base
    if remap == "rgb_mask":
        _remap_rgb_palette_pixels(
            scratch.rgb_work,
            source=scratch.rgb_base,
            from_colors=ctx.base_colors,
            to_colors=colors,
        )
    elif remap == "packed_u32":
        _remap_with_packed_u32(
            scratch.rgb_work,
            packed=scratch.packed,
            mask=scratch.mask,
            from_colors=ctx.base_colors,
            to_colors=colors,
        )
    elif remap == "gray_ch0_mask":
        _remap_with_gray_ch0(
            scratch.rgb_work,
            source_ch0=scratch.rgb_base[:, :, 0],
            mask=scratch.mask,
            from_colors=ctx.base_colors,
            to_colors=colors,
        )
    else:
        raise ValueError(f"unknown remap mode: {remap}")
    _draw_source_state_rgb_bonuses(
        scratch.rgb_work,
        ctx.arrays,
        ctx.row,
        ctx.map_size,
        bonus_render_mode=ctx.bonus_render_mode,
    )
    _draw_source_state_rgb_heads(
        scratch.rgb_work,
        ctx.arrays,
        ctx.row,
        ctx.map_size,
        colors=colors,
    )
    rgb_canvas_like_to_gray64(scratch.rgb_work, out=scratch.frames[player])


def _render_identity_rgb_player(ctx: RenderContext, scratch: Scratch, *, player: int) -> None:
    colors = _player_rgb_values(ctx.arrays, ctx.row, player_rgb=ctx.palettes[player])
    _draw_source_state_rgb_bonuses(
        scratch.rgb_base,
        ctx.arrays,
        ctx.row,
        ctx.map_size,
        bonus_render_mode=ctx.bonus_render_mode,
    )
    _draw_source_state_rgb_heads(
        scratch.rgb_base,
        ctx.arrays,
        ctx.row,
        ctx.map_size,
        colors=colors,
    )
    rgb_canvas_like_to_gray64(scratch.rgb_base, out=scratch.frames[player])


def _pack_rgb24_to_u32(rgb: np.ndarray, packed: np.ndarray) -> None:
    np.copyto(packed, rgb[:, :, 0], casting="unsafe")
    packed |= rgb[:, :, 1].astype(np.uint32) << np.uint32(8)
    packed |= rgb[:, :, 2].astype(np.uint32) << np.uint32(16)


def _remap_with_packed_u32(
    target: np.ndarray,
    *,
    packed: np.ndarray,
    mask: np.ndarray,
    from_colors: np.ndarray,
    to_colors: np.ndarray,
) -> None:
    for from_rgb, to_rgb in zip(from_colors, to_colors, strict=True):
        from_triplet = _rgb_triplet(from_rgb)
        to_triplet = _rgb_triplet(to_rgb)
        if bool(np.array_equal(from_triplet, to_triplet)):
            continue
        from_key = _rgb_u32_key(from_triplet)
        np.equal(packed, from_key, out=mask)
        if bool(mask.any()):
            target[mask] = to_triplet


def _remap_with_gray_ch0(
    target: np.ndarray,
    *,
    source_ch0: np.ndarray,
    mask: np.ndarray,
    from_colors: np.ndarray,
    to_colors: np.ndarray,
) -> None:
    for from_rgb, to_rgb in zip(from_colors, to_colors, strict=True):
        from_triplet = _rgb_triplet(from_rgb)
        to_triplet = _rgb_triplet(to_rgb)
        if bool(np.array_equal(from_triplet, to_triplet)):
            continue
        np.equal(source_ch0, np.uint8(from_triplet[0]), out=mask)
        if bool(mask.any()):
            target[mask] = to_triplet


def _fill_gray_lut(scratch: Scratch, from_colors: np.ndarray, to_colors: np.ndarray) -> None:
    np.copyto(scratch.lut, scratch.identity)
    for from_rgb, to_rgb in zip(from_colors, to_colors, strict=True):
        scratch.lut[int(from_rgb[0])] = np.uint8(to_rgb[0])


def _fill_gray_rgb_lut(scratch: Scratch, from_colors: np.ndarray, to_colors: np.ndarray) -> None:
    scratch.lut_rgb[:, 0] = scratch.identity
    scratch.lut_rgb[:, 1] = scratch.identity
    scratch.lut_rgb[:, 2] = scratch.identity
    for from_rgb, to_rgb in zip(from_colors, to_colors, strict=True):
        scratch.lut_rgb[int(from_rgb[0])] = _rgb_triplet(to_rgb)


def _draw_heads_luma(canvas: np.ndarray, ctx: RenderContext, *, colors: np.ndarray) -> None:
    for player in range(ctx.player_count):
        if not bool(ctx.arrays["present"][ctx.row, player]):
            continue
        if not bool(ctx.arrays["alive"][ctx.row, player]):
            continue
        _draw_world_circle_luma(
            canvas,
            ctx.arrays["pos"][ctx.row, player, 0],
            ctx.arrays["pos"][ctx.row, player, 1],
            ctx.arrays["radius"][ctx.row, player],
            ctx.map_size,
            value=int(colors[player, 0]),
        )


def _draw_world_circle_luma(
    canvas: np.ndarray,
    x_value: Any,
    y_value: Any,
    radius_value: Any,
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

    size = int(canvas.shape[0])
    radius_px = int(max(0, np.ceil((radius / map_size) * float(size))))
    px = int(np.clip(np.rint((x / map_size) * float(size - 1)), 0, size - 1))
    py = int(np.clip(np.rint((y / map_size) * float(size - 1)), 0, size - 1))
    if radius_px == 0:
        canvas[py, px] = np.uint8(value)
        return

    x0 = max(0, px - radius_px)
    x1 = min(size - 1, px + radius_px)
    y0 = max(0, py - radius_px)
    y1 = min(size - 1, py + radius_px)
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius_px**2
    if bool(mask.any()):
        canvas[y0 : y1 + 1, x0 : x1 + 1][mask] = np.uint8(value)


def _luma_canvas_to_gray64(luma: np.ndarray, *, out: np.ndarray, scratch: Scratch) -> None:
    target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
    source_size = int(luma.shape[0])
    ratio = source_size // target_size
    np.mean(
        luma.reshape(target_size, ratio, target_size, ratio),
        axis=(1, 3),
        dtype=np.float32,
        out=scratch.downsample,
    )
    np.rint(scratch.downsample, out=scratch.downsample)
    np.clip(scratch.downsample, 0.0, 255.0, out=scratch.downsample)
    np.copyto(out[0], scratch.downsample, casting="unsafe")


def _has_active_bonus(ctx: RenderContext) -> bool:
    return "bonus_active" in ctx.arrays and bool(np.any(ctx.arrays["bonus_active"][ctx.row]))


def _is_grayscale_rgb(color: Sequence[int] | np.ndarray) -> bool:
    triplet = _rgb_triplet(color)
    return int(triplet[0]) == int(triplet[1]) == int(triplet[2])


def _rgb_u32_key(rgb: Sequence[int] | np.ndarray) -> np.uint32:
    triplet = _rgb_triplet(rgb)
    return np.uint32(int(triplet[0]) | (int(triplet[1]) << 8) | (int(triplet[2]) << 16))


def _best_variant(reports: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        report
        for report in reports
        if report["name"] != "production_helper"
        and report["status"] == "exact"
        and bool(report["parity"])
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda report: float(report["us_per_call"]))
    return {
        "name": best["name"],
        "us_per_call": best["us_per_call"],
        "us_per_policy_row": best["us_per_policy_row"],
        "speedup_vs_production": best["speedup_vs_production"],
    }


def _best_summary(workloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for workload in workloads:
        baseline = next(
            report
            for report in workload["variants"]
            if report["name"] == "production_helper"
        )
        best = workload["best_exact"]
        summary.append(
            {
                "trail_length": workload["trail_length"],
                "baseline_us_per_call": baseline["us_per_call"],
                "best_exact": best,
                "all_variants_parity": all(
                    bool(report["parity"]) for report in workload["variants"]
                ),
            }
        )
    return summary


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    try:
        return tuple(_nonnegative_int(item, "trail_length") for item in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_arg(value: str) -> int:
    try:
        return _positive_int(int(value), value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative_arg(value: str) -> int:
    try:
        return _nonnegative_int(int(value), value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-lengths", type=_parse_int_csv, default=DEFAULT_TRAIL_LENGTHS)
    parser.add_argument("--iterations", type=_positive_arg, default=3)
    parser.add_argument("--warmup-iterations", type=_nonnegative_arg, default=1)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(
        trail_lengths=args.trail_lengths,
        iterations=int(args.iterations),
        warmup_iterations=int(args.warmup_iterations),
        seed=int(args.seed),
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_plain(report)
    return 0


def _print_plain(report: dict[str, Any]) -> None:
    config = report["config"]
    print(
        f"{report['schema_id']} P={config['player_count']} "
        f"mode={config['trail_render_mode']} iterations={config['iterations']} "
        f"warmup={config['warmup_iterations']}"
    )
    print("variants:")
    for variant in report["variants"]:
        print(f"  {variant['name']}: exact_when={variant['exact_when']}")
        print(f"    fallback={variant['fallback_when']}")
    print("timings:")
    for workload in report["workloads"]:
        print(f"  L{workload['trail_length']} active_bodies={workload['active_bodies']}")
        baseline = next(
            variant
            for variant in workload["variants"]
            if variant["name"] == "production_helper"
        )
        for variant in workload["variants"]:
            status = variant["status"]
            parity = "OK" if variant["parity"] else f"FAIL max_abs_diff={variant['max_abs_diff']}"
            fallback = (
                f" fallback_reason={variant['fallback_reason']}"
                if variant["fallback_reason"]
                else ""
            )
            print(
                "    "
                f"{variant['name']}: {variant['us_per_call']:.1f} us/call "
                f"({variant['us_per_policy_row']:.1f} us/policy-row) "
                f"speedup={variant['speedup_vs_production']:.2f}x "
                f"status={status} parity={parity}{fallback}"
            )
        best = workload["best_exact"]
        if best is None:
            print(
                f"    baseline_vs_best: baseline={baseline['us_per_call']:.1f} us, "
                "no exact parity variant"
            )
        else:
            print(
                f"    baseline_vs_best: baseline={baseline['us_per_call']:.1f} us, "
                f"best={best['name']} {best['us_per_call']:.1f} us "
                f"({best['speedup_vs_production']:.2f}x)"
            )
    print("summary:")
    for item in report["summary"]:
        best = item["best_exact"]
        best_text = (
            "none"
            if best is None
            else f"{best['name']} {best['us_per_call']:.1f} us {best['speedup_vs_production']:.2f}x"
        )
        print(
            f"  L{item['trail_length']}: baseline={item['baseline_us_per_call']:.1f} us "
            f"best={best_text} parity_all={item['all_variants_parity']}"
        )
    print(f"files_changed={', '.join(report['files_changed'])}")


if __name__ == "__main__":
    raise SystemExit(main())
