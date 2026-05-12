"""Prototype RGB canvas-like to gray64 downsample implementations.

This is intentionally render-lane scratch work, not production code. It compares
candidate implementations against the production ``rgb_canvas_like_to_gray64``
byte output on random 704x704 RGB frames and RGB frames emitted by the source
state renderer.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_CANVAS_GRAY64_SHAPE,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    TRAIL_RENDER_MODE_BROWSER_LINES,
    render_source_state_rgb_canvas_like,
    rgb_canvas_like_to_gray64,
)


REPORT_SCHEMA_ID = "curvyzero_downsample_fast_prototype_report/v0"
TARGET_SIZE = int(SOURCE_STATE_CANVAS_GRAY64_SHAPE[1])
LUMA_R = np.float32(0.299)
LUMA_G = np.float32(0.587)
LUMA_B = np.float32(0.114)
INTEGER_LUMA_WEIGHTS = (299, 587, 114)
FILES_CHANGED = ("scripts/prototype_downsample_fast_bench.py",)

Exactness = Literal["production", "exact", "non-exact"]
FrameSetName = Literal["random", "renderer", "mixed"]
DownsampleFunc = Callable[
    [np.ndarray],
    np.ndarray,
]


@dataclass(frozen=True)
class Variant:
    name: str
    exactness: Exactness
    description: str
    uses_scratch: bool
    func: Callable[
        [np.ndarray, np.ndarray | None, "DownsampleScratch | None"],
        np.ndarray,
    ]


class DownsampleScratch:
    """Reusable work buffers for the prototype variants."""

    def __init__(self, source_size: int) -> None:
        if source_size < TARGET_SIZE or source_size % TARGET_SIZE != 0:
            raise ValueError(
                f"source_size must be {TARGET_SIZE} or an integer multiple of it"
            )
        ratio = source_size // TARGET_SIZE
        self.source_size = int(source_size)
        self.ratio = int(ratio)
        self.gray = np.empty((source_size, source_size), dtype=np.float32)
        self.temp = np.empty_like(self.gray)
        self.down = np.empty((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        self.sum_float = np.empty((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        self.total_u64 = np.empty((TARGET_SIZE, TARGET_SIZE), dtype=np.uint64)
        self.sum_u64 = np.empty((TARGET_SIZE, TARGET_SIZE), dtype=np.uint64)
        self.quot_u64 = np.empty((TARGET_SIZE, TARGET_SIZE), dtype=np.uint64)
        self.rem_u64 = np.empty((TARGET_SIZE, TARGET_SIZE), dtype=np.uint64)
        self.round_mask = np.empty((TARGET_SIZE, TARGET_SIZE), dtype=bool)
        self.stripe = np.empty((ratio, source_size), dtype=np.float32)
        self.stripe_temp = np.empty_like(self.stripe)


def downsample_production_baseline(
    rgb: np.ndarray,
    out: np.ndarray | None = None,
    scratch: DownsampleScratch | None = None,
) -> np.ndarray:
    del scratch
    return rgb_canvas_like_to_gray64(rgb, out=out)


def downsample_exact_float32_full_scratch(
    rgb: np.ndarray,
    out: np.ndarray | None = None,
    scratch: DownsampleScratch | None = None,
) -> np.ndarray:
    """Order-preserving float32 luma with reusable full-frame scratch buffers."""

    rgb_array, ratio, frame = _validated_call(rgb, out)
    work = _scratch_for(scratch, rgb_array.shape[0])

    np.multiply(rgb_array[:, :, 0], LUMA_R, out=work.gray)
    np.multiply(rgb_array[:, :, 1], LUMA_G, out=work.temp)
    np.add(work.gray, work.temp, out=work.gray)
    np.multiply(rgb_array[:, :, 2], LUMA_B, out=work.temp)
    np.add(work.gray, work.temp, out=work.gray)

    blocks = work.gray.reshape(TARGET_SIZE, ratio, TARGET_SIZE, ratio)
    np.mean(blocks, axis=(1, 3), dtype=np.float32, out=work.down)
    np.rint(work.down, out=work.down)
    np.clip(work.down, np.float32(0.0), np.float32(255.0), out=work.down)
    np.copyto(frame[0], work.down, casting="unsafe")
    return frame


def downsample_exact_float32_stripe_scratch(
    rgb: np.ndarray,
    out: np.ndarray | None = None,
    scratch: DownsampleScratch | None = None,
) -> np.ndarray:
    """Order-preserving float32 luma using one source block-row at a time."""

    rgb_array, ratio, frame = _validated_call(rgb, out)
    work = _scratch_for(scratch, rgb_array.shape[0])

    for target_row in range(TARGET_SIZE):
        source_rows = rgb_array[target_row * ratio : (target_row + 1) * ratio]
        stripe = work.stripe[:ratio, : rgb_array.shape[0]]
        stripe_temp = work.stripe_temp[:ratio, : rgb_array.shape[0]]
        np.multiply(source_rows[:, :, 0], LUMA_R, out=stripe)
        np.multiply(source_rows[:, :, 1], LUMA_G, out=stripe_temp)
        np.add(stripe, stripe_temp, out=stripe)
        np.multiply(source_rows[:, :, 2], LUMA_B, out=stripe_temp)
        np.add(stripe, stripe_temp, out=stripe)
        stripe_blocks = stripe.reshape(ratio, TARGET_SIZE, ratio)
        np.mean(
            stripe_blocks,
            axis=(0, 2),
            dtype=np.float32,
            out=work.down[target_row],
        )

    np.rint(work.down, out=work.down)
    np.clip(work.down, np.float32(0.0), np.float32(255.0), out=work.down)
    np.copyto(frame[0], work.down, casting="unsafe")
    return frame


def downsample_nonexact_float32_channel_blocksum(
    rgb: np.ndarray,
    out: np.ndarray | None = None,
    scratch: DownsampleScratch | None = None,
) -> np.ndarray:
    """Non-exact: sum channels per block first, then apply float32 luma weights."""

    rgb_array, ratio, frame = _validated_call(rgb, out)
    work = _scratch_for(scratch, rgb_array.shape[0])
    blocks = rgb_array.reshape(TARGET_SIZE, ratio, TARGET_SIZE, ratio, 3)
    block_area = np.float32(ratio * ratio)

    np.sum(blocks[:, :, :, :, 0], axis=(1, 3), dtype=np.float32, out=work.down)
    np.multiply(work.down, LUMA_R / block_area, out=work.down)

    np.sum(blocks[:, :, :, :, 1], axis=(1, 3), dtype=np.float32, out=work.sum_float)
    np.multiply(work.sum_float, LUMA_G / block_area, out=work.sum_float)
    np.add(work.down, work.sum_float, out=work.down)

    np.sum(blocks[:, :, :, :, 2], axis=(1, 3), dtype=np.float32, out=work.sum_float)
    np.multiply(work.sum_float, LUMA_B / block_area, out=work.sum_float)
    np.add(work.down, work.sum_float, out=work.down)

    np.rint(work.down, out=work.down)
    np.clip(work.down, np.float32(0.0), np.float32(255.0), out=work.down)
    np.copyto(frame[0], work.down, casting="unsafe")
    return frame


def downsample_nonexact_integer_weight_blocksum(
    rgb: np.ndarray,
    out: np.ndarray | None = None,
    scratch: DownsampleScratch | None = None,
) -> np.ndarray:
    """Non-exact: integer 299/587/114 block sums with round-to-even rational divide."""

    rgb_array, ratio, frame = _validated_call(rgb, out)
    work = _scratch_for(scratch, rgb_array.shape[0])
    blocks = rgb_array.reshape(TARGET_SIZE, ratio, TARGET_SIZE, ratio, 3)

    np.sum(blocks[:, :, :, :, 0], axis=(1, 3), dtype=np.uint64, out=work.total_u64)
    np.multiply(work.total_u64, np.uint64(INTEGER_LUMA_WEIGHTS[0]), out=work.total_u64)

    np.sum(blocks[:, :, :, :, 1], axis=(1, 3), dtype=np.uint64, out=work.sum_u64)
    np.multiply(work.sum_u64, np.uint64(INTEGER_LUMA_WEIGHTS[1]), out=work.sum_u64)
    np.add(work.total_u64, work.sum_u64, out=work.total_u64)

    np.sum(blocks[:, :, :, :, 2], axis=(1, 3), dtype=np.uint64, out=work.sum_u64)
    np.multiply(work.sum_u64, np.uint64(INTEGER_LUMA_WEIGHTS[2]), out=work.sum_u64)
    np.add(work.total_u64, work.sum_u64, out=work.total_u64)

    denominator = np.uint64(ratio * ratio * 1000)
    np.floor_divide(work.total_u64, denominator, out=work.quot_u64)
    np.remainder(work.total_u64, denominator, out=work.rem_u64)
    np.multiply(work.rem_u64, np.uint64(2), out=work.rem_u64)
    np.greater(work.rem_u64, denominator, out=work.round_mask)
    ties = work.rem_u64 == denominator
    odd = (work.quot_u64 & np.uint64(1)) == np.uint64(1)
    np.logical_or(work.round_mask, ties & odd, out=work.round_mask)
    np.copyto(work.total_u64, work.quot_u64)
    np.add(work.total_u64, np.uint64(1), out=work.total_u64, where=work.round_mask)

    np.copyto(frame[0], work.total_u64, casting="unsafe")
    return frame


VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="production_baseline",
        exactness="production",
        description="Production rgb_canvas_like_to_gray64",
        uses_scratch=False,
        func=downsample_production_baseline,
    ),
    Variant(
        name="exact_float32_full_scratch",
        exactness="exact",
        description="Same float32 luma/mean order with reusable full-frame gray/temp buffers",
        uses_scratch=True,
        func=downsample_exact_float32_full_scratch,
    ),
    Variant(
        name="exact_float32_stripe_scratch",
        exactness="exact",
        description="Same float32 luma/mean math using one 11-row source stripe at a time",
        uses_scratch=True,
        func=downsample_exact_float32_stripe_scratch,
    ),
    Variant(
        name="nonexact_float32_channel_blocksum",
        exactness="non-exact",
        description="Block-sum RGB channels first, then apply float32 luma weights",
        uses_scratch=True,
        func=downsample_nonexact_float32_channel_blocksum,
    ),
    Variant(
        name="nonexact_integer_weight_blocksum",
        exactness="non-exact",
        description="Block-sum RGB channels with integer 299/587/114 weights",
        uses_scratch=True,
        func=downsample_nonexact_integer_weight_blocksum,
    ),
)


def run_probe(
    *,
    iterations: int,
    warmup_iterations: int,
    random_frames: int,
    renderer_frames: int,
    frame_size: int,
    trail_length: int,
    bonus_count: int,
    benchmark_set: FrameSetName,
    seed: int,
) -> dict[str, Any]:
    checked_iterations = _positive_value(iterations, name="iterations")
    checked_warmup = _nonnegative_value(warmup_iterations, name="warmup_iterations")
    checked_random_frames = _nonnegative_value(random_frames, name="random_frames")
    checked_renderer_frames = _nonnegative_value(renderer_frames, name="renderer_frames")
    checked_frame_size = _frame_size(frame_size)
    checked_trail_length = _nonnegative_value(trail_length, name="trail_length")
    checked_bonus_count = _nonnegative_value(bonus_count, name="bonus_count")
    if checked_random_frames == 0 and checked_renderer_frames == 0:
        raise ValueError("at least one random or renderer frame is required")

    rng = np.random.default_rng(int(seed))
    random_rgb = make_random_rgb_frames(
        count=checked_random_frames,
        frame_size=checked_frame_size,
        rng=rng,
    )
    renderer_rgb = make_renderer_rgb_frames(
        count=checked_renderer_frames,
        frame_size=checked_frame_size,
        trail_length=checked_trail_length,
        bonus_count=checked_bonus_count,
        seed=int(seed) + 1,
    )
    frame_sets = {
        "random": random_rgb,
        "renderer": renderer_rgb,
        "mixed": [*random_rgb, *renderer_rgb],
    }
    benchmark_frames = frame_sets[benchmark_set]
    if not benchmark_frames:
        raise ValueError(f"benchmark set {benchmark_set!r} has no frames")

    parity = {
        variant.name: {
            "random": compare_variant_to_production(variant, random_rgb),
            "renderer": compare_variant_to_production(variant, renderer_rgb),
        }
        for variant in VARIANTS
    }
    timings = {
        variant.name: time_variant(
            variant,
            benchmark_frames,
            iterations=checked_iterations,
            warmup_iterations=checked_warmup,
        )
        for variant in VARIANTS
    }
    baseline_us = timings["production_baseline"]["us_per_call"]
    variants = []
    for variant in VARIANTS:
        timing = timings[variant.name]
        speedup = baseline_us / timing["us_per_call"] if timing["us_per_call"] > 0.0 else 0.0
        variants.append(
            {
                "name": variant.name,
                "exactness": variant.exactness,
                "description": variant.description,
                "uses_scratch": variant.uses_scratch,
                "parity": parity[variant.name],
                "timing": timing,
                "speedup_vs_baseline": float(speedup),
            }
        )

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "config": {
            "iterations": checked_iterations,
            "warmup_iterations": checked_warmup,
            "random_frames": checked_random_frames,
            "renderer_frames": checked_renderer_frames,
            "frame_size": checked_frame_size,
            "target_size": TARGET_SIZE,
            "downsample_ratio": checked_frame_size // TARGET_SIZE,
            "trail_length": checked_trail_length,
            "bonus_count": checked_bonus_count,
            "benchmark_set": benchmark_set,
            "seed": int(seed),
        },
        "frame_sets": {
            "random": len(random_rgb),
            "renderer": len(renderer_rgb),
            "mixed": len(frame_sets["mixed"]),
        },
        "variants": variants,
        "best": best_reports(variants),
        "files_changed": [path for path in FILES_CHANGED if (REPO_ROOT / path).exists()],
    }


def make_random_rgb_frames(
    *,
    count: int,
    frame_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if count == 0:
        return []
    frames = rng.integers(
        0,
        256,
        size=(count, frame_size, frame_size, 3),
        dtype=np.uint8,
    )
    return [np.ascontiguousarray(frame) for frame in frames]


def make_renderer_rgb_frames(
    *,
    count: int,
    frame_size: int,
    trail_length: int,
    bonus_count: int,
    seed: int,
) -> list[np.ndarray]:
    if count == 0:
        return []
    state = make_renderer_state(
        batch_size=count,
        player_count=2,
        trail_length=trail_length,
        bonus_count=bonus_count,
        seed=seed,
    )
    modes = (TRAIL_RENDER_MODE_BROWSER_LINES, TRAIL_RENDER_MODE_BODY_CIRCLES_FAST)
    frames: list[np.ndarray] = []
    for row in range(count):
        rgb = render_source_state_rgb_canvas_like(
            state,
            row=row,
            frame_size=frame_size,
            trail_render_mode=modes[row % len(modes)],
        )
        frames.append(np.ascontiguousarray(rgb.copy()))
    return frames


def make_renderer_state(
    *,
    batch_size: int,
    player_count: int,
    trail_length: int,
    bonus_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = _positive_value(batch_size, name="batch_size")
    players = _positive_value(player_count, name="player_count")
    trail = _nonnegative_value(trail_length, name="trail_length")
    bonuses = _nonnegative_value(bonus_count, name="bonus_count")

    state: dict[str, np.ndarray] = {
        "tick": np.arange(batch, dtype=np.int32),
        "elapsed_ms": np.arange(batch, dtype=np.float64) * (1000.0 / 60.0),
        "map_size": np.full(batch, 64.0, dtype=np.float64),
        "present": np.ones((batch, players), dtype=bool),
        "alive": np.ones((batch, players), dtype=bool),
        "pos": np.zeros((batch, players, 2), dtype=np.float64),
        "radius": np.full((batch, players), 0.85, dtype=np.float64),
        "avatar_color": np.tile(np.arange(players, dtype=np.int16), (batch, 1)),
        "body_active": np.zeros((batch, trail), dtype=bool),
        "body_pos": np.zeros((batch, trail, 2), dtype=np.float64),
        "body_radius": np.full((batch, trail), 0.75, dtype=np.float64),
        "body_owner": np.full((batch, trail), -1, dtype=np.int16),
        "body_num": np.full((batch, trail), -1, dtype=np.int32),
        "body_break_before": np.zeros((batch, trail), dtype=bool),
        "body_write_cursor": np.full(batch, trail, dtype=np.int32),
        "done": np.zeros(batch, dtype=bool),
        "terminated": np.zeros(batch, dtype=bool),
        "truncated": np.zeros(batch, dtype=bool),
        "terminal_reason": np.zeros(batch, dtype=np.int16),
    }

    if trail:
        owner_counts = np.full(players, trail // players, dtype=np.int32)
        owner_counts[: trail % players] += 1
        for row in range(batch):
            slot = 0
            row_phase = float(rng.uniform(0.0, 2.0 * np.pi)) + row * 0.09
            for player in range(players):
                count = int(owner_counts[player])
                if count == 0:
                    continue
                phase = row_phase + (2.0 * np.pi * player / players)
                radius = 7.5 + 2.5 * player
                drift_x = np.cos(phase) * 7.0
                drift_y = np.sin(phase) * 7.0
                for body_num in range(count):
                    progress = body_num / max(1, count - 1)
                    angle = phase + progress * np.pi * 2.75
                    wobble = np.sin(progress * np.pi * 5.0 + row_phase) * 3.0
                    x = 32.0 + radius * np.cos(angle) + drift_x * (progress - 0.5)
                    y = 32.0 + radius * np.sin(angle) + drift_y * (progress - 0.5)
                    state["body_active"][row, slot] = True
                    state["body_pos"][row, slot] = np.clip((x + wobble, y - wobble), 1.0, 63.0)
                    state["body_radius"][row, slot] = 0.65 + 0.12 * ((body_num + player) % 3)
                    state["body_owner"][row, slot] = player
                    state["body_num"][row, slot] = body_num
                    state["body_break_before"][row, slot] = body_num > 0 and body_num % 97 == 0
                    slot += 1

    for row in range(batch):
        for player in range(players):
            player_slots = np.flatnonzero(state["body_owner"][row] == player)
            if player_slots.size:
                state["pos"][row, player] = state["body_pos"][row, player_slots[-1]]
            else:
                angle = (2.0 * np.pi * player / players) + row * 0.17
                state["pos"][row, player] = (
                    32.0 + 13.0 * np.cos(angle),
                    32.0 + 13.0 * np.sin(angle),
                )

    if bonuses:
        bonus_pos = np.zeros((batch, bonuses, 2), dtype=np.float64)
        for row in range(batch):
            for bonus in range(bonuses):
                angle = (2.0 * np.pi * bonus / bonuses) + row * 0.21
                ring = 10.0 + 2.5 * ((bonus + row) % 4)
                bonus_pos[row, bonus] = (
                    32.0 + ring * np.cos(angle),
                    32.0 + ring * np.sin(angle),
                )
        state.update(
            {
                "bonus_active": np.ones((batch, bonuses), dtype=bool),
                "bonus_pos": np.clip(bonus_pos, 2.0, 62.0),
                "bonus_radius": np.full((batch, bonuses), 1.15, dtype=np.float64),
                "bonus_type": (
                    (np.arange(bonuses, dtype=np.int16)[None, :] % 12) + 1
                ).repeat(batch, axis=0),
            }
        )

    return state


def compare_variant_to_production(
    variant: Variant,
    frames: Sequence[np.ndarray],
) -> dict[str, Any]:
    if not frames:
        return {
            "frames": 0,
            "byte_equal": True,
            "mismatch_frames": 0,
            "mismatch_pixels": 0,
            "max_abs_diff": 0,
            "first_mismatch": None,
        }

    source_size = int(frames[0].shape[0])
    scratch = DownsampleScratch(source_size) if variant.uses_scratch else None
    expected_out = np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
    actual_out = np.empty_like(expected_out)
    mismatch_frames = 0
    mismatch_pixels = 0
    max_abs_diff = 0
    first_mismatch: dict[str, int] | None = None

    for index, frame in enumerate(frames):
        expected = rgb_canvas_like_to_gray64(frame, out=expected_out)
        actual = variant.func(frame, actual_out, scratch)
        diff = actual.astype(np.int16) - expected.astype(np.int16)
        abs_diff = np.abs(diff)
        frame_mismatch_pixels = int(np.count_nonzero(abs_diff))
        if frame_mismatch_pixels:
            mismatch_frames += 1
            mismatch_pixels += frame_mismatch_pixels
            max_abs_diff = max(max_abs_diff, int(np.max(abs_diff)))
            if first_mismatch is None:
                coord = np.argwhere(abs_diff > 0)[0]
                channel, y, x = (int(value) for value in coord)
                first_mismatch = {
                    "frame_index": int(index),
                    "channel": channel,
                    "y": y,
                    "x": x,
                    "expected": int(expected[channel, y, x]),
                    "actual": int(actual[channel, y, x]),
                }

    return {
        "frames": len(frames),
        "byte_equal": mismatch_frames == 0,
        "mismatch_frames": mismatch_frames,
        "mismatch_pixels": mismatch_pixels,
        "max_abs_diff": max_abs_diff,
        "first_mismatch": first_mismatch,
    }


def time_variant(
    variant: Variant,
    frames: Sequence[np.ndarray],
    *,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("at least one frame is required for timing")
    source_size = int(frames[0].shape[0])
    scratch = DownsampleScratch(source_size) if variant.uses_scratch else None
    out = np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
    checksum = 0

    for index in range(warmup_iterations):
        variant.func(frames[index % len(frames)], out, scratch)

    started = time.perf_counter()
    for index in range(iterations):
        result = variant.func(frames[index % len(frames)], out, scratch)
        checksum ^= int(result[0, index % TARGET_SIZE, (index * 17) % TARGET_SIZE])
    elapsed = time.perf_counter() - started
    return {
        "calls": int(iterations),
        "seconds": float(elapsed),
        "us_per_call": float((elapsed / iterations) * 1_000_000.0),
        "calls_per_second": float(iterations / elapsed) if elapsed > 0.0 else 0.0,
        "checksum": int(checksum),
    }


def best_reports(variants: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline = _variant_by_name(variants, "production_baseline")
    exact_candidates = [
        variant
        for variant in variants
        if variant["exactness"] == "exact" and _variant_parity_ok(variant)
    ]
    nonexact_candidates = [
        variant for variant in variants if variant["exactness"] == "non-exact"
    ]
    return {
        "baseline": _compact_best(baseline),
        "best_exact_with_full_parity": _compact_best(_fastest(exact_candidates)),
        "best_non_exact": _compact_best(_fastest(nonexact_candidates)),
    }


def _variant_parity_ok(variant: dict[str, Any]) -> bool:
    return all(report["byte_equal"] for report in variant["parity"].values())


def _fastest(variants: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not variants:
        return None
    return min(variants, key=lambda item: item["timing"]["us_per_call"])


def _variant_by_name(variants: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    for variant in variants:
        if variant["name"] == name:
            return variant
    raise KeyError(name)


def _compact_best(variant: dict[str, Any] | None) -> dict[str, Any] | None:
    if variant is None:
        return None
    return {
        "name": variant["name"],
        "exactness": variant["exactness"],
        "us_per_call": variant["timing"]["us_per_call"],
        "speedup_vs_baseline": variant["speedup_vs_baseline"],
        "parity_ok": _variant_parity_ok(variant),
        "parity": variant["parity"],
    }


def _validated_call(
    rgb: np.ndarray,
    out: np.ndarray | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    rgb_array = np.asarray(rgb)
    if (
        rgb_array.ndim != 3
        or rgb_array.shape[2] != 3
        or rgb_array.shape[0] != rgb_array.shape[1]
    ):
        raise ValueError(f"rgb must have square HWC RGB shape, got {rgb_array.shape}")
    if rgb_array.dtype != np.uint8:
        raise ValueError(f"rgb dtype must be uint8, got {rgb_array.dtype}")
    if rgb_array.shape[0] < TARGET_SIZE or rgb_array.shape[0] % TARGET_SIZE != 0:
        raise ValueError(
            f"rgb frame size must be {TARGET_SIZE} or an integer multiple of {TARGET_SIZE}"
        )
    frame = _validated_out(out)
    return rgb_array, int(rgb_array.shape[0] // TARGET_SIZE), frame


def _validated_out(out: np.ndarray | None) -> np.ndarray:
    if out is None:
        return np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
    array = np.asarray(out)
    if array.shape != SOURCE_STATE_CANVAS_GRAY64_SHAPE:
        raise ValueError(f"out must have shape {SOURCE_STATE_CANVAS_GRAY64_SHAPE}")
    if array.dtype != np.uint8:
        raise ValueError(f"out dtype must be uint8, got {array.dtype}")
    return array


def _scratch_for(scratch: DownsampleScratch | None, source_size: int) -> DownsampleScratch:
    if scratch is not None and scratch.source_size == source_size:
        return scratch
    return DownsampleScratch(source_size)


def _frame_size(value: int) -> int:
    parsed = _positive_value(value, name="frame_size")
    if parsed < TARGET_SIZE or parsed % TARGET_SIZE != 0:
        raise ValueError(
            f"frame_size must be {TARGET_SIZE} or an integer multiple of {TARGET_SIZE}"
        )
    return parsed


def _positive_value(value: int, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_value(value: int, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_arg(value: str) -> int:
    try:
        return _positive_value(int(value), name=value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonnegative_arg(value: str) -> int:
    try:
        return _nonnegative_value(int(value), name=value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=_positive_arg, default=1000)
    parser.add_argument("--warmup-iterations", type=_nonnegative_arg, default=50)
    parser.add_argument("--random-frames", type=_nonnegative_arg, default=8)
    parser.add_argument("--renderer-frames", type=_nonnegative_arg, default=6)
    parser.add_argument(
        "--frame-size",
        type=_positive_arg,
        default=SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    )
    parser.add_argument("--trail-length", type=_nonnegative_arg, default=256)
    parser.add_argument("--bonus-count", type=_nonnegative_arg, default=2)
    parser.add_argument("--benchmark-set", choices=("random", "renderer", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(
        iterations=int(args.iterations),
        warmup_iterations=int(args.warmup_iterations),
        random_frames=int(args.random_frames),
        renderer_frames=int(args.renderer_frames),
        frame_size=int(args.frame_size),
        trail_length=int(args.trail_length),
        bonus_count=int(args.bonus_count),
        benchmark_set=args.benchmark_set,
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
        " ".join(
            (
                report["schema_id"],
                f"size={config['frame_size']}",
                f"ratio={config['downsample_ratio']}",
                f"iterations={config['iterations']}",
                f"benchmark_set={config['benchmark_set']}",
            )
        )
    )
    print("parity:")
    for variant in report["variants"]:
        if variant["exactness"] == "production":
            continue
        random_report = variant["parity"]["random"]
        renderer_report = variant["parity"]["renderer"]
        print(
            "  "
            + " ".join(
                (
                    f"{variant['name']} [{variant['exactness']}]",
                    f"random={_parity_text(random_report)}",
                    f"renderer={_parity_text(renderer_report)}",
                )
            )
        )

    print("timings:")
    ordered = sorted(report["variants"], key=lambda item: item["timing"]["us_per_call"])
    for variant in ordered:
        parity = "production" if variant["exactness"] == "production" else (
            "OK" if _variant_parity_ok(variant) else "MISMATCH"
        )
        print(
            "  "
            + " ".join(
                (
                    f"{variant['name']} [{variant['exactness']}]",
                    f"{variant['timing']['us_per_call']:.1f} us/call",
                    f"speedup={variant['speedup_vs_baseline']:.2f}x",
                    f"parity={parity}",
                )
            )
        )

    best = report["best"]
    print("best:")
    print(f"  baseline={_best_text(best['baseline'])}")
    print(f"  best_exact_with_full_parity={_best_text(best['best_exact_with_full_parity'])}")
    print(f"  best_non_exact={_best_text(best['best_non_exact'])}")
    print("files_changed=" + ", ".join(report["files_changed"]))


def _parity_text(report: dict[str, Any]) -> str:
    if report["frames"] == 0:
        return "n/a"
    if report["byte_equal"]:
        return f"OK/{report['frames']}"
    return (
        f"MISMATCH frames={report['mismatch_frames']}/{report['frames']} "
        f"pixels={report['mismatch_pixels']} max_abs={report['max_abs_diff']}"
    )


def _best_text(report: dict[str, Any] | None) -> str:
    if report is None:
        return "none"
    parity = "OK" if report["parity_ok"] else "MISMATCH"
    return (
        f"{report['name']} {report['us_per_call']:.1f} us/call "
        f"speedup={report['speedup_vs_baseline']:.2f}x parity={parity}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
