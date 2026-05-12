"""Small render-lane microbench for current CurvyZero visual policy input.

This intentionally avoids LightZero/DI-engine and renderer internals. The first
probe answers a narrow question: is ``browser_lines`` cost mostly duplicate
per-seat redraw, trail-length redraw slope, RGB draw, RGB-to-gray64 conversion,
perspective reuse, gray64 render, or stack copy?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Literal, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
    SOURCE_STATE_CANVAS_GRAY64_SHAPE,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    SourceStateBrowserLineTrailLayerCache,
    SourceStateGray64DownsampleScratch,
    TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    TRAIL_RENDER_MODE_BROWSER_LINES,
    TRAIL_RENDER_MODE_ORDER,
    normalize_source_state_gray64,
    render_source_state_canvas_gray64,
    render_source_state_rgb_canvas_like,
    rgb_canvas_like_to_gray64,
)
try:  # noqa: E402
    from curvyzero.env.vector_visual_observation import (
        render_source_state_canvas_gray64_player_perspectives
        as _render_source_state_canvas_gray64_player_perspectives,
    )
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests.
    _render_source_state_canvas_gray64_player_perspectives = None
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (  # noqa: E402
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
    SourceStateGray64Stack4,
    player_perspective_rgb_palette,
)


REPORT_SCHEMA_ID = "curvyzero_render_lane_microbench_report/v0"
DEFAULT_PLAN_ID = "render_lane_first_probe_v0"
REQUIRED_SYNTHETIC_STATE_FIELDS = (
    "tick",
    "elapsed_ms",
    "map_size",
    "present",
    "alive",
    "pos",
    "radius",
    "avatar_color",
    "body_active",
    "body_pos",
    "body_radius",
    "body_owner",
    "body_num",
    "body_write_cursor",
    "done",
    "terminated",
    "truncated",
    "terminal_reason",
)
OPTIONAL_BONUS_FIELDS = (
    "bonus_active",
    "bonus_pos",
    "bonus_radius",
    "bonus_type",
)

FULL_STACK = "full_stack_update"
GRAY64_RENDER_ONLY = "gray64_render_only"
GRAY64_RENDER_NORMALIZE = "gray64_render_normalize"
RGB_RENDER_ONLY = "rgb_render_only"
RGB_TO_GRAY64_ONLY = "rgb_to_gray64_only"
PERSPECTIVE_REUSE_GRAY64 = "perspective_reuse_gray64"
STACK_COPY_ONLY = "stack_shift_insert_return_copy_only"
CELL_KINDS = (
    FULL_STACK,
    GRAY64_RENDER_ONLY,
    GRAY64_RENDER_NORMALIZE,
    RGB_RENDER_ONLY,
    RGB_TO_GRAY64_ONLY,
    PERSPECTIVE_REUSE_GRAY64,
    STACK_COPY_ONLY,
)
CELL_KIND_ALIASES = {
    "render_only_two_perspective": GRAY64_RENDER_ONLY,
    "normalize_stack_copy_only": STACK_COPY_ONLY,
}

AllocationMode = Literal["reuse", "allocate"]
GpuTransferMode = Literal["on", "off", "auto"]
TrailSource = Literal["body", "visual"]


def default_probe_cells() -> list[dict[str, Any]]:
    """Return the deliberately small first-probe plan."""

    cells: list[dict[str, Any]] = []
    for trail_length in (0, 64, 256, 1024, 4096):
        cells.append(
            _cell(
                kind=FULL_STACK,
                batch_size=16,
                player_count=2,
                trail_length=trail_length,
                bonus_count=0,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
                label=f"browser_stack_B16_P2_L{trail_length}",
            )
        )
    for player_count in (1,):
        for trail_length in (1024, 4096):
            cells.append(
                _cell(
                    kind=FULL_STACK,
                    batch_size=16,
                    player_count=player_count,
                    trail_length=trail_length,
                    bonus_count=0,
                    trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
                    label=f"browser_stack_B16_P{player_count}_L{trail_length}",
                )
            )
    for trail_length in (64, 256, 1024, 4096):
        cells.append(
            _cell(
                kind=FULL_STACK,
                batch_size=16,
                player_count=2,
                trail_length=trail_length,
                bonus_count=0,
                trail_render_mode=TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
                label=f"fast_stack_B16_P2_L{trail_length}",
            )
        )
    for trail_length in (0, 64, 256, 1024, 4096):
        cells.append(
            _cell(
                kind=GRAY64_RENDER_ONLY,
                batch_size=16,
                player_count=2,
                trail_length=trail_length,
                bonus_count=0,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
                label=f"browser_render_only_B16_P2_L{trail_length}",
            )
        )
    for player_count in (1,):
        for trail_length in (1024, 4096):
            cells.append(
                _cell(
                    kind=GRAY64_RENDER_ONLY,
                    batch_size=16,
                    player_count=player_count,
                    trail_length=trail_length,
                    bonus_count=0,
                    trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
                    label=f"browser_render_only_B16_P{player_count}_L{trail_length}",
                )
            )
    for kind, label_prefix in (
        (RGB_RENDER_ONLY, "browser_rgb_render_only"),
        (RGB_TO_GRAY64_ONLY, "browser_rgb_to_gray64_only"),
        (PERSPECTIVE_REUSE_GRAY64, "browser_perspective_reuse_gray64"),
    ):
        cells.append(
            _cell(
                kind=kind,
                batch_size=16,
                player_count=2,
                trail_length=1024,
                bonus_count=0,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
                label=f"{label_prefix}_B16_P2_L1024",
            )
        )
    cells.append(
        _cell(
            kind=GRAY64_RENDER_NORMALIZE,
            batch_size=16,
            player_count=2,
            trail_length=0,
            bonus_count=0,
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            label="browser_gray64_render_normalize_B16_P2_L0",
        )
    )
    cells.append(
        _cell(
            kind=STACK_COPY_ONLY,
            batch_size=16,
            player_count=2,
            trail_length=1024,
            bonus_count=0,
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            label="stack_shift_insert_return_copy_only_B16_P2",
        )
    )
    for bonus_count in (0, 4):
        cells.append(
            _cell(
                kind=FULL_STACK,
                batch_size=8,
                player_count=2,
                trail_length=1024,
                bonus_count=bonus_count,
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
                label=f"bonus_delta_stack_B8_P2_L1024_bonus{bonus_count}",
            )
        )
    return cells


def make_synthetic_source_state(
    *,
    batch_size: int,
    player_count: int,
    trail_length: int,
    bonus_count: int,
    trail_source: TrailSource = "body",
    seed: int,
) -> dict[str, np.ndarray]:
    """Build vector-runtime-shaped source-state arrays without an env/runtime."""

    batch = _positive_value(batch_size, name="batch_size")
    players = _positive_value(player_count, name="player_count")
    trail = _nonnegative_value(trail_length, name="trail_length")
    bonuses = _nonnegative_value(bonus_count, name="bonus_count")
    source = _trail_source(trail_source)
    rng = np.random.default_rng(int(seed))

    state: dict[str, np.ndarray] = {
        "tick": np.arange(batch, dtype=np.int32),
        "elapsed_ms": np.arange(batch, dtype=np.float64) * (1000.0 / 60.0),
        "map_size": np.full(batch, 64.0, dtype=np.float64),
        "present": np.ones((batch, players), dtype=bool),
        "alive": np.ones((batch, players), dtype=bool),
        "pos": np.zeros((batch, players, 2), dtype=np.float64),
        "radius": np.full((batch, players), 1.0, dtype=np.float64),
        "avatar_color": np.tile(np.arange(players, dtype=np.int16), (batch, 1)),
        "body_active": np.zeros((batch, trail), dtype=bool),
        "body_pos": np.zeros((batch, trail, 2), dtype=np.float64),
        "body_radius": np.full((batch, trail), 0.8, dtype=np.float64),
        "body_owner": np.full((batch, trail), -1, dtype=np.int16),
        "body_num": np.full((batch, trail), -1, dtype=np.int32),
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
            row_phase = float(rng.uniform(0.0, 2.0 * np.pi))
            for player in range(players):
                count = int(owner_counts[player])
                if count == 0:
                    continue
                player_phase = row_phase + (2.0 * np.pi * player / players)
                center_x = 32.0 + 5.0 * np.cos(player_phase)
                center_y = 32.0 + 5.0 * np.sin(player_phase)
                orbit = 9.0 + 2.0 * (player % 4)
                for body_num in range(count):
                    progress = body_num / max(1, count - 1)
                    angle = player_phase + progress * np.pi * 1.35
                    drift = (progress - 0.5) * 14.0
                    x = center_x + orbit * np.cos(angle) + drift * np.cos(player_phase)
                    y = center_y + orbit * np.sin(angle) + drift * np.sin(player_phase)
                    state["body_active"][row, slot] = True
                    state["body_pos"][row, slot] = np.clip((x, y), 1.0, 63.0)
                    state["body_radius"][row, slot] = 0.7 + 0.15 * (player % 3)
                    state["body_owner"][row, slot] = player
                    state["body_num"][row, slot] = body_num
                    slot += 1

    for row in range(batch):
        for player in range(players):
            player_slots = np.flatnonzero(state["body_owner"][row] == player)
            if player_slots.size:
                state["pos"][row, player] = state["body_pos"][row, player_slots[-1]]
            else:
                angle = (2.0 * np.pi * player / players) + row * 0.17
                state["pos"][row, player] = (
                    32.0 + 14.0 * np.cos(angle),
                    32.0 + 14.0 * np.sin(angle),
                )

    if bonuses:
        bonus_pos = np.zeros((batch, bonuses, 2), dtype=np.float64)
        for row in range(batch):
            for bonus in range(bonuses):
                angle = (2.0 * np.pi * bonus / bonuses) + row * 0.11
                ring = 12.0 + 3.0 * ((bonus + row) % 3)
                bonus_pos[row, bonus] = (
                    32.0 + ring * np.cos(angle),
                    32.0 + ring * np.sin(angle),
                )
        state.update(
            {
                "bonus_active": np.ones((batch, bonuses), dtype=bool),
                "bonus_pos": np.clip(bonus_pos, 2.0, 62.0),
                "bonus_radius": np.full((batch, bonuses), 1.1, dtype=np.float64),
                "bonus_type": (
                    (np.arange(bonuses, dtype=np.int16)[None, :] % 12) + 1
                ).repeat(batch, axis=0),
            }
        )

    if source == "visual":
        state["visual_trail_active"] = state["body_active"].copy()
        state["visual_trail_write_cursor"] = state["body_write_cursor"].copy()
        state["visual_trail_pos"] = state["body_pos"].copy()
        state["visual_trail_radius"] = state["body_radius"].copy()
        state["visual_trail_owner"] = state["body_owner"].copy()
        break_before = np.zeros((batch, trail), dtype=bool)
        if trail:
            for row in range(batch):
                active_slots = np.flatnonzero(state["body_active"][row])
                if active_slots.size:
                    break_before[row, active_slots[0]] = True
                    previous_owner = state["body_owner"][row, active_slots[:-1]]
                    current_owner = state["body_owner"][row, active_slots[1:]]
                    break_before[row, active_slots[1:]] = current_owner != previous_owner
        state["visual_trail_break_before"] = break_before

    return state


def run_default_probe(
    *,
    iterations: int,
    warmup_iterations: int,
    allocation_mode: AllocationMode,
    gpu_transfer: GpuTransferMode,
    seed: int,
) -> dict[str, Any]:
    return run_probe_cells(
        cells=default_probe_cells(),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        allocation_mode=allocation_mode,
        gpu_transfer=gpu_transfer,
        seed=seed,
        plan_id=DEFAULT_PLAN_ID,
    )


def run_probe_cells(
    *,
    cells: Sequence[dict[str, Any]],
    iterations: int,
    warmup_iterations: int,
    allocation_mode: AllocationMode,
    gpu_transfer: GpuTransferMode,
    seed: int,
    plan_id: str = "custom_grid",
) -> dict[str, Any]:
    checked_iterations = _positive_value(iterations, name="iterations")
    checked_warmups = _nonnegative_value(warmup_iterations, name="warmup_iterations")
    checked_allocation = _allocation_mode(allocation_mode)
    checked_gpu_transfer = _gpu_transfer_mode(gpu_transfer)
    if checked_gpu_transfer != "off":
        raise NotImplementedError(
            "GPU transfer is deliberately deferred for the first render-lane microbench"
        )

    results: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        workload = _validated_cell(cell)
        state = make_synthetic_source_state(
            batch_size=workload["batch_size"],
            player_count=workload["player_count"],
            trail_length=workload["trail_length"],
            bonus_count=workload["bonus_count"],
            trail_source=workload["trail_source"],
            seed=int(seed) + index,
        )
        results.append(
            run_benchmark_cell(
                state=state,
                cell=workload,
                iterations=checked_iterations,
                warmup_iterations=checked_warmups,
                allocation_mode=checked_allocation,
                seed=int(seed) + index,
            )
        )

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "plan_id": plan_id,
        "source_observation_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
        "apis": {
            "rgb_render": "render_source_state_rgb_canvas_like",
            "rgb_to_gray64": "rgb_canvas_like_to_gray64",
            "gray64_render": "render_source_state_canvas_gray64",
            "perspective_reuse_gray64": (
                "render_source_state_canvas_gray64_player_perspectives"
                if _render_source_state_canvas_gray64_player_perspectives is not None
                else None
            ),
            "normalize": "normalize_source_state_gray64",
            "stack": "SourceStateGray64Stack4.update",
        },
        "probe_questions": {
            "duplicate_per_seat_redraw": "compare matched P1 and P2 full_stack_update/gray64_render_only cells",
            "trail_length_redraw_slope": "compare full_stack_update latency across trail lengths",
            "rgb_draw_overhead": "use rgb_render_only cells to isolate browser-like RGB drawing",
            "rgb_to_gray64_overhead": "use rgb_to_gray64_only cells to isolate luma/downsample conversion",
            "perspective_reuse_overhead": "use perspective_reuse_gray64 cells for the render-once-plus-remap path",
            "stack_copy_overhead": "compare stack_shift_insert_return_copy_only to full stack at matching shape",
            "normalize_overhead": "use gray64_render_normalize cells; do not infer from zeros",
            "bonus_overhead": "compare B8/P2/L1024 bonus_count 0 vs 4",
        },
        "synthetic_state": {
            "required_fields": list(REQUIRED_SYNTHETIC_STATE_FIELDS),
            "optional_bonus_fields": list(OPTIONAL_BONUS_FIELDS),
            "trail_source": "body uses body_* fallback; visual adds visual_trail_* fields",
        },
        "config": {
            "iterations": checked_iterations,
            "warmup_iterations": checked_warmups,
            "allocation_mode": checked_allocation,
            "gpu_transfer": {
                "requested": checked_gpu_transfer,
                "performed": False,
                "deferred": checked_gpu_transfer != "off",
            },
            "seed": int(seed),
        },
        "cell_count": len(results),
        "cells": results,
    }


def run_grid_probe(
    *,
    batch_sizes: Sequence[int],
    player_counts: Sequence[int],
    trail_lengths: Sequence[int],
    bonus_counts: Sequence[int],
    trail_render_modes: Sequence[str],
    kinds: Sequence[str],
    trail_source: TrailSource = "body",
    iterations: int,
    warmup_iterations: int,
    allocation_mode: AllocationMode,
    gpu_transfer: GpuTransferMode,
    seed: int,
) -> dict[str, Any]:
    """Small flexible grid for later local follow-ups."""

    cells: list[dict[str, Any]] = []
    for kind in kinds:
        for batch_size in batch_sizes:
            for player_count in player_counts:
                for trail_length in trail_lengths:
                    for bonus_count in bonus_counts:
                        for trail_render_mode in trail_render_modes:
                            cells.append(
                                _cell(
                                    kind=_cell_kind(kind),
                                    batch_size=int(batch_size),
                                    player_count=int(player_count),
                                    trail_length=int(trail_length),
                                    bonus_count=int(bonus_count),
                                    trail_source=_trail_source(trail_source),
                                    trail_render_mode=_trail_render_mode(trail_render_mode),
                                    label=(
                                        f"{kind}_B{batch_size}_P{player_count}_"
                                        f"L{trail_length}_bonus{bonus_count}_{trail_render_mode}_"
                                        f"{trail_source}"
                                    ),
                                )
                            )
    return run_probe_cells(
        cells=cells,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        allocation_mode=allocation_mode,
        gpu_transfer=gpu_transfer,
        seed=seed,
    )


def run_benchmark_cell(
    *,
    state: dict[str, np.ndarray],
    cell: dict[str, Any],
    iterations: int,
    warmup_iterations: int,
    allocation_mode: AllocationMode,
    seed: int,
) -> dict[str, Any]:
    kind = _cell_kind(cell["kind"])
    batch_size = int(cell["batch_size"])
    player_count = int(cell["player_count"])
    trail_render_mode = _trail_render_mode(str(cell["trail_render_mode"]))
    if (
        kind == PERSPECTIVE_REUSE_GRAY64
        and _render_source_state_canvas_gray64_player_perspectives is None
    ):
        return _unsupported_cell_report(
            cell=cell,
            kind=kind,
            batch_size=batch_size,
            player_count=player_count,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            allocation_mode=allocation_mode,
            seed=seed,
            reason="render_source_state_canvas_gray64_player_perspectives is not importable",
        )
    env = SimpleNamespace(batch_size=batch_size, player_count=player_count, state=state)
    palettes = _palettes(state, batch_size=batch_size, player_count=player_count)
    scratch = (
        _DirectScratch(batch_size=batch_size, player_count=player_count)
        if allocation_mode == "reuse"
        else None
    )

    stack = (
        SourceStateGray64Stack4(
            batch_size=batch_size,
            player_count=player_count,
            trail_render_mode=trail_render_mode,
        )
        if kind in (FULL_STACK, STACK_COPY_ONLY)
        else None
    )
    overhead_frame = None
    if kind == STACK_COPY_ONLY:
        overhead_frame = _one_normalized_frame(
            state=state,
            trail_render_mode=trail_render_mode,
            palette=palettes[0][0],
            allocation_mode=allocation_mode,
            scratch=scratch,
        )

    for _ in range(warmup_iterations):
        pre_rendered_rgb = _pre_rendered_rgb_frame(
            state=state,
            trail_render_mode=trail_render_mode,
            palette=palettes[0][0],
            allocation_mode=allocation_mode,
            scratch=scratch,
        ) if kind == RGB_TO_GRAY64_ONLY else None
        _run_cell_once(
            kind=kind,
            state=state,
            env=env,
            stack=stack,
            overhead_frame=overhead_frame,
            pre_rendered_rgb=pre_rendered_rgb,
            batch_size=batch_size,
            player_count=player_count,
            trail_render_mode=trail_render_mode,
            palettes=palettes,
            allocation_mode=allocation_mode,
            scratch=scratch,
            collect_latency=False,
        )
        _advance_static_state_clock(state)

    timing_sec = _empty_timing()
    latencies = _empty_latencies()
    integrity_counter = {"nonzero_pixels": 0}
    last_raw: np.ndarray | None = None
    last_rgb: np.ndarray | None = None
    last_normalized: np.ndarray | None = None
    last_perspective_frames: np.ndarray | None = None
    last_stack: np.ndarray | None = None
    pre_rendered_rgb = (
        _pre_rendered_rgb_frame(
            state=state,
            trail_render_mode=trail_render_mode,
            palette=palettes[0][0],
            allocation_mode=allocation_mode,
            scratch=scratch,
        )
        if kind == RGB_TO_GRAY64_ONLY
        else None
    )
    measured_started = time.perf_counter()

    for _ in range(iterations):
        result = _run_cell_once(
            kind=kind,
            state=state,
            env=env,
            stack=stack,
            overhead_frame=overhead_frame,
            pre_rendered_rgb=pre_rendered_rgb,
            batch_size=batch_size,
            player_count=player_count,
            trail_render_mode=trail_render_mode,
            palettes=palettes,
            allocation_mode=allocation_mode,
            scratch=scratch,
            collect_latency=True,
        )
        for key, value in result["timing_sec"].items():
            timing_sec[key] += float(value)
        for key, values in result["latency_sec"].items():
            latencies[key].extend(values)
        integrity_counter["nonzero_pixels"] += int(result["nonzero_pixels"])
        last_raw = result["last_raw"] if result["last_raw"] is not None else last_raw
        last_rgb = result["last_rgb"] if result["last_rgb"] is not None else last_rgb
        last_normalized = (
            result["last_normalized"]
            if result["last_normalized"] is not None
            else last_normalized
        )
        last_perspective_frames = (
            result["last_perspective_frames"]
            if result["last_perspective_frames"] is not None
            else last_perspective_frames
        )
        last_stack = result["last_stack"] if result["last_stack"] is not None else last_stack
        _advance_static_state_clock(state)

    timing_sec["measured_elapsed"] = time.perf_counter() - measured_started
    denominators = _denominators(
        kind=kind,
        batch_size=batch_size,
        player_count=player_count,
        iterations=iterations,
        stack_present=last_stack is not None,
    )
    if last_stack is not None:
        denominators["stack_bytes_copied"] = _estimated_stack_bytes_copied(
            batch_size,
            player_count,
        ) * iterations

    return {
        "label": cell["label"],
        "kind": kind,
        "status": "ok",
        "workload": {
            "batch_size": batch_size,
            "player_count": player_count,
            "trail_length": int(cell["trail_length"]),
            "bonus_count": int(cell["bonus_count"]),
            "trail_source": _trail_source(str(cell["trail_source"])),
            "trail_render_mode": trail_render_mode,
            "iterations": int(iterations),
            "warmup_iterations": int(warmup_iterations),
            "allocation_mode": allocation_mode,
            "seed": int(seed),
        },
        "timing_sec": timing_sec,
        "latency_us": _latency_report(latencies),
        "denominators": denominators,
        "throughput": _throughput_report(timing_sec=timing_sec, denominators=denominators),
        "density": _density_report(state),
        "integrity": {
            "nonzero_pixels_total": int(integrity_counter["nonzero_pixels"]),
            "last_raw_nonzero_pixels": (
                int(np.count_nonzero(last_raw)) if last_raw is not None else 0
            ),
            "last_stack_nonzero_pixels": (
                int(np.count_nonzero(last_stack)) if last_stack is not None else 0
            ),
            "last_raw_frame_hash": _array_digest(last_raw),
            "last_rgb_frame_hash": _array_digest(last_rgb),
            "last_normalized_frame_hash": _array_digest(last_normalized),
            "last_perspective_frames_hash": _array_digest(last_perspective_frames),
            "last_stack_hash": _array_digest(last_stack),
        },
        "observation": _observation_report(
            batch_size=batch_size,
            player_count=player_count,
        ),
    }


def _unsupported_cell_report(
    *,
    cell: dict[str, Any],
    kind: str,
    batch_size: int,
    player_count: int,
    iterations: int,
    warmup_iterations: int,
    allocation_mode: AllocationMode,
    seed: int,
    reason: str,
) -> dict[str, Any]:
    timing_sec = _empty_timing()
    denominators = _denominators(
        kind=kind,
        batch_size=batch_size,
        player_count=player_count,
        iterations=iterations,
        stack_present=False,
    )
    denominators["perspective_reuse_gray64_calls"] = 0
    denominators["perspective_reuse_gray64_frames"] = 0
    return {
        "label": cell["label"],
        "kind": kind,
        "status": "unsupported",
        "unsupported_reason": reason,
        "workload": {
            "batch_size": batch_size,
            "player_count": player_count,
            "trail_length": int(cell["trail_length"]),
            "bonus_count": int(cell["bonus_count"]),
            "trail_source": _trail_source(str(cell.get("trail_source", "body"))),
            "trail_render_mode": _trail_render_mode(str(cell["trail_render_mode"])),
            "iterations": int(iterations),
            "warmup_iterations": int(warmup_iterations),
            "allocation_mode": allocation_mode,
            "seed": int(seed),
        },
        "timing_sec": timing_sec,
        "latency_us": {},
        "denominators": denominators,
        "throughput": _throughput_report(
            timing_sec=timing_sec,
            denominators=denominators,
        ),
        "density": {
            "active_bodies_per_row": 0.0,
            "body_capacity": int(cell["trail_length"]),
            "bonus_count": int(cell["bonus_count"]),
            "active_bonuses_per_row": 0.0,
        },
        "integrity": {
            "nonzero_pixels_total": 0,
            "last_raw_nonzero_pixels": 0,
            "last_stack_nonzero_pixels": 0,
            "last_raw_frame_hash": None,
            "last_rgb_frame_hash": None,
            "last_normalized_frame_hash": None,
            "last_perspective_frames_hash": None,
            "last_stack_hash": None,
        },
        "observation": _observation_report(
            batch_size=batch_size,
            player_count=player_count,
        ),
    }


def _observation_report(*, batch_size: int, player_count: int) -> dict[str, Any]:
    rgb_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    return {
        "rgb_frame_shape": [rgb_size, rgb_size, 3],
        "rgb_frame_dtype": "uint8",
        "raw_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
        "raw_frame_dtype": "uint8",
        "perspective_raw_frames_shape": [
            player_count,
            *SOURCE_STATE_CANVAS_GRAY64_SHAPE,
        ],
        "normalized_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
        "normalized_frame_dtype": "float32",
        "stack_shape": [batch_size, player_count, *STACKED_SOURCE_STATE_GRAY64_SHAPE],
        "stack_dtype": "float32",
    }


def _run_cell_once(
    *,
    kind: str,
    state: dict[str, np.ndarray],
    env: SimpleNamespace,
    stack: SourceStateGray64Stack4 | None,
    overhead_frame: np.ndarray | None,
    pre_rendered_rgb: np.ndarray | None,
    batch_size: int,
    player_count: int,
    trail_render_mode: str,
    palettes: Sequence[Sequence[Sequence[Sequence[int]]]],
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
    collect_latency: bool,
) -> dict[str, Any]:
    timing = _empty_timing()
    latencies = _empty_latencies()
    last_raw: np.ndarray | None = None
    last_rgb: np.ndarray | None = None
    last_normalized: np.ndarray | None = None
    last_perspective_frames: np.ndarray | None = None
    last_stack: np.ndarray | None = None
    nonzero_pixels = 0

    if kind in (GRAY64_RENDER_ONLY, GRAY64_RENDER_NORMALIZE):
        direct = _run_direct_render_normalize_pass(
            normalize=kind == GRAY64_RENDER_NORMALIZE,
            state=state,
            batch_size=batch_size,
            player_count=player_count,
            trail_render_mode=trail_render_mode,
            palettes=palettes,
            allocation_mode=allocation_mode,
            scratch=scratch,
            collect_latency=collect_latency,
        )
        for key, value in direct["timing_sec"].items():
            timing[key] += float(value)
        for key, values in direct["latency_sec"].items():
            latencies[key].extend(values)
        nonzero_pixels += int(direct["nonzero_pixels"])
        last_raw = direct["last_raw"]
        last_normalized = direct["last_normalized"]
    elif kind == RGB_RENDER_ONLY:
        direct = _run_rgb_render_pass(
            state=state,
            batch_size=batch_size,
            player_count=player_count,
            trail_render_mode=trail_render_mode,
            palettes=palettes,
            allocation_mode=allocation_mode,
            scratch=scratch,
            collect_latency=collect_latency,
        )
        for key, value in direct["timing_sec"].items():
            timing[key] += float(value)
        for key, values in direct["latency_sec"].items():
            latencies[key].extend(values)
        nonzero_pixels += int(direct["nonzero_pixels"])
        last_rgb = direct["last_rgb"]
    elif kind == RGB_TO_GRAY64_ONLY:
        if pre_rendered_rgb is None:
            raise ValueError("pre_rendered_rgb is required for rgb-to-gray64 cells")
        direct = _run_rgb_to_gray64_pass(
            rgb=pre_rendered_rgb,
            batch_size=batch_size,
            player_count=player_count,
            allocation_mode=allocation_mode,
            scratch=scratch,
            collect_latency=collect_latency,
        )
        for key, value in direct["timing_sec"].items():
            timing[key] += float(value)
        for key, values in direct["latency_sec"].items():
            latencies[key].extend(values)
        nonzero_pixels += int(direct["nonzero_pixels"])
        last_rgb = pre_rendered_rgb
        last_raw = direct["last_raw"]
    elif kind == PERSPECTIVE_REUSE_GRAY64:
        direct = _run_perspective_reuse_gray64_pass(
            state=state,
            batch_size=batch_size,
            player_count=player_count,
            trail_render_mode=trail_render_mode,
            palettes=palettes,
            allocation_mode=allocation_mode,
            scratch=scratch,
            collect_latency=collect_latency,
        )
        for key, value in direct["timing_sec"].items():
            timing[key] += float(value)
        for key, values in direct["latency_sec"].items():
            latencies[key].extend(values)
        nonzero_pixels += int(direct["nonzero_pixels"])
        last_raw = direct["last_raw"]
        last_perspective_frames = direct["last_perspective_frames"]

    if kind == FULL_STACK:
        if stack is None:
            raise ValueError("stack is required for full stack cells")
        started = time.perf_counter()
        last_stack = stack.update(env)
        duration = time.perf_counter() - started
        timing["stack_update"] += duration
        if collect_latency:
            latencies["stack_update"].append(duration)
    elif kind == STACK_COPY_ONLY:
        if stack is None or overhead_frame is None:
            raise ValueError("stack and overhead_frame are required for stack-copy cells")
        started = time.perf_counter()
        last_stack = _update_stack_with_frame_only(
            stack,
            overhead_frame,
            batch_size=batch_size,
            player_count=player_count,
        )
        duration = time.perf_counter() - started
        timing["stack_copy_only"] += duration
        if collect_latency:
            latencies["stack_copy_only"].append(duration)

    return {
        "timing_sec": timing,
        "latency_sec": latencies,
        "nonzero_pixels": nonzero_pixels,
        "last_raw": last_raw,
        "last_rgb": last_rgb,
        "last_normalized": last_normalized,
        "last_perspective_frames": last_perspective_frames,
        "last_stack": last_stack,
    }


def _run_direct_render_normalize_pass(
    *,
    normalize: bool,
    state: dict[str, np.ndarray],
    batch_size: int,
    player_count: int,
    trail_render_mode: str,
    palettes: Sequence[Sequence[Sequence[Sequence[int]]]],
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
    collect_latency: bool,
) -> dict[str, Any]:
    render_sec = 0.0
    normalize_sec = 0.0
    render_latencies: list[float] = []
    normalize_latencies: list[float] = []
    nonzero_pixels = 0
    last_raw: np.ndarray | None = None
    last_normalized: np.ndarray | None = None

    for row in range(batch_size):
        for player in range(player_count):
            render_started = time.perf_counter()
            raw = _render_one(
                state=state,
                row=row,
                palette=palettes[row][player],
                trail_render_mode=trail_render_mode,
                allocation_mode=allocation_mode,
                scratch=scratch,
            )
            render_duration = time.perf_counter() - render_started
            render_sec += render_duration
            if collect_latency:
                render_latencies.append(render_duration)
            nonzero_pixels += int(np.count_nonzero(raw))
            last_raw = raw

            if not normalize:
                continue

            normalize_started = time.perf_counter()
            if allocation_mode == "reuse":
                if scratch is None:
                    raise ValueError("scratch is required in reuse allocation mode")
                normalized = normalize_source_state_gray64(raw, out=scratch.normalized)
            else:
                normalized = normalize_source_state_gray64(raw)
            normalize_duration = time.perf_counter() - normalize_started
            normalize_sec += normalize_duration
            if collect_latency:
                normalize_latencies.append(normalize_duration)
            last_normalized = normalized

    return {
        "timing_sec": {
            **_empty_timing(),
            "gray64_render": render_sec,
            "normalize": normalize_sec,
        },
        "latency_sec": {
            **_empty_latencies(),
            "gray64_render": render_latencies,
            "normalize": normalize_latencies,
        },
        "nonzero_pixels": nonzero_pixels,
        "last_raw": last_raw,
        "last_normalized": last_normalized,
    }


def _run_rgb_render_pass(
    *,
    state: dict[str, np.ndarray],
    batch_size: int,
    player_count: int,
    trail_render_mode: str,
    palettes: Sequence[Sequence[Sequence[Sequence[int]]]],
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
    collect_latency: bool,
) -> dict[str, Any]:
    render_sec = 0.0
    render_latencies: list[float] = []
    nonzero_pixels = 0
    last_rgb: np.ndarray | None = None

    for row in range(batch_size):
        for player in range(player_count):
            render_started = time.perf_counter()
            rgb = _render_one_rgb(
                state=state,
                row=row,
                palette=palettes[row][player],
                trail_render_mode=trail_render_mode,
                allocation_mode=allocation_mode,
                scratch=scratch,
            )
            render_duration = time.perf_counter() - render_started
            render_sec += render_duration
            if collect_latency:
                render_latencies.append(render_duration)
            nonzero_pixels += int(np.count_nonzero(rgb))
            last_rgb = rgb

    return {
        "timing_sec": {
            **_empty_timing(),
            "rgb_render": render_sec,
        },
        "latency_sec": {
            **_empty_latencies(),
            "rgb_render": render_latencies,
        },
        "nonzero_pixels": nonzero_pixels,
        "last_rgb": last_rgb,
    }


def _run_rgb_to_gray64_pass(
    *,
    rgb: np.ndarray,
    batch_size: int,
    player_count: int,
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
    collect_latency: bool,
) -> dict[str, Any]:
    convert_sec = 0.0
    convert_latencies: list[float] = []
    nonzero_pixels = 0
    last_raw: np.ndarray | None = None

    for _ in range(batch_size * player_count):
        convert_started = time.perf_counter()
        if allocation_mode == "reuse":
            if scratch is None:
                raise ValueError("scratch is required in reuse allocation mode")
            raw = rgb_canvas_like_to_gray64(
                rgb,
                out=scratch.raw,
                scratch=scratch.downsample,
            )
        else:
            raw = rgb_canvas_like_to_gray64(rgb)
        convert_duration = time.perf_counter() - convert_started
        convert_sec += convert_duration
        if collect_latency:
            convert_latencies.append(convert_duration)
        nonzero_pixels += int(np.count_nonzero(raw))
        last_raw = raw

    return {
        "timing_sec": {
            **_empty_timing(),
            "rgb_to_gray64": convert_sec,
        },
        "latency_sec": {
            **_empty_latencies(),
            "rgb_to_gray64": convert_latencies,
        },
        "nonzero_pixels": nonzero_pixels,
        "last_raw": last_raw,
    }


def _run_perspective_reuse_gray64_pass(
    *,
    state: dict[str, np.ndarray],
    batch_size: int,
    player_count: int,
    trail_render_mode: str,
    palettes: Sequence[Sequence[Sequence[Sequence[int]]]],
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
    collect_latency: bool,
) -> dict[str, Any]:
    if _render_source_state_canvas_gray64_player_perspectives is None:
        raise ValueError("perspective-reuse renderer is not importable")

    render_sec = 0.0
    render_latencies: list[float] = []
    nonzero_pixels = 0
    last_raw: np.ndarray | None = None
    last_perspective_frames: np.ndarray | None = None

    for row in range(batch_size):
        render_started = time.perf_counter()
        if allocation_mode == "reuse":
            if scratch is None:
                raise ValueError("scratch is required in reuse allocation mode")
            frames = _render_source_state_canvas_gray64_player_perspectives(
                state,
                row=row,
                player_rgbs=palettes[row],
                out=scratch.raw_perspectives,
                rgb_base_out=scratch.rgb_base,
                rgb_work_out=scratch.rgb,
                trail_cache=scratch.trail_caches[row],
                downsample_scratch=scratch.downsample,
                trail_render_mode=trail_render_mode,
            )
        else:
            frames = _render_source_state_canvas_gray64_player_perspectives(
                state,
                row=row,
                player_rgbs=palettes[row],
                trail_render_mode=trail_render_mode,
            )
        render_duration = time.perf_counter() - render_started
        render_sec += render_duration
        if collect_latency:
            render_latencies.append(render_duration)
        nonzero_pixels += int(np.count_nonzero(frames))
        last_raw = frames[-1]
        last_perspective_frames = frames

    return {
        "timing_sec": {
            **_empty_timing(),
            "perspective_reuse_gray64": render_sec,
        },
        "latency_sec": {
            **_empty_latencies(),
            "perspective_reuse_gray64": render_latencies,
        },
        "nonzero_pixels": nonzero_pixels,
        "last_raw": last_raw,
        "last_perspective_frames": last_perspective_frames,
    }


def _render_one(
    *,
    state: dict[str, np.ndarray],
    row: int,
    palette: Sequence[Sequence[int]],
    trail_render_mode: str,
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
) -> np.ndarray:
    if allocation_mode == "reuse":
        if scratch is None:
            raise ValueError("scratch is required in reuse allocation mode")
        return render_source_state_canvas_gray64(
            state,
            row=row,
            out=scratch.raw,
            rgb_out=scratch.rgb,
            player_rgb=palette,
            trail_render_mode=trail_render_mode,
            downsample_scratch=scratch.downsample,
        )
    return render_source_state_canvas_gray64(
        state,
        row=row,
        player_rgb=palette,
        trail_render_mode=trail_render_mode,
    )


def _render_one_rgb(
    *,
    state: dict[str, np.ndarray],
    row: int,
    palette: Sequence[Sequence[int]],
    trail_render_mode: str,
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
) -> np.ndarray:
    if allocation_mode == "reuse":
        if scratch is None:
            raise ValueError("scratch is required in reuse allocation mode")
        return render_source_state_rgb_canvas_like(
            state,
            row=row,
            out=scratch.rgb,
            frame_size=SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
            player_rgb=palette,
            trail_render_mode=trail_render_mode,
        )
    return render_source_state_rgb_canvas_like(
        state,
        row=row,
        frame_size=SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        player_rgb=palette,
        trail_render_mode=trail_render_mode,
    )


def _pre_rendered_rgb_frame(
    *,
    state: dict[str, np.ndarray],
    trail_render_mode: str,
    palette: Sequence[Sequence[int]],
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
) -> np.ndarray:
    return _render_one_rgb(
        state=state,
        row=0,
        palette=palette,
        trail_render_mode=trail_render_mode,
        allocation_mode=allocation_mode,
        scratch=scratch,
    ).copy()


def _one_normalized_frame(
    *,
    state: dict[str, np.ndarray],
    trail_render_mode: str,
    palette: Sequence[Sequence[int]],
    allocation_mode: AllocationMode,
    scratch: "_DirectScratch | None",
) -> np.ndarray:
    raw = _render_one(
        state=state,
        row=0,
        palette=palette,
        trail_render_mode=trail_render_mode,
        allocation_mode=allocation_mode,
        scratch=scratch,
    )
    if allocation_mode == "reuse":
        if scratch is None:
            raise ValueError("scratch is required in reuse allocation mode")
        return normalize_source_state_gray64(raw, out=scratch.normalized).copy()
    return normalize_source_state_gray64(raw)


def _update_stack_with_frame_only(
    stack: SourceStateGray64Stack4,
    frame: np.ndarray,
    *,
    batch_size: int,
    player_count: int,
) -> np.ndarray:
    for row in range(batch_size):
        stack.stack[row, :, :-1] = stack.stack[row, :, 1:]
        for player in range(player_count):
            stack.stack[row, player, -1] = frame[0]
    return stack.stack.copy()


class _DirectScratch:
    def __init__(self, *, batch_size: int, player_count: int) -> None:
        self.raw = np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
        self.raw_perspectives = np.empty(
            (int(player_count), *SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            dtype=np.uint8,
        )
        self.rgb = np.empty(
            (
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                3,
            ),
            dtype=np.uint8,
        )
        self.rgb_base = np.empty_like(self.rgb)
        self.normalized = np.empty(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.float32)
        self.downsample = SourceStateGray64DownsampleScratch(
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        )
        self.trail_caches = [
            SourceStateBrowserLineTrailLayerCache()
            for _ in range(int(batch_size))
        ]


def _cell(
    *,
    kind: str,
    batch_size: int,
    player_count: int,
    trail_length: int,
    bonus_count: int,
    trail_source: TrailSource = "body",
    trail_render_mode: str,
    label: str,
) -> dict[str, Any]:
    return {
        "kind": _cell_kind(kind),
        "batch_size": _positive_value(batch_size, name="batch_size"),
        "player_count": _positive_value(player_count, name="player_count"),
        "trail_length": _nonnegative_value(trail_length, name="trail_length"),
        "bonus_count": _nonnegative_value(bonus_count, name="bonus_count"),
        "trail_source": _trail_source(trail_source),
        "trail_render_mode": _trail_render_mode(trail_render_mode),
        "label": label,
    }


def _validated_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return _cell(
        kind=str(cell["kind"]),
        batch_size=int(cell["batch_size"]),
        player_count=int(cell["player_count"]),
        trail_length=int(cell["trail_length"]),
        bonus_count=int(cell["bonus_count"]),
        trail_source=_trail_source(str(cell.get("trail_source", "body"))),
        trail_render_mode=str(cell["trail_render_mode"]),
        label=str(cell.get("label") or cell["kind"]),
    )


def _palettes(
    state: dict[str, np.ndarray],
    *,
    batch_size: int,
    player_count: int,
) -> list[list[tuple[tuple[int, int, int], ...]]]:
    return [
        [
            player_perspective_rgb_palette(
                state,
                row=row,
                controlled_player=player,
                player_count=player_count,
            )
            for player in range(player_count)
        ]
        for row in range(batch_size)
    ]


def _advance_static_state_clock(state: dict[str, np.ndarray]) -> None:
    state["tick"] += np.int32(1)
    state["elapsed_ms"] += np.float64(1000.0 / 60.0)


def _empty_timing() -> dict[str, float]:
    return {
        "rgb_render": 0.0,
        "rgb_to_gray64": 0.0,
        "gray64_render": 0.0,
        "perspective_reuse_gray64": 0.0,
        "normalize": 0.0,
        "stack_update": 0.0,
        "stack_copy_only": 0.0,
        "gpu_transfer": 0.0,
        "measured_elapsed": 0.0,
    }


def _empty_latencies() -> dict[str, list[float]]:
    return {
        "rgb_render": [],
        "rgb_to_gray64": [],
        "gray64_render": [],
        "perspective_reuse_gray64": [],
        "normalize": [],
        "stack_update": [],
        "stack_copy_only": [],
        "gpu_transfer": [],
    }


def _denominators(
    *,
    kind: str,
    batch_size: int,
    player_count: int,
    iterations: int,
    stack_present: bool,
) -> dict[str, int]:
    policy_rows = batch_size * player_count * iterations
    render_calls = (
        policy_rows
        if kind in (GRAY64_RENDER_ONLY, GRAY64_RENDER_NORMALIZE)
        else 0
    )
    rgb_render_calls = policy_rows if kind == RGB_RENDER_ONLY else 0
    rgb_to_gray64_calls = policy_rows if kind == RGB_TO_GRAY64_ONLY else 0
    perspective_reuse_calls = (
        batch_size * iterations if kind == PERSPECTIVE_REUSE_GRAY64 else 0
    )
    perspective_reuse_frames = policy_rows if kind == PERSPECTIVE_REUSE_GRAY64 else 0
    stack_render_calls = policy_rows if kind == FULL_STACK else 0
    direct_normalized = policy_rows if kind == GRAY64_RENDER_NORMALIZE else 0
    stack_normalized = policy_rows if kind == FULL_STACK else 0
    stack_copy_only_frames = policy_rows if kind == STACK_COPY_ONLY else 0
    return {
        "env_rows": int(batch_size * iterations),
        "policy_rows": int(policy_rows),
        "render_calls": int(render_calls + rgb_render_calls + stack_render_calls),
        "normalized_frames": int(direct_normalized + stack_normalized),
        "stack_updates": int(iterations if stack_present else 0),
        "rgb_render_calls": int(rgb_render_calls),
        "rgb_to_gray64_calls": int(rgb_to_gray64_calls),
        "gray64_render_calls": int(render_calls),
        "perspective_reuse_gray64_calls": int(perspective_reuse_calls),
        "perspective_reuse_gray64_frames": int(perspective_reuse_frames),
        "direct_normalized_frames": int(direct_normalized),
        "stack_render_calls_estimated": int(stack_render_calls),
        "stack_normalized_frames_estimated": int(stack_normalized),
        "stack_copy_only_frames_estimated": int(stack_copy_only_frames),
        "stack_bytes_copied": 0,
        "gpu_transfer_bytes": 0,
    }


def _estimated_stack_bytes_copied(batch_size: int, player_count: int) -> int:
    frame_bytes = np.dtype(np.float32).itemsize * 64 * 64
    shift_frames = batch_size * player_count * 3
    insert_frames = batch_size * player_count
    returned_frames = batch_size * player_count * 4
    return int((shift_frames + insert_frames + returned_frames) * frame_bytes)


def _latency_report(latency_sec: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for key, values in latency_sec.items():
        if not values:
            continue
        arr_us = np.asarray(values, dtype=np.float64) * 1_000_000.0
        report[key] = {
            "samples": int(arr_us.size),
            "p50": float(np.percentile(arr_us, 50)),
            "p95": float(np.percentile(arr_us, 95)),
            "p99": float(np.percentile(arr_us, 99)),
            "max": float(np.max(arr_us)),
        }
    return report


def _throughput_report(
    *,
    timing_sec: dict[str, float],
    denominators: dict[str, int],
) -> dict[str, float]:
    return {
        "rgb_render_calls_per_sec": _rate(
            denominators["rgb_render_calls"],
            timing_sec["rgb_render"],
        ),
        "rgb_render_us_per_call": _unit_us(
            timing_sec["rgb_render"],
            denominators["rgb_render_calls"],
        ),
        "rgb_to_gray64_calls_per_sec": _rate(
            denominators["rgb_to_gray64_calls"],
            timing_sec["rgb_to_gray64"],
        ),
        "rgb_to_gray64_us_per_call": _unit_us(
            timing_sec["rgb_to_gray64"],
            denominators["rgb_to_gray64_calls"],
        ),
        "gray64_render_calls_per_sec": _rate(
            denominators["gray64_render_calls"],
            timing_sec["gray64_render"],
        ),
        "gray64_render_us_per_call": _unit_us(
            timing_sec["gray64_render"],
            denominators["gray64_render_calls"],
        ),
        "perspective_reuse_gray64_calls_per_sec": _rate(
            denominators["perspective_reuse_gray64_calls"],
            timing_sec["perspective_reuse_gray64"],
        ),
        "perspective_reuse_gray64_us_per_call": _unit_us(
            timing_sec["perspective_reuse_gray64"],
            denominators["perspective_reuse_gray64_calls"],
        ),
        "perspective_reuse_gray64_us_per_policy_row": _unit_us(
            timing_sec["perspective_reuse_gray64"],
            denominators["perspective_reuse_gray64_frames"],
        ),
        "normalize_us_per_frame": _unit_us(
            timing_sec["normalize"],
            denominators["direct_normalized_frames"],
        ),
        "stack_update_us_per_update": _unit_us(
            timing_sec["stack_update"],
            denominators["stack_updates"],
        ),
        "stack_update_us_per_policy_row": _unit_us(
            timing_sec["stack_update"],
            denominators["policy_rows"],
        ),
        "stack_copy_only_us_per_update": _unit_us(
            timing_sec["stack_copy_only"],
            denominators["stack_updates"],
        ),
        "stack_copy_only_us_per_policy_row": _unit_us(
            timing_sec["stack_copy_only"],
            denominators["policy_rows"],
        ),
        "stack_copy_gib_per_sec": _byte_rate_gib(
            denominators["stack_bytes_copied"],
            timing_sec["stack_update"] + timing_sec["stack_copy_only"],
        ),
        "measured_policy_rows_per_sec": _rate(
            denominators["policy_rows"],
            timing_sec["measured_elapsed"],
        ),
    }


def _density_report(state: dict[str, np.ndarray]) -> dict[str, float | int]:
    env_rows = int(state["body_active"].shape[0])
    active_bodies = int(np.count_nonzero(state["body_active"]))
    active_visual_trails = (
        int(np.count_nonzero(state["visual_trail_active"]))
        if "visual_trail_active" in state
        else 0
    )
    active_bonuses = (
        int(np.count_nonzero(state["bonus_active"]))
        if "bonus_active" in state
        else 0
    )
    return {
        "active_bodies_per_row": active_bodies / env_rows if env_rows else 0.0,
        "body_capacity": int(state["body_active"].shape[1]),
        "active_visual_trails_per_row": (
            active_visual_trails / env_rows if env_rows else 0.0
        ),
        "visual_trail_capacity": (
            int(state["visual_trail_active"].shape[1])
            if "visual_trail_active" in state
            else 0
        ),
        "bonus_count": int(state["bonus_active"].shape[1]) if "bonus_active" in state else 0,
        "active_bonuses_per_row": active_bonuses / env_rows if env_rows else 0.0,
    }


def _array_digest(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


def _rate(count: int, seconds: float) -> float:
    return float(count / seconds) if seconds > 0.0 else 0.0


def _unit_us(seconds: float, count: int) -> float:
    return float((seconds / count) * 1_000_000.0) if count > 0 else 0.0


def _byte_rate_gib(byte_count: int, seconds: float) -> float:
    return float((byte_count / (1024.0**3)) / seconds) if seconds > 0.0 else 0.0


def _parse_int_csv(value: str, *, positive: bool) -> list[int]:
    parsed: list[int] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        number = int(text)
        if positive:
            _positive_value(number, name=text)
        else:
            _nonnegative_value(number, name=text)
        parsed.append(number)
    if not parsed:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    return parsed


def _parse_positive_int_csv(value: str) -> list[int]:
    try:
        return _parse_int_csv(value, positive=True)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_nonnegative_int_csv(value: str) -> list[int]:
    try:
        return _parse_int_csv(value, positive=False)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_trail_render_modes(value: str) -> list[str]:
    if value.strip() == "all":
        return list(TRAIL_RENDER_MODE_ORDER)
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("must contain at least one trail render mode")
    try:
        return [_trail_render_mode(mode) for mode in modes]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_cell_kinds(value: str) -> list[str]:
    kinds = [item.strip() for item in value.split(",") if item.strip()]
    if not kinds:
        raise argparse.ArgumentTypeError("must contain at least one cell kind")
    try:
        return [_cell_kind(kind) for kind in kinds]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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


def _allocation_mode(value: str) -> AllocationMode:
    if value not in ("reuse", "allocate"):
        raise ValueError("allocation_mode must be reuse or allocate")
    return value  # type: ignore[return-value]


def _gpu_transfer_mode(value: str) -> GpuTransferMode:
    if value not in ("on", "off", "auto"):
        raise ValueError("gpu_transfer must be on, off, or auto")
    return value  # type: ignore[return-value]


def _trail_render_mode(value: str) -> str:
    if value not in TRAIL_RENDER_MODE_ORDER:
        supported = ", ".join(TRAIL_RENDER_MODE_ORDER)
        raise ValueError(f"trail_render_mode must be one of [{supported}], got {value!r}")
    return value


def _trail_source(value: str) -> TrailSource:
    if value not in ("body", "visual"):
        raise ValueError(f"trail_source must be 'body' or 'visual', got {value!r}")
    return value  # type: ignore[return-value]


def _cell_kind(value: str) -> str:
    value = CELL_KIND_ALIASES.get(value, value)
    if value not in CELL_KINDS:
        supported = ", ".join(CELL_KINDS)
        raise ValueError(f"cell kind must be one of [{supported}], got {value!r}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=("default", "grid"), default="default")
    parser.add_argument("--batch-sizes", type=_parse_positive_int_csv, default=[16])
    parser.add_argument("--player-counts", type=_parse_positive_int_csv, default=[2])
    parser.add_argument("--trail-lengths", type=_parse_nonnegative_int_csv, default=[0, 64, 256, 1024])
    parser.add_argument("--bonus-counts", type=_parse_nonnegative_int_csv, default=[0])
    parser.add_argument(
        "--trail-render-modes",
        type=_parse_trail_render_modes,
        default=[TRAIL_RENDER_MODE_BROWSER_LINES],
        help=(
            "Comma-separated modes or 'all'. Supported: "
            f"{TRAIL_RENDER_MODE_BROWSER_LINES}, {TRAIL_RENDER_MODE_BODY_CIRCLES_FAST}"
        ),
    )
    parser.add_argument(
        "--cell-kinds",
        type=_parse_cell_kinds,
        default=[FULL_STACK],
        help=f"Comma-separated grid cell kinds: {', '.join(CELL_KINDS)}",
    )
    parser.add_argument(
        "--trail-source",
        choices=("body", "visual"),
        default="body",
        help="Synthetic trail source: body_* fallback or current visual_trail_* fields.",
    )
    parser.add_argument("--iterations", type=_positive_arg, default=1)
    parser.add_argument("--warmup-iterations", type=_nonnegative_arg, default=0)
    parser.add_argument("--allocation-mode", choices=("reuse", "allocate"), default="reuse")
    parser.add_argument("--gpu-transfer", choices=("on", "off", "auto"), default="off")
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan == "default":
        report = run_default_probe(
            iterations=int(args.iterations),
            warmup_iterations=int(args.warmup_iterations),
            allocation_mode=args.allocation_mode,
            gpu_transfer=args.gpu_transfer,
            seed=int(args.seed),
        )
    else:
        report = run_grid_probe(
            batch_sizes=args.batch_sizes,
            player_counts=args.player_counts,
            trail_lengths=args.trail_lengths,
            bonus_counts=args.bonus_counts,
            trail_render_modes=args.trail_render_modes,
            kinds=args.cell_kinds,
            trail_source=args.trail_source,
            iterations=int(args.iterations),
            warmup_iterations=int(args.warmup_iterations),
            allocation_mode=args.allocation_mode,
            gpu_transfer=args.gpu_transfer,
            seed=int(args.seed),
        )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    _print_plain(report)
    return 0


def _print_plain(report: dict[str, Any]) -> None:
    print(f"{report['schema_id']} plan={report['plan_id']} cells={report['cell_count']}")
    for cell in report["cells"]:
        workload = cell["workload"]
        if cell.get("status") == "unsupported":
            print(
                " ".join(
                    (
                        f"{cell['label']}:",
                        f"kind={cell['kind']}",
                        f"B={workload['batch_size']}",
                        f"P={workload['player_count']}",
                        f"L={workload['trail_length']}",
                        f"trail_source={workload.get('trail_source', 'body')}",
                        "status=unsupported",
                        f"reason={cell.get('unsupported_reason', 'unsupported')}",
                    )
                )
            )
            continue
        throughput = cell["throughput"]
        denominators = cell["denominators"]
        timing = cell["timing_sec"]
        density = cell["density"]
        visible_nonzero = (
            cell["integrity"]["last_stack_nonzero_pixels"]
            or cell["integrity"]["last_raw_nonzero_pixels"]
        )
        print(
            " ".join(
                (
                    f"{cell['label']}:",
                    f"kind={cell['kind']}",
                    f"B={workload['batch_size']}",
                    f"P={workload['player_count']}",
                    f"L={workload['trail_length']}",
                    f"bonus={workload['bonus_count']}",
                    f"trail_source={workload.get('trail_source', 'body')}",
                    f"mode={workload['trail_render_mode']}",
                    "rgb_us="
                    f"{_format_optional_us(throughput['rgb_render_us_per_call'], denominators['rgb_render_calls'])}",
                    "rgb_to_gray64_us="
                    f"{_format_optional_us(throughput['rgb_to_gray64_us_per_call'], denominators['rgb_to_gray64_calls'])}",
                    "gray64_us="
                    f"{_format_optional_us(throughput['gray64_render_us_per_call'], denominators['gray64_render_calls'])}",
                    "perspective_reuse_us_per_policy_row="
                    f"{_format_optional_us(throughput['perspective_reuse_gray64_us_per_policy_row'], denominators['perspective_reuse_gray64_frames'])}",
                    "normalize_us="
                    f"{_format_optional_us(throughput['normalize_us_per_frame'], denominators['direct_normalized_frames'])}",
                    "stack_ms_per_update="
                    f"{_format_optional_ms(throughput['stack_update_us_per_update'], denominators['stack_updates'])}",
                    "stack_copy_ms_per_update="
                    f"{_format_optional_ms(throughput['stack_copy_only_us_per_update'], denominators['stack_updates'])}",
                    f"measured_total_ms={timing['measured_elapsed'] * 1000.0:.3f}",
                    f"policy_rows_per_sec={throughput['measured_policy_rows_per_sec']:.1f}",
                    f"active_bodies={density['active_bodies_per_row']:.1f}",
                    f"active_visual={density.get('active_visual_trails_per_row', 0.0):.1f}",
                    f"nonzero={visible_nonzero}",
                )
            )
        )


def _format_optional_us(value_us: float, denominator: int) -> str:
    return f"{value_us:.1f}" if int(denominator) > 0 else "n/a"


def _format_optional_ms(value_us: float, denominator: int) -> str:
    return f"{value_us / 1000.0:.3f}" if int(denominator) > 0 else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
