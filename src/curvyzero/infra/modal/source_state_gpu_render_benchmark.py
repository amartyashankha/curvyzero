"""Isolated Modal GPU benchmark for source-state render prototypes.

This does not import trainers, checkpoints, runs, or Modal Volumes. It measures
whether batched GPU rendering is worth pursuing for CurvyTron observations.

Run from repo root:

    uv run --extra modal modal run -m curvyzero.infra.modal.source_state_gpu_render_benchmark
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal.mctx_dependency_smoke import JAX_VERSION


APP_NAME = "curvyzero-source-state-gpu-render-benchmark"
REMOTE_ROOT = Path("/repo")

RENDER_MODE_BROWSER_LINES = "browser_lines"
RENDER_MODE_IDS = {
    RENDER_MODE_BROWSER_LINES: 0,
}

RENDER_SURFACE_DIRECT_GRAY64 = "direct_gray64"
RENDER_SURFACE_BLOCK_704_GRAY64 = "block_704_gray64"
RENDER_SURFACES = {
    RENDER_SURFACE_DIRECT_GRAY64,
    RENDER_SURFACE_BLOCK_704_GRAY64,
}

SYNTHETIC_BACKGROUND_LUMA = 34
SYNTHETIC_INVALID_OWNER_LUMA = 120
SYNTHETIC_PLAYER_LUMA_BY_INDEX = (76, 150, 76, 217)
SYNTHETIC_BACKGROUND_LUMA_FLOAT = 34.0
SYNTHETIC_INVALID_OWNER_LUMA_FLOAT = 120.0
SYNTHETIC_PLAYER_LUMA_FLOAT_BY_INDEX = (76.245, 149.685, 75.945, 217.335)

SCHEMA_ID = "curvyzero_source_state_gpu_render_benchmark/v0"
RENDERER_IMPL_ID = "synthetic_source_state_jax/v1"

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"jax[cuda12]=={JAX_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=2.0)
def run_source_state_gpu_render_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_source_state_gpu_render_benchmark_impl(config)
    except Exception as exc:  # pragma: no cover - Modal remote diagnostics.
        import traceback

        return {
            "schema_id": SCHEMA_ID,
            "ok": False,
            "app_name": APP_NAME,
            "renderer_impl_id": RENDERER_IMPL_ID,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _run_source_state_gpu_render_benchmark_impl(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    checked = _validate_config(config)
    render_mode_id = RENDER_MODE_IDS[checked["render_mode"]]
    state = _synthetic_source_state(np=np, config=checked)
    render_fn = _make_jax_render_fn(
        jax=jax,
        jnp=jnp,
        config=checked,
        render_mode_id=render_mode_id,
    )

    timings = _benchmark_render(
        jax=jax,
        np=np,
        render_fn=render_fn,
        state=state,
        config=checked,
    )
    verification = _verify_against_cpu(
        jax=jax,
        np=np,
        state=state,
        config=checked,
        render_mode_id=render_mode_id,
    )

    return {
        "schema_id": SCHEMA_ID,
        "ok": True,
        "app_name": APP_NAME,
        "renderer_impl_id": RENDERER_IMPL_ID,
        "nvidia_smi": _nvidia_smi(),
        "packages": {
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "numpy": _version_or_missing("numpy"),
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "device_count": len(jax.devices()),
            "devices": [str(device) for device in jax.devices()],
        },
        "config": checked,
        "semantics": {
            "input_shape": "synthetic source-state arrays, not live env rows",
            "output_shape": [
                checked["batch_size"],
                1,
                checked["target_size"],
                checked["target_size"],
            ],
            "render_surface": checked["render_surface"],
            "direct_gray64": (
                "samples one point per 64x64 cell; fast economics probe, "
                "not trusted browser fidelity"
            ),
            "block_704_gray64": (
                "checks all 11x11 source pixels for each 64x64 cell with "
                "production-like pixel-space trail/head raster; closer to 704->64 "
                "cost but still no full RGB canvas"
            ),
            "browser_lines": "approximates connected same-owner source-state segments",
            "composition": "luma overwrite approximation; production parity must be checked",
        },
        "known_gaps": [
            "No live CurvyTron env or LightZero trainer/checkpoint imports.",
            "No Modal Volume access.",
            "No full RGB canvas parity yet; block_704_gray64 can be compared to production CPU render.",
            "Synthetic trail rows preserve tensor shape, not exact game histories.",
            "JAX path does not yet implement exact owner draw grouping, RGB overwrite, or bonus sprites.",
        ],
        "timings": timings,
        "verification": verification,
    }


def _benchmark_render(
    *,
    jax: Any,
    np: Any,
    render_fn: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    transfer_times: list[float] = []
    render_times: list[float] = []
    readback_times: list[float] = []
    end_to_end_times: list[float] = []
    compile_first_render_sec = None
    warmup_runs = int(config["warmup_runs"])
    steady_runs = int(config["steady_runs"])

    for iteration in range(warmup_runs + steady_runs):
        started = time.perf_counter()
        device_state = _copy_state_to_device(jax=jax, state=state)
        transfer_sec = time.perf_counter() - started

        started = time.perf_counter()
        output_device = render_fn(device_state)
        output_device.block_until_ready()
        render_sec = time.perf_counter() - started
        if iteration == 0:
            compile_first_render_sec = render_sec

        readback_sec = 0.0
        if bool(config["transfer_output"]):
            started = time.perf_counter()
            _ = np.asarray(output_device)
            readback_sec = time.perf_counter() - started

        if iteration >= warmup_runs:
            transfer_times.append(transfer_sec)
            render_times.append(render_sec)
            readback_times.append(readback_sec)
            end_to_end_times.append(transfer_sec + render_sec + readback_sec)

    target_size = int(config["target_size"])
    pixels = int(config["batch_size"]) * target_size * target_size
    if config["render_surface"] == RENDER_SURFACE_BLOCK_704_GRAY64:
        block = int(config["frame_size"]) // target_size
        effective_pixel_tests = pixels * block * block * int(config["trail_slots"])
    else:
        effective_pixel_tests = pixels * int(config["trail_slots"])

    return {
        "warmup_runs": warmup_runs,
        "steady_runs": steady_runs,
        "pixel_count": pixels,
        "effective_pixel_trail_slot_tests": effective_pixel_tests,
        "host_to_device_sec_median": _median(transfer_times),
        "compile_first_render_sec": compile_first_render_sec,
        "device_render_sec_median": _median(render_times),
        "device_to_host_sec_median": _median(readback_times)
        if config["transfer_output"]
        else None,
        "end_to_end_sec_median": _median(end_to_end_times),
        "device_frames_per_sec": _rate(float(config["batch_size"]), _median(render_times)),
        "end_to_end_frames_per_sec": _rate(
            float(config["batch_size"]),
            _median(end_to_end_times),
        ),
        "transfer_output_measured": bool(config["transfer_output"]),
    }


def _make_jax_render_fn(
    *,
    jax: Any,
    jnp: Any,
    config: dict[str, Any],
    render_mode_id: int,
) -> Any:
    target_size = int(config["target_size"])
    frame_size = int(config["frame_size"])
    map_size = float(config["map_size"])
    bonus_count = int(config["bonus_count"])
    render_surface = str(config["render_surface"])

    @jax.jit
    def render(device_state: dict[str, Any]) -> Any:
        if render_surface == RENDER_SURFACE_BLOCK_704_GRAY64:
            return _jax_render_block_704_gray64(
                jnp=jnp,
                lax=jax.lax,
                state=device_state,
                frame_size=frame_size,
                target_size=target_size,
                map_size=map_size,
                bonus_count=bonus_count,
                render_mode_id=render_mode_id,
            )
        return _jax_render_direct_gray64(
            jnp=jnp,
            state=device_state,
            target_size=target_size,
            map_size=map_size,
            bonus_count=bonus_count,
            render_mode_id=render_mode_id,
        )

    return render


def _jax_render_direct_gray64(
    *,
    jnp: Any,
    state: dict[str, Any],
    target_size: int,
    map_size: float,
    bonus_count: int,
    render_mode_id: int,
) -> Any:
    grid_x = (
        (jnp.arange(target_size, dtype=jnp.float32) + 0.5)
        * float(map_size)
        / float(target_size)
    )
    grid_y = (
        (jnp.arange(target_size, dtype=jnp.float32) + 0.5)
        * float(map_size)
        / float(target_size)
    )
    world_x = grid_x[None, None, None, :]
    world_y = grid_y[None, None, :, None]

    trail_dx = world_x - state["trail_x"][:, :, None, None]
    trail_dy = world_y - state["trail_y"][:, :, None, None]
    trail_radius_sq = state["trail_radius"][:, :, None, None] ** 2
    hit = trail_dx * trail_dx + trail_dy * trail_dy <= trail_radius_sq

    if render_mode_id == RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]:
        prev_x = jnp.concatenate([state["trail_x"][:, :1], state["trail_x"][:, :-1]], axis=1)
        prev_y = jnp.concatenate([state["trail_y"][:, :1], state["trail_y"][:, :-1]], axis=1)
        prev_owner = jnp.concatenate(
            [state["trail_owner"][:, :1], state["trail_owner"][:, :-1]],
            axis=1,
        )
        prev_active = jnp.concatenate(
            [jnp.zeros_like(state["trail_active"][:, :1]), state["trail_active"][:, :-1]],
            axis=1,
        )
        hit = hit | _segment_hits(
            jnp=jnp,
            world_x=world_x,
            world_y=world_y,
            x=state["trail_x"],
            y=state["trail_y"],
            prev_x=prev_x,
            prev_y=prev_y,
            radius_sq=trail_radius_sq,
            owner=state["trail_owner"],
            prev_owner=prev_owner,
            active=prev_active,
            break_before=state["trail_break_before"],
        )

    hit = hit & (state["trail_active"][:, :, None, None] != 0)
    player_luma = jnp.asarray(SYNTHETIC_PLAYER_LUMA_BY_INDEX, dtype=jnp.uint8)
    owner = state["trail_owner"]
    trail_value = jnp.where(
        owner < 0,
        jnp.uint8(SYNTHETIC_INVALID_OWNER_LUMA),
        jnp.take(player_luma, jnp.mod(owner, player_luma.shape[0])),
    ).astype(jnp.uint8)
    trail_layer = jnp.max(
        jnp.where(hit, trail_value[:, :, None, None], jnp.zeros((), dtype=jnp.uint8)),
        axis=1,
    )
    out = jnp.maximum(
        jnp.full_like(trail_layer, SYNTHETIC_BACKGROUND_LUMA, dtype=jnp.uint8),
        trail_layer,
    )
    out = _draw_direct_bonus_and_heads(
        jnp=jnp,
        out=out,
        state=state,
        world_x=world_x,
        world_y=world_y,
        bonus_count=bonus_count,
    )
    return out[:, None, :, :].astype(jnp.uint8)


def _jax_render_block_704_gray64(
    *,
    jnp: Any,
    lax: Any,
    state: dict[str, Any],
    frame_size: int,
    target_size: int,
    map_size: float,
    bonus_count: int,
    render_mode_id: int,
) -> Any:
    if frame_size % target_size != 0:
        raise ValueError("frame_size must be divisible by target_size")
    block = frame_size // target_size
    source = (jnp.arange(frame_size, dtype=jnp.float32) + 0.5) * float(map_size) / float(
        frame_size
    )
    source_blocks = source.reshape(target_size, block)
    world_x = source_blocks[None, None, :, None, :]
    world_y = source_blocks[None, :, None, :, None]
    pixel_blocks = jnp.arange(frame_size, dtype=jnp.float32).reshape(target_size, block)
    pixel_x = pixel_blocks[None, None, :, None, :]
    pixel_y = pixel_blocks[None, :, None, :, None]
    pixel_scale = float(frame_size) / float(map_size)
    image = jnp.full(
        (state["trail_x"].shape[0], target_size, target_size, block, block),
        SYNTHETIC_BACKGROUND_LUMA_FLOAT,
        dtype=jnp.float32,
    )

    def draw_trail_slot(slot: Any, current: Any) -> Any:
        x = state["trail_x"][:, slot] * pixel_scale
        y = state["trail_y"][:, slot] * pixel_scale
        radius = state["trail_radius"][:, slot] * pixel_scale
        owner = state["trail_owner"][:, slot]
        active = state["trail_active"][:, slot] != 0
        dx = pixel_x - x[:, None, None, None, None]
        dy = pixel_y - y[:, None, None, None, None]
        radius_sq = radius[:, None, None, None, None] ** 2
        hit = dx * dx + dy * dy <= radius_sq

        if render_mode_id == RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]:
            previous = jnp.maximum(slot - 1, 0)
            prev_x = state["trail_x"][:, previous] * pixel_scale
            prev_y = state["trail_y"][:, previous] * pixel_scale
            prev_owner = state["trail_owner"][:, previous]
            prev_active = state["trail_active"][:, previous] != 0
            hit = hit | _segment_hits_one_slot(
                jnp=jnp,
                world_x=pixel_x,
                world_y=pixel_y,
                x=x,
                y=y,
                prev_x=prev_x,
                prev_y=prev_y,
                radius_sq=radius_sq,
                owner=owner,
                prev_owner=prev_owner,
                active=prev_active,
                break_before=state["trail_break_before"][:, slot],
                slot=slot,
            )

        player_luma = jnp.asarray(SYNTHETIC_PLAYER_LUMA_FLOAT_BY_INDEX, dtype=jnp.float32)
        value = jnp.where(
            owner < 0,
            jnp.float32(SYNTHETIC_INVALID_OWNER_LUMA_FLOAT),
            jnp.take(player_luma, jnp.mod(owner, player_luma.shape[0])),
        ).astype(jnp.float32)
        return jnp.where(
            hit & active[:, None, None, None, None],
            value[:, None, None, None, None],
            current,
        )

    image = lax.fori_loop(0, state["trail_x"].shape[1], draw_trail_slot, image)
    image = _draw_block_bonus_and_heads(
        jnp=jnp,
        lax=lax,
        image=image,
        state=state,
        world_x=world_x,
        world_y=world_y,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        frame_size=frame_size,
        map_size=map_size,
        bonus_count=bonus_count,
    )
    gray = jnp.rint(jnp.mean(image.astype(jnp.float32), axis=(3, 4)))
    return jnp.clip(gray, 0, 255).astype(jnp.uint8)[:, None, :, :]


def _segment_hits(
    *,
    jnp: Any,
    world_x: Any,
    world_y: Any,
    x: Any,
    y: Any,
    prev_x: Any,
    prev_y: Any,
    radius_sq: Any,
    owner: Any,
    prev_owner: Any,
    active: Any,
    break_before: Any,
) -> Any:
    vx = x - prev_x
    vy = y - prev_y
    length_sq = vx * vx + vy * vy
    raw_t = (
        (world_x - prev_x[:, :, None, None]) * vx[:, :, None, None]
        + (world_y - prev_y[:, :, None, None]) * vy[:, :, None, None]
    ) / jnp.maximum(length_sq[:, :, None, None], 0.0001)
    t = jnp.clip(raw_t, 0.0, 1.0)
    nearest_x = prev_x[:, :, None, None] + t * vx[:, :, None, None]
    nearest_y = prev_y[:, :, None, None] + t * vy[:, :, None, None]
    dx = world_x - nearest_x
    dy = world_y - nearest_y
    ok = (
        (break_before == 0)
        & (active != 0)
        & (owner == prev_owner)
        & (length_sq > 0.0001)
    )
    return (dx * dx + dy * dy <= radius_sq) & ok[:, :, None, None]


def _segment_hits_one_slot(
    *,
    jnp: Any,
    world_x: Any,
    world_y: Any,
    x: Any,
    y: Any,
    prev_x: Any,
    prev_y: Any,
    radius_sq: Any,
    owner: Any,
    prev_owner: Any,
    active: Any,
    break_before: Any,
    slot: Any,
) -> Any:
    vx = x - prev_x
    vy = y - prev_y
    length_sq = vx * vx + vy * vy
    raw_t = (
        (world_x - prev_x[:, None, None, None, None]) * vx[:, None, None, None, None]
        + (world_y - prev_y[:, None, None, None, None]) * vy[:, None, None, None, None]
    ) / jnp.maximum(length_sq[:, None, None, None, None], 0.0001)
    t = jnp.clip(raw_t, 0.0, 1.0)
    nearest_x = prev_x[:, None, None, None, None] + t * vx[:, None, None, None, None]
    nearest_y = prev_y[:, None, None, None, None] + t * vy[:, None, None, None, None]
    dx = world_x - nearest_x
    dy = world_y - nearest_y
    ok = (
        (slot > 0)
        & (break_before == 0)
        & active
        & (owner == prev_owner)
        & (length_sq > 0.0001)
    )
    return (dx * dx + dy * dy <= radius_sq) & ok[:, None, None, None, None]


def _draw_direct_bonus_and_heads(
    *,
    jnp: Any,
    out: Any,
    state: dict[str, Any],
    world_x: Any,
    world_y: Any,
    bonus_count: int,
) -> Any:
    if bonus_count > 0:
        bonus_dx = world_x - state["bonus_x"][:, :, None, None]
        bonus_dy = world_y - state["bonus_y"][:, :, None, None]
        bonus_hit = (
            bonus_dx * bonus_dx + bonus_dy * bonus_dy
            <= state["bonus_radius"][:, :, None, None] ** 2
        ) & (state["bonus_active"][:, :, None, None] != 0)
        bonus_value = (144 + (state["bonus_type"] % 4) * 24).astype(jnp.uint8)
        bonus_layer = jnp.max(
            jnp.where(
                bonus_hit,
                bonus_value[:, :, None, None],
                jnp.zeros((), dtype=jnp.uint8),
            ),
            axis=1,
        )
        out = jnp.maximum(out, bonus_layer)

    player_count = state["head_x"].shape[1]
    player_luma = jnp.asarray(SYNTHETIC_PLAYER_LUMA_BY_INDEX, dtype=jnp.uint8)
    player_indices = jnp.arange(player_count, dtype=jnp.int32)
    player_values = jnp.take(player_luma, jnp.mod(player_indices, player_luma.shape[0]))
    head_dx = world_x - state["head_x"][:, :, None, None]
    head_dy = world_y - state["head_y"][:, :, None, None]
    head_hit = (
        head_dx * head_dx + head_dy * head_dy <= state["head_radius"][:, :, None, None] ** 2
    ) & (state["head_alive"][:, :, None, None] != 0)
    head_layer = jnp.max(
        jnp.where(head_hit, player_values[None, :, None, None], jnp.zeros((), dtype=jnp.uint8)),
        axis=1,
    )
    return jnp.maximum(out, head_layer)


def _draw_block_bonus_and_heads(
    *,
    jnp: Any,
    lax: Any,
    image: Any,
    state: dict[str, Any],
    world_x: Any,
    world_y: Any,
    pixel_x: Any,
    pixel_y: Any,
    frame_size: int,
    map_size: float,
    bonus_count: int,
) -> Any:
    def draw_bonus(index: Any, current: Any) -> Any:
        dx = world_x - state["bonus_x"][:, index, None, None, None, None]
        dy = world_y - state["bonus_y"][:, index, None, None, None, None]
        radius = state["bonus_radius"][:, index, None, None, None, None]
        hit = (
            dx * dx + dy * dy <= radius * radius
        ) & (state["bonus_active"][:, index, None, None, None, None] != 0)
        value = (144 + (state["bonus_type"][:, index] % 4) * 24).astype(jnp.float32)
        return jnp.where(hit, value[:, None, None, None, None], current)

    def draw_head(index: Any, current: Any) -> Any:
        px = jnp.rint(
            state["head_x"][:, index] * float(frame_size - 1) / float(map_size)
        )
        py = jnp.rint(
            state["head_y"][:, index] * float(frame_size - 1) / float(map_size)
        )
        radius = jnp.ceil(
            state["head_radius"][:, index] * float(frame_size) / float(map_size)
        )
        dx = pixel_x - px[:, None, None, None, None]
        dy = pixel_y - py[:, None, None, None, None]
        hit = (dx * dx + dy * dy <= radius[:, None, None, None, None] ** 2) & (
            state["head_alive"][:, index, None, None, None, None] != 0
        )
        player_luma = jnp.asarray(SYNTHETIC_PLAYER_LUMA_FLOAT_BY_INDEX, dtype=jnp.float32)
        value = jnp.take(player_luma, jnp.mod(index, player_luma.shape[0]))
        return jnp.where(hit, value, current)

    if bonus_count > 0:
        image = lax.fori_loop(0, bonus_count, draw_bonus, image)
    return lax.fori_loop(0, state["head_x"].shape[1], draw_head, image)


def _copy_state_to_device(*, jax: Any, state: dict[str, Any]) -> dict[str, Any]:
    copied = {key: jax.device_put(value) for key, value in state.items()}
    for value in copied.values():
        value.block_until_ready()
    return copied


def _verify_against_cpu(
    *,
    jax: Any,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
    render_mode_id: int,
) -> dict[str, Any]:
    import jax.numpy as jnp

    verify_rows = min(int(config["verify_rows"]), int(config["batch_size"]))
    if verify_rows <= 0:
        return {"rows": 0, "status": "skipped"}

    target_size = int(config["target_size"])
    operations = verify_rows * target_size * target_size * int(config["trail_slots"])
    if operations > int(config["cpu_verify_max_pixel_trail_tests"]):
        return {
            "rows": verify_rows,
            "status": "skipped_cpu_reference_too_large",
            "pixel_trail_slot_tests": operations,
            "max_pixel_trail_slot_tests": int(config["cpu_verify_max_pixel_trail_tests"]),
        }

    sliced_state = {key: value[:verify_rows].copy() for key, value in state.items()}
    verify_render_fn = _make_jax_render_fn(
        jax=jax,
        jnp=jnp,
        config={**config, "batch_size": verify_rows},
        render_mode_id=render_mode_id,
    )
    device_state = _copy_state_to_device(jax=jax, state=sliced_state)
    started = time.perf_counter()
    gpu_device = verify_render_fn(device_state)
    gpu_device.block_until_ready()
    gpu_sec = time.perf_counter() - started
    gpu = np.asarray(gpu_device)

    started = time.perf_counter()
    cpu_reference_kind = "synthetic_direct_gray64"
    if config["render_surface"] == RENDER_SURFACE_DIRECT_GRAY64:
        cpu = _cpu_render_direct_gray64(
            np=np,
            state=sliced_state,
            config={**config, "batch_size": verify_rows},
            render_mode_id=render_mode_id,
        )
    else:
        cpu_reference_kind = "production_render_source_state_canvas_gray64_browser_lines"
        if int(config["target_size"]) != 64 or int(config["frame_size"]) != 704:
            return {
                "rows": verify_rows,
                "status": "skipped_production_reference_requires_704_to_64",
                "render_surface": config["render_surface"],
                "frame_size": int(config["frame_size"]),
                "target_size": int(config["target_size"]),
            }
        cpu = _cpu_render_production_canvas_gray64(
            np=np,
            state=sliced_state,
            config={**config, "batch_size": verify_rows},
        )
    cpu_sec = time.perf_counter() - started

    diff = np.abs(gpu.astype(np.int16) - cpu.astype(np.int16))
    mismatch_count = int(np.count_nonzero(diff))
    total_values = int(diff.size)
    return {
        "rows": verify_rows,
        "status": "checked",
        "cpu_reference_kind": cpu_reference_kind,
        "exact_parity": bool(mismatch_count == 0),
        "mismatch_count": mismatch_count,
        "mismatch_fraction": float(mismatch_count / total_values) if total_values else 0.0,
        "max_abs_diff": int(diff.max()) if diff.size else 0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "cpu_reference_sec": cpu_sec,
        "gpu_verify_render_sec": gpu_sec,
        "pixel_trail_slot_tests": operations,
    }


def _cpu_render_direct_gray64(
    *,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
    render_mode_id: int,
) -> Any:
    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    trail_slots = int(config["trail_slots"])
    bonus_count = int(config["bonus_count"])
    target_size = int(config["target_size"])
    map_size = float(config["map_size"])
    out = np.full(
        (batch_size, 1, target_size, target_size),
        SYNTHETIC_BACKGROUND_LUMA,
        dtype=np.uint8,
    )

    for row in range(batch_size):
        for py in range(target_size):
            world_y = (float(py) + 0.5) * map_size / float(target_size)
            for px in range(target_size):
                world_x = (float(px) + 0.5) * map_size / float(target_size)
                value = SYNTHETIC_BACKGROUND_LUMA
                for slot in range(trail_slots):
                    if not bool(state["trail_active"][row, slot]):
                        continue
                    owner = int(state["trail_owner"][row, slot])
                    radius = float(state["trail_radius"][row, slot])
                    dx = world_x - float(state["trail_x"][row, slot])
                    dy = world_y - float(state["trail_y"][row, slot])
                    hit = dx * dx + dy * dy <= radius * radius
                    if (
                        not hit
                        and render_mode_id == RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]
                        and slot > 0
                        and not bool(state["trail_break_before"][row, slot])
                        and bool(state["trail_active"][row, slot - 1])
                        and int(state["trail_owner"][row, slot - 1]) == owner
                    ):
                        hit = _point_hits_segment(
                            world_x=world_x,
                            world_y=world_y,
                            ax=float(state["trail_x"][row, slot - 1]),
                            ay=float(state["trail_y"][row, slot - 1]),
                            bx=float(state["trail_x"][row, slot]),
                            by=float(state["trail_y"][row, slot]),
                            radius=radius,
                        )
                    if hit:
                        value = max(value, _synthetic_owner_luma(owner))

                for bonus in range(bonus_count):
                    if not bool(state["bonus_active"][row, bonus]):
                        continue
                    dx = world_x - float(state["bonus_x"][row, bonus])
                    dy = world_y - float(state["bonus_y"][row, bonus])
                    radius = float(state["bonus_radius"][row, bonus])
                    if dx * dx + dy * dy <= radius * radius:
                        value = max(value, 144 + int(state["bonus_type"][row, bonus] % 4) * 24)

                for player in range(player_count):
                    if not bool(state["head_alive"][row, player]):
                        continue
                    dx = world_x - float(state["head_x"][row, player])
                    dy = world_y - float(state["head_y"][row, player])
                    radius = float(state["head_radius"][row, player])
                    if dx * dx + dy * dy <= radius * radius:
                        value = max(value, _synthetic_owner_luma(player))

                out[row, 0, py, px] = np.uint8(value)
    return out


def _synthetic_owner_luma(owner: int) -> int:
    if owner < 0:
        return SYNTHETIC_INVALID_OWNER_LUMA
    return SYNTHETIC_PLAYER_LUMA_BY_INDEX[owner % len(SYNTHETIC_PLAYER_LUMA_BY_INDEX)]


def _cpu_render_production_canvas_gray64(
    *,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    from curvyzero.env.vector_visual_observation import (
        TRAIL_RENDER_MODE_BROWSER_LINES,
        render_source_state_canvas_gray64,
    )

    batch_size = int(config["batch_size"])
    production_state = _synthetic_to_production_source_state(np=np, state=state, config=config)
    out = np.empty((batch_size, 1, 64, 64), dtype=np.uint8)
    for row in range(batch_size):
        out[row] = render_source_state_canvas_gray64(
            production_state,
            row=row,
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
        )
    return out


def _synthetic_to_production_source_state(
    *,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    bonus_count = int(config["bonus_count"])
    map_size = float(config["map_size"])
    trail_pos = np.stack([state["trail_x"], state["trail_y"]], axis=2).astype(np.float64)
    head_pos = np.stack([state["head_x"], state["head_y"]], axis=2).astype(np.float64)
    trail_active = state["trail_active"].astype(bool, copy=False)
    bonus_pos = np.stack([state["bonus_x"], state["bonus_y"]], axis=2).astype(np.float64)

    return {
        "tick": np.zeros((batch_size,), dtype=np.int64),
        "elapsed_ms": np.zeros((batch_size,), dtype=np.float64),
        "map_size": np.full((batch_size,), map_size, dtype=np.float64),
        "present": np.ones((batch_size, player_count), dtype=bool),
        "alive": state["head_alive"].astype(bool, copy=False),
        "pos": head_pos,
        "radius": state["head_radius"].astype(np.float64),
        "body_active": trail_active,
        "body_write_cursor": np.sum(trail_active, axis=1).astype(np.int32),
        "body_pos": trail_pos,
        "body_radius": state["trail_radius"].astype(np.float64),
        "body_owner": state["trail_owner"].astype(np.int16),
        "body_break_before": state["trail_break_before"].astype(bool, copy=False),
        "done": np.zeros((batch_size,), dtype=bool),
        "terminated": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "terminal_reason": np.zeros((batch_size,), dtype=np.int16),
        "avatar_color": (
            np.arange(player_count, dtype=np.int16)[None, :]
            .repeat(batch_size, axis=0)
        ),
        "visual_trail_active": trail_active,
        "visual_trail_write_cursor": np.sum(trail_active, axis=1).astype(np.int32),
        "visual_trail_pos": trail_pos,
        "visual_trail_radius": state["trail_radius"].astype(np.float64),
        "visual_trail_owner": state["trail_owner"].astype(np.int16),
        "visual_trail_break_before": state["trail_break_before"].astype(bool, copy=False),
        "bonus_active": state["bonus_active"].astype(bool, copy=False),
        "bonus_pos": bonus_pos.reshape(batch_size, bonus_count, 2),
        "bonus_radius": state["bonus_radius"].astype(np.float64),
        "bonus_type": state["bonus_type"].astype(np.int16),
    }


def _point_hits_segment(
    *,
    world_x: float,
    world_y: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    radius: float,
) -> bool:
    vx = bx - ax
    vy = by - ay
    length_sq = vx * vx + vy * vy
    if length_sq <= 0.0001:
        return False
    t = ((world_x - ax) * vx + (world_y - ay) * vy) / length_sq
    t = min(1.0, max(0.0, t))
    nearest_x = ax + t * vx
    nearest_y = ay + t * vy
    dx = world_x - nearest_x
    dy = world_y - nearest_y
    return dx * dx + dy * dy <= radius * radius


def _synthetic_source_state(*, np: Any, config: dict[str, Any]) -> dict[str, Any]:
    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    trail_slots = int(config["trail_slots"])
    bonus_count = int(config["bonus_count"])
    map_size = float(config["map_size"])
    trail_radius_value = float(config["trail_radius"])
    rng = np.random.default_rng(int(config["seed"]))

    trail_x = np.zeros((batch_size, trail_slots), dtype=np.float32)
    trail_y = np.zeros((batch_size, trail_slots), dtype=np.float32)
    trail_radius = np.zeros((batch_size, trail_slots), dtype=np.float32)
    trail_owner = np.zeros((batch_size, trail_slots), dtype=np.int32)
    trail_active = np.ones((batch_size, trail_slots), dtype=np.uint8)
    trail_break_before = np.zeros((batch_size, trail_slots), dtype=np.uint8)

    for row in range(batch_size):
        slot = 0
        for player in range(player_count):
            remaining_players = player_count - player
            remaining_slots = trail_slots - slot
            count = max(1, remaining_slots // remaining_players)
            x = float(rng.uniform(0.15 * map_size, 0.85 * map_size))
            y = float(rng.uniform(0.15 * map_size, 0.85 * map_size))
            heading = float(rng.uniform(0.0, 6.283185307179586))
            for local_index in range(count):
                if slot >= trail_slots:
                    break
                if local_index == 0:
                    trail_break_before[row, slot] = 1
                heading += float(rng.normal(0.0, 0.18))
                step = float(rng.uniform(5.0, 13.0))
                x = min(map_size - 1.0, max(1.0, x + step * float(np.cos(heading))))
                y = min(map_size - 1.0, max(1.0, y + step * float(np.sin(heading))))
                trail_x[row, slot] = x
                trail_y[row, slot] = y
                trail_radius[row, slot] = trail_radius_value
                trail_owner[row, slot] = player
                slot += 1
        if slot < trail_slots:
            trail_active[row, slot:] = 0

    head_x = np.zeros((batch_size, player_count), dtype=np.float32)
    head_y = np.zeros((batch_size, player_count), dtype=np.float32)
    head_radius = np.full((batch_size, player_count), 8.0, dtype=np.float32)
    head_alive = np.ones((batch_size, player_count), dtype=np.uint8)
    for row in range(batch_size):
        for player in range(player_count):
            player_slots = np.flatnonzero(
                (trail_owner[row] == player) & (trail_active[row].astype(bool))
            )
            if player_slots.size:
                last = int(player_slots[-1])
                head_x[row, player] = trail_x[row, last]
                head_y[row, player] = trail_y[row, last]

    bonus_x = rng.uniform(0.1 * map_size, 0.9 * map_size, size=(batch_size, bonus_count)).astype(
        np.float32
    )
    bonus_y = rng.uniform(0.1 * map_size, 0.9 * map_size, size=(batch_size, bonus_count)).astype(
        np.float32
    )
    bonus_radius = rng.uniform(10.0, 18.0, size=(batch_size, bonus_count)).astype(np.float32)
    bonus_active = np.ones((batch_size, bonus_count), dtype=np.uint8)
    bonus_type = rng.integers(0, 4, size=(batch_size, bonus_count), dtype=np.int32)

    return {
        "trail_x": trail_x,
        "trail_y": trail_y,
        "trail_radius": trail_radius,
        "trail_owner": trail_owner,
        "trail_active": trail_active,
        "trail_break_before": trail_break_before,
        "head_x": head_x,
        "head_y": head_y,
        "head_radius": head_radius,
        "head_alive": head_alive,
        "bonus_x": bonus_x,
        "bonus_y": bonus_y,
        "bonus_radius": bonus_radius,
        "bonus_active": bonus_active,
        "bonus_type": bonus_type,
    }


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    render_mode = str(config.get("render_mode", RENDER_MODE_BROWSER_LINES))
    if render_mode not in RENDER_MODE_IDS:
        allowed = ", ".join(sorted(RENDER_MODE_IDS))
        raise ValueError(f"render_mode must be one of {allowed}, got {render_mode!r}")
    render_surface = str(config.get("render_surface", RENDER_SURFACE_DIRECT_GRAY64))
    if render_surface not in RENDER_SURFACES:
        allowed = ", ".join(sorted(RENDER_SURFACES))
        raise ValueError(f"render_surface must be one of {allowed}, got {render_surface!r}")

    checked = {
        "batch_size": _positive_int(config.get("batch_size", 64), "batch_size"),
        "player_count": _positive_int(config.get("player_count", 2), "player_count"),
        "trail_slots": _positive_int(config.get("trail_slots", 256), "trail_slots"),
        "render_mode": render_mode,
        "render_surface": render_surface,
        "bonus_count": _nonnegative_int(config.get("bonus_count", 8), "bonus_count"),
        "frame_size": _positive_int(config.get("frame_size", 704), "frame_size"),
        "target_size": _positive_int(config.get("target_size", 64), "target_size"),
        "seed": int(config.get("seed", 20260513)),
        "warmup_runs": _nonnegative_int(config.get("warmup_runs", 3), "warmup_runs"),
        "steady_runs": _positive_int(config.get("steady_runs", 10), "steady_runs"),
        "verify_rows": _nonnegative_int(config.get("verify_rows", 2), "verify_rows"),
        "transfer_output": bool(config.get("transfer_output", False)),
        "map_size": float(config.get("map_size", 1000.0)),
        "trail_radius": float(config.get("trail_radius", 4.0)),
        "cpu_verify_max_pixel_trail_tests": int(
            config.get("cpu_verify_max_pixel_trail_tests", 25_000_000)
        ),
    }
    if checked["map_size"] <= 0.0:
        raise ValueError(f"map_size must be positive, got {checked['map_size']}")
    if checked["trail_radius"] <= 0.0:
        raise ValueError(f"trail_radius must be positive, got {checked['trail_radius']}")
    if checked["target_size"] > checked["frame_size"]:
        raise ValueError("target_size must not exceed frame_size")
    if render_surface == RENDER_SURFACE_BLOCK_704_GRAY64 and (
        checked["frame_size"] % checked["target_size"] != 0
    ):
        raise ValueError("block_704_gray64 requires frame_size divisible by target_size")
    return checked


def _positive_int(value: Any, name: str) -> int:
    checked = int(value)
    if checked < 1:
        raise ValueError(f"{name} must be positive, got {checked}")
    return checked


def _nonnegative_int(value: Any, name: str) -> int:
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative, got {checked}")
    return checked


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _rate(count: float, seconds: float) -> float | None:
    if seconds <= 0.0:
        return None
    return count / seconds


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _nvidia_smi() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


@app.local_entrypoint()
def main(
    batch_size: int = 64,
    player_count: int = 2,
    trail_slots: int = 256,
    render_mode: str = RENDER_MODE_BROWSER_LINES,
    render_surface: str = RENDER_SURFACE_DIRECT_GRAY64,
    bonus_count: int = 8,
    frame_size: int = 704,
    target_size: int = 64,
    seed: int = 20260513,
    warmup_runs: int = 3,
    steady_runs: int = 10,
    verify_rows: int = 2,
    transfer_output: bool = False,
    trail_radius: float = 4.0,
) -> None:
    config = {
        "batch_size": batch_size,
        "player_count": player_count,
        "trail_slots": trail_slots,
        "render_mode": render_mode,
        "render_surface": render_surface,
        "bonus_count": bonus_count,
        "frame_size": frame_size,
        "target_size": target_size,
        "seed": seed,
        "warmup_runs": warmup_runs,
        "steady_runs": steady_runs,
        "verify_rows": verify_rows,
        "transfer_output": transfer_output,
        "trail_radius": trail_radius,
    }
    result = run_source_state_gpu_render_benchmark.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
