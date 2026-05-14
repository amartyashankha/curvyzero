"""Standalone Modal GPU probes for CurvyTron render economics.

This does not touch training runs or Modal Volumes. It answers a narrow
optimizer question: can a batched device-side renderer beat CPU rendering once
transfer, launch, and optional readback are measured?

Run from repo root:

    uv run --extra modal modal run -m curvyzero.infra.modal.curvytron_gpu_render_probe
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


APP_NAME = "curvyzero-curvytron-gpu-render-probe"
REMOTE_ROOT = Path("/repo")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"jax[cuda12]=={JAX_VERSION}",
        "numpy>=1.26",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=4.0)
def run_gpu_render_probe(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_gpu_render_probe_impl(config)
    except Exception as exc:  # pragma: no cover - Modal remote diagnostics.
        import traceback

        return {
            "schema_id": "curvyzero_gpu_render_probe/v0",
            "ok": False,
            "scope": (
                "standalone synthetic direct64 splat renderer; not training "
                "and not browser-line fidelity"
            ),
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _run_gpu_render_probe_impl(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    batch_sizes = _positive_int_list(config.get("batch_sizes", [1, 8, 64]))
    primitive_counts = _positive_int_list(config.get("primitive_counts", [64, 256]))
    repeats = _positive_int(config.get("repeats", 5), "repeats")
    warmup = _nonnegative_int(config.get("warmup", 3), "warmup")
    seed = int(config.get("seed", 20260513))
    map_size = float(config.get("map_size", 1000.0))
    height = int(config.get("height", 64))
    width = int(config.get("width", 64))
    cpu_baseline_max_ops = int(config.get("cpu_baseline_max_ops", 12_000_000))

    nvidia_smi = _nvidia_smi()
    cells: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    backend = jax.default_backend()

    for batch in batch_sizes:
        for primitives in primitive_counts:
            inputs = _synthetic_primitives(
                rng,
                batch=batch,
                primitives=primitives,
                map_size=map_size,
            )
            cell = _run_cell(
                jax=jax,
                jnp=jnp,
                np=np,
                inputs=inputs,
                batch=batch,
                primitives=primitives,
                height=height,
                width=width,
                map_size=map_size,
                repeats=repeats,
                warmup=warmup,
                cpu_baseline_max_ops=cpu_baseline_max_ops,
            )
            cells.append(cell)
            print(json.dumps(cell, sort_keys=True), flush=True)

    report = {
        "schema_id": "curvyzero_gpu_render_probe/v0",
        "ok": True,
        "scope": (
            "standalone synthetic JAX direct64 splat renderer; not training "
            "and not browser-line fidelity"
        ),
        "nvidia_smi": nvidia_smi,
        "packages": {
            "jax": _version_or_missing("jax"),
            "jaxlib": _version_or_missing("jaxlib"),
            "numpy": _version_or_missing("numpy"),
        },
        "jax": {
            "default_backend": backend,
            "devices": [str(device) for device in jax.devices()],
            "device_count": len(jax.devices()),
        },
        "config": {
            "batch_sizes": batch_sizes,
            "primitive_counts": primitive_counts,
            "repeats": repeats,
            "warmup": warmup,
            "seed": seed,
            "map_size": map_size,
            "height": height,
            "width": width,
            "cpu_baseline_max_ops": cpu_baseline_max_ops,
        },
        "interpretation": (
            "Use this to find transfer/kernel/readback crossover points. "
            "A useful production GPU renderer still needs real CurvyTron state "
            "arrays, browser-line parity or an explicit approximation contract, "
            "and policy/search consumption without immediate CPU readback."
        ),
        "cells": cells,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _run_cell(
    *,
    jax: Any,
    jnp: Any,
    np: Any,
    inputs: dict[str, Any],
    batch: int,
    primitives: int,
    height: int,
    width: int,
    map_size: float,
    repeats: int,
    warmup: int,
    cpu_baseline_max_ops: int,
) -> dict[str, Any]:
    transfer_times = []
    render_times = []
    readback_times = []
    end_to_end_readback_times = []
    compile_first_render_sec = None

    render_fn = jax.jit(
        lambda x_dev, y_dev, radius_dev, owner_dev: _jax_direct64_splats(
            jnp=jnp,
            x=x_dev,
            y=y_dev,
            radius=radius_dev,
            owner=owner_dev,
            height=height,
            width=width,
            map_size=map_size,
        )
    )

    output_host = None
    for iteration in range(warmup + repeats):
        started = time.perf_counter()
        x_dev = jax.device_put(inputs["x"])
        y_dev = jax.device_put(inputs["y"])
        radius_dev = jax.device_put(inputs["radius"])
        owner_dev = jax.device_put(inputs["owner"])
        x_dev.block_until_ready()
        y_dev.block_until_ready()
        radius_dev.block_until_ready()
        owner_dev.block_until_ready()
        transfer_sec = time.perf_counter() - started

        started = time.perf_counter()
        output_device = render_fn(x_dev, y_dev, radius_dev, owner_dev)
        output_device.block_until_ready()
        render_sec = time.perf_counter() - started
        if iteration == 0:
            compile_first_render_sec = render_sec

        started = time.perf_counter()
        output_host = np.asarray(output_device)
        readback_sec = time.perf_counter() - started

        if iteration >= warmup:
            transfer_times.append(transfer_sec)
            render_times.append(render_sec)
            readback_times.append(readback_sec)
            end_to_end_readback_times.append(transfer_sec + render_sec + readback_sec)

    cpu_result = None
    cpu_render_sec = None
    cpu_ops = int(batch) * int(primitives) * int(height) * int(width)
    if cpu_ops <= cpu_baseline_max_ops:
        started = time.perf_counter()
        cpu_result = _cpu_direct64_splats(
            np=np,
            inputs=inputs,
            batch=batch,
            primitives=primitives,
            height=height,
            width=width,
            map_size=map_size,
        )
        cpu_render_sec = time.perf_counter() - started

    parity = None
    if cpu_result is not None and output_host is not None:
        parity = bool(np.array_equal(output_host, cpu_result))

    render_median = statistics.median(render_times)
    transfer_median = statistics.median(transfer_times)
    readback_median = statistics.median(readback_times)
    end_to_end_readback_median = statistics.median(end_to_end_readback_times)
    return {
        "batch": int(batch),
        "primitives": int(primitives),
        "height": int(height),
        "width": int(width),
        "pixel_primitive_tests": int(cpu_ops),
        "transfer_h2d_sec_median": transfer_median,
        "compile_first_render_sec": compile_first_render_sec,
        "device_render_sec_median": render_median,
        "readback_d2h_sec_median": readback_median,
        "end_to_end_with_readback_sec_median": end_to_end_readback_median,
        "device_frames_per_sec": float(batch) / render_median if render_median > 0 else None,
        "end_to_end_frames_per_sec_with_readback": (
            float(batch) / end_to_end_readback_median
            if end_to_end_readback_median > 0
            else None
        ),
        "cpu_render_sec": cpu_render_sec,
        "cpu_frames_per_sec": (
            float(batch) / cpu_render_sec
            if cpu_render_sec is not None and cpu_render_sec > 0
            else None
        ),
        "gpu_vs_cpu_render_speedup": (
            cpu_render_sec / render_median
            if cpu_render_sec is not None and render_median > 0
            else None
        ),
        "gpu_vs_cpu_with_readback_speedup": (
            cpu_render_sec / end_to_end_readback_median
            if cpu_render_sec is not None and end_to_end_readback_median > 0
            else None
        ),
        "parity_with_cpu_toy": parity,
    }


def _synthetic_primitives(
    rng: Any,
    *,
    batch: int,
    primitives: int,
    map_size: float,
) -> dict[str, Any]:
    import numpy as np

    x = rng.uniform(0.0, map_size, size=(batch, primitives)).astype(np.float32)
    y = rng.uniform(0.0, map_size, size=(batch, primitives)).astype(np.float32)
    radius = rng.uniform(3.0, 16.0, size=(batch, primitives)).astype(np.float32)
    owner = rng.integers(0, 2, size=(batch, primitives), dtype=np.int32)
    return {"x": x, "y": y, "radius": radius, "owner": owner}


def _jax_direct64_splats(
    *,
    jnp: Any,
    x: Any,
    y: Any,
    radius: Any,
    owner: Any,
    height: int,
    width: int,
    map_size: float,
) -> Any:
    grid_x = (jnp.arange(width, dtype=jnp.float32) + 0.5) * float(map_size) / float(width)
    grid_y = (jnp.arange(height, dtype=jnp.float32) + 0.5) * float(map_size) / float(height)
    world_x = grid_x[None, None, None, :]
    world_y = grid_y[None, None, :, None]
    dx = world_x - x[:, :, None, None]
    dy = world_y - y[:, :, None, None]
    inside = dx * dx + dy * dy <= radius[:, :, None, None] * radius[:, :, None, None]
    values = jnp.where(owner[:, :, None, None] == 0, 96, 128).astype(jnp.uint8)
    painted = jnp.where(inside, values, jnp.zeros((), dtype=jnp.uint8))
    return jnp.max(painted, axis=1).astype(jnp.uint8)


def _cpu_direct64_splats(
    *,
    np: Any,
    inputs: dict[str, Any],
    batch: int,
    primitives: int,
    height: int,
    width: int,
    map_size: float,
) -> Any:
    out = np.zeros((batch, height, width), dtype=np.uint8)
    scale_x = float(width) / float(map_size)
    scale_y = float(height) / float(map_size)
    for b in range(batch):
        for p in range(primitives):
            cx = float(inputs["x"][b, p]) * scale_x - 0.5
            cy = float(inputs["y"][b, p]) * scale_y - 0.5
            radius_px = float(inputs["radius"][b, p]) * scale_x
            min_x = max(0, int(np.floor(cx - radius_px)))
            max_x = min(width - 1, int(np.ceil(cx + radius_px)))
            min_y = max(0, int(np.floor(cy - radius_px)))
            max_y = min(height - 1, int(np.ceil(cy + radius_px)))
            value = np.uint8(96 if int(inputs["owner"][b, p]) == 0 else 128)
            radius_sq = float(inputs["radius"][b, p]) ** 2
            for yy in range(min_y, max_y + 1):
                world_y = (float(yy) + 0.5) * float(map_size) / float(height)
                dy = world_y - float(inputs["y"][b, p])
                for xx in range(min_x, max_x + 1):
                    world_x = (float(xx) + 0.5) * float(map_size) / float(width)
                    dx = world_x - float(inputs["x"][b, p])
                    if dx * dx + dy * dy <= radius_sq and value > out[b, yy, xx]:
                        out[b, yy, xx] = value
    return out


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


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


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


def _positive_int_list(value: Any) -> list[int]:
    values = [_positive_int(item, "list item") for item in value]
    if not values:
        raise ValueError("list must be non-empty")
    return values


@app.local_entrypoint()
def main(
    batch_sizes: str = "1,8,64",
    primitive_counts: str = "64,256",
    repeats: int = 5,
    warmup: int = 3,
    seed: int = 20260513,
) -> None:
    config = {
        "batch_sizes": _parse_csv_ints(batch_sizes),
        "primitive_counts": _parse_csv_ints(primitive_counts),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "seed": int(seed),
    }
    result = run_gpu_render_probe.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))


def _parse_csv_ints(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return [_positive_int(value, "csv item") for value in values]
