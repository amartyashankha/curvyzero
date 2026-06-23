"""Profile persistent policy-space trail buffers versus stateless redraw.

This is an isolated optimizer benchmark. It does not import trainers, touch
Modal Volumes, change observation defaults, or claim CurvyTron browser parity.

The synthetic target is deliberately small: append one segment per player per
row per step, render two controlled-player views, and compare:

- stateless redraw: fresh frame, loop over all previous segments each step;
- persistent update: keep a uint8 framebuffer and stamp only the new segment.

Run from repo root:

    uv run --extra modal modal run -m curvyzero.infra.modal.source_state_persistent_framebuffer_benchmark
"""

from __future__ import annotations

import json
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal.mctx_dependency_smoke import JAX_VERSION


APP_NAME = "curvyzero-source-state-persistent-framebuffer-benchmark"
REMOTE_ROOT = Path("/repo")
SCHEMA_ID = "curvyzero_source_state_persistent_framebuffer_benchmark/v0"
COMPUTE_L4_T4 = "gpu-l4-t4"
COMPUTE_H100 = "gpu-h100"
COMPUTE_CHOICES = {COMPUTE_L4_T4, COMPUTE_H100}


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"jax[cuda12]=={JAX_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=20 * 60, cpu=2.0)
def run_persistent_framebuffer_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    return _run_safe(config)


@app.function(image=gpu_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_persistent_framebuffer_benchmark_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_safe(config)


def _run_safe(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_impl(config)
    except Exception as exc:  # pragma: no cover - remote diagnostics
        import traceback

        return {
            "schema_id": SCHEMA_ID,
            "ok": False,
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _run_impl(config: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    checked = _validate_config(config)
    batch_size = int(checked["batch_size"])
    steps = int(checked["steps"])
    warmup_steps = int(checked["warmup_steps"])
    target_size = int(checked["target_size"])
    radius = float(checked["radius"])
    transfer_output = bool(checked["transfer_output"])

    segments = _make_segments(
        np=np,
        batch_size=batch_size,
        total_steps=steps + warmup_steps,
        target_size=target_size,
    )
    segments_device = jax.device_put(segments)
    stateless_step = _make_stateless_step(jax=jax, jnp=jnp, target_size=target_size, radius=radius)
    persistent_step = _make_persistent_step(
        jax=jax,
        jnp=jnp,
        target_size=target_size,
        radius=radius,
    )

    compile_timings: dict[str, float] = {}
    started = time.perf_counter()
    stateless_first = stateless_step(segments_device, jnp.asarray(0, dtype=jnp.int32))
    stateless_first.block_until_ready()
    compile_timings["stateless_first_call_sec"] = time.perf_counter() - started

    layer = jnp.full(
        (batch_size, 2, target_size, target_size),
        jnp.uint8(34),
        dtype=jnp.uint8,
    )
    started = time.perf_counter()
    layer = persistent_step(layer, segments_device, jnp.asarray(0, dtype=jnp.int32))
    layer.block_until_ready()
    compile_timings["persistent_first_call_sec"] = time.perf_counter() - started

    layer = jnp.full_like(layer, jnp.uint8(34))
    for step in range(warmup_steps):
        layer = persistent_step(layer, segments_device, jnp.asarray(step, dtype=jnp.int32))
        layer.block_until_ready()
        warmed_stateless = stateless_step(segments_device, jnp.asarray(step, dtype=jnp.int32))
        warmed_stateless.block_until_ready()

    stateless_device_secs: list[float] = []
    stateless_readback_secs: list[float] = []
    stateless_total_secs: list[float] = []
    persistent_device_secs: list[float] = []
    persistent_readback_secs: list[float] = []
    persistent_total_secs: list[float] = []
    mismatch_checks: list[dict[str, Any]] = []

    for offset in range(steps):
        step = warmup_steps + offset
        step_idx = jnp.asarray(step, dtype=jnp.int32)

        started_total = time.perf_counter()
        started = time.perf_counter()
        stateless = stateless_step(segments_device, step_idx)
        stateless.block_until_ready()
        stateless_device = time.perf_counter() - started
        stateless_readback = 0.0
        stateless_host = None
        if transfer_output:
            started = time.perf_counter()
            stateless_host = np.asarray(stateless)
            stateless_readback = time.perf_counter() - started
        stateless_total = time.perf_counter() - started_total

        started_total = time.perf_counter()
        started = time.perf_counter()
        layer = persistent_step(layer, segments_device, step_idx)
        layer.block_until_ready()
        persistent_device = time.perf_counter() - started
        persistent_readback = 0.0
        persistent_host = None
        if transfer_output:
            started = time.perf_counter()
            persistent_host = np.asarray(layer)
            persistent_readback = time.perf_counter() - started
        persistent_total = time.perf_counter() - started_total

        stateless_device_secs.append(stateless_device)
        stateless_readback_secs.append(stateless_readback)
        stateless_total_secs.append(stateless_total)
        persistent_device_secs.append(persistent_device)
        persistent_readback_secs.append(persistent_readback)
        persistent_total_secs.append(persistent_total)

        if (
            offset == 0
            or offset == steps - 1
            or (int(checked["parity_interval"]) > 0 and offset % int(checked["parity_interval"]) == 0)
        ):
            if stateless_host is None:
                stateless_host = np.asarray(stateless)
            if persistent_host is None:
                persistent_host = np.asarray(layer)
            diff = np.abs(stateless_host.astype(np.int16) - persistent_host.astype(np.int16))
            mismatch_checks.append(
                {
                    "step": int(step),
                    "mismatch_count": int(np.count_nonzero(diff)),
                    "max_abs_diff": int(diff.max()) if diff.size else 0,
                    "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                }
            )

    stateless_total_median = _median(stateless_total_secs)
    persistent_total_median = _median(persistent_total_secs)
    stateless_device_median = _median(stateless_device_secs)
    persistent_device_median = _median(persistent_device_secs)
    frame_count = float(batch_size * 2)
    return {
        "schema_id": SCHEMA_ID,
        "ok": True,
        "app_name": APP_NAME,
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
            "scope": "synthetic policy-space trail framebuffer benchmark",
            "not_claims": [
                "not browser pixel parity",
                "not source-state physics",
                "not LightZero training",
                "not tournament/checkpoint observation metadata",
            ],
            "shape": [batch_size, 2, target_size, target_size],
            "dtype": "uint8",
            "views": "controlled-player view 0 and view 1 with self/other luma",
            "stateless_cost_model": "redraw all prior synthetic segments every frame",
            "persistent_cost_model": "stamp only the new synthetic segment into a persistent frame",
        },
        "compile_timings": compile_timings,
        "timings": {
            "stateless_device_sec_median": stateless_device_median,
            "stateless_readback_sec_median": _median(stateless_readback_secs),
            "stateless_total_sec_median": stateless_total_median,
            "persistent_device_sec_median": persistent_device_median,
            "persistent_readback_sec_median": _median(persistent_readback_secs),
            "persistent_total_sec_median": persistent_total_median,
            "device_speedup": (
                stateless_device_median / persistent_device_median
                if persistent_device_median > 0.0
                else None
            ),
            "total_speedup": (
                stateless_total_median / persistent_total_median
                if persistent_total_median > 0.0
                else None
            ),
            "stateless_frames_per_sec": _rate(frame_count, stateless_total_median),
            "persistent_frames_per_sec": _rate(frame_count, persistent_total_median),
            "stateless_env_rows_per_sec": _rate(float(batch_size), stateless_total_median),
            "persistent_env_rows_per_sec": _rate(float(batch_size), persistent_total_median),
        },
        "parity": {
            "checks": mismatch_checks,
            "exact": all(check["mismatch_count"] == 0 for check in mismatch_checks),
        },
        "plain_read": (
            "This isolates the long-trail cost model. If persistent total speedup is "
            "large with readback enabled, a profile-only dirty/device renderer is worth "
            "building. If only device_speedup is large, observation ownership/readback "
            "is still the boundary."
        ),
    }


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    batch_size = _positive_int(config.get("batch_size", 512), "batch_size")
    steps = _positive_int(config.get("steps", 512), "steps")
    warmup_steps = _nonnegative_int(config.get("warmup_steps", 32), "warmup_steps")
    target_size = _positive_int(config.get("target_size", 64), "target_size")
    if target_size < 8:
        raise ValueError("target_size must be at least 8")
    radius = float(config.get("radius", 1.25))
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return {
        "batch_size": batch_size,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "target_size": target_size,
        "radius": radius,
        "seed": int(config.get("seed", 20260521)),
        "transfer_output": bool(config.get("transfer_output", True)),
        "parity_interval": _nonnegative_int(config.get("parity_interval", 64), "parity_interval"),
    }


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def _make_segments(*, np: Any, batch_size: int, total_steps: int, target_size: int) -> Any:
    rng = np.random.default_rng(20260521)
    rows = np.arange(batch_size, dtype=np.float32)[:, None]
    steps = np.arange(total_steps, dtype=np.float32)[None, :]
    segments = np.zeros((total_steps, batch_size, 2, 4), dtype=np.float32)
    for player in range(2):
        phase = rows * (0.017 + player * 0.003) + steps * (0.071 + player * 0.011)
        radius = (target_size * 0.30) + (player * target_size * 0.03)
        center_x = target_size * (0.48 + 0.04 * np.sin(rows * 0.031 + player))
        center_y = target_size * (0.50 + 0.04 * np.cos(rows * 0.029 + player))
        x0 = center_x + radius * np.cos(phase)
        y0 = center_y + radius * np.sin(phase)
        x1 = center_x + radius * np.cos(phase + 0.09 + rng.normal(0.0, 0.002, phase.shape))
        y1 = center_y + radius * np.sin(phase + 0.09 + rng.normal(0.0, 0.002, phase.shape))
        segments[:, :, player, 0] = np.clip(x0.T, 0.0, float(target_size - 1))
        segments[:, :, player, 1] = np.clip(y0.T, 0.0, float(target_size - 1))
        segments[:, :, player, 2] = np.clip(x1.T, 0.0, float(target_size - 1))
        segments[:, :, player, 3] = np.clip(y1.T, 0.0, float(target_size - 1))
    return segments


def _make_stateless_step(*, jax: Any, jnp: Any, target_size: int, radius: float) -> Any:
    background = jnp.uint8(34)

    @jax.jit
    def stateless_step(segments: Any, step_idx: Any) -> Any:
        frame = jnp.full(
            (segments.shape[1], 2, target_size, target_size),
            background,
            dtype=jnp.uint8,
        )

        def draw_slot(slot: Any, current: Any) -> Any:
            segment = segments[slot]
            update = slot <= step_idx
            current = _draw_owner_segment(
                jnp=jnp,
                current=current,
                segment=segment[:, 0],
                owner=0,
                target_size=target_size,
                radius=radius,
            )
            current = _draw_owner_segment(
                jnp=jnp,
                current=current,
                segment=segment[:, 1],
                owner=1,
                target_size=target_size,
                radius=radius,
            )
            return jnp.where(update, current, current)

        return jax.lax.fori_loop(0, step_idx + 1, draw_slot, frame)

    return stateless_step


def _make_persistent_step(*, jax: Any, jnp: Any, target_size: int, radius: float) -> Any:
    @jax.jit
    def persistent_step(layer: Any, segments: Any, step_idx: Any) -> Any:
        segment = segments[step_idx]
        layer = _draw_owner_segment(
            jnp=jnp,
            current=layer,
            segment=segment[:, 0],
            owner=0,
            target_size=target_size,
            radius=radius,
        )
        return _draw_owner_segment(
            jnp=jnp,
            current=layer,
            segment=segment[:, 1],
            owner=1,
            target_size=target_size,
            radius=radius,
        )

    return persistent_step


def _draw_owner_segment(
    *,
    jnp: Any,
    current: Any,
    segment: Any,
    owner: int,
    target_size: int,
    radius: float,
) -> Any:
    grid = jnp.arange(target_size, dtype=jnp.float32)
    x = grid[None, None, :]
    y = grid[None, :, None]
    x0 = segment[:, 0, None, None]
    y0 = segment[:, 1, None, None]
    x1 = segment[:, 2, None, None]
    y1 = segment[:, 3, None, None]
    vx = x1 - x0
    vy = y1 - y0
    length_sq = jnp.maximum(vx * vx + vy * vy, 1.0e-4)
    t = jnp.clip(((x - x0) * vx + (y - y0) * vy) / length_sq, 0.0, 1.0)
    nearest_x = x0 + t * vx
    nearest_y = y0 + t * vy
    dx = x - nearest_x
    dy = y - nearest_y
    hit = dx * dx + dy * dy <= float(radius * radius)
    values = jnp.asarray(((96, 128), (128, 96)), dtype=jnp.uint8)
    owner_values = values[:, int(owner)]
    update = jnp.where(hit[:, None, :, :], owner_values[None, :, None, None], current)
    return jnp.maximum(current, update)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def _rate(count: float, seconds: float) -> float | None:
    return float(count) / float(seconds) if seconds > 0.0 else None


def _version_or_missing(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def _nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        ).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"


@app.local_entrypoint()
def main(
    batch_size: int = 512,
    steps: int = 256,
    warmup_steps: int = 32,
    target_size: int = 64,
    radius: float = 1.25,
    seed: int = 20260521,
    transfer_output: bool = True,
    parity_interval: int = 64,
    compute: str = COMPUTE_H100,
) -> None:
    config = {
        "batch_size": batch_size,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "target_size": target_size,
        "radius": radius,
        "seed": seed,
        "transfer_output": transfer_output,
        "parity_interval": parity_interval,
    }
    if compute == COMPUTE_L4_T4:
        result = run_persistent_framebuffer_benchmark.remote(config)
    elif compute == COMPUTE_H100:
        result = run_persistent_framebuffer_benchmark_h100.remote(config)
    else:
        allowed = ", ".join(sorted(COMPUTE_CHOICES))
        raise ValueError(f"compute must be one of {allowed}; got {compute!r}")
    print(json.dumps(result, indent=2, sort_keys=True))

