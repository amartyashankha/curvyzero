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

BONUS_RENDER_MODE_BROWSER_SPRITES = "browser_sprites"
BONUS_RENDER_MODE_CIRCLES_FAST = "circles_fast"
BONUS_RENDER_MODE_SIMPLE_SYMBOLS = "simple_symbols"
BONUS_RENDER_MODE_IDS = {
    BONUS_RENDER_MODE_BROWSER_SPRITES: 0,
    BONUS_RENDER_MODE_CIRCLES_FAST: 1,
    BONUS_RENDER_MODE_SIMPLE_SYMBOLS: 2,
}

RENDER_SURFACE_DIRECT_GRAY64 = "direct_gray64"
RENDER_SURFACE_BLOCK_704_GRAY64 = "block_704_gray64"
RENDER_SURFACES = {
    RENDER_SURFACE_DIRECT_GRAY64,
    RENDER_SURFACE_BLOCK_704_GRAY64,
}
TRAIL_COMPOSITION_PRIORITY_BUFFER = "priority_buffer"
TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT = "owner_ordered_compact"
TRAIL_COMPOSITIONS = {
    TRAIL_COMPOSITION_PRIORITY_BUFFER,
    TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT,
}
RENDER_VIEWS_SINGLE = "single"
RENDER_VIEWS_BOTH = "both"
RENDER_VIEWS_CHOICES = {
    RENDER_VIEWS_SINGLE,
    RENDER_VIEWS_BOTH,
}
RENDER_OUTPUT_ORDER_VIEW_MAJOR = "view_major"
GEOMETRY_DTYPE_FLOAT32 = "float32"
GEOMETRY_DTYPE_FLOAT64 = "float64"
GEOMETRY_DTYPES = {
    GEOMETRY_DTYPE_FLOAT32,
    GEOMETRY_DTYPE_FLOAT64,
}
STATE_SOURCE_SYNTHETIC = "synthetic"
STATE_SOURCE_REAL_ENV_ROLLOUT = "real_env_rollout"
STATE_SOURCE_ADVERSARIAL_FIXTURE = "adversarial_fixture"
STATE_SOURCES = {
    STATE_SOURCE_SYNTHETIC,
    STATE_SOURCE_REAL_ENV_ROLLOUT,
    STATE_SOURCE_ADVERSARIAL_FIXTURE,
}

SYNTHETIC_BACKGROUND_LUMA = 34
SYNTHETIC_INVALID_OWNER_LUMA = 120
SYNTHETIC_PLAYER_LUMA_BY_INDEX = (76, 150, 76, 217)
SYNTHETIC_BACKGROUND_LUMA_FLOAT = 34.0
SYNTHETIC_INVALID_OWNER_LUMA_FLOAT = 120.0
SYNTHETIC_PLAYER_LUMA_FLOAT_BY_INDEX = (76.245, 149.685, 75.945, 217.335)
PERSPECTIVE_SELF_LUMA = 96
PERSPECTIVE_OTHER_LUMA = 128
PERSPECTIVE_SELF_LUMA_FLOAT = 96.0
PERSPECTIVE_OTHER_LUMA_FLOAT = 128.0
BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE = (68.0, 148.0, 196.0)
BONUS_SYMBOL_INNER_LUMA_BY_SHAPE = (212.0, 48.0, 48.0)
BONUS_SYMBOL_BASE_SIZE = 7

SCHEMA_ID = "curvyzero_source_state_gpu_render_benchmark/v0"
SCALAR_PROFILE_SCHEMA_ID = "curvyzero_source_state_gpu_scalar_component_profile/v1"
RENDERER_IMPL_ID = "synthetic_source_state_jax/v2"
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
def run_source_state_gpu_render_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    return _run_source_state_gpu_render_benchmark_safe(config)


@app.function(image=gpu_image, gpu="H100", timeout=20 * 60, cpu=4.0)
def run_source_state_gpu_render_benchmark_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _run_source_state_gpu_render_benchmark_safe(config)


def _run_source_state_gpu_render_benchmark_safe(config: dict[str, Any]) -> dict[str, Any]:
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
    import numpy as np

    if str(config.get("geometry_dtype", GEOMETRY_DTYPE_FLOAT32)) == GEOMETRY_DTYPE_FLOAT64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    checked = _validate_config(config)
    if bool(config.get("scalar_component_profile", False)):
        return _run_scalar_component_profile_impl(jax=jax, jnp=jnp, np=np, config=checked)
    render_mode_id = RENDER_MODE_IDS[checked["render_mode"]]
    bonus_render_mode_id = BONUS_RENDER_MODE_IDS[checked["bonus_render_mode"]]
    (
        state,
        production_reference_state,
        setup_timings,
    ) = _source_state_reference_and_setup_timings_for_benchmark(
        np=np,
        config=checked,
    )
    if checked["render_views"] == RENDER_VIEWS_BOTH:
        render_fn = _make_jax_two_view_render_fn(
            jax=jax,
            jnp=jnp,
            config=checked,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
        )
    else:
        render_fn = _make_jax_render_fn(
            jax=jax,
            jnp=jnp,
            config=checked,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
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
        production_reference_state=production_reference_state,
        config=checked,
        render_mode_id=render_mode_id,
        bonus_render_mode_id=bonus_render_mode_id,
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
            "input_shape": (
                "synthetic source-state arrays"
                if checked["state_source"] == STATE_SOURCE_SYNTHETIC
                else (
                    "hand-authored adversarial production state converted to compact render arrays"
                    if checked["state_source"] == STATE_SOURCE_ADVERSARIAL_FIXTURE
                    else "real VectorMultiplayerEnv.state converted to compact render arrays"
                )
            ),
            "state_source": checked["state_source"],
            "controlled_player": checked["controlled_player"],
            "output_shape": [
                checked["batch_size"] * (2 if checked["render_views"] == RENDER_VIEWS_BOTH else 1),
                1,
                checked["target_size"],
                checked["target_size"],
            ],
            "output_order": (
                RENDER_OUTPUT_ORDER_VIEW_MAJOR
                if checked["render_views"] == RENDER_VIEWS_BOTH
                else "single_view_batch_major"
            ),
            "render_views": checked["render_views"],
            "geometry_dtype": checked["geometry_dtype"],
            "render_surface": checked["render_surface"],
            "direct_gray64": (
                "samples one point per 64x64 cell; fast economics probe, "
                "not trusted browser fidelity"
            ),
            "block_704_gray64": (
                f"outputs final {checked['target_size']}x{checked['target_size']} while "
                "checking all high-resolution sample positions for each cell with "
                "production-like pixel-space trail/head raster; no materialized "
                "source RGB canvas"
            ),
            "browser_lines": (
                "approximates connected source-state segments using the previous active "
                "same-owner point, with break_before suppressing a new segment"
            ),
            "bonus_render_mode": checked["bonus_render_mode"],
            "perspective": (
                "two controlled-player views for players 0 and 1"
                if checked["render_views"] == RENDER_VIEWS_BOTH
                else "controlled-player self/other luma"
                if checked["controlled_player"] is not None
                else "source/player-color luma"
            ),
            "trail_composition": checked["trail_composition"],
            "composition": (
                (
                    "benchmark-only CPU-packed owner draw order; block_704_gray64 "
                    "trail pass overwrites without carrying a priority buffer"
                )
                if checked["trail_composition"] == TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT
                else (
                    "policy-grayscale owner-priority overwrite; production parity "
                    "must still be checked for every rollout shape"
                )
            ),
        },
        "setup_timings": setup_timings,
        "known_gaps": [
            (
                "Real env rows are benchmark-only when state_source=real_env_rollout; "
                "there is still no LightZero trainer/checkpoint integration."
            ),
            "No Modal Volume access.",
            "No full RGB canvas parity yet; block_704_gray64 can be compared to production CPU render.",
            "Synthetic trail rows preserve tensor shape, not exact game histories.",
            (
                "JAX path now uses previous active same-owner browser-line connectivity, "
                "and block_704_gray64 uses owner-priority composition for the current "
                "policy-grayscale target. It is not a full RGB browser renderer."
            ),
            (
                "owner_ordered_compact, when selected, is benchmark-only and only "
                "changes compact render input ordering inside this Modal benchmark."
            ),
            "Browser sprite parity is intentionally out of scope when bonus_render_mode=simple_symbols.",
        ],
        "timings": timings,
        "verification": verification,
    }


def _run_scalar_component_profile_impl(
    *,
    jax: Any,
    jnp: Any,
    np: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Profile the current one-env, two-player JAX observation shape."""

    if int(config["batch_size"]) != 1:
        raise ValueError("scalar_component_profile requires batch_size=1")
    if int(config["player_count"]) != 2:
        raise ValueError("scalar_component_profile currently expects player_count=2")
    if config["render_surface"] != RENDER_SURFACE_BLOCK_704_GRAY64:
        raise ValueError("scalar_component_profile currently expects block_704_gray64")

    render_mode_id = RENDER_MODE_IDS[config["render_mode"]]
    bonus_render_mode_id = BONUS_RENDER_MODE_IDS[config["bonus_render_mode"]]
    render_config_p0 = {**config, "controlled_player": 0}
    render_config_p1 = {**config, "controlled_player": 1}
    render_fn_p0 = _make_jax_render_fn(
        jax=jax,
        jnp=jnp,
        config=render_config_p0,
        render_mode_id=render_mode_id,
        bonus_render_mode_id=bonus_render_mode_id,
    )
    render_fn_p1 = _make_jax_render_fn(
        jax=jax,
        jnp=jnp,
        config=render_config_p1,
        render_mode_id=render_mode_id,
        bonus_render_mode_id=bonus_render_mode_id,
    )
    fused_render_fn = _make_jax_two_view_render_fn(
        jax=jax,
        jnp=jnp,
        config=config,
        render_mode_id=render_mode_id,
        bonus_render_mode_id=bonus_render_mode_id,
    )
    production_state = _real_env_rollout_production_state(np=np, config=config)
    warmup_runs = int(config["warmup_runs"])
    steady_runs = int(config["steady_runs"])

    timings: dict[str, list[float]] = {
        "production_to_compact_sec": [],
        "host_to_device_sec": [],
        "render_player0_sec": [],
        "readback_player0_sec": [],
        "render_player1_sec": [],
        "readback_player1_sec": [],
        "separate_render_readback_sec": [],
        "two_view_total_sec": [],
        "fused_render_sec": [],
        "fused_readback_sec": [],
        "fused_render_readback_sec": [],
        "fused_two_view_total_sec": [],
    }
    compile_two_view_sec: float | None = None
    compile_fused_two_view_sec: float | None = None
    fused_comparison: dict[str, Any] | None = None

    for iteration in range(warmup_runs + steady_runs):
        total_started = time.perf_counter()

        started = time.perf_counter()
        compact_state = _production_to_benchmark_source_state(
            np=np,
            production_state=production_state,
            config=config,
        )
        compact_state = _prepare_compact_state_for_render(
            np=np,
            state=compact_state,
            config=config,
        )
        conversion_sec = time.perf_counter() - started

        started = time.perf_counter()
        device_state = _copy_state_to_device(jax=jax, state=compact_state)
        transfer_sec = time.perf_counter() - started

        started = time.perf_counter()
        output_p0 = render_fn_p0(device_state)
        output_p0.block_until_ready()
        render_p0_sec = time.perf_counter() - started

        started = time.perf_counter()
        host_p0 = np.asarray(output_p0)
        readback_p0_sec = time.perf_counter() - started

        started = time.perf_counter()
        output_p1 = render_fn_p1(device_state)
        output_p1.block_until_ready()
        render_p1_sec = time.perf_counter() - started

        started = time.perf_counter()
        host_p1 = np.asarray(output_p1)
        readback_p1_sec = time.perf_counter() - started

        separate_total_sec = time.perf_counter() - total_started

        started = time.perf_counter()
        fused_output = fused_render_fn(device_state)
        fused_output.block_until_ready()
        fused_render_sec = time.perf_counter() - started

        started = time.perf_counter()
        host_fused = np.asarray(fused_output)
        fused_readback_sec = time.perf_counter() - started

        separate_render_readback_sec = (
            render_p0_sec + readback_p0_sec + render_p1_sec + readback_p1_sec
        )
        fused_render_readback_sec = fused_render_sec + fused_readback_sec
        fused_total_sec = conversion_sec + transfer_sec + fused_render_readback_sec
        if iteration == 0:
            compile_two_view_sec = separate_total_sec
            compile_fused_two_view_sec = fused_total_sec
        if iteration >= warmup_runs:
            if fused_comparison is None:
                separate = np.concatenate([host_p0, host_p1], axis=0)
                diff = np.abs(host_fused.astype(np.int16) - separate.astype(np.int16))
                cpu_started = time.perf_counter()
                cpu_p0 = _cpu_render_original_production_canvas_gray64(
                    np=np,
                    production_state=production_state,
                    config={**config, "controlled_player": 0},
                )
                cpu_p1 = _cpu_render_original_production_canvas_gray64(
                    np=np,
                    production_state=production_state,
                    config={**config, "controlled_player": 1},
                )
                cpu_pair = np.concatenate([cpu_p0, cpu_p1], axis=0)
                cpu_sec = time.perf_counter() - cpu_started
                cpu_diff = np.abs(host_fused.astype(np.int16) - cpu_pair.astype(np.int16))
                cpu_mismatch_count = int(np.count_nonzero(cpu_diff))
                fused_comparison = {
                    "shape": list(host_fused.shape),
                    "matches_separate": bool(not np.count_nonzero(diff)),
                    "mismatch_count": int(np.count_nonzero(diff)),
                    "max_abs_diff": int(diff.max()) if diff.size else 0,
                    "cpu_reference_kind": (
                        "production_render_source_state_canvas_gray64_player0_player1"
                    ),
                    "cpu_reference_sec": float(cpu_sec),
                    "cpu_exact_parity": bool(cpu_mismatch_count == 0),
                    "cpu_mismatch_count": cpu_mismatch_count,
                    "cpu_mismatch_fraction": (
                        float(cpu_mismatch_count / cpu_diff.size) if cpu_diff.size else 0.0
                    ),
                    "cpu_max_abs_diff": int(cpu_diff.max()) if cpu_diff.size else 0,
                    "cpu_mean_abs_diff": float(cpu_diff.mean()) if cpu_diff.size else 0.0,
                }
            timings["production_to_compact_sec"].append(conversion_sec)
            timings["host_to_device_sec"].append(transfer_sec)
            timings["render_player0_sec"].append(render_p0_sec)
            timings["readback_player0_sec"].append(readback_p0_sec)
            timings["render_player1_sec"].append(render_p1_sec)
            timings["readback_player1_sec"].append(readback_p1_sec)
            timings["separate_render_readback_sec"].append(separate_render_readback_sec)
            timings["two_view_total_sec"].append(separate_total_sec)
            timings["fused_render_sec"].append(fused_render_sec)
            timings["fused_readback_sec"].append(fused_readback_sec)
            timings["fused_render_readback_sec"].append(fused_render_readback_sec)
            timings["fused_two_view_total_sec"].append(fused_total_sec)

    medians = {key: _median(value) for key, value in timings.items()}
    fused_total = medians["fused_two_view_total_sec"]
    fused_render_readback = medians["fused_render_readback_sec"]
    return {
        "schema_id": SCALAR_PROFILE_SCHEMA_ID,
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
        "config": {**config, "scalar_component_profile": True},
        "timings": {
            "warmup_runs": warmup_runs,
            "steady_runs": steady_runs,
            "compile_first_two_view_sec": compile_two_view_sec,
            "compile_first_fused_two_view_sec": compile_fused_two_view_sec,
            "median": medians,
            "frames_per_sec_two_views": _rate(2.0, medians["two_view_total_sec"]),
            "env_steps_per_sec_two_views": _rate(1.0, medians["two_view_total_sec"]),
            "fused_frames_per_sec_two_views": _rate(2.0, fused_total),
            "fused_env_steps_per_sec_two_views": _rate(1.0, fused_total),
            "fused_vs_separate_total_speedup": (
                medians["two_view_total_sec"] / fused_total if fused_total > 0.0 else None
            ),
            "fused_vs_separate_render_readback_speedup": (
                medians["separate_render_readback_sec"] / fused_render_readback
                if fused_render_readback > 0.0
                else None
            ),
        },
        "fused_comparison": fused_comparison or {"status": "not_checked"},
        "semantic_notes": {
            "previous_point_algorithm": (
                "JAX browser_lines scans trail slots once and carries the last active "
                "position per non-negative owner. Each current active slot connects to "
                "that carried same-owner point unless break_before is set or the trail "
                "radius changed."
            ),
            "fused_two_view_path": (
                "For block_704_gray64, the fused JIT keeps a leading two-view axis and "
                "shares trail, bonus, and head geometry across player-perspective luma "
                "palettes before returning [2,1,64,64] for the scalar batch=1 profile."
            ),
            "remaining_fidelity_gap": (
                "This is still a benchmark prototype. The block_704_gray64 JAX path "
                "now matches the production policy-grayscale owner ordering on the "
                "checked smoke rows, but it is not a full RGB browser renderer and "
                "is not wired into the trainer."
            ),
        },
        "plain_read": (
            "This profiles the current scalar trainer shape against a fused two-view "
            "JIT: one production env row, one host-to-device copy, either two "
            "controlled-player renders plus two NumPy readbacks, or one JIT render "
            "returning [2,1,64,64] plus one readback."
        ),
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
    view_count = 2 if config.get("render_views") == RENDER_VIEWS_BOTH else 1
    frame_count = float(config["batch_size"]) * float(view_count)

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
    env_row_count = int(config["batch_size"])
    frame_count_int = int(env_row_count * view_count)
    env_row_pixels = env_row_count * target_size * target_size
    rendered_frame_pixels = frame_count_int * target_size * target_size
    if config["render_surface"] == RENDER_SURFACE_BLOCK_704_GRAY64:
        block = int(config["frame_size"]) // target_size
        effective_pixel_tests = (
            rendered_frame_pixels
            * block
            * block
            * int(config["trail_slots"])
        )
    else:
        effective_pixel_tests = rendered_frame_pixels * int(config["trail_slots"])

    return {
        "warmup_runs": warmup_runs,
        "steady_runs": steady_runs,
        "env_row_count": env_row_count,
        "frame_count": frame_count_int,
        "env_row_pixel_count": env_row_pixels,
        "rendered_frame_pixel_count": rendered_frame_pixels,
        "pixel_count": rendered_frame_pixels,
        "effective_pixel_trail_slot_tests": effective_pixel_tests,
        "host_to_device_sec_median": _median(transfer_times),
        "compile_first_render_sec": compile_first_render_sec,
        "device_render_sec_median": _median(render_times),
        "device_to_host_sec_median": _median(readback_times) if config["transfer_output"] else None,
        "end_to_end_sec_median": _median(end_to_end_times),
        "device_frames_per_sec": _rate(frame_count, _median(render_times)),
        "end_to_end_frames_per_sec": _rate(
            frame_count,
            _median(end_to_end_times),
        ),
        "env_rows_per_sec": _rate(float(config["batch_size"]), _median(end_to_end_times)),
        "render_views": str(config.get("render_views", RENDER_VIEWS_SINGLE)),
        "transfer_output_measured": bool(config["transfer_output"]),
    }


def _make_jax_render_fn(
    *,
    jax: Any,
    jnp: Any,
    config: dict[str, Any],
    render_mode_id: int,
    bonus_render_mode_id: int,
) -> Any:
    target_size = int(config["target_size"])
    frame_size = int(config["frame_size"])
    map_size = float(config["map_size"])
    bonus_count = int(config["bonus_count"])
    render_surface = str(config["render_surface"])
    trail_composition = str(config.get("trail_composition", TRAIL_COMPOSITION_PRIORITY_BUFFER))
    use_priority_buffer = trail_composition != TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT
    controlled_player = config.get("controlled_player")
    player_luma_by_index = _player_luma_by_index(config)
    player_luma_float_by_index = _player_luma_float_by_index(config)

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
                bonus_render_mode_id=bonus_render_mode_id,
                controlled_player=controlled_player,
                player_luma_float_by_index=player_luma_float_by_index,
                use_priority_buffer=use_priority_buffer,
            )
        return _jax_render_direct_gray64(
            jnp=jnp,
            lax=jax.lax,
            state=device_state,
            target_size=target_size,
            map_size=map_size,
            bonus_count=bonus_count,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
            controlled_player=controlled_player,
            player_luma_by_index=player_luma_by_index,
        )

    return render


def _make_jax_two_view_render_fn(
    *,
    jax: Any,
    jnp: Any,
    config: dict[str, Any],
    render_mode_id: int,
    bonus_render_mode_id: int,
) -> Any:
    target_size = int(config["target_size"])
    frame_size = int(config["frame_size"])
    map_size = float(config["map_size"])
    bonus_count = int(config["bonus_count"])
    render_surface = str(config["render_surface"])
    trail_composition = str(config.get("trail_composition", TRAIL_COMPOSITION_PRIORITY_BUFFER))
    use_priority_buffer = trail_composition != TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT
    p0_luma = _player_luma_by_index({**config, "controlled_player": 0})
    p1_luma = _player_luma_by_index({**config, "controlled_player": 1})
    p0_luma_float = _player_luma_float_by_index({**config, "controlled_player": 0})
    p1_luma_float = _player_luma_float_by_index({**config, "controlled_player": 1})

    @jax.jit
    def render(device_state: dict[str, Any]) -> Any:
        if render_surface == RENDER_SURFACE_BLOCK_704_GRAY64:
            return _jax_render_block_704_gray64_two_views(
                jnp=jnp,
                lax=jax.lax,
                state=device_state,
                frame_size=frame_size,
                target_size=target_size,
                map_size=map_size,
                bonus_count=bonus_count,
                render_mode_id=render_mode_id,
                bonus_render_mode_id=bonus_render_mode_id,
                player_luma_float_by_view=(p0_luma_float, p1_luma_float),
                use_priority_buffer=use_priority_buffer,
            )
        p0 = _jax_render_direct_gray64(
            jnp=jnp,
            lax=jax.lax,
            state=device_state,
            target_size=target_size,
            map_size=map_size,
            bonus_count=bonus_count,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
            controlled_player=0,
            player_luma_by_index=p0_luma,
        )
        p1 = _jax_render_direct_gray64(
            jnp=jnp,
            lax=jax.lax,
            state=device_state,
            target_size=target_size,
            map_size=map_size,
            bonus_count=bonus_count,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
            controlled_player=1,
            player_luma_by_index=p1_luma,
        )
        return jnp.concatenate([p0, p1], axis=0)

    return render


def _jax_render_direct_gray64(
    *,
    jnp: Any,
    lax: Any,
    state: dict[str, Any],
    target_size: int,
    map_size: float,
    bonus_count: int,
    render_mode_id: int,
    bonus_render_mode_id: int,
    controlled_player: int | None,
    player_luma_by_index: tuple[int, ...],
) -> Any:
    grid_x = (
        (jnp.arange(target_size, dtype=jnp.float32) + 0.5) * float(map_size) / float(target_size)
    )
    grid_y = (
        (jnp.arange(target_size, dtype=jnp.float32) + 0.5) * float(map_size) / float(target_size)
    )
    world_x = grid_x[None, None, None, :]
    world_y = grid_y[None, None, :, None]

    trail_dx = world_x - state["trail_x"][:, :, None, None]
    trail_dy = world_y - state["trail_y"][:, :, None, None]
    trail_radius_sq = state["trail_radius"][:, :, None, None] ** 2
    hit = trail_dx * trail_dx + trail_dy * trail_dy <= trail_radius_sq

    if render_mode_id == RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]:
        prev_x, prev_y, prev_active = _previous_owner_trail_slots(
            jnp=jnp,
            lax=lax,
            state=state,
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
            prev_owner=state["trail_owner"],
            active=prev_active,
            break_before=state["trail_break_before"],
        )

    hit = hit & (state["trail_active"][:, :, None, None] != 0)
    owner = state["trail_owner"]
    player_luma = _player_luma_for_state(
        jnp=jnp,
        state=state,
        controlled_player=controlled_player,
        fallback_luma=player_luma_by_index,
        dtype=jnp.uint8,
    )
    trail_value = jnp.where(
        owner < 0,
        jnp.uint8(SYNTHETIC_INVALID_OWNER_LUMA),
        jnp.take_along_axis(
            player_luma,
            jnp.mod(owner, player_luma.shape[1]),
            axis=1,
        ),
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
        lax=lax,
        out=out,
        state=state,
        world_x=world_x,
        world_y=world_y,
        target_size=target_size,
        map_size=map_size,
        bonus_count=bonus_count,
        bonus_render_mode_id=bonus_render_mode_id,
        controlled_player=controlled_player,
        player_luma_by_index=player_luma_by_index,
    )
    out = jnp.reshape(out, (state["trail_x"].shape[0], target_size, target_size))
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
    bonus_render_mode_id: int,
    controlled_player: int | None,
    player_luma_float_by_index: tuple[float, ...],
    use_priority_buffer: bool,
) -> Any:
    if frame_size % target_size != 0:
        raise ValueError("frame_size must be divisible by target_size")
    block = frame_size // target_size
    geometry_dtype = state["trail_x"].dtype
    source = (
        jnp.arange(frame_size, dtype=geometry_dtype) + jnp.asarray(0.5, dtype=geometry_dtype)
    ) * jnp.asarray(map_size / float(frame_size), dtype=geometry_dtype)
    source_blocks = source.reshape(target_size, block)
    world_x = source_blocks[None, None, :, None, :]
    world_y = source_blocks[None, :, None, :, None]
    pixel_blocks = jnp.arange(frame_size, dtype=geometry_dtype).reshape(target_size, block)
    pixel_x = pixel_blocks[None, None, :, None, :]
    pixel_y = pixel_blocks[None, :, None, :, None]
    pixel_scale = jnp.asarray(float(frame_size) / float(map_size), dtype=geometry_dtype)
    image = jnp.full(
        (state["trail_x"].shape[0], target_size, target_size, block, block),
        SYNTHETIC_BACKGROUND_LUMA_FLOAT,
        dtype=jnp.float32,
    )
    if use_priority_buffer:
        trail_priority = jnp.full(
            image.shape,
            jnp.int32(-1),
            dtype=jnp.int32,
        )
    prev_trail_x = None
    prev_trail_y = None
    prev_trail_active = None
    if render_mode_id == RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]:
        prev_trail_x, prev_trail_y, prev_trail_active = _previous_owner_trail_slots(
            jnp=jnp,
            lax=lax,
            state=state,
        )

    def draw_trail_slot(slot: Any, current: Any) -> Any:
        if use_priority_buffer:
            current_luma, current_priority = current
        else:
            current_luma = current
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
            hit = hit | _segment_hits_one_slot(
                jnp=jnp,
                world_x=pixel_x,
                world_y=pixel_y,
                x=x,
                y=y,
                prev_x=prev_trail_x[:, slot] * pixel_scale,
                prev_y=prev_trail_y[:, slot] * pixel_scale,
                radius_sq=radius_sq,
                owner=owner,
                prev_owner=owner,
                active=prev_trail_active[:, slot],
                break_before=state["trail_break_before"][:, slot],
                slot=slot,
            )

        player_luma = _player_luma_for_state(
            jnp=jnp,
            state=state,
            controlled_player=controlled_player,
            fallback_luma=player_luma_float_by_index,
            dtype=jnp.float32,
        )
        value = jnp.where(
            owner < 0,
            jnp.float32(SYNTHETIC_INVALID_OWNER_LUMA_FLOAT),
            jnp.take_along_axis(
                player_luma,
                jnp.mod(owner, player_luma.shape[1])[:, None],
                axis=1,
            )[:, 0],
        ).astype(jnp.float32)
        update = hit & active[:, None, None, None, None]
        if use_priority_buffer:
            priority = _owner_draw_priority(
                jnp=jnp,
                owner=owner,
                player_count=state["head_x"].shape[1],
            )
            update = update & (priority[:, None, None, None, None] >= current_priority)
            return (
                jnp.where(
                    update,
                    value[:, None, None, None, None],
                    current_luma,
                ),
                jnp.where(
                    update,
                    priority[:, None, None, None, None],
                    current_priority,
                ),
            )
        return jnp.where(
            update,
            value[:, None, None, None, None],
            current_luma,
        )

    if use_priority_buffer:
        image, _ = lax.fori_loop(
            0,
            state["trail_x"].shape[1],
            draw_trail_slot,
            (image, trail_priority),
        )
    else:
        image = lax.fori_loop(
            0,
            state["trail_x"].shape[1],
            draw_trail_slot,
            image,
        )

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
        bonus_render_mode_id=bonus_render_mode_id,
        controlled_player=controlled_player,
        player_luma_float_by_index=player_luma_float_by_index,
    )
    gray = jnp.rint(jnp.mean(image.astype(jnp.float32), axis=(3, 4)))
    return jnp.clip(gray, 0, 255).astype(jnp.uint8)[:, None, :, :]


def _jax_render_block_704_gray64_two_views(
    *,
    jnp: Any,
    lax: Any,
    state: dict[str, Any],
    frame_size: int,
    target_size: int,
    map_size: float,
    bonus_count: int,
    render_mode_id: int,
    bonus_render_mode_id: int,
    player_luma_float_by_view: tuple[tuple[float, ...], ...],
    use_priority_buffer: bool,
) -> Any:
    if frame_size % target_size != 0:
        raise ValueError("frame_size must be divisible by target_size")
    block = frame_size // target_size
    view_count = len(player_luma_float_by_view)
    geometry_dtype = state["trail_x"].dtype
    source = (
        jnp.arange(frame_size, dtype=geometry_dtype) + jnp.asarray(0.5, dtype=geometry_dtype)
    ) * jnp.asarray(map_size / float(frame_size), dtype=geometry_dtype)
    source_blocks = source.reshape(target_size, block)
    world_x = source_blocks[None, None, :, None, :]
    world_y = source_blocks[None, :, None, :, None]
    pixel_blocks = jnp.arange(frame_size, dtype=geometry_dtype).reshape(target_size, block)
    pixel_x = pixel_blocks[None, None, :, None, :]
    pixel_y = pixel_blocks[None, :, None, :, None]
    pixel_scale = jnp.asarray(float(frame_size) / float(map_size), dtype=geometry_dtype)
    image = jnp.full(
        (view_count, state["trail_x"].shape[0], target_size, target_size, block, block),
        SYNTHETIC_BACKGROUND_LUMA_FLOAT,
        dtype=jnp.float32,
    )
    if use_priority_buffer:
        trail_priority = jnp.full(
            image.shape,
            jnp.int32(-1),
            dtype=jnp.int32,
        )
    prev_trail_x = None
    prev_trail_y = None
    prev_trail_active = None
    if render_mode_id == RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]:
        prev_trail_x, prev_trail_y, prev_trail_active = _previous_owner_trail_slots(
            jnp=jnp,
            lax=lax,
            state=state,
        )
    player_luma = _player_luma_for_controlled_players(
        jnp=jnp,
        state=state,
        controlled_players=(0, 1),
        fallback_luma_by_view=player_luma_float_by_view,
        dtype=jnp.float32,
    )

    def draw_trail_slot(slot: Any, current: Any) -> Any:
        if use_priority_buffer:
            current_luma, current_priority = current
        else:
            current_luma = current
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
            hit = hit | _segment_hits_one_slot(
                jnp=jnp,
                world_x=pixel_x,
                world_y=pixel_y,
                x=x,
                y=y,
                prev_x=prev_trail_x[:, slot] * pixel_scale,
                prev_y=prev_trail_y[:, slot] * pixel_scale,
                radius_sq=radius_sq,
                owner=owner,
                prev_owner=owner,
                active=prev_trail_active[:, slot],
                break_before=state["trail_break_before"][:, slot],
                slot=slot,
            )

        owner_value = jnp.where(
            owner < 0,
            jnp.float32(SYNTHETIC_INVALID_OWNER_LUMA_FLOAT),
            jnp.take_along_axis(
                player_luma,
                jnp.mod(owner, player_luma.shape[2])[None, :, None],
                axis=2,
            )[:, :, 0],
        ).astype(jnp.float32)
        update = hit[None, :, :, :, :, :] & active[None, :, None, None, None, None]
        if use_priority_buffer:
            priority = _owner_draw_priority(
                jnp=jnp,
                owner=owner,
                player_count=state["head_x"].shape[1],
            )
            update = update & (priority[None, :, None, None, None, None] >= current_priority)
            return (
                jnp.where(
                    update,
                    owner_value[:, :, None, None, None, None],
                    current_luma,
                ),
                jnp.where(
                    update,
                    priority[None, :, None, None, None, None],
                    current_priority,
                ),
            )
        return jnp.where(
            update,
            owner_value[:, :, None, None, None, None],
            current_luma,
        )

    if use_priority_buffer:
        image, _ = lax.fori_loop(
            0,
            state["trail_x"].shape[1],
            draw_trail_slot,
            (image, trail_priority),
        )
    else:
        image = lax.fori_loop(
            0,
            state["trail_x"].shape[1],
            draw_trail_slot,
            image,
        )
    image = _draw_block_bonus_and_heads_two_views(
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
        bonus_render_mode_id=bonus_render_mode_id,
        player_luma_float_by_view=player_luma_float_by_view,
    )
    gray = jnp.rint(jnp.mean(image.astype(jnp.float32), axis=(4, 5)))
    out = jnp.clip(gray, 0, 255).astype(jnp.uint8)[:, :, None, :, :]
    return out.reshape((view_count * state["trail_x"].shape[0], 1, target_size, target_size))


def _owner_draw_priority(*, jnp: Any, owner: Any, player_count: int) -> Any:
    return jnp.where(
        owner < 0,
        jnp.int32(0),
        (jnp.int32(player_count) - owner.astype(jnp.int32)),
    )


def _player_luma_for_state(
    *,
    jnp: Any,
    state: dict[str, Any],
    controlled_player: int | None,
    fallback_luma: tuple[int, ...] | tuple[float, ...],
    dtype: Any,
) -> Any:
    batch_size = state["head_x"].shape[0]
    player_count = state["head_x"].shape[1]
    fallback = jnp.asarray(fallback_luma, dtype=dtype)
    if controlled_player is None or "avatar_color" not in state:
        return jnp.broadcast_to(fallback[None, :player_count], (batch_size, player_count))

    controlled = int(controlled_player)
    avatar_color = state["avatar_color"][:, :player_count].astype(jnp.int32)
    self_color = avatar_color[:, controlled]
    owns_self_color = avatar_color == self_color[:, None]
    self_value = jnp.asarray(PERSPECTIVE_SELF_LUMA_FLOAT, dtype=dtype)
    other_value = jnp.asarray(PERSPECTIVE_OTHER_LUMA_FLOAT, dtype=dtype)
    return jnp.where(owns_self_color, self_value, other_value).astype(dtype)


def _player_luma_for_controlled_players(
    *,
    jnp: Any,
    state: dict[str, Any],
    controlled_players: tuple[int, ...],
    fallback_luma_by_view: tuple[tuple[float, ...], ...],
    dtype: Any,
) -> Any:
    return jnp.stack(
        [
            _player_luma_for_state(
                jnp=jnp,
                state=state,
                controlled_player=player,
                fallback_luma=fallback_luma_by_view[index],
                dtype=dtype,
            )
            for index, player in enumerate(controlled_players)
        ],
        axis=0,
    )


def _previous_owner_trail_slots(
    *,
    jnp: Any,
    lax: Any,
    state: dict[str, Any],
) -> tuple[Any, Any, Any]:
    batch_size = state["trail_x"].shape[0]
    player_count = state["head_x"].shape[1]
    rows = jnp.arange(batch_size)
    prev_x = jnp.zeros_like(state["trail_x"])
    prev_y = jnp.zeros_like(state["trail_y"])
    prev_active = jnp.zeros(state["trail_active"].shape, dtype=bool)
    last_x = jnp.zeros((batch_size, player_count), dtype=state["trail_x"].dtype)
    last_y = jnp.zeros((batch_size, player_count), dtype=state["trail_y"].dtype)
    last_radius = jnp.zeros((batch_size, player_count), dtype=state["trail_radius"].dtype)
    last_active = jnp.zeros((batch_size, player_count), dtype=bool)

    def scan_slot(slot: Any, carry: Any) -> Any:
        (
            prev_x_acc,
            prev_y_acc,
            prev_active_acc,
            last_x_acc,
            last_y_acc,
            last_radius_acc,
            last_active_acc,
        ) = carry
        owner = state["trail_owner"][:, slot]
        owner_valid = (owner >= 0) & (owner < player_count)
        owner_index = jnp.clip(owner, 0, player_count - 1)
        gathered_x = last_x_acc[rows, owner_index]
        gathered_y = last_y_acc[rows, owner_index]
        gathered_radius = last_radius_acc[rows, owner_index]
        radius_matches = jnp.abs(gathered_radius - state["trail_radius"][:, slot]) <= 1.0e-6
        gathered_active = last_active_acc[rows, owner_index] & owner_valid & radius_matches

        prev_x_acc = prev_x_acc.at[:, slot].set(gathered_x)
        prev_y_acc = prev_y_acc.at[:, slot].set(gathered_y)
        prev_active_acc = prev_active_acc.at[:, slot].set(gathered_active)

        update = (state["trail_active"][:, slot] != 0) & owner_valid
        last_x_acc = last_x_acc.at[rows, owner_index].set(
            jnp.where(update, state["trail_x"][:, slot], gathered_x)
        )
        last_y_acc = last_y_acc.at[rows, owner_index].set(
            jnp.where(update, state["trail_y"][:, slot], gathered_y)
        )
        last_radius_acc = last_radius_acc.at[rows, owner_index].set(
            jnp.where(update, state["trail_radius"][:, slot], gathered_radius)
        )
        last_active_acc = last_active_acc.at[rows, owner_index].set(
            jnp.where(update, jnp.ones_like(gathered_active), last_active_acc[rows, owner_index])
        )
        return (
            prev_x_acc,
            prev_y_acc,
            prev_active_acc,
            last_x_acc,
            last_y_acc,
            last_radius_acc,
            last_active_acc,
        )

    prev_x, prev_y, prev_active, _, _, _, _ = lax.fori_loop(
        0,
        state["trail_x"].shape[1],
        scan_slot,
        (prev_x, prev_y, prev_active, last_x, last_y, last_radius, last_active),
    )
    return prev_x, prev_y, prev_active


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
    ok = (break_before == 0) & (active != 0) & (owner == prev_owner) & (length_sq > 0.0001)
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
    ok = (slot > 0) & (break_before == 0) & active & (owner == prev_owner) & (length_sq > 0.0001)
    return (dx * dx + dy * dy <= radius_sq) & ok[:, None, None, None, None]


def _draw_direct_bonus_and_heads(
    *,
    jnp: Any,
    lax: Any,
    out: Any,
    state: dict[str, Any],
    world_x: Any,
    world_y: Any,
    target_size: int,
    map_size: float,
    bonus_count: int,
    bonus_render_mode_id: int,
    controlled_player: int | None,
    player_luma_by_index: tuple[int, ...],
) -> Any:
    pixel_x = jnp.arange(target_size, dtype=state["trail_x"].dtype)[None, None, :]
    pixel_y = jnp.arange(target_size, dtype=state["trail_x"].dtype)[None, :, None]

    def draw_bonus(index: Any, current: Any) -> Any:
        if bonus_render_mode_id == BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]:
            hit, value = _direct_simple_symbol_bonus_hit_value(
                jnp=jnp,
                state=state,
                index=index,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                target_size=target_size,
                map_size=map_size,
            )
            return jnp.where(hit, value, current)

        bonus_x = jnp.take(state["bonus_x"], index, axis=1)
        bonus_y = jnp.take(state["bonus_y"], index, axis=1)
        radius = jnp.take(state["bonus_radius"], index, axis=1)
        active = jnp.take(state["bonus_active"], index, axis=1)
        bonus_type = jnp.take(state["bonus_type"], index, axis=1)
        dx = world_x - bonus_x[:, None, None]
        dy = world_y - bonus_y[:, None, None]
        radius = radius[:, None, None]
        hit = (dx * dx + dy * dy <= radius * radius) & (
            active[:, None, None] != 0
        )
        value = (144 + (bonus_type % 4) * 24).astype(jnp.uint8)
        return jnp.where(hit, value[:, None, None], current)

    for index in range(bonus_count):
        out = draw_bonus(index, out)

    player_count = state["head_x"].shape[1]
    player_luma = _player_luma_for_state(
        jnp=jnp,
        state=state,
        controlled_player=controlled_player,
        fallback_luma=player_luma_by_index,
        dtype=jnp.uint8,
    )

    def draw_head(index: Any, current: Any) -> Any:
        head_x = jnp.take(state["head_x"], index, axis=1)
        head_y = jnp.take(state["head_y"], index, axis=1)
        radius = jnp.take(state["head_radius"], index, axis=1)
        alive = jnp.take(state["head_alive"], index, axis=1)
        dx = world_x - head_x[:, None, None]
        dy = world_y - head_y[:, None, None]
        radius = radius[:, None, None]
        hit = (dx * dx + dy * dy <= radius * radius) & (
            alive[:, None, None] != 0
        )
        value = jnp.take(player_luma, jnp.mod(index, player_luma.shape[1]), axis=1)
        return jnp.where(hit, value[:, None, None], current)

    for index in range(player_count):
        out = draw_head(index, out)
    return out


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
    bonus_render_mode_id: int,
    controlled_player: int | None,
    player_luma_float_by_index: tuple[float, ...],
) -> Any:
    def draw_bonus(index: Any, current: Any) -> Any:
        if bonus_render_mode_id == BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]:
            hit, value = _block_simple_symbol_bonus_hit_value(
                jnp=jnp,
                state=state,
                index=index,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                frame_size=frame_size,
                map_size=map_size,
            )
            return jnp.where(hit, value, current)
        else:
            dx = world_x - state["bonus_x"][:, index, None, None, None, None]
            dy = world_y - state["bonus_y"][:, index, None, None, None, None]
            radius = state["bonus_radius"][:, index, None, None, None, None]
            hit = (dx * dx + dy * dy <= radius * radius) & (
                state["bonus_active"][:, index, None, None, None, None] != 0
            )
            value = (144 + (state["bonus_type"][:, index] % 4) * 24).astype(jnp.float32)
        return jnp.where(hit, value[:, None, None, None, None], current)

    def draw_head(index: Any, current: Any) -> Any:
        px = jnp.rint(state["head_x"][:, index] * float(frame_size - 1) / float(map_size))
        py = jnp.rint(state["head_y"][:, index] * float(frame_size - 1) / float(map_size))
        radius = jnp.ceil(state["head_radius"][:, index] * float(frame_size) / float(map_size))
        dx = pixel_x - px[:, None, None, None, None]
        dy = pixel_y - py[:, None, None, None, None]
        hit = (dx * dx + dy * dy <= radius[:, None, None, None, None] ** 2) & (
            state["head_alive"][:, index, None, None, None, None] != 0
        )
        player_luma = _player_luma_for_state(
            jnp=jnp,
            state=state,
            controlled_player=controlled_player,
            fallback_luma=player_luma_float_by_index,
            dtype=jnp.float32,
        )
        value = player_luma[:, jnp.mod(index, player_luma.shape[1])]
        return jnp.where(hit, value[:, None, None, None, None], current)

    if bonus_count > 0:
        image = lax.fori_loop(0, bonus_count, draw_bonus, image)
    return lax.fori_loop(0, state["head_x"].shape[1], draw_head, image)


def _draw_block_bonus_and_heads_two_views(
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
    bonus_render_mode_id: int,
    player_luma_float_by_view: tuple[tuple[float, ...], ...],
) -> Any:
    def draw_bonus(index: Any, current: Any) -> Any:
        if bonus_render_mode_id == BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]:
            hit, value = _block_simple_symbol_bonus_hit_value(
                jnp=jnp,
                state=state,
                index=index,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                frame_size=frame_size,
                map_size=map_size,
            )
            return jnp.where(hit[None, :, :, :, :, :], value[None, :, :, :, :, :], current)
        else:
            dx = world_x - state["bonus_x"][:, index, None, None, None, None]
            dy = world_y - state["bonus_y"][:, index, None, None, None, None]
            radius = state["bonus_radius"][:, index, None, None, None, None]
            hit = (dx * dx + dy * dy <= radius * radius) & (
                state["bonus_active"][:, index, None, None, None, None] != 0
            )
            value = (144 + (state["bonus_type"][:, index] % 4) * 24).astype(jnp.float32)
        return jnp.where(
            hit[None, :, :, :, :, :],
            value[None, :, None, None, None, None],
            current,
        )

    def draw_head(index: Any, current: Any) -> Any:
        px = jnp.rint(state["head_x"][:, index] * float(frame_size - 1) / float(map_size))
        py = jnp.rint(state["head_y"][:, index] * float(frame_size - 1) / float(map_size))
        radius = jnp.ceil(state["head_radius"][:, index] * float(frame_size) / float(map_size))
        dx = pixel_x - px[:, None, None, None, None]
        dy = pixel_y - py[:, None, None, None, None]
        hit = (dx * dx + dy * dy <= radius[:, None, None, None, None] ** 2) & (
            state["head_alive"][:, index, None, None, None, None] != 0
        )
        player_luma = _player_luma_for_controlled_players(
            jnp=jnp,
            state=state,
            controlled_players=(0, 1),
            fallback_luma_by_view=player_luma_float_by_view,
            dtype=jnp.float32,
        )
        value = player_luma[:, :, jnp.mod(index, player_luma.shape[2])]
        return jnp.where(
            hit[None, :, :, :, :, :],
            value[:, :, None, None, None, None],
            current,
        )

    if bonus_count > 0:
        image = lax.fori_loop(0, bonus_count, draw_bonus, image)
    return lax.fori_loop(0, state["head_x"].shape[1], draw_head, image)


def _block_simple_symbol_bonus_hit_value(
    *,
    jnp: Any,
    state: dict[str, Any],
    index: Any,
    pixel_x: Any,
    pixel_y: Any,
    frame_size: int,
    map_size: float,
) -> tuple[Any, Any]:
    x = jnp.take(state["bonus_x"], index, axis=1)
    y = jnp.take(state["bonus_y"], index, axis=1)
    radius = jnp.take(state["bonus_radius"], index, axis=1)
    active = jnp.take(state["bonus_active"], index, axis=1)
    bonus_type = jnp.take(state["bonus_type"], index, axis=1)
    radius_px = jnp.maximum(
        3.0,
        jnp.ceil(radius * float(frame_size) / float(map_size)),
    )
    center_x = jnp.clip(
        jnp.rint(x * float(frame_size - 1) / float(map_size)),
        0.0,
        float(frame_size - 1),
    )
    center_y = jnp.clip(
        jnp.rint(y * float(frame_size - 1) / float(map_size)),
        0.0,
        float(frame_size - 1),
    )
    dst_size = radius_px * 2.0 + 1.0
    local_x = pixel_x - (center_x[:, None, None, None, None] - radius_px[:, None, None, None, None])
    local_y = pixel_y - (center_y[:, None, None, None, None] - radius_px[:, None, None, None, None])
    in_bounds = (
        (local_x >= 0.0)
        & (local_y >= 0.0)
        & (local_x < dst_size[:, None, None, None, None])
        & (local_y < dst_size[:, None, None, None, None])
        & (active[:, None, None, None, None] != 0)
        & (radius[:, None, None, None, None] > 0.0)
    )
    scale = float(BONUS_SYMBOL_BASE_SIZE - 1)
    base_x = jnp.rint(local_x * scale / jnp.maximum(dst_size[:, None, None, None, None] - 1.0, 1.0))
    base_y = jnp.rint(local_y * scale / jnp.maximum(dst_size[:, None, None, None, None] - 1.0, 1.0))
    base_x = jnp.clip(base_x, 0.0, scale).astype(jnp.int32)
    base_y = jnp.clip(base_y, 0.0, scale).astype(jnp.int32)
    stamp_value = _simple_symbol_luma(
        jnp=jnp,
        base_x=base_x,
        base_y=base_y,
        bonus_type=bonus_type,
    )
    hit = in_bounds & (stamp_value > 0.0)
    return hit, stamp_value


def _direct_simple_symbol_bonus_hit_value(
    *,
    jnp: Any,
    state: dict[str, Any],
    index: Any,
    pixel_x: Any,
    pixel_y: Any,
    target_size: int,
    map_size: float,
) -> tuple[Any, Any]:
    x = jnp.take(state["bonus_x"], index, axis=1)
    y = jnp.take(state["bonus_y"], index, axis=1)
    radius = jnp.take(state["bonus_radius"], index, axis=1)
    active = jnp.take(state["bonus_active"], index, axis=1)
    bonus_type = jnp.take(state["bonus_type"], index, axis=1)
    radius_px = jnp.maximum(
        3.0,
        jnp.ceil(radius * float(target_size) / float(map_size)),
    )
    center_x = jnp.clip(
        jnp.rint(x * float(target_size - 1) / float(map_size)),
        0.0,
        float(target_size - 1),
    )
    center_y = jnp.clip(
        jnp.rint(y * float(target_size - 1) / float(map_size)),
        0.0,
        float(target_size - 1),
    )
    dst_size = radius_px * 2.0 + 1.0
    local_x = pixel_x - (center_x[:, None, None] - radius_px[:, None, None])
    local_y = pixel_y - (center_y[:, None, None] - radius_px[:, None, None])
    in_bounds = (
        (local_x >= 0.0)
        & (local_y >= 0.0)
        & (local_x < dst_size[:, None, None])
        & (local_y < dst_size[:, None, None])
        & (active[:, None, None] != 0)
        & (radius[:, None, None] > 0.0)
    )
    scale = float(BONUS_SYMBOL_BASE_SIZE - 1)
    base_x = jnp.rint(local_x * scale / jnp.maximum(dst_size[:, None, None] - 1.0, 1.0))
    base_y = jnp.rint(local_y * scale / jnp.maximum(dst_size[:, None, None] - 1.0, 1.0))
    base_x = jnp.clip(base_x, 0.0, scale).astype(jnp.int32)
    base_y = jnp.clip(base_y, 0.0, scale).astype(jnp.int32)
    stamp_value = _simple_symbol_luma(
        jnp=jnp,
        base_x=base_x,
        base_y=base_y,
        bonus_type=bonus_type,
    )
    hit = in_bounds & (stamp_value > 0.0)
    return hit, stamp_value.astype(jnp.uint8)


def _simple_symbol_luma(*, jnp: Any, base_x: Any, base_y: Any, bonus_type: Any) -> Any:
    code = _bonus_type_code(jnp=jnp, bonus_type=bonus_type)
    symbol_index = code - 1
    outer_index = symbol_index // 4
    inner_index = symbol_index % 4
    expand_shape = outer_index.shape + (1,) * (base_x.ndim - outer_index.ndim)
    outer = outer_index.reshape(expand_shape)
    center = (BONUS_SYMBOL_BASE_SIZE - 1) // 2
    dx = jnp.abs(base_x - center)
    dy = jnp.abs(base_y - center)
    outer_circle = dx * dx + dy * dy <= center * center
    outer_diamond = dx + dy <= center + 1
    outer_square = jnp.ones_like(outer_circle, dtype=bool)
    outer_mask = jnp.where(
        outer == 0,
        outer_circle,
        jnp.where(outer == 1, outer_diamond, outer_square),
    )
    inner = _simple_symbol_inner_mask(
        jnp=jnp,
        base_x=base_x,
        base_y=base_y,
        outer_index=outer_index,
        inner_index=inner_index,
    )
    outer_luma = jnp.asarray(BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE, dtype=jnp.float32)
    inner_luma = jnp.asarray(BONUS_SYMBOL_INNER_LUMA_BY_SHAPE, dtype=jnp.float32)
    outer_value = jnp.take(outer_luma, outer_index).reshape(expand_shape)
    inner_value = jnp.take(inner_luma, outer_index).reshape(expand_shape)
    return jnp.where(outer_mask & inner, inner_value, jnp.where(outer_mask, outer_value, 0.0))


def _simple_symbol_inner_mask(
    *,
    jnp: Any,
    base_x: Any,
    base_y: Any,
    outer_index: Any,
    inner_index: Any,
) -> Any:
    expand_shape = outer_index.shape + (1,) * (base_x.ndim - outer_index.ndim)
    outer = outer_index.reshape(expand_shape)
    inner = inner_index.reshape(expand_shape)
    center = (BONUS_SYMBOL_BASE_SIZE - 1) // 2
    plus_01 = ((base_y >= center - 1) & (base_y <= center + 1) & (base_x >= 1) & (base_x <= 5)) | (
        (base_x >= center - 1) & (base_x <= center + 1) & (base_y >= 1) & (base_y <= 5)
    )
    plus_2 = ((base_y >= center) & (base_y <= 5)) | ((base_x >= center) & (base_x <= 5))
    plus = jnp.where(outer == 2, plus_2, plus_01)

    x0 = (jnp.abs(base_x - base_y) <= 1) | (
        ((base_y == 0) & (base_x == 5))
        | ((base_y == 1) & (base_x == 4))
        | ((base_y == 2) & (base_x == 3))
        | ((base_y == 3) & (base_x == 2))
        | ((base_y == 4) & (base_x == 1))
        | ((base_y == 5) & (base_x == 0))
    )
    x1 = (
        (base_x == base_y)
        | (base_x == 6 - base_y)
        | (base_x == base_y + 1)
        | ((base_y > 0) & (base_x == 7 - base_y))
    )
    x2 = (
        (base_x == 6 - base_y)
        | (base_x == 5 - base_y)
        | ((base_y >= 2) & (base_x == base_y - 1))
        | ((base_y >= 2) & (base_x == base_y))
    )
    cross = jnp.where(outer == 0, x0, jnp.where(outer == 1, x1, x2))

    horiz0 = (base_y >= center - 2) & (base_y <= center)
    horiz1 = (base_y >= 1) & (base_y <= center)
    horiz2 = (base_y >= center) & (base_y <= 5)
    horizontal = jnp.where(outer == 0, horiz0, jnp.where(outer == 1, horiz1, horiz2))

    vert0 = (base_x >= center) & (base_x <= center + 2)
    vert1 = (base_x >= 1) & (base_x <= center)
    vert2 = (base_x >= center) & (base_x <= 5)
    vertical = jnp.where(outer == 0, vert0, jnp.where(outer == 1, vert1, vert2))

    return jnp.where(
        inner == 0,
        plus,
        jnp.where(inner == 1, cross, jnp.where(inner == 2, horizontal, vertical)),
    )


def _bonus_type_code(*, jnp: Any, bonus_type: Any) -> Any:
    code = bonus_type.astype(jnp.int32)
    return jnp.where((code >= 1) & (code <= 12), code, jnp.ones_like(code))


def _copy_state_to_device(
    *,
    jax: Any,
    state: dict[str, Any],
    block_until_ready: bool = True,
) -> dict[str, Any]:
    copied = {key: jax.device_put(value) for key, value in state.items()}
    if block_until_ready:
        for value in copied.values():
            value.block_until_ready()
    return copied


def _verify_against_cpu(
    *,
    jax: Any,
    np: Any,
    state: dict[str, Any],
    production_reference_state: dict[str, Any] | None,
    config: dict[str, Any],
    render_mode_id: int,
    bonus_render_mode_id: int,
) -> dict[str, Any]:
    import jax.numpy as jnp

    verify_rows = min(int(config["verify_rows"]), int(config["batch_size"]))
    if verify_rows <= 0:
        return {"rows": 0, "status": "skipped"}

    target_size = int(config["target_size"])
    view_count = 2 if config.get("render_views") == RENDER_VIEWS_BOTH else 1
    operations = (
        verify_rows
        * view_count
        * target_size
        * target_size
        * int(config["trail_slots"])
    )
    if operations > int(config["cpu_verify_max_pixel_trail_tests"]):
        return {
            "rows": verify_rows,
            "status": "skipped_cpu_reference_too_large",
            "pixel_trail_slot_tests": operations,
            "max_pixel_trail_slot_tests": int(config["cpu_verify_max_pixel_trail_tests"]),
        }

    sliced_state = _slice_batch_state(np=np, state=state, rows=verify_rows)
    verify_config = {**config, "batch_size": verify_rows}
    if config.get("render_views") == RENDER_VIEWS_BOTH:
        verify_render_fn = _make_jax_two_view_render_fn(
            jax=jax,
            jnp=jnp,
            config=verify_config,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
        )
    else:
        verify_render_fn = _make_jax_render_fn(
            jax=jax,
            jnp=jnp,
            config=verify_config,
            render_mode_id=render_mode_id,
            bonus_render_mode_id=bonus_render_mode_id,
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
        if config.get("render_views") == RENDER_VIEWS_BOTH:
            cpu = np.concatenate(
                [
                    _cpu_render_direct_gray64(
                        np=np,
                        state=sliced_state,
                        config={**config, "batch_size": verify_rows, "controlled_player": 0},
                        render_mode_id=render_mode_id,
                        bonus_render_mode_id=bonus_render_mode_id,
                        player_luma_by_index=_player_luma_by_index(
                            {**config, "controlled_player": 0}
                        ),
                    ),
                    _cpu_render_direct_gray64(
                        np=np,
                        state=sliced_state,
                        config={**config, "batch_size": verify_rows, "controlled_player": 1},
                        render_mode_id=render_mode_id,
                        bonus_render_mode_id=bonus_render_mode_id,
                        player_luma_by_index=_player_luma_by_index(
                            {**config, "controlled_player": 1}
                        ),
                    ),
                ],
                axis=0,
            )
        else:
            cpu = _cpu_render_direct_gray64(
                np=np,
                state=sliced_state,
                config={**config, "batch_size": verify_rows},
                render_mode_id=render_mode_id,
                bonus_render_mode_id=bonus_render_mode_id,
                player_luma_by_index=_player_luma_by_index(config),
            )
    else:
        cpu_reference_kind = (
            "production_render_source_state_canvas_gray64_"
            f"{config['render_mode']}_{config['bonus_render_mode']}"
        )
        if int(config["target_size"]) != 64 or int(config["frame_size"]) != 704:
            return {
                "rows": verify_rows,
                "status": "skipped_production_reference_requires_704_to_64",
                "render_surface": config["render_surface"],
                "frame_size": int(config["frame_size"]),
                "target_size": int(config["target_size"]),
            }
        if production_reference_state is not None:
            sliced_production = _slice_batch_state(
                np=np,
                state=production_reference_state,
                rows=verify_rows,
            )
            if config.get("render_views") == RENDER_VIEWS_BOTH:
                cpu = np.concatenate(
                    [
                        _cpu_render_original_production_canvas_gray64(
                            np=np,
                            production_state=sliced_production,
                            config={
                                **config,
                                "batch_size": verify_rows,
                                "controlled_player": 0,
                            },
                        ),
                        _cpu_render_original_production_canvas_gray64(
                            np=np,
                            production_state=sliced_production,
                            config={
                                **config,
                                "batch_size": verify_rows,
                                "controlled_player": 1,
                            },
                        ),
                    ],
                    axis=0,
                )
            else:
                cpu = _cpu_render_original_production_canvas_gray64(
                    np=np,
                    production_state=sliced_production,
                    config={**config, "batch_size": verify_rows},
                )
        else:
            if config.get("render_views") == RENDER_VIEWS_BOTH:
                cpu = np.concatenate(
                    [
                        _cpu_render_production_canvas_gray64(
                            np=np,
                            state=sliced_state,
                            config={
                                **config,
                                "batch_size": verify_rows,
                                "controlled_player": 0,
                            },
                        ),
                        _cpu_render_production_canvas_gray64(
                            np=np,
                            state=sliced_state,
                            config={
                                **config,
                                "batch_size": verify_rows,
                                "controlled_player": 1,
                            },
                        ),
                    ],
                    axis=0,
                )
            else:
                cpu = _cpu_render_production_canvas_gray64(
                    np=np,
                    state=sliced_state,
                    config={**config, "batch_size": verify_rows},
                )
    cpu_sec = time.perf_counter() - started

    diff = np.abs(gpu.astype(np.int16) - cpu.astype(np.int16))
    mismatch_count = int(np.count_nonzero(diff))
    total_values = int(diff.size)
    mismatch_samples: list[dict[str, int]] = []
    if mismatch_count:
        for row, channel, y, x in np.argwhere(diff)[:16]:
            mismatch_samples.append(
                {
                    "row": int(row),
                    "channel": int(channel),
                    "y": int(y),
                    "x": int(x),
                    "gpu": int(gpu[row, channel, y, x]),
                    "cpu": int(cpu[row, channel, y, x]),
                    "abs_diff": int(diff[row, channel, y, x]),
                }
            )
    return {
        "rows": verify_rows,
        "status": "checked",
        "cpu_reference_kind": cpu_reference_kind,
        "render_views": str(config.get("render_views", RENDER_VIEWS_SINGLE)),
        "output_order": (
            RENDER_OUTPUT_ORDER_VIEW_MAJOR
            if config.get("render_views") == RENDER_VIEWS_BOTH
            else "single_view_batch_major"
        ),
        "exact_parity": bool(mismatch_count == 0),
        "mismatch_count": mismatch_count,
        "mismatch_fraction": float(mismatch_count / total_values) if total_values else 0.0,
        "max_abs_diff": int(diff.max()) if diff.size else 0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "cpu_reference_sec": cpu_sec,
        "gpu_verify_render_sec": gpu_sec,
        "pixel_trail_slot_tests": operations,
        "mismatch_samples": mismatch_samples,
    }


def _cpu_render_direct_gray64(
    *,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
    render_mode_id: int,
    bonus_render_mode_id: int,
    player_luma_by_index: tuple[int, ...],
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
    prev_x, prev_y, prev_active = _cpu_previous_owner_trail_slots(np=np, state=state)

    for row in range(batch_size):
        row_player_luma = _cpu_player_luma_for_state_row(
            state=state,
            row=row,
            player_count=player_count,
            controlled_player=config.get("controlled_player"),
            fallback_luma=player_luma_by_index,
        )
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
                        and bool(prev_active[row, slot])
                    ):
                        hit = _point_hits_segment(
                            world_x=world_x,
                            world_y=world_y,
                            ax=float(prev_x[row, slot]),
                            ay=float(prev_y[row, slot]),
                            bx=float(state["trail_x"][row, slot]),
                            by=float(state["trail_y"][row, slot]),
                            radius=radius,
                        )
                    if hit:
                        value = max(value, _owner_luma(owner, row_player_luma))

                for bonus in range(bonus_count):
                    if not bool(state["bonus_active"][row, bonus]):
                        continue
                    radius = float(state["bonus_radius"][row, bonus])
                    if bonus_render_mode_id == BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]:
                        symbol_value = _cpu_direct_simple_symbol_bonus_luma(
                            np=np,
                            state=state,
                            row=row,
                            bonus=bonus,
                            px=px,
                            py=py,
                            target_size=target_size,
                            map_size=map_size,
                        )
                        if symbol_value > 0:
                            value = symbol_value
                    else:
                        dx = world_x - float(state["bonus_x"][row, bonus])
                        dy = world_y - float(state["bonus_y"][row, bonus])
                        if dx * dx + dy * dy <= radius * radius:
                            value = 144 + int(state["bonus_type"][row, bonus] % 4) * 24

                for player in range(player_count):
                    if not bool(state["head_alive"][row, player]):
                        continue
                    dx = world_x - float(state["head_x"][row, player])
                    dy = world_y - float(state["head_y"][row, player])
                    radius = float(state["head_radius"][row, player])
                    if dx * dx + dy * dy <= radius * radius:
                        value = _owner_luma(player, row_player_luma)

                out[row, 0, py, px] = np.uint8(value)
    return out


def _cpu_player_luma_for_state_row(
    *,
    state: dict[str, Any],
    row: int,
    player_count: int,
    controlled_player: int | None,
    fallback_luma: tuple[int, ...],
) -> tuple[int, ...]:
    if controlled_player is None or "avatar_color" not in state:
        return fallback_luma[:player_count]
    controlled = int(controlled_player)
    avatar_color = state["avatar_color"][row, :player_count]
    if controlled < 0 or controlled >= player_count:
        return fallback_luma[:player_count]
    self_color = int(avatar_color[controlled])
    return tuple(
        PERSPECTIVE_SELF_LUMA if int(color) == self_color else PERSPECTIVE_OTHER_LUMA
        for color in avatar_color
    )


def _cpu_direct_simple_symbol_bonus_luma(
    *,
    np: Any,
    state: dict[str, Any],
    row: int,
    bonus: int,
    px: int,
    py: int,
    target_size: int,
    map_size: float,
) -> int:
    radius = float(state["bonus_radius"][row, bonus])
    if radius <= 0.0:
        return 0
    radius_px = max(3.0, float(np.ceil(radius * float(target_size) / float(map_size))))
    center_x = float(
        np.clip(
            np.rint(float(state["bonus_x"][row, bonus]) * float(target_size - 1) / float(map_size)),
            0.0,
            float(target_size - 1),
        )
    )
    center_y = float(
        np.clip(
            np.rint(float(state["bonus_y"][row, bonus]) * float(target_size - 1) / float(map_size)),
            0.0,
            float(target_size - 1),
        )
    )
    dst_size = radius_px * 2.0 + 1.0
    local_x = float(px) - (center_x - radius_px)
    local_y = float(py) - (center_y - radius_px)
    if local_x < 0.0 or local_y < 0.0 or local_x >= dst_size or local_y >= dst_size:
        return 0
    scale = float(BONUS_SYMBOL_BASE_SIZE - 1)
    base_x = int(np.clip(np.rint(local_x * scale / max(dst_size - 1.0, 1.0)), 0.0, scale))
    base_y = int(np.clip(np.rint(local_y * scale / max(dst_size - 1.0, 1.0)), 0.0, scale))
    return _cpu_simple_symbol_luma(
        base_x=base_x,
        base_y=base_y,
        bonus_type=int(state["bonus_type"][row, bonus]),
    )


def _cpu_simple_symbol_luma(*, base_x: int, base_y: int, bonus_type: int) -> int:
    code = bonus_type if 1 <= bonus_type <= 12 else 1
    symbol_index = code - 1
    outer_index = symbol_index // 4
    inner_index = symbol_index % 4
    center = (BONUS_SYMBOL_BASE_SIZE - 1) // 2
    dx = abs(base_x - center)
    dy = abs(base_y - center)
    if outer_index == 0:
        outer_mask = dx * dx + dy * dy <= center * center
    elif outer_index == 1:
        outer_mask = dx + dy <= center + 1
    else:
        outer_mask = True
    if not outer_mask:
        return 0

    if inner_index == 0:
        if outer_index == 2:
            inner_mask = (base_y >= center and base_y <= 5) or (base_x >= center and base_x <= 5)
        else:
            inner_mask = (
                base_y >= center - 1
                and base_y <= center + 1
                and base_x >= 1
                and base_x <= 5
            ) or (
                base_x >= center - 1
                and base_x <= center + 1
                and base_y >= 1
                and base_y <= 5
            )
    elif inner_index == 1:
        if outer_index == 0:
            inner_mask = abs(base_x - base_y) <= 1 or (
                (base_y == 0 and base_x == 5)
                or (base_y == 1 and base_x == 4)
                or (base_y == 2 and base_x == 3)
                or (base_y == 3 and base_x == 2)
                or (base_y == 4 and base_x == 1)
                or (base_y == 5 and base_x == 0)
            )
        elif outer_index == 1:
            inner_mask = (
                base_x == base_y
                or base_x == 6 - base_y
                or base_x == base_y + 1
                or (base_y > 0 and base_x == 7 - base_y)
            )
        else:
            inner_mask = (
                base_x == 6 - base_y
                or base_x == 5 - base_y
                or (base_y >= 2 and base_x == base_y - 1)
                or (base_y >= 2 and base_x == base_y)
            )
    elif inner_index == 2:
        if outer_index == 0:
            inner_mask = base_y >= center - 2 and base_y <= center
        elif outer_index == 1:
            inner_mask = base_y >= 1 and base_y <= center
        else:
            inner_mask = base_y >= center and base_y <= 5
    else:
        if outer_index == 0:
            inner_mask = base_x >= center and base_x <= center + 2
        elif outer_index == 1:
            inner_mask = base_x >= 1 and base_x <= center
        else:
            inner_mask = base_x >= center and base_x <= 5

    if inner_mask:
        return int(BONUS_SYMBOL_INNER_LUMA_BY_SHAPE[outer_index])
    return int(BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE[outer_index])


def _cpu_previous_owner_trail_slots(
    *,
    np: Any,
    state: dict[str, Any],
) -> tuple[Any, Any, Any]:
    trail_x = np.asarray(state["trail_x"])
    trail_y = np.asarray(state["trail_y"])
    trail_radius = np.asarray(state["trail_radius"])
    trail_owner = np.asarray(state["trail_owner"])
    trail_active = np.asarray(state["trail_active"]).astype(bool, copy=False)
    batch_size, trail_slots = trail_x.shape
    player_count = int(np.asarray(state["head_x"]).shape[1])
    prev_x = np.zeros_like(trail_x)
    prev_y = np.zeros_like(trail_y)
    prev_active = np.zeros(trail_active.shape, dtype=bool)
    last_x = np.zeros((batch_size, player_count), dtype=trail_x.dtype)
    last_y = np.zeros((batch_size, player_count), dtype=trail_y.dtype)
    last_radius = np.zeros((batch_size, player_count), dtype=trail_radius.dtype)
    last_active = np.zeros((batch_size, player_count), dtype=bool)

    for row in range(batch_size):
        for slot in range(trail_slots):
            owner = int(trail_owner[row, slot])
            if owner < 0 or owner >= player_count:
                continue
            if last_active[row, owner] and np.isclose(
                last_radius[row, owner],
                trail_radius[row, slot],
                rtol=0.0,
                atol=1.0e-6,
            ):
                prev_x[row, slot] = last_x[row, owner]
                prev_y[row, slot] = last_y[row, owner]
                prev_active[row, slot] = True
            if bool(trail_active[row, slot]):
                last_x[row, owner] = trail_x[row, slot]
                last_y[row, owner] = trail_y[row, slot]
                last_radius[row, owner] = trail_radius[row, slot]
                last_active[row, owner] = True
    return prev_x, prev_y, prev_active


def _owner_luma(owner: int, player_luma_by_index: tuple[int, ...]) -> int:
    if owner < 0:
        return SYNTHETIC_INVALID_OWNER_LUMA
    return player_luma_by_index[owner % len(player_luma_by_index)]


def _cpu_render_production_canvas_gray64(
    *,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    production_state = _synthetic_to_production_source_state(np=np, state=state, config=config)
    return _cpu_render_original_production_canvas_gray64(
        np=np,
        production_state=production_state,
        config=config,
    )


def _cpu_render_original_production_canvas_gray64(
    *,
    np: Any,
    production_state: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    from curvyzero.env.vector_visual_observation import (
        BONUS_RENDER_MODE_BROWSER_SPRITES,
        BONUS_RENDER_MODE_CIRCLES_FAST,
        BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
        TRAIL_RENDER_MODE_BROWSER_LINES,
        render_source_state_canvas_gray64,
    )

    bonus_render_mode = str(config.get("bonus_render_mode", BONUS_RENDER_MODE_SIMPLE_SYMBOLS))
    if bonus_render_mode not in {
        BONUS_RENDER_MODE_BROWSER_SPRITES,
        BONUS_RENDER_MODE_CIRCLES_FAST,
        BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    }:
        raise ValueError(f"unsupported production bonus_render_mode {bonus_render_mode!r}")
    batch_size = int(config["batch_size"])
    out = np.empty((batch_size, 1, 64, 64), dtype=np.uint8)
    for row in range(batch_size):
        out[row] = render_source_state_canvas_gray64(
            production_state,
            row=row,
            player_rgb=_benchmark_player_rgb_palette_for_state(
                np=np,
                state=production_state,
                row=row,
                config=config,
            ),
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            bonus_render_mode=bonus_render_mode,
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
    trail_write_cursor = np.asarray(
        state.get("trail_write_cursor", np.sum(trail_active, axis=1)),
        dtype=np.int32,
    )
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
        "body_write_cursor": trail_write_cursor.copy(),
        "body_pos": trail_pos,
        "body_radius": state["trail_radius"].astype(np.float64),
        "body_owner": state["trail_owner"].astype(np.int16),
        "body_break_before": state["trail_break_before"].astype(bool, copy=False),
        "done": np.zeros((batch_size,), dtype=bool),
        "terminated": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "terminal_reason": np.zeros((batch_size,), dtype=np.int16),
        "avatar_color": np.asarray(
            state.get(
                "avatar_color",
                np.arange(player_count, dtype=np.int16)[None, :].repeat(batch_size, axis=0),
            ),
            dtype=np.int16,
        )[:batch_size, :player_count].copy(),
        "visual_trail_active": trail_active,
        "visual_trail_write_cursor": trail_write_cursor.copy(),
        "visual_trail_pos": trail_pos,
        "visual_trail_radius": state["trail_radius"].astype(np.float64),
        "visual_trail_owner": state["trail_owner"].astype(np.int16),
        "visual_trail_break_before": state["trail_break_before"].astype(bool, copy=False),
        "bonus_active": state["bonus_active"].astype(bool, copy=False),
        "bonus_pos": bonus_pos.reshape(batch_size, bonus_count, 2),
        "bonus_radius": state["bonus_radius"].astype(np.float64),
        "bonus_type": state["bonus_type"].astype(np.int16),
    }


def _source_state_for_benchmark(*, np: Any, config: dict[str, Any]) -> dict[str, Any]:
    state, _ = _source_state_and_reference_for_benchmark(np=np, config=config)
    return state


def _source_state_and_reference_for_benchmark(
    *,
    np: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    state, production_reference_state, _ = _source_state_reference_and_setup_timings_for_benchmark(
        np=np,
        config=config,
    )
    return state, production_reference_state


def _source_state_reference_and_setup_timings_for_benchmark(
    *,
    np: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, float | bool]]:
    total_started = time.perf_counter()
    production_reference_state = None
    production_source_sec = 0.0
    production_to_compact_sec = 0.0
    synthetic_source_sec = 0.0
    owner_ordered_pack_sec = 0.0

    if config["state_source"] == STATE_SOURCE_REAL_ENV_ROLLOUT:
        started = time.perf_counter()
        production_state = _real_env_rollout_production_state(np=np, config=config)
        production_source_sec = time.perf_counter() - started
        production_reference_state = production_state
        started = time.perf_counter()
        state = _production_to_benchmark_source_state(
            np=np,
            production_state=production_state,
            config=config,
        )
        production_to_compact_sec = time.perf_counter() - started
    elif config["state_source"] == STATE_SOURCE_ADVERSARIAL_FIXTURE:
        started = time.perf_counter()
        production_state = _adversarial_fixture_production_state(np=np, config=config)
        production_source_sec = time.perf_counter() - started
        production_reference_state = production_state
        started = time.perf_counter()
        state = _production_to_benchmark_source_state(
            np=np,
            production_state=production_state,
            config=config,
        )
        production_to_compact_sec = time.perf_counter() - started
    else:
        started = time.perf_counter()
        state = _synthetic_source_state(np=np, config=config)
        synthetic_source_sec = time.perf_counter() - started

    if config.get("trail_composition") == TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT:
        started = time.perf_counter()
        state = _pack_compact_trails_in_owner_draw_order(
            np=np,
            state=state,
            config=config,
        )
        owner_ordered_pack_sec = time.perf_counter() - started

    setup_timings = {
        "production_source_sec": production_source_sec,
        "production_to_compact_sec": production_to_compact_sec,
        "synthetic_source_sec": synthetic_source_sec,
        "owner_ordered_pack_sec": owner_ordered_pack_sec,
        "total_setup_sec": time.perf_counter() - total_started,
        "included_in_render_timing": False,
    }
    return state, production_reference_state, setup_timings


def _prepare_compact_state_for_render(
    *,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if config.get("trail_composition") != TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT:
        return state
    return _pack_compact_trails_in_owner_draw_order(
        np=np,
        state=state,
        config=config,
    )


def _owner_ordered_active_trail_slots(
    *,
    np: Any,
    owners: Any,
    active: Any,
) -> Any:
    owner_array = np.asarray(owners)
    active_array = np.asarray(active).astype(bool, copy=False)
    slots = np.flatnonzero(active_array).astype(np.int32, copy=False)
    if slots.size == 0:
        return slots
    active_owners = owner_array[slots].astype(np.int64, copy=False)
    unique_owners = tuple(int(owner) for owner in np.unique(active_owners))
    invalid_owners = tuple(sorted(owner for owner in unique_owners if owner < 0))
    valid_owners = tuple(sorted((owner for owner in unique_owners if owner >= 0), reverse=True))
    ordered = [
        slots[active_owners == owner]
        for owner in (*invalid_owners, *valid_owners)
    ]
    if not ordered:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(ordered).astype(np.int32, copy=False)


def _owner_ordered_compact_trail_order(
    *,
    np: Any,
    owners: Any,
    active: Any,
) -> Any:
    active_slots = _owner_ordered_active_trail_slots(
        np=np,
        owners=owners,
        active=active,
    )
    active_mask = np.asarray(active).astype(bool, copy=False)
    inactive_slots = np.flatnonzero(~active_mask).astype(np.int32, copy=False)
    if active_slots.size == 0:
        return inactive_slots
    if inactive_slots.size == 0:
        return active_slots
    return np.concatenate([active_slots, inactive_slots]).astype(np.int32, copy=False)


def _pack_compact_trails_in_owner_draw_order(
    *,
    np: Any,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    batch_size = int(config["batch_size"])
    trail_keys = (
        "trail_x",
        "trail_y",
        "trail_radius",
        "trail_owner",
        "trail_active",
        "trail_break_before",
    )
    packed = dict(state)
    packed_trails = {
        key: np.empty_like(np.asarray(state[key]))
        for key in trail_keys
    }
    active_counts = np.zeros((batch_size,), dtype=np.int32)
    for row in range(batch_size):
        order = _owner_ordered_compact_trail_order(
            np=np,
            owners=state["trail_owner"][row],
            active=state["trail_active"][row],
        )
        active_count = int(np.count_nonzero(state["trail_active"][row]))
        active_counts[row] = active_count
        for key in trail_keys:
            packed_trails[key][row] = state[key][row, order]
    packed.update(packed_trails)
    packed["trail_write_cursor"] = active_counts
    return packed


def _slice_batch_state(*, np: Any, state: dict[str, Any], rows: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] >= rows:
            sliced[key] = value[:rows].copy()
        else:
            sliced[key] = value.copy() if hasattr(value, "copy") else value
    return sliced


def _player_luma_by_index(config: dict[str, Any]) -> tuple[int, ...]:
    controlled = config.get("controlled_player")
    player_count = int(config["player_count"])
    if controlled is None:
        repeats = (player_count + len(SYNTHETIC_PLAYER_LUMA_BY_INDEX) - 1) // len(
            SYNTHETIC_PLAYER_LUMA_BY_INDEX
        )
        return (SYNTHETIC_PLAYER_LUMA_BY_INDEX * repeats)[:player_count]
    player = int(controlled)
    return tuple(
        PERSPECTIVE_SELF_LUMA if index == player else PERSPECTIVE_OTHER_LUMA
        for index in range(player_count)
    )


def _player_luma_float_by_index(config: dict[str, Any]) -> tuple[float, ...]:
    controlled = config.get("controlled_player")
    player_count = int(config["player_count"])
    if controlled is None:
        repeats = (player_count + len(SYNTHETIC_PLAYER_LUMA_FLOAT_BY_INDEX) - 1) // len(
            SYNTHETIC_PLAYER_LUMA_FLOAT_BY_INDEX
        )
        return (SYNTHETIC_PLAYER_LUMA_FLOAT_BY_INDEX * repeats)[:player_count]
    player = int(controlled)
    return tuple(
        PERSPECTIVE_SELF_LUMA_FLOAT if index == player else PERSPECTIVE_OTHER_LUMA_FLOAT
        for index in range(player_count)
    )


def _benchmark_player_rgb_palette(
    config: dict[str, Any],
) -> tuple[tuple[int, int, int], ...] | None:
    controlled = config.get("controlled_player")
    if controlled is None:
        return None
    player = int(controlled)
    return tuple(
        (
            (PERSPECTIVE_SELF_LUMA, PERSPECTIVE_SELF_LUMA, PERSPECTIVE_SELF_LUMA)
            if index == player
            else (PERSPECTIVE_OTHER_LUMA, PERSPECTIVE_OTHER_LUMA, PERSPECTIVE_OTHER_LUMA)
        )
        for index in range(int(config["player_count"]))
    )


def _benchmark_player_rgb_palette_for_state(
    *,
    np: Any,
    state: dict[str, Any],
    row: int,
    config: dict[str, Any],
) -> tuple[tuple[int, int, int], ...] | None:
    controlled = config.get("controlled_player")
    if controlled is None:
        return None
    player = int(controlled)
    player_count = int(config["player_count"])
    color_indices = np.arange(player_count, dtype=np.int64)
    if "avatar_color" in state:
        avatar_color = np.asarray(state["avatar_color"])
        if avatar_color.ndim >= 2:
            color_indices = np.asarray(avatar_color[int(row), :player_count], dtype=np.int64)
    if bool((color_indices < 0).any()):
        raise ValueError("avatar_color indices must be non-negative")
    max_color_index = int(color_indices.max()) if color_indices.size else player_count - 1
    self_rgb = (PERSPECTIVE_SELF_LUMA, PERSPECTIVE_SELF_LUMA, PERSPECTIVE_SELF_LUMA)
    other_rgb = (PERSPECTIVE_OTHER_LUMA, PERSPECTIVE_OTHER_LUMA, PERSPECTIVE_OTHER_LUMA)
    palette = [other_rgb for _ in range(max(player_count, max_color_index + 1))]
    palette[int(color_indices[player])] = self_rgb
    return tuple(palette)


def _real_env_rollout_source_state(*, np: Any, config: dict[str, Any]) -> dict[str, Any]:
    production_state = _real_env_rollout_production_state(np=np, config=config)
    return _prepare_compact_state_for_render(
        np=np,
        state=_production_to_benchmark_source_state(
            np=np,
            production_state=production_state,
            config=config,
        ),
        config=config,
    )


def _real_env_rollout_production_state(*, np: Any, config: dict[str, Any]) -> dict[str, Any]:
    from curvyzero.env import vector_runtime
    from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS
    from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv

    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    trail_slots = int(config["trail_slots"])
    bonus_count = int(config["bonus_count"])
    seed = int(config["seed"])
    env = VectorMultiplayerEnv(
        batch_size=batch_size,
        player_count=player_count,
        seed=seed,
        decision_source_frames=1,
        body_capacity=trail_slots,
        map_size=float(config["map_size"]),
        death_mode=vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
        natural_bonus_spawn=False,
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 17)
    if bonus_count > 0:
        for row in range(batch_size):
            env.seed_active_bonus(
                row=row,
                bonus_type=int((row % 12) + 1),
                x=float(rng.uniform(0.2 * env.map_size, 0.8 * env.map_size)),
                y=float(rng.uniform(0.2 * env.map_size, 0.8 * env.map_size)),
                radius=3.0,
                bonus_id=row + 1,
                slot=0,
                bonus_capacity=bonus_count,
            )
    for _ in range(int(config["real_env_steps"])):
        actions = rng.integers(
            0,
            3,
            size=(batch_size, player_count),
            dtype=np.int16,
        )
        env.step(actions, timer_advance_ms=SOURCE_PHYSICS_STEP_MS)
    return env.state


def _production_to_benchmark_source_state(
    *,
    np: Any,
    production_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    trail_slots = int(config["trail_slots"])
    bonus_count = int(config["bonus_count"])
    geometry_dtype = (
        np.float64
        if str(config.get("geometry_dtype", GEOMETRY_DTYPE_FLOAT32)) == GEOMETRY_DTYPE_FLOAT64
        else np.float32
    )

    if "visual_trail_pos" in production_state and "visual_trail_active" in production_state:
        trail_pos = np.asarray(production_state["visual_trail_pos"], dtype=geometry_dtype)
        trail_radius = np.asarray(production_state["visual_trail_radius"], dtype=geometry_dtype)
        trail_owner = np.asarray(production_state["visual_trail_owner"], dtype=np.int32)
        trail_active = np.asarray(production_state["visual_trail_active"], dtype=np.uint8)
        trail_write_cursor = np.asarray(
            production_state.get(
                "visual_trail_write_cursor",
                np.full((batch_size,), trail_active.shape[1], dtype=np.int32),
            ),
            dtype=np.int32,
        )
        trail_break = np.asarray(
            production_state.get(
                "visual_trail_break_before",
                np.zeros(trail_active.shape, dtype=bool),
            ),
            dtype=np.uint8,
        )
    else:
        trail_pos = np.asarray(production_state["body_pos"], dtype=geometry_dtype)
        trail_radius = np.asarray(production_state["body_radius"], dtype=geometry_dtype)
        trail_owner = np.asarray(production_state["body_owner"], dtype=np.int32)
        trail_active = np.asarray(production_state["body_active"], dtype=np.uint8)
        trail_write_cursor = np.asarray(
            production_state.get(
                "body_write_cursor",
                np.full((batch_size,), trail_active.shape[1], dtype=np.int32),
            ),
            dtype=np.int32,
        )
        trail_break = np.asarray(
            production_state.get(
                "body_break_before",
                np.zeros(trail_active.shape, dtype=bool),
            ),
            dtype=np.uint8,
        )

    pos = np.asarray(production_state["pos"], dtype=geometry_dtype)
    radius = np.asarray(production_state["radius"], dtype=geometry_dtype)
    alive = np.asarray(production_state["alive"], dtype=np.uint8)
    present = np.asarray(production_state.get("present", alive), dtype=np.uint8)
    head_alive = (alive[:, :player_count] & present[:, :player_count]).astype(np.uint8)
    avatar_color = np.asarray(
        production_state.get(
            "avatar_color",
            np.arange(player_count, dtype=np.int16)[None, :].repeat(batch_size, axis=0),
        ),
        dtype=np.int32,
    )

    state = {
        "trail_x": np.zeros((batch_size, trail_slots), dtype=geometry_dtype),
        "trail_y": np.zeros((batch_size, trail_slots), dtype=geometry_dtype),
        "trail_radius": np.zeros((batch_size, trail_slots), dtype=geometry_dtype),
        "trail_owner": np.full((batch_size, trail_slots), -1, dtype=np.int32),
        "trail_active": np.zeros((batch_size, trail_slots), dtype=np.uint8),
        "trail_break_before": np.zeros((batch_size, trail_slots), dtype=np.uint8),
        "head_x": pos[:batch_size, :player_count, 0].astype(geometry_dtype, copy=True),
        "head_y": pos[:batch_size, :player_count, 1].astype(geometry_dtype, copy=True),
        "head_radius": radius[:batch_size, :player_count].astype(geometry_dtype, copy=True),
        "head_alive": head_alive[:batch_size, :player_count].astype(np.uint8, copy=True),
        "avatar_color": avatar_color[:batch_size, :player_count].astype(np.int32, copy=True),
        "trail_write_cursor": np.clip(
            trail_write_cursor[:batch_size],
            0,
            trail_slots,
        ).astype(np.int32, copy=True),
        "bonus_x": np.zeros((batch_size, bonus_count), dtype=geometry_dtype),
        "bonus_y": np.zeros((batch_size, bonus_count), dtype=geometry_dtype),
        "bonus_radius": np.zeros((batch_size, bonus_count), dtype=geometry_dtype),
        "bonus_active": np.zeros((batch_size, bonus_count), dtype=np.uint8),
        "bonus_type": np.ones((batch_size, bonus_count), dtype=np.int32),
    }
    copied_trails = min(trail_slots, int(trail_active.shape[1]))
    if copied_trails:
        active_copy = trail_active[:batch_size, :copied_trails].copy()
        for row in range(batch_size):
            cursor = int(state["trail_write_cursor"][row])
            if cursor < copied_trails:
                active_copy[row, max(0, cursor) :] = 0
        state["trail_x"][:, :copied_trails] = trail_pos[:batch_size, :copied_trails, 0]
        state["trail_y"][:, :copied_trails] = trail_pos[:batch_size, :copied_trails, 1]
        state["trail_radius"][:, :copied_trails] = trail_radius[:batch_size, :copied_trails]
        state["trail_owner"][:, :copied_trails] = trail_owner[:batch_size, :copied_trails]
        state["trail_active"][:, :copied_trails] = active_copy
        state["trail_break_before"][:, :copied_trails] = trail_break[:batch_size, :copied_trails]

    if bonus_count and "bonus_active" in production_state:
        active = np.asarray(production_state["bonus_active"], dtype=np.uint8)
        copied_bonuses = min(bonus_count, int(active.shape[1]))
        if copied_bonuses:
            bonus_pos = np.asarray(production_state["bonus_pos"], dtype=geometry_dtype)
            state["bonus_x"][:, :copied_bonuses] = bonus_pos[:batch_size, :copied_bonuses, 0]
            state["bonus_y"][:, :copied_bonuses] = bonus_pos[:batch_size, :copied_bonuses, 1]
            state["bonus_radius"][:, :copied_bonuses] = np.asarray(
                production_state["bonus_radius"],
                dtype=geometry_dtype,
            )[:batch_size, :copied_bonuses]
            state["bonus_active"][:, :copied_bonuses] = active[:batch_size, :copied_bonuses]
            state["bonus_type"][:, :copied_bonuses] = np.asarray(
                production_state["bonus_type"],
                dtype=np.int32,
            )[:batch_size, :copied_bonuses]
    return state


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
    bonus_type = rng.integers(1, 13, size=(batch_size, bonus_count), dtype=np.int32)

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


def _adversarial_fixture_production_state(
    *,
    np: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    batch_size = int(config["batch_size"])
    player_count = int(config["player_count"])
    trail_slots = int(config["trail_slots"])
    bonus_count = int(config["bonus_count"])
    map_size = float(config["map_size"])
    if player_count < 2:
        raise ValueError("adversarial_fixture requires player_count >= 2")
    if trail_slots < 10:
        raise ValueError("adversarial_fixture requires trail_slots >= 10")

    pos = np.zeros((batch_size, player_count, 2), dtype=np.float64)
    radius = np.full((batch_size, player_count), 8.0, dtype=np.float64)
    alive = np.ones((batch_size, player_count), dtype=bool)
    present = np.ones((batch_size, player_count), dtype=bool)

    trail_pos = np.zeros((batch_size, trail_slots, 2), dtype=np.float64)
    trail_radius = np.zeros((batch_size, trail_slots), dtype=np.float64)
    trail_owner = np.full((batch_size, trail_slots), -1, dtype=np.int16)
    trail_active = np.zeros((batch_size, trail_slots), dtype=bool)
    trail_break = np.zeros((batch_size, trail_slots), dtype=bool)
    trail_cursor = np.full((batch_size,), trail_slots, dtype=np.int32)

    bonus_pos = np.zeros((batch_size, bonus_count, 2), dtype=np.float64)
    bonus_radius = np.zeros((batch_size, bonus_count), dtype=np.float64)
    bonus_active = np.zeros((batch_size, bonus_count), dtype=bool)
    bonus_type = np.ones((batch_size, bonus_count), dtype=np.int16)
    avatar_color = np.arange(player_count, dtype=np.int16)[None, :].repeat(batch_size, axis=0)

    def xy(x_fraction: float, y_fraction: float) -> tuple[float, float]:
        return map_size * x_fraction, map_size * y_fraction

    def set_head(
        row: int,
        player: int,
        x_fraction: float,
        y_fraction: float,
        *,
        head_radius: float,
        is_alive: bool = True,
        is_present: bool = True,
    ) -> None:
        if player >= player_count:
            return
        pos[row, player] = xy(x_fraction, y_fraction)
        radius[row, player] = float(head_radius)
        alive[row, player] = bool(is_alive)
        present[row, player] = bool(is_present)

    def set_trail(
        row: int,
        slot: int,
        owner: int,
        x_fraction: float,
        y_fraction: float,
        *,
        point_radius: float,
        active: bool = True,
        break_before: bool = False,
    ) -> None:
        if slot >= trail_slots:
            return
        trail_pos[row, slot] = xy(x_fraction, y_fraction)
        trail_radius[row, slot] = float(point_radius)
        trail_owner[row, slot] = int(owner)
        trail_active[row, slot] = bool(active)
        trail_break[row, slot] = bool(break_before)

    def set_bonus(
        row: int,
        slot: int,
        x_fraction: float,
        y_fraction: float,
        *,
        item_radius: float,
        item_type: int,
        active: bool = True,
    ) -> None:
        if slot >= bonus_count:
            return
        bonus_pos[row, slot] = xy(x_fraction, y_fraction)
        bonus_radius[row, slot] = float(item_radius)
        bonus_type[row, slot] = np.int16(item_type)
        bonus_active[row, slot] = bool(active)

    owner2 = 2 if player_count > 2 else 1
    for row in range(batch_size):
        pattern = row % 4
        if player_count >= 3:
            if pattern == 0:
                avatar_color[row, :3] = np.asarray([2, 0, 1], dtype=np.int16)
            elif pattern == 1:
                avatar_color[row, :3] = np.asarray([0, 0, 2], dtype=np.int16)
            elif pattern == 2:
                avatar_color[row, :3] = np.asarray([7, 3, 5], dtype=np.int16)
            else:
                avatar_color[row, :3] = np.asarray([1, 2, 1], dtype=np.int16)
        elif player_count == 2:
            avatar_color[row, :2] = (
                np.asarray([1, 0], dtype=np.int16)
                if pattern % 2 == 0
                else np.asarray([0, 0], dtype=np.int16)
            )
        if pattern == 0:
            trail_cursor[row] = min(trail_slots, 8)
            set_trail(row, 0, 0, 0.25, 0.25, point_radius=5.0, break_before=True)
            set_trail(row, 1, 1, 0.75, 0.25, point_radius=7.0, break_before=True)
            set_trail(row, 2, 0, 0.75, 0.75, point_radius=5.0)
            set_trail(row, 3, 1, 0.25, 0.75, point_radius=7.0)
            set_trail(row, 4, 0, 0.32, 0.75, point_radius=9.0, break_before=True)
            set_trail(row, 5, 0, 0.43, 0.75, point_radius=9.0)
            set_trail(row, 6, 1, 0.50, 0.50, point_radius=4.0, break_before=True)
            set_trail(row, 7, owner2, 0.50, 0.50, point_radius=10.0, break_before=True)
            set_trail(row, 8, 1, 0.50, 0.15, point_radius=18.0, break_before=True)
            set_trail(row, 9, 0, 0.50, 0.85, point_radius=18.0)
            set_head(row, 0, 0.50, 0.50, head_radius=12.0)
            set_head(row, 1, 0.50, 0.50, head_radius=6.0)
            set_head(row, owner2, 0.58, 0.50, head_radius=8.0)
            set_bonus(row, 0, 0.50, 0.50, item_radius=14.0, item_type=1)
            set_bonus(row, 1, 0.50, 0.35, item_radius=12.0, item_type=5)
            set_bonus(row, 2, 0.02, 0.02, item_radius=10.0, item_type=12)
        elif pattern == 1:
            set_trail(row, 0, 0, 0.14, 0.16, point_radius=4.0, break_before=True)
            set_trail(row, 1, 0, 0.25, 0.16, point_radius=4.0)
            set_trail(row, 2, 0, 0.36, 0.16, point_radius=4.0, break_before=True)
            set_trail(row, 3, 1, 0.47, 0.16, point_radius=4.0, active=False)
            set_trail(row, 4, 0, 0.58, 0.16, point_radius=4.0)
            set_trail(row, 5, 1, 0.58, 0.30, point_radius=8.0, break_before=True)
            set_trail(row, 6, 1, 0.36, 0.30, point_radius=8.0)
            set_trail(row, 7, 1, 0.25, 0.30, point_radius=3.0)
            set_trail(row, 8, 1, 0.14, 0.30, point_radius=3.0)
            set_trail(row, 9, -1, 0.50, 0.45, point_radius=12.0, break_before=True)
            set_head(row, 0, 0.58, 0.16, head_radius=8.0)
            set_head(row, 1, 0.42, 0.42, head_radius=9.0, is_alive=False)
            set_head(row, owner2, 0.70, 0.42, head_radius=7.0, is_present=False)
            set_bonus(row, 0, 0.42, 0.42, item_radius=13.0, item_type=2)
            set_bonus(row, 1, 0.14, 0.30, item_radius=10.0, item_type=6)
            set_bonus(row, 2, 0.97, 0.50, item_radius=11.0, item_type=10)
        elif pattern == 2:
            trail_cursor[row] = 0
            for slot in range(min(trail_slots, 10)):
                owner = slot % min(player_count, 3)
                set_trail(
                    row,
                    slot,
                    owner,
                    0.12 + 0.08 * slot,
                    0.82 - 0.05 * (slot % 4),
                    point_radius=14.0 if slot % 2 else 5.0,
                    break_before=(slot % 3 == 0),
                )
            set_head(row, 0, 0.25, 0.62, head_radius=10.0)
            set_head(row, 1, 0.35, 0.62, head_radius=10.0)
            set_head(row, owner2, 0.45, 0.62, head_radius=10.0)
            set_bonus(row, 0, 0.25, 0.62, item_radius=12.0, item_type=3)
            set_bonus(row, 1, 0.02, 0.98, item_radius=11.0, item_type=7)
            set_bonus(row, 2, 0.50, 0.82, item_radius=12.0, item_type=11)
        else:
            set_trail(row, 0, 0, 0.20, 0.70, point_radius=6.0, break_before=True)
            set_trail(row, 1, 1, 0.30, 0.62, point_radius=6.0, break_before=True)
            set_trail(row, 2, 1, 0.45, 0.62, point_radius=6.0)
            set_trail(row, 3, 0, 0.60, 0.70, point_radius=6.0)
            set_trail(row, 4, owner2, 0.60, 0.30, point_radius=5.0, break_before=True)
            set_trail(row, 5, 0, 0.50, 0.30, point_radius=9.0, break_before=True)
            set_trail(row, 6, owner2, 0.40, 0.30, point_radius=5.0)
            set_trail(row, 7, 0, 0.30, 0.30, point_radius=9.0)
            set_trail(row, 8, -1, 0.70, 0.70, point_radius=7.0, break_before=True)
            set_trail(row, 9, owner2, 0.76, 0.70, point_radius=7.0, active=False)
            set_head(row, 0, 0.30, 0.30, head_radius=9.0)
            set_head(row, 1, 0.45, 0.62, head_radius=7.0)
            set_head(row, owner2, 0.40, 0.30, head_radius=8.0, is_present=False)
            set_bonus(row, 0, 0.30, 0.30, item_radius=13.0, item_type=4)
            set_bonus(row, 1, 0.45, 0.62, item_radius=9.0, item_type=8)
            set_bonus(row, 2, 0.70, 0.70, item_radius=10.0, item_type=9)

        for slot in range(3, bonus_count):
            item_type = ((row * max(1, bonus_count) + slot) % 12) + 1
            x_fraction = 0.08 + 0.12 * (slot % 7)
            y_fraction = 0.08 + 0.14 * ((slot // 7) % 6)
            set_bonus(
                row,
                slot,
                x_fraction,
                y_fraction,
                item_radius=8.0 + float(slot % 5),
                item_type=item_type,
            )

    return {
        "tick": np.arange(batch_size, dtype=np.int64),
        "elapsed_ms": np.zeros((batch_size,), dtype=np.float64),
        "map_size": np.full((batch_size,), map_size, dtype=np.float64),
        "present": present,
        "alive": alive,
        "pos": pos,
        "radius": radius,
        "body_active": trail_active.copy(),
        "body_write_cursor": trail_cursor.copy(),
        "body_pos": trail_pos.copy(),
        "body_radius": trail_radius.copy(),
        "body_owner": trail_owner.copy(),
        "body_break_before": trail_break.copy(),
        "done": np.zeros((batch_size,), dtype=bool),
        "terminated": np.zeros((batch_size,), dtype=bool),
        "truncated": np.zeros((batch_size,), dtype=bool),
        "terminal_reason": np.zeros((batch_size,), dtype=np.int16),
        "avatar_color": avatar_color,
        "visual_trail_active": trail_active,
        "visual_trail_write_cursor": trail_cursor,
        "visual_trail_pos": trail_pos,
        "visual_trail_radius": trail_radius,
        "visual_trail_owner": trail_owner,
        "visual_trail_break_before": trail_break,
        "bonus_active": bonus_active,
        "bonus_pos": bonus_pos,
        "bonus_radius": bonus_radius,
        "bonus_type": bonus_type,
    }


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    state_source = str(config.get("state_source", STATE_SOURCE_SYNTHETIC))
    if state_source not in STATE_SOURCES:
        allowed = ", ".join(sorted(STATE_SOURCES))
        raise ValueError(f"state_source must be one of {allowed}, got {state_source!r}")
    render_mode = str(config.get("render_mode", RENDER_MODE_BROWSER_LINES))
    if render_mode not in RENDER_MODE_IDS:
        allowed = ", ".join(sorted(RENDER_MODE_IDS))
        raise ValueError(f"render_mode must be one of {allowed}, got {render_mode!r}")
    bonus_render_mode = str(config.get("bonus_render_mode", BONUS_RENDER_MODE_SIMPLE_SYMBOLS))
    if bonus_render_mode not in BONUS_RENDER_MODE_IDS:
        allowed = ", ".join(sorted(BONUS_RENDER_MODE_IDS))
        raise ValueError(f"bonus_render_mode must be one of {allowed}, got {bonus_render_mode!r}")
    render_surface = str(config.get("render_surface", RENDER_SURFACE_DIRECT_GRAY64))
    if render_surface not in RENDER_SURFACES:
        allowed = ", ".join(sorted(RENDER_SURFACES))
        raise ValueError(f"render_surface must be one of {allowed}, got {render_surface!r}")
    trail_composition = str(config.get("trail_composition", TRAIL_COMPOSITION_PRIORITY_BUFFER))
    if trail_composition not in TRAIL_COMPOSITIONS:
        allowed = ", ".join(sorted(TRAIL_COMPOSITIONS))
        raise ValueError(
            f"trail_composition must be one of {allowed}, got {trail_composition!r}"
        )
    geometry_dtype = str(config.get("geometry_dtype", GEOMETRY_DTYPE_FLOAT32))
    if geometry_dtype not in GEOMETRY_DTYPES:
        allowed = ", ".join(sorted(GEOMETRY_DTYPES))
        raise ValueError(f"geometry_dtype must be one of {allowed}, got {geometry_dtype!r}")
    render_views = str(config.get("render_views", RENDER_VIEWS_SINGLE))
    if render_views not in RENDER_VIEWS_CHOICES:
        allowed = ", ".join(sorted(RENDER_VIEWS_CHOICES))
        raise ValueError(f"render_views must be one of {allowed}, got {render_views!r}")
    if (
        trail_composition == TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT
        and render_surface != RENDER_SURFACE_BLOCK_704_GRAY64
    ):
        raise ValueError("owner_ordered_compact trail_composition requires block_704_gray64")

    player_count = _positive_int(config.get("player_count", 2), "player_count")
    controlled_player = _optional_player_index(
        config.get("controlled_player", 0),
        player_count=player_count,
    )
    if "map_size" in config:
        map_size = float(config["map_size"])
    elif state_source == STATE_SOURCE_REAL_ENV_ROLLOUT:
        from curvyzero.env.config import CurvyTronReferenceDefaults

        map_size = float(CurvyTronReferenceDefaults().arena_size_for_players(player_count))
    elif state_source == STATE_SOURCE_ADVERSARIAL_FIXTURE:
        map_size = 704.0
    else:
        map_size = 1000.0
    checked = {
        "state_source": state_source,
        "batch_size": _positive_int(config.get("batch_size", 64), "batch_size"),
        "player_count": player_count,
        "controlled_player": controlled_player,
        "trail_slots": _positive_int(config.get("trail_slots", 256), "trail_slots"),
        "render_mode": render_mode,
        "bonus_render_mode": bonus_render_mode,
        "render_surface": render_surface,
        "trail_composition": trail_composition,
        "render_views": render_views,
        "geometry_dtype": geometry_dtype,
        "bonus_count": _nonnegative_int(config.get("bonus_count", 8), "bonus_count"),
        "frame_size": _positive_int(config.get("frame_size", 704), "frame_size"),
        "target_size": _positive_int(config.get("target_size", 64), "target_size"),
        "seed": int(config.get("seed", 20260513)),
        "warmup_runs": _nonnegative_int(config.get("warmup_runs", 3), "warmup_runs"),
        "steady_runs": _positive_int(config.get("steady_runs", 10), "steady_runs"),
        "real_env_steps": _nonnegative_int(config.get("real_env_steps", 128), "real_env_steps"),
        "verify_rows": _nonnegative_int(config.get("verify_rows", 2), "verify_rows"),
        "transfer_output": bool(config.get("transfer_output", False)),
        "map_size": map_size,
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


def _optional_player_index(value: Any, *, player_count: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "world", "source"}:
        return None
    player = int(value)
    if player < 0 or player >= player_count:
        raise ValueError(f"controlled_player must be in [0, {player_count}) or none, got {value!r}")
    return player


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
    state_source: str = STATE_SOURCE_SYNTHETIC,
    batch_size: int = 64,
    player_count: int = 2,
    controlled_player: int | None = 0,
    trail_slots: int = 256,
    compute: str = COMPUTE_L4_T4,
    render_mode: str = RENDER_MODE_BROWSER_LINES,
    bonus_render_mode: str = BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    render_surface: str = RENDER_SURFACE_DIRECT_GRAY64,
    trail_composition: str = TRAIL_COMPOSITION_PRIORITY_BUFFER,
    render_views: str = RENDER_VIEWS_SINGLE,
    geometry_dtype: str = GEOMETRY_DTYPE_FLOAT32,
    bonus_count: int = 8,
    frame_size: int = 704,
    target_size: int = 64,
    seed: int = 20260513,
    warmup_runs: int = 3,
    steady_runs: int = 10,
    real_env_steps: int = 128,
    verify_rows: int = 2,
    transfer_output: bool = False,
    trail_radius: float = 4.0,
    scalar_component_profile: bool = False,
) -> None:
    if compute not in COMPUTE_CHOICES:
        allowed = ", ".join(sorted(COMPUTE_CHOICES))
        raise ValueError(f"compute must be one of {allowed}; got {compute!r}")
    config = {
        "state_source": state_source,
        "batch_size": batch_size,
        "player_count": player_count,
        "controlled_player": controlled_player,
        "trail_slots": trail_slots,
        "compute": compute,
        "render_mode": render_mode,
        "bonus_render_mode": bonus_render_mode,
        "render_surface": render_surface,
        "trail_composition": trail_composition,
        "render_views": render_views,
        "geometry_dtype": geometry_dtype,
        "bonus_count": bonus_count,
        "frame_size": frame_size,
        "target_size": target_size,
        "seed": seed,
        "warmup_runs": warmup_runs,
        "steady_runs": steady_runs,
        "real_env_steps": real_env_steps,
        "verify_rows": verify_rows,
        "transfer_output": transfer_output,
        "trail_radius": trail_radius,
        "scalar_component_profile": scalar_component_profile,
    }
    fn = (
        run_source_state_gpu_render_benchmark_h100
        if compute == COMPUTE_H100
        else run_source_state_gpu_render_benchmark
    )
    result = fn.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
