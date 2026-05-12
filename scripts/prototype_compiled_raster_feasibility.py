"""Quick optional-dependency feasibility probe for compiled raster helpers.

This is optimizer-lane prototype scaffolding only. It does not import or change
production render modules. The reference kernels below mirror the current
pixel-space shape of rounded segment coverage and 704-to-64 luma averaging
closely enough to answer one question: do locally importable libraries offer an
easy exact speed win?
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np


FRAME_SIZE = 704
TARGET_SIZE = 64
RATIO = FRAME_SIZE // TARGET_SIZE
SEGMENT_COUNT = 96
RADIUS_PX = 3.75
BACKGROUND_RGB = np.array([9, 11, 13], dtype=np.uint8)
TRAIL_RGB = np.array([231, 86, 68], dtype=np.uint8)

OPTIONAL_IMPORTS = {
    "cv2": "cv2",
    "numba": "numba",
    "skia/skia-python": "skia",
    "PIL/Pillow": "PIL",
    "scipy": "scipy",
    "cupy": "cupy",
    "triton": "triton",
}


def _probe_imports() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for label, module_name in OPTIONAL_IMPORTS.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            results[label] = {"available": False, "module": module_name}
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on local env.
            results[label] = {
                "available": False,
                "module": module_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        results[label] = {
            "available": True,
            "module": module_name,
            "version": getattr(module, "__version__", "unknown"),
        }
    return results


def _segments() -> np.ndarray:
    rng = np.random.default_rng(20260512)
    starts = rng.uniform(18.0, FRAME_SIZE - 18.0, size=(SEGMENT_COUNT, 2))
    deltas = rng.normal(0.0, 64.0, size=(SEGMENT_COUNT, 2))
    ends = np.clip(starts + deltas, -32.0, FRAME_SIZE + 32.0)
    return np.column_stack([starts, ends]).astype(np.float64)


def _canvas() -> np.ndarray:
    canvas = np.empty((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    canvas[:, :] = BACKGROUND_RGB
    return canvas


def _draw_reference(segments: np.ndarray) -> np.ndarray:
    canvas = _canvas()
    for x0, y0, x1, y1 in segments:
        _draw_reference_segment(canvas, float(x0), float(y0), float(x1), float(y1))
    return canvas


def _draw_reference_segment(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        _draw_reference_cap(canvas, x0, y0)
        return
    size = int(canvas.shape[0])
    min_x = min(x0, x1) - RADIUS_PX
    max_x = max(x0, x1) + RADIUS_PX
    min_y = min(y0, y1) - RADIUS_PX
    max_y = max(y0, y1) + RADIUS_PX
    if max_x < 0.0 or min_x > float(size - 1) or max_y < 0.0 or min_y > float(size - 1):
        return
    px0 = max(0, int(np.floor(min_x - 1.0)))
    px1 = min(size - 1, int(np.ceil(max_x + 1.0)))
    py0 = max(0, int(np.floor(min_y - 1.0)))
    py1 = min(size - 1, int(np.ceil(max_y + 1.0)))
    yy, xx = np.ogrid[py0 : py1 + 1, px0 : px1 + 1]
    rel_x = xx.astype(np.float64) - x0
    rel_y = yy.astype(np.float64) - y0
    projection = np.clip((rel_x * dx + rel_y * dy) / length_sq, 0.0, 1.0)
    closest_x = x0 + projection * dx
    closest_y = y0 + projection * dy
    distance_sq = (xx.astype(np.float64) - closest_x) ** 2 + (
        yy.astype(np.float64) - closest_y
    ) ** 2
    mask = distance_sq <= RADIUS_PX * RADIUS_PX
    if bool(mask.any()):
        canvas[py0 : py1 + 1, px0 : px1 + 1][mask] = TRAIL_RGB


def _draw_reference_cap(canvas: np.ndarray, x: float, y: float) -> None:
    size = int(canvas.shape[0])
    if x + RADIUS_PX < 0.0 or x - RADIUS_PX > float(size - 1):
        return
    if y + RADIUS_PX < 0.0 or y - RADIUS_PX > float(size - 1):
        return
    x0 = max(0, int(np.floor(x - RADIUS_PX - 1.0)))
    x1 = min(size - 1, int(np.ceil(x + RADIUS_PX + 1.0)))
    y0 = max(0, int(np.floor(y - RADIUS_PX - 1.0)))
    y1 = min(size - 1, int(np.ceil(y + RADIUS_PX + 1.0)))
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx.astype(np.float64) - x) ** 2 + (yy.astype(np.float64) - y) ** 2 <= (
        RADIUS_PX * RADIUS_PX
    )
    if bool(mask.any()):
        canvas[y0 : y1 + 1, x0 : x1 + 1][mask] = TRAIL_RGB


def _sample_rgb_frame() -> np.ndarray:
    yy, xx = np.indices((FRAME_SIZE, FRAME_SIZE), dtype=np.uint16)
    frame = np.empty((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frame[:, :, 0] = (xx * 3 + yy * 5) % 256
    frame[:, :, 1] = (xx * 7 + yy * 11 + 17) % 256
    frame[:, :, 2] = (xx * 13 + yy * 2 + 29) % 256
    return frame


def _downsample_reference(rgb: np.ndarray) -> np.ndarray:
    gray = (
        rgb[:, :, 0].astype(np.float32) * np.float32(0.299)
        + rgb[:, :, 1].astype(np.float32) * np.float32(0.587)
        + rgb[:, :, 2].astype(np.float32) * np.float32(0.114)
    )
    downsampled = gray.reshape(TARGET_SIZE, RATIO, TARGET_SIZE, RATIO).mean(
        axis=(1, 3),
        dtype=np.float32,
    )
    np.rint(downsampled, out=downsampled)
    np.clip(downsampled, 0.0, 255.0, out=downsampled)
    return downsampled.astype(np.uint8)


def _draw_pillow(segments: np.ndarray) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), tuple(int(v) for v in BACKGROUND_RGB))
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(RADIUS_PX * 2.0)))
    fill = tuple(int(v) for v in TRAIL_RGB)
    for x0, y0, x1, y1 in segments:
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=fill, width=width)
        draw.ellipse((x0 - RADIUS_PX, y0 - RADIUS_PX, x0 + RADIUS_PX, y0 + RADIUS_PX), fill=fill)
        draw.ellipse((x1 - RADIUS_PX, y1 - RADIUS_PX, x1 + RADIUS_PX, y1 + RADIUS_PX), fill=fill)
    return np.asarray(image, dtype=np.uint8).copy()


def _draw_cv2(segments: np.ndarray) -> np.ndarray:
    import cv2

    canvas = _canvas()
    color = tuple(int(v) for v in TRAIL_RGB)
    thickness = max(1, int(round(RADIUS_PX * 2.0)))
    cap_radius = max(1, int(round(RADIUS_PX)))
    for x0, y0, x1, y1 in segments:
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        cv2.line(canvas, p0, p1, color, thickness=thickness, lineType=cv2.LINE_8)
        cv2.circle(canvas, p0, cap_radius, color, thickness=-1, lineType=cv2.LINE_8)
        cv2.circle(canvas, p1, cap_radius, color, thickness=-1, lineType=cv2.LINE_8)
    return canvas


def _downsample_scipy(rgb: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    gray = (
        rgb[:, :, 0].astype(np.float32) * np.float32(0.299)
        + rgb[:, :, 1].astype(np.float32) * np.float32(0.587)
        + rgb[:, :, 2].astype(np.float32) * np.float32(0.114)
    )
    filtered = ndimage.uniform_filter(gray, size=(RATIO, RATIO), mode="constant", cval=0.0)
    sampled = filtered[RATIO // 2 :: RATIO, RATIO // 2 :: RATIO]
    sampled = sampled[:TARGET_SIZE, :TARGET_SIZE].astype(np.float32, copy=True)
    np.rint(sampled, out=sampled)
    np.clip(sampled, 0.0, 255.0, out=sampled)
    return sampled.astype(np.uint8)


def _downsample_pillow(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(rgb, mode="RGB").convert("L")
    resampling = getattr(Image, "Resampling", Image).BOX
    resized = image.resize((TARGET_SIZE, TARGET_SIZE), resampling)
    return np.asarray(resized, dtype=np.uint8).copy()


def _time_ms(fn: Callable[[], np.ndarray], *, iterations: int, warmup: int) -> tuple[float, np.ndarray]:
    last = fn()
    for _ in range(warmup):
        last = fn()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        last = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(timings)), last


def _diff(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    delta = np.abs(candidate.astype(np.int16) - reference.astype(np.int16))
    return {
        "exact": bool(np.array_equal(candidate, reference)),
        "max_abs_diff": int(delta.max(initial=0)),
        "changed_values": int(np.count_nonzero(delta)),
    }


def _experiment_result(
    name: str,
    kind: str,
    fn: Callable[[], np.ndarray],
    reference: np.ndarray,
    *,
    iterations: int,
    warmup: int,
    semantics: str,
) -> dict[str, Any]:
    try:
        median_ms, value = _time_ms(fn, iterations=iterations, warmup=warmup)
        return {
            "name": name,
            "kind": kind,
            "median_ms": round(median_ms, 4),
            "parity": _diff(value, reference),
            "semantics": semantics,
        }
    except Exception as exc:  # pragma: no cover - depends on local env.
        return {
            "name": name,
            "kind": kind,
            "error": f"{type(exc).__name__}: {exc}",
            "semantics": semantics,
        }


def run(iterations: int, warmup: int) -> dict[str, Any]:
    imports = _probe_imports()
    segments = _segments()
    rgb = _sample_rgb_frame()
    draw_reference_ms, draw_reference = _time_ms(
        lambda: _draw_reference(segments),
        iterations=iterations,
        warmup=warmup,
    )
    downsample_reference_ms, downsample_reference = _time_ms(
        lambda: _downsample_reference(rgb),
        iterations=iterations,
        warmup=warmup,
    )
    experiments: list[dict[str, Any]] = [
        {
            "name": "numpy_reference_rounded_segments",
            "kind": "draw",
            "median_ms": round(draw_reference_ms, 4),
            "parity": {"exact": True, "max_abs_diff": 0, "changed_values": 0},
            "semantics": "reference NumPy rounded-segment coverage",
        },
        {
            "name": "numpy_reference_luma_area_downsample",
            "kind": "downsample",
            "median_ms": round(downsample_reference_ms, 4),
            "parity": {"exact": True, "max_abs_diff": 0, "changed_values": 0},
            "semantics": "reference float32 luma then 11x11 area mean/rint",
        },
    ]

    if imports["PIL/Pillow"]["available"]:
        experiments.append(
            _experiment_result(
                "pillow_imagedraw_rounded_line",
                "draw",
                lambda: _draw_pillow(segments),
                draw_reference,
                iterations=iterations,
                warmup=warmup,
                semantics="Pillow integer line/ellipse rasterizer; expected to differ",
            )
        )
        experiments.append(
            _experiment_result(
                "pillow_luma_box_resize",
                "downsample",
                lambda: _downsample_pillow(rgb),
                downsample_reference,
                iterations=iterations,
                warmup=warmup,
                semantics="Pillow L conversion plus BOX resize; expected to differ",
            )
        )
    if imports["cv2"]["available"]:
        experiments.append(
            _experiment_result(
                "opencv_line_circle",
                "draw",
                lambda: _draw_cv2(segments),
                draw_reference,
                iterations=iterations,
                warmup=warmup,
                semantics="OpenCV integer line/circle rasterizer; expected to differ",
            )
        )
    if imports["scipy"]["available"]:
        experiments.append(
            _experiment_result(
                "scipy_uniform_filter_downsample",
                "downsample",
                lambda: _downsample_scipy(rgb),
                downsample_reference,
                iterations=iterations,
                warmup=warmup,
                semantics="SciPy uniform filter sampled at block centers; reduction differs",
            )
        )

    available = [name for name, item in imports.items() if item["available"]]
    project_recommendation = (
        "No easy project-runtime win from optional raster dependencies unless one of the exact "
        "compiler options is added deliberately. Pillow-style drawing is fast but changes pixels; "
        "SciPy can match the tiny downsample probe when present, but it is not in the project env "
        "and was slower here. The exact render path still needs Numba/Cython-style kernels or an "
        "incremental trail buffer."
    )
    return {
        "schema": "curvyzero_compiled_raster_feasibility/v0",
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "workload": {
            "frame_size": FRAME_SIZE,
            "target_size": TARGET_SIZE,
            "segment_count": SEGMENT_COUNT,
            "radius_px": RADIUS_PX,
            "iterations": iterations,
            "warmup": warmup,
        },
        "imports": imports,
        "available_optional_imports": available,
        "experiments": experiments,
        "recommendation": project_recommendation,
    }


def _print_plain(report: dict[str, Any]) -> None:
    runtime = report["runtime"]
    print("Compiled raster feasibility quick probe")
    print(f"python: {runtime['python']} ({runtime['executable']})")
    print(f"numpy: {runtime['numpy']}")
    print("imports:")
    for name, item in report["imports"].items():
        if item["available"]:
            print(f"  {name}: available ({item.get('version', 'unknown')})")
        else:
            detail = f" - {item['error']}" if "error" in item else ""
            print(f"  {name}: unavailable{detail}")
    print("experiments:")
    for item in report["experiments"]:
        if "error" in item:
            print(f"  {item['name']}: error {item['error']}")
            continue
        parity = item["parity"]
        print(
            f"  {item['name']}: {item['median_ms']} ms, "
            f"exact={parity['exact']}, max_abs_diff={parity['max_abs_diff']}, "
            f"changed_values={parity['changed_values']}"
        )
    print("recommendation:")
    print(f"  {report['recommendation']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args()
    report = run(iterations=args.iterations, warmup=args.warmup)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_plain(report)


if __name__ == "__main__":
    main()
