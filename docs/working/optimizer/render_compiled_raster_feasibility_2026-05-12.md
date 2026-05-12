# Compiled Raster Feasibility Quick Matrix

Date: 2026-05-12

Scope: optimizer render-only lane. No package installs. No production module
edits. Prototype/report files only.

## Commands

```text
uv run python -c "import sys, importlib.util; print(sys.executable); mods=['cv2','numba','skia','PIL','scipy','cupy','triton']; print({m: importlib.util.find_spec(m) is not None for m in mods})"
python -c "import sys, importlib.util; print(sys.executable); mods=['cv2','numba','skia','PIL','scipy','cupy','triton']; print({m: importlib.util.find_spec(m) is not None for m in mods})"
uv run python scripts/prototype_compiled_raster_feasibility.py --iterations 15 --warmup 2 --format plain
python scripts/prototype_compiled_raster_feasibility.py --iterations 15 --warmup 2 --format plain
```

## Importability

Project runtime, `uv run python` at `/Users/shankha/curvy/.venv/bin/python3`:

| Package | Available |
| --- | --- |
| cv2 | no |
| numba | no |
| skia/skia-python | no |
| PIL/Pillow | no |
| scipy | no |
| cupy | no |
| triton | no |

System Python at `/opt/homebrew/opt/python@3.11/bin/python3.11`:

| Package | Available |
| --- | --- |
| cv2 | no |
| numba | no |
| skia/skia-python | no |
| PIL/Pillow | yes, 12.1.0 |
| scipy | yes, 1.16.3 |
| cupy | no |
| triton | no |

## Tiny Timing/Parity

Workload: synthetic 704x704 RGB frame, 96 thick rounded segments with
`radius_px=3.75`, and 704-to-64 luma/area downsample. These are feasibility
smoke numbers, not a trainer profile.

Project `uv` runtime:

| Probe | Median | Exact vs reference |
| --- | ---: | --- |
| NumPy rounded segments reference | 5.6318 ms | yes |
| NumPy luma area downsample reference | 1.1130 ms | yes |

System Python, where Pillow/SciPy are importable:

| Probe | Median | Exact vs reference | Diff |
| --- | ---: | --- | --- |
| NumPy rounded segments reference | 5.2833 ms | yes | 0 |
| Pillow ImageDraw rounded line | 0.7632 ms | no | max 222, 22935 changed values |
| NumPy luma area downsample reference | 1.0504 ms | yes | 0 |
| Pillow L + BOX resize | 0.4490 ms | no | max 1, 219 changed values |
| SciPy uniform_filter downsample | 2.9503 ms | yes on this toy | 0 |

## Recommendation

No easy speed win is available in the project runtime today: all requested
optional raster/compiler/GPU packages are absent from `uv run python`.

Do not pursue Pillow as an exact replacement. It can be fast on the tiny local
draw probe, but it changes segment pixels. SciPy matched the tiny downsample
probe, but it is not in the project runtime and was slower than the current
NumPy downsample here. OpenCV, Skia, CuPy, Triton, and Numba are not locally
importable in the project runtime.

Worth pursuing now: none of these optional dependency swaps.

Worth keeping on the backlog: an exact Numba/Cython-style CPU kernel or an
incremental trail buffer, but only if the team deliberately accepts the
dependency/deployment work.
