# Incremental Render Toy Bench

Date: 2026-05-21

Scope: quick local synthetic benchmark only. No production code was modified.

## Context

Latest optimizer notes in this folder point back to long-trajectory render cost
as the dominant observation wall. In the B512/A16 dynamic-slot ladder, 500
no-death steps spent about `821.5s` in observation and `802.4s` in renderer
render, roughly `94%` of measured wall in renderer work. Dynamic active-prefix
width avoids paying for empty capacity, but it still redraws a growing active
trail prefix as trajectories get longer.

Relevant code inspected:

- `src/curvyzero/training/source_state_batched_observation_profile.py`:
  profile-only batched facade emits `[B, 1, 64, 64]` frames and exposes
  `render_sec`, `readback_sec`, and `stack_sec`.
- `src/curvyzero/env/vector_visual_observation.py`: current CPU visual renderer
  already has an exact append-only `SourceStateBrowserLineTrailLayerCache` for
  browser-line trails in supported cases.
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py` and the
  boundary-profile notes: current GPU candidate uses dense/direct batched
  render lanes, with dynamic trail-slot width but not a persistent image layer.

## Toy Setup

Script:

```text
/private/tmp/curvy_incremental_render_toybench_20260521.py
```

Result JSON:

```text
/private/tmp/curvy_incremental_render_toybench_20260521.json
```

Command:

```text
uv run python /private/tmp/curvy_incremental_render_toybench_20260521.py --batch-sizes 64,512 --steps 20,100,200,500 --repeats 5 --warmups 1 --output /private/tmp/curvy_incremental_render_toybench_20260521.json
```

Machine/runtime reported by the script:

- macOS 15.6 arm64
- Python `3.11.13`
- NumPy `2.4.4`

The benchmark precomputes CurvyTron-ish two-player line pixels on a direct
`[B, 64, 64]` uint8 grayscale canvas. That intentionally removes line-geometry
computation from the timed section, so the measured delta is old-trail replay
volume:

- **Full redraw:** clear `[B,64,64]`, then replay every old trail pixel every
  frame.
- **Incremental:** keep a persistent `[B,64,64]` canvas and write only newly
  appended trail pixels.
- Both paths use the same append order and final-frame parity check.

## Results

All cells had final-frame parity: `diff_pixels=0`.

| batch | steps | full redraw median sec | incremental median sec | speedup | full/incremental write ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 20 | `0.000103` | `0.0000102` | `10.1x` | `10.7x` |
| 64 | 100 | `0.001104` | `0.0000478` | `23.1x` | `53.2x` |
| 64 | 200 | `0.003424` | `0.0000974` | `35.1x` | `104.2x` |
| 64 | 500 | `0.019670` | `0.0002448` | `80.4x` | `254.9x` |
| 512 | 20 | `0.000863` | `0.0000429` | `20.1x` | `10.6x` |
| 512 | 100 | `0.010087` | `0.0002129` | `47.4x` | `53.3x` |
| 512 | 200 | `0.039180` | `0.0004555` | `86.0x` | `104.5x` |
| 512 | 500 | `0.229214` | `0.0010887` | `210.5x` | `254.9x` |

The B512/500 cell is the cleanest pressure read:

- full redraw replayed about `204.1M` prefix pixel writes;
- incremental wrote about `0.801M` newly appended pixels;
- final canvas parity still matched exactly.

## Interpretation

The toy supports an incremental-render lane. It shows the expected algorithmic
shape: full redraw grows with the sum of active-prefix lengths over time, while
persistent update grows with only the newly appended trail pixels. As
trajectory length increases, the write ratio and speedup both widen.

This is directionally consistent with the latest long-trajectory profile:
dynamic trail slots can shrink inactive capacity, but once active trails grow,
full active-prefix redraw again becomes the wall. A persistent trail image or
per-owner trail layer is the natural next renderer-side experiment for long
survival rows.

## Limitations

This is not a production benchmark.

- Direct `64x64` grayscale only; no 704 RGB canvas, antialiasing,
  area-downsample, or browser pixel parity.
- Precomputed line pixels; it isolates redraw volume and does not measure line
  rasterization math.
- CPU/NumPy local timing; no JAX kernel launch, GPU memory layout,
  host-to-device/device-to-host transfer, or compilation behavior.
- Append-only trails only; no clear/reset/wrap, cursor regression, prefix
  mutation, death/autoreset, trail gaps, radius/color changes, or bonus/head
  redraw complexity.
- No stack shift/update timing and no LightZero scalar materialization.
- Collisions/overwrites are simple grayscale assignment. Production draw order
  and per-owner overlap semantics still need explicit parity gates.

## Recommendation

Use this as a cheap algorithmic yes, not as a speed claim. The next useful lane
is a profile-only persistent/incremental renderer at the current batched
observation boundary, with fail-closed rebuilds for non-append cases and parity
checks against the trusted CPU renderer on reset, wrap/clear, gap, owner overlap,
radius/color change, and terminal/autoreset rows.

