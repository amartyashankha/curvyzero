# CurvyTron Render Acceleration Research

Date: 2026-05-13

Scope: trusted stock LightZero `source_state_fixed_opponent` path. Live training
runs are read-only for this note. Numbers below are local env/prototype signals,
not full-training measurements.

Current target: full-fidelity `browser_lines` only. That means source-state RGB
on the 704-style canvas, BT.601 luma, then 11x11 downsample to 64x64.
`body_circles_fast` is historical/control evidence only and is not a current
optimization lane.

## Short Read

The current render bottleneck is full-frame source-state drawing. The rich
`browser_lines` path redraws the source-style RGB canvas, then converts it to
the 64x64 gray observation. In long-survival states, most of the old trail has
not changed, but the scalar render path can still pay to redraw it.

That makes render cost grow with trail history. Stack shifting and ordinary
wrapper bookkeeping are small compared with the redraw.

## Evidence

Local fixed-length no-death profiling of the stock fixed-opponent wrapper showed
`browser_lines` spending about 76% of env-only wall time in render at 100 steps,
and about 88-89% at 500-2000 steps. The faster `body_circles_fast` comparison
was still render-heavy, about 58-75%, but it is an approximation and not a
semantic replacement.

Local active-path microbenches also point at the same shape: gray64 rendering
dominates once trails are nontrivial, while stack shift/copy work is near noise.
Earlier full-redraw probes scaled roughly with batch, player views, and trail
length. Dirty/incremental cache probes showed parity in supported cases and much
better long-trail behavior, but these were local/prototype or narrow production
signals rather than full training verdicts.

## 2026-05-13 Landing

The exact dirty/incremental cache is now wired into the trusted stock
fixed-opponent `browser_lines` wrapper. A focused parity test checks that after
multiple no-death blank-canvas steps, the cached frame matches the full scalar
renderer byte-for-byte for both raw RGB and gray64.

Latest local env-only profile artifact:

```text
artifacts/local/curvytron_render_profiles/render_trajectory_lengths_dirty_scalar_20260513.json
```

Plain result: cached `browser_lines` is a real long-trajectory win. It moved
the 500-step local wall time from `39.1s` to `10.5s`, and the 2000-step local
wall time from `175.9s` to `46.9s`. In the same latest table,
`body_circles_fast` is still faster at 100-200 steps, but cached
`browser_lines` is faster at 500+ steps while keeping the richer visual
surface.

## 2026-05-13 Bonus Fix

Natural bonus spawning exposed a second cache problem. Active bonus sprites were
stationary for many steps, but the dirty cache marked every old/current bonus
sprite box dirty on every observation. In a 500-step no-death local profile that
turned `6,068` no-bonus dirty blocks into `162,185` bonus-on dirty blocks.

The fix now snapshots active bonus slot/id/type/position/radius. Bonus boxes are
marked dirty only when the visible bonus snapshot changes. Current bonuses
still expand any dirty region that intersects them, so if a trail or head changes
under a translucent sprite, the whole sprite box is redrawn once in the right
order. Unchanged far-away sprites are skipped.

Measured local result for the 500-step bonus-on row:

| metric | before | after |
| --- | ---: | ---: |
| dirty blocks | 162,185 | 12,153 |
| render time | 4.43s | 1.54s |
| wall time | 9.39s | 6.13s |
| bonus draw bucket | 0.300s | 0.070s |

The after row is still env-only and local; it is not a full training speedup
claim. It does show that the natural-bonus cost is no longer an obvious runaway
dirty-block bug.

## Body Circles Clarification

`body_circles_fast` is not inherently useless. It is simpler than browser-style
lines, and it wins on short trajectories today. Keep those numbers as
historical/control evidence. They are not recommendations because the current
target is full-fidelity `browser_lines`.

The reason `body_circles_fast` loses at long lengths is simple: the current code
still redraws all active body circles every observation, while cached
`browser_lines` mostly reuses the already-rendered trail and updates only
changed blocks.

The same broad dirty-cache idea can probably be applied to `body_circles_fast`,
but it is not the same patch. The existing cache is built around append-only
visual trail layers and is explicitly gated to `browser_lines`. A body-circles
cache would need its own invalidation rules over body slots, owner changes,
clears, death/reset, radius changes, and bonus/head overlays. It is a reasonable
future experiment, not something we have already proven.

Plain choice from now on: optimize the full-fidelity `browser_lines` lane. Do
not spend recommendation effort on a `body_circles_fast` cache unless the target
explicitly changes.

## Candidate Routes

- Treat the existing exact dirty/incremental cache as the current first win.
  It keeps the full renderer as the fallback and only recomposes changed source
  blocks when state is append-only and supported.
- Keep the stationary-bonus dirty-block fix. It is parity-protected and removes
  the largest measured natural-bonus render regression.
- Add component profiling around the stock wrapper render call: full RGB draw,
  player/perspective normalization, bonus/head overlay, luma/downsample, stack
  insert, and cache hit/fallback/dirty-block counts.
- Consider compiled CPU kernels only after profiling says which bucket remains
  hot. Good candidates are dirty-block luma/downsample, stamp/mask overlay, and
  line/raster loops that can stay exact.
- Consider GPU render only if batching and device residency beat transfer and
  launch cost. A GPU path that renders one small frame at a time, then copies it
  back to CPU, is unlikely to win.
- Prototype exact block-local 11x11 rendering: compose only dirty output cells
  in RGB draw order, then luma/downsample the tile. This is the most coherent
  next CPU/GPU architecture.

## GPU Renderer Read

The GPU idea is not rejected. The first isolated Modal probes now make the
read more concrete.

The first CuPy RawKernel attempt was blocked by CUDA/NVRTC image plumbing on
Modal, so the probes now use the repo's known-good JAX CUDA stack. That is fine
for economics. It is not yet the final implementation choice.

Current L4 probe results:

- direct64 toy, `B=64`, `primitives=256`: device render about `0.31ms`,
  host->device about `0.79ms`, readback about `0.28ms`;
- synthetic source-state direct-gray64, `browser_lines`, `B=64`,
  `trail_slots=500`: device render about `0.98ms`, host->device about `3.91ms`;
- same with output readback measured: readback about `0.29ms`;
- synthetic source-state direct-gray64, `browser_lines`, `B=128`,
  `trail_slots=2000`: device render about `5.18ms`, host->device about
  `3.47ms`.
- timing-only block-704 source-state, `browser_lines`, `B=16`,
  `trail_slots=100`: device render about `6.37ms`, host->device about
  `3.17ms`;
- timing-only block-704 source-state, `browser_lines`, `B=8`,
  `trail_slots=500`: device render about `16.50ms`, host->device about
  `3.85ms`.
- block-704 source-state with production CPU oracle, no bonuses, `B=1`,
  `trail_slots=64`: after switching to production-like 704-pixel coordinates
  and keeping float luma until the final block average, device render was about
  `0.87ms`, host->device about `2.89ms`, readback about `0.39ms`; output
  matched production gray64 byte-for-byte on the tiny oracle.
- the same `B=1`, `trail_slots=64` shape with `8` active bonuses differed on
  `51/4096` pixels, about `1.25%`, with max diff `90`, because the prototype
  still uses simple bonus circles instead of production sprite stamps.
- block-704 source-state batched, no bonuses, `B=16`, `trail_slots=64`:
  device render about `4.74ms`, host->device about `3.34ms`, end-to-end about
  `8.03ms` without output readback.
- longer no-bonus rows show the naive full-redraw GPU path still scales with
  trail history: `B=16`, `trail_slots=256` took about `23.1ms` device render
  plus `3.2ms` host->device; `B=16`, `trail_slots=500` took about `46.7ms`
  device render plus `3.2ms` host->device.
- parity remains good for no-bonus rows but not universal yet: `B=2`,
  `trail_slots=256` matched production exactly; `B=2`, `trail_slots=500`
  differed on `5/8192` pixels.

Plain read: raw GPU rendering is fast enough to keep investigating. The larger
risk is copying and packing state each step. For short and moderate trail
histories, transfer can dominate render. For long trails, render and transfer
are both meaningful. A serious GPU renderer should therefore be batched and
device-resident: many env rows at once, direct output to `[B,P,4,64,64]` or the
policy input, and minimal host-device synchronization.

Important gap: the block-704 probe now has a tiny production CPU comparison,
and the no-bonus line/head path can be byte-exact on that oracle. It is still
not a full replacement because it does not yet cover exact owner grouping,
path splitting under harder cases, RGB overwrite interactions, or bonus sprites.

The Amdahl recommendation did not change after these probes. A CPU->GPU->CPU
full redraw is the wrong production shape unless policy/search can consume the
tensor on device. The strongest next production path is still to instrument and
improve the CPU dirty/cache renderer, then compile only the measured hot dirty
loops if needed.

## Full-Fidelity Next Plan

1. Keep the exact dirty/incremental cache on for trusted `browser_lines`
   profiles. Measure hit rate, fallbacks, dirty blocks per hit, and exact parity
   against full redraw when changing render code.
2. Profile components in the live stock wrapper shape, without changing
   semantics: RGB draw, BT.601 luma, 11x11 downsample, overlays, Python glue,
   cache hits, and fallbacks.
3. Build the next GPU prototype against the 704-style RGB -> BT.601 luma ->
   11x11 downsample shape, still isolated from training.
4. Try compiled CPU kernels for any measured hot bucket that remains simpler
   than a GPU integration, keeping full-render parity tests as the gate.
5. Only integrate GPU rendering with training if the output can feed policy
   input without an immediate CPU round trip.
6. If continuing GPU fidelity, switch `block_704_gray64` to true RGB block
   composition and add production bonus sprite parity. The smallest useful test
   is a no-trail row with one active bonus of each type, then a trail-under-sprite
   and head-over-sprite ordering case.

## Direct-To-64 Sprite Toy Result

Scratch-only toy probe:

```text
/private/tmp/curvy_sprite_probe/sprite_block_probe.py
```

It compared full `704 RGB -> luma -> 11x11 downsample` with direct sprite
contribution into 64x64 block sums. The exact per-704-pixel accumulator and an
exact interval/count accumulator both matched the full path on `250/250`
randomized sprite placements. The naive 64x64 center-sampling approximation
failed in `247/250` cases, and luma-space alpha blending failed in `127/250`
cases, although the luma error was usually small.

Plain read: we can avoid materializing a full 704 image, but only if we preserve
704-pixel semantics. Exact sprite handling still needs the underlying RGB for
the touched pixels. That confirms the next architecture: block-local RGB tile
composition, then immediate luma/downsample.

Plain recommendation: keep the exact CPU cache in production now, but continue
the GPU lane. The GPU lane has enough signal to justify the next prototype; it
does not yet have enough fidelity to replace the trusted renderer.
