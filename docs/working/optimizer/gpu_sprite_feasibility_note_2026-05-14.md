# GPU Sprite Feasibility Note

Date: 2026-05-14

Scope: side-lane critique only. No code changes, Modal runs, live runs, or new
measurements.

## Bottom Line

We cannot truthfully say "just put the original bonus sprites on GPU" because
the missing work is not the upload of `bonus.png`. The missing work is matching
the current CPU reference's sprite composition semantics inside the GPU
`block_704_gray64` path.

Trail/head parity is already close enough to make the GPU lane interesting.
Bonus sprites are the known semantic gap. The current GPU benchmark still draws
bonuses as type-coded scalar circles/luma values, while the CPU reference draws
RGBA atlas tiles, alpha-blends them over trails, draws heads on top, converts
RGB to BT.601 luma, and area-downsamples 704 source pixels into 64x64.

Simple symbols are therefore a reasonable bridge for an immediate fast lane, as
long as they are explicit opt-in approximation mode and not presented as original
sprite parity.

## Current Evidence

- `source_state_gpu_render_benchmark.py` declares the JAX path's known gaps:
  no full RGB canvas parity and no bonus sprites. Its block renderer compares
  against production CPU `render_source_state_canvas_gray64`, but bonus drawing
  remains a circle/luma shortcut.
- `vector_visual_observation.py` has the CPU reference behavior: atlas path
  `third_party/curvytron-reference/web/images/bonus.png`, 3x4 tiles, type to
  sprite mapping, cached stamp resize by destination size, source-over alpha
  blend, RGB rounding/clipping, then gray64 downsample.
- The optimizer GPU notes show no-bonus line/head rows can be exact or near
  exact, while bonus rows are the visible mismatch: about `1.2%` to `1.6%` of
  checked gray64 pixels in synthetic bonus comparisons, with large max diffs
  around missing or differently shaped sprite pixels.
- The sprite-shape probe says the original 12 icons carry class identity through
  both color/luma group and internal alpha pattern. Collapsing to filled circles
  discards that pattern, even if a class-coded luma shortcut may remain learnable.

## What Original Sprite GPU Parity Requires

Minimum faithful GPU work:

1. Put the 3x4 RGBA atlas, type-to-sprite mapping, and alpha data on device.
2. Match CPU placement exactly: world center/radius, `_canvas_round`, clipping,
   destination size, nearest source sampling, and tile selection.
3. Compose in the same draw order: background/trails, active bonus sprites,
   live heads.
4. Preserve source-over RGB alpha blending and rounding before luma. A pure
   luma-domain shortcut is not byte-equivalent because the CPU blends RGB first.
5. Produce the same gray64 contract: BT.601 luma and 11x11 area average from
   704-style source pixels.
6. Handle overlapping bonuses deterministically. Sprite-parallel unordered
   writes or atomics are risky unless tests prove overlap order cannot matter.
7. Compare against CPU `render_source_state_canvas_gray64(...,
   trail_render_mode="browser_lines", bonus_render_mode="browser_sprites")`
   on real source-state rows, not only synthetic rows.

Implementation shape options:

- Full 704 RGB scratch on GPU, then luma/downsample. This is conceptually
  simplest and easiest to parity-test, but likely wastes work and memory.
- Direct 64 output with per-11x11-block RGB recomposition. This preserves exact
  semantics without materializing all 704 pixels, but the kernel is more complex.
- JAX-only version with atlas constants and per-pixel gathers. Possible, but
  awkward around variable destination sizes, alpha composition, and ordered
  overlapping sprites.
- Custom CUDA/Pallas/Torch extension. Likely cleaner for production if the
  observation stays on GPU, but a larger engineering commitment.

## Likely Complexity

This is moderate-to-high complexity, not a one-line GPU texture upload.

The hard parts are exact composition and integration economics:

- Current JAX benchmark state is synthetic and host-fed; production needs real
  env rows and ideally device-resident output near policy/search.
- Host-to-device copy is already milliseconds in the benchmark. A sprite pass
  that copies whole RGB frames or bounces through CPU would erase much of the
  benefit.
- The CPU renderer caches resized stamps by `(sprite_index, dst_size)`. A GPU
  parity path needs either equivalent cached device stamps for expected sizes or
  exact atlas sampling inside the kernel.
- The GPU path must still beat the improved CPU cached `browser_lines` baseline,
  not the older full-redraw renderer.

I would treat original-sprite GPU rendering as a real subproject with parity
fixtures, not as a small patch to the current prototype.

## Are Simple Symbols A Reasonable Bridge?

Yes, for the immediate fast lane.

Simple symbols remove the expensive/awkward parts of original sprites:
RGBA atlas sampling, alpha blending, RGB round-before-luma, variable sprite
stamp sizes, and overlap-sensitive translucent composition. A small deterministic
gray64 mask atlas can be much easier to implement on CPU and GPU with exact
equality, while still preserving the facts the policy needs: bonus type, group,
location, and footprint.

The caveat is semantic honesty. Simple symbols are an approximation surface, not
the trusted CPU reference. They need:

- an explicit `simple_symbols`/equivalent bonus mode;
- train/eval metadata so render modes cannot silently mismatch;
- separability tests across offset, radius, clipping, downsample phase, trails,
  heads, and player-perspective remap;
- CPU/GPU equality tests if both implementations exist.

So the bridge is reasonable because it gives the main thread a fast,
machine-visible bonus code while the harder original-sprite GPU lane remains
properly gated. It should not block or replace the full-fidelity GPU work unless
profiles show original sprite handling is the decisive blocker.

## Recommendation

Proceed with simple symbols as an explicit fast approximation if that is already
the main-thread path. Keep `browser_sprites` as the CPU reference/default.

For original sprites on GPU, the next credible step is a small isolated parity
prototype: one or a few real source-state rows, atlas on device, exact sprite
over trail plus head-over-sprite cases, byte comparison to CPU gray64, and
separate timing for transfer, render, and readback. Until that passes, the GPU
renderer should continue to advertise bonus sprites as the known gap.
