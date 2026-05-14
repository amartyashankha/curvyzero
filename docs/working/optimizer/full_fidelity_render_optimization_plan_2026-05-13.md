# Full-Fidelity Render Optimization Plan

Date: 2026-05-13

Scope: optimizer-owned render speed work for the trusted CurvyTron visual
surface. The target is CPU-reference fidelity for `browser_lines`: source-state
browser-like rendering, 704-style source pixels, browser-sprite bonuses,
BT.601 luma, and 11x11 area downsample to gray64. This is not a browser-canvas
pixel parity claim. Do not use `body_circles_fast` as a recommendation lane.

## Plain Goal

Make the trusted visual renderer much faster without changing what the policy
sees.

The current good baseline is CPU cached `browser_lines`. It already avoids
redrawing old trail history in supported append-only cases. The next wins must
beat that baseline or remove a different cost such as state packing,
host/device transfer, or downstream CPU round trips.

Optimizer parity rule: dirty-cache and GPU paths must match the CPU full
reference exactly for the surface they replace. If a path returns gray64, the
gray64 tensor must be byte-exact. If a path stores RGB cache state, the RGB
cache must be byte-exact too. Browser golden-frame proof belongs to Environment
Reconstruction and is a separate lane.

Downsample note: the current 704 RGB -> BT.601 luma -> 11x11 local mean is a
standard anti-aliased decimation shape. It should not be replaced by center
sampling or luma-space sprite shortcuts. The current 2P self/other palette is
visible in one-channel luma, but future 3P/4P color identity is not guaranteed.
See [downsample/reference fidelity](downsample_reference_fidelity_2026-05-13.md).

## Current Read

- Full redraw is the enemy for long trajectories.
- GPU math is promising when batched, but copying state to the GPU can already
  cost milliseconds.
- The closer 704-style GPU surface is not catastrophic, but it is not
  production-ready. The first production CPU comparison is close, not exact.
- A naive GPU full redraw every observation is probably not the final answer.
  The best shape is dirty/incremental source blocks or a fused renderer whose
  output stays near policy/search on GPU.
- After the dirty-cache fixes, render-only optimization is not automatically
  the biggest full-loop win. Use full-loop Amdahl profiles before choosing the
  next production target.

## Latest Probe

`src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py` now compares
the synthetic `block_704_gray64` GPU output against production
`render_source_state_canvas_gray64(..., trail_render_mode="browser_lines")` for
tiny rows. After switching the block path to production-like 704-pixel
coordinates:

- `B=1`, `trail_slots=64`, no bonuses: exact byte parity on the tiny oracle.
- `B=1`, `trail_slots=64`, 8 bonuses: 51 of 4096 pixels differed, max diff 90.
- `B=16`, `trail_slots=64`, no bonuses: about 4.74ms device render and 3.34ms host-to-device
  copy on L4, without output readback.
- `B=2`, `trail_slots=256`, no bonuses: exact byte parity on the two-row oracle.
- `B=16`, `trail_slots=500`, no bonuses: about 46.7ms device render and 3.2ms
  host-to-device copy on L4.

Plain read: the GPU lane is plausible, but the next optimization must close
the exactness gap or keep the CPU cached renderer as the trusted production
path. The line/head geometry has a first exact tiny oracle; bonus sprites and
harder overlap/draw-order cases remain open. Naive full GPU redraw still scales
with trail history, so the likely production shape is dirty/incremental CPU
first, or dirty/device-resident GPU later.

## Next Gates

0. **Profile Comparability**
   Every profile must name code state, render mode, bonus mode, natural bonus
   setting, death mode, trajectory length, warmup, env/search/learner inclusion,
   dirty-cache stats, and whether artifact/checkpoint/eval/GIF work is included.

1. **Real State Feed**
   Feed the GPU benchmark from actual vector runtime state arrays instead of
   synthetic rows. Keep the run isolated: no trainer, no volumes.

2. **Tiny Parity Gate**
   Compare one or two simple real states against CPU `render_source_state_canvas_gray64`.
   Start with no bonuses and known line geometry. The first acceptable goal is
   bounded edge-pixel mismatch, then exactness where possible.

3. **Component Timing**
   Split the CPU trusted path into trail draw, bonus/head overlay, luma/downsample,
   stack insert, cache hit/fallback, and state packing. Do not optimize blind.

4. **Dirty 11x11 Blocks**
   Prototype updating only dirty target blocks after a new trail segment/head
   move. This matches Amdahl better than redrawing all source samples.

5. **Device-Resident Handoff**
   Test whether a rendered GPU tensor can feed the model/search path without
   immediate CPU readback. If not, GPU render is capped by transfer/sync costs.

6. **Bonus Sprite Parity**
   If the GPU lane continues, move `block_704_gray64` from luma-only to RGB
   block composition and match production bonus sprite stamping before claiming
   full-fidelity replacement.

7. **Downsample Signal Tests**
   Add tests that sweep line/head/sprite positions across 11x11 block
   boundaries, audit player-color luma collisions, and measure all 12 bonus
   sprite gray64 signatures.

## What To Avoid

- Do not switch the training surface to a simpler renderer just because it is
  faster.
- Do not claim fidelity from the synthetic GPU probes.
- Do not optimize the old custom two-seat trainer path.
- Do not add a heavy graphics stack before a small exactness and transfer
  benchmark says it is worth it.

## Open Questions

- How often does the CPU dirty cache fall back in real training states?
- How many dirty 64x64 target blocks change per environment step at realistic
  survival lengths?
- Is state packing or renderer math the larger cost after the CPU cache lands?
- Can LightZero consume a GPU-rendered observation without forcing CPU
  conversion in the env manager boundary?

## Immediate Tasks

- [ ] Keep docs aligned around full-fidelity `browser_lines`.
- [ ] Remove approximation-first language from optimizer recommendations.
- [ ] Run a real-state isolated GPU smoke.
- [x] Add a tiny CPU-vs-GPU production comparison for the 704-style timing surface.
- [x] Tighten that comparison into an exact no-bonus production-shaped parity test.
- [ ] Add bonus-sprite parity or explicitly exclude active bonuses from any GPU claim.
- [ ] Reprofile current CPU cached full-fidelity path after any environment
      changes land.
