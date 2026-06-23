# CurvyTron Render Optimization Research

> [!IMPORTANT]
> Superseded/archive note (2026-05-15): production policy observation is CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU `browser_lines + simple_symbols` remains lab-only until trainer contract parity. `body_circles_fast` is historical/control only.

Date: 2026-05-12

Status: active research note. Do not treat this as an Environment fidelity
claim or as a final training recommendation.

## Current Truth

Current trusted Coach training/profiling uses the stock LightZero path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
env_variant=source_state_fixed_opponent
```

For this stock fixed-opponent lane, the trusted render comparison is:

- `browser_lines + simple_symbols` through CPU `cpu_oracle`: current production
  policy observation surface, stacked as `[4,64,64]`.
- `body_circles_fast`: faster approximation lens for profiling or explicit
  speed/fidelity experiments.

The old `fast_gray64_direct` mode below was for the superseded custom
`two-seat-selfplay` adapter. Do not copy it into current stock-path commands.
Use `profile_no_death` only for optimizer timing.

Use `scripts/profile_curvytron_render_trajectory_lengths.py` for local
fixed-length no-death render tables over 100/200/500/1000/2000 steps. It
exercises the current source-state fixed-opponent env wrapper and reports
render time versus other env work without touching live training runs.

Environment Reconstruction is still changing rich rendering, so optimizer should
pause landing renderer rewrites and focus on understanding the cost shape,
contracts, and benchmark tooling.

Current render correction: the active canvas-gray64 renderer now renders the
source/browser-resolution RGB frame, currently 704x704, then computes luma and
area-averages 11x11 blocks into the 64x64 training tensor. Older direct-64
numbers below are stale shape evidence only. Use fresh 704-to-64 profiles for
optimizer decisions.

Verified active call path:

```text
SourceStateGray64Stack4.update
  -> render_source_state_canvas_gray64_player_perspectives for P=2
      -> render_source_state_rgb_canvas_like(frame_size=704) for the shared trail base
      -> palette-remap the full RGB frame per player perspective
      -> draw per-perspective bonuses/heads
      -> rgb_canvas_like_to_gray64
  -> render_source_state_canvas_gray64 per player for other P counts or unsafe palettes
```

Historical note: older text below may mention `browser_sprites` as the default.
That is stale for policy observations. Current production policy observations
use `browser_lines + simple_symbols` through CPU `cpu_oracle`; browser sprites
are artifact/reference views. Browser pixel parity is still not claimed.

2026-05-12 optimization landed: the two-seat stack now avoids the worst
duplicate trail redraw when the palette is safe. The helper renders the
expensive trail layer once, recolors it for each player perspective, then draws
bonuses and heads before downsampling. It falls back to independent renders if
the base palette is ambiguous, duplicated, or collides with background or the
invalid-owner gray. This is an optimizer equivalence optimization, not a new
Environment fidelity claim.

2026-05-12 follow-up landing: the active two-seat stack now also passes a
stateful browser-line trail cache into that helper. The cache is deliberately
narrow: it only handles non-empty `visual_trail_*` rows in `browser_lines`
mode, rebuilds on cursor regression, prefix mutation, map-size change, or
palette change, and otherwise falls back to the existing full renderer. It also
uses a reusable exact downsample scratch buffer for the 704-to-64 luma path.
The production cache has a minimum active-trail threshold because the prototype
was slower on tiny trails and faster once trail history is long enough to make
redraw avoidance matter.

2026-05-12 dirty-block production landing: the active two-seat path now uses
stateful source-state dirty gray64 rendering for supported cache state. This is
exact-pixel intended and does not change render semantics. It cold-starts or
falls back to the full renderer on reset or unsupported cache state, then reuses
previous RGB/gray frames and recomposes only dirty 11x11 source blocks. Trail
layer cache append now refreshes only the dirty bbox instead of rescanning the
full 704 mask.

Historical custom-adapter note, 2026-05-12: `fast_gray64_direct` was an
explicit approximation mode on the custom two-seat stack. It did not pretend to
be browser pixel parity. It rendered directly at 64x64, used the same BT.601
RGB-to-gray luma rule as the full RGB path, preserved self/other contrast for
trails and heads, and preserved bonus type as luma instead of collapsing every
bonus to one gray value. This is not the fast render knob in the current stock
fixed-opponent lane; use `body_circles_fast` there.

Fresh no-death sentinel A/B, 2026-05-12, canonical Modal launcher, L4/T4,
B8/sim2, 8 iterations, 64 collect steps, no learner updates, no checkpoints,
background eval/GIF off:

```text
browser_lines:
  elapsed_sec=53.6
  visual_stack_update_sec sum=31.2
  policy_search_sec sum=9.0

fast_gray64_direct:
  elapsed_sec=25.5
  visual_stack_update_sec sum=2.4
  policy_search_sec sum=9.1
```

Plain read: the user's reminder is correct for the rich long-survival regime.
Rendering is still the bottleneck when we use `browser_lines` and disable death
to mimic trained long-lived games. The fast-direct approximation cuts visual
time by about `13x` in this matched sentinel and makes search the larger bucket.
That supports fast-direct as the main speed surface plus matched browser-lines
sentinels as approximation checks.

Post-dirty-cache next step: verify production `visual_stack_dirty_render` hit
rate, fallback count, and dirty-block count before landing another renderer
rewrite. If browser-lines is still render-bound after dirty hits are high, the
next low-risk target is fixed per-row overlay work: clipped head redraw, clipped
bonus redraw, and sprite stamping/blending before gray64. A GPU renderer or
vectorized CPU renderer is still plausible, but only after post-dirty profiles
show render remains the largest bucket.

Post-dirty Modal profile, 2026-05-12, canonical launcher, L4/T4, B16/sim4,
8 iterations, 64 collect steps, `profile_no_death`, `browser_lines`, no learner
updates, background eval/GIF off:

```text
elapsed_sec=92.7
visual_stack_update_sec sum=61.3
policy_search_sec sum=14.4
env_step_sec sum=3.2
dirty_render hit_rate=0.9965
dirty_render fallbacks=0
last dirty_blocks_per_hit=28.3
```

Plain read: the dirty cache is active and reliable in this stress profile. It
does not make browser-lines cheap enough for the main training matrix. The next
browser-lines optimization target is fixed per-row overlay/downsample work, not
old full trail redraw.

Fresh local dirty/reuse microbench, 2026-05-12:

```text
full_stack_update, dirty/reuse path:
  B8/P2/L1024 bonus4: 31.2ms/update, 513 rows/sec
  B8/P2/L4096 bonus4: 28.5ms/update, 562 rows/sec
  B16/P2/L1024 bonus4: 53.4ms/update, 599 rows/sec
  B16/P2/L4096 bonus4: 56.9ms/update, 562 rows/sec

gray64_render_only, independent full render:
  B8/P2/L1024 bonus4: 28.8ms per player-view call, 34.7 rows/sec
  B8/P2/L4096 bonus4: 106.7ms per player-view call, 9.4 rows/sec
  B16/P2/L1024 bonus4: 29.0ms per player-view call, 34.5 rows/sec
  B16/P2/L4096 bonus4: 107.8ms per player-view call, 9.3 rows/sec

perspective_reuse_gray64:
  B16/P2/L1024 bonus4: 6.1ms per policy row, 162.8 rows/sec
  B16/P2/L4096 bonus4: 6.3ms per policy row, 158.5 rows/sec
```

Plain read: trail length no longer explodes the dirty stack path the way it did
before. Browser-lines is still slow in the full Modal loop because every step
still has many rows, player perspectives, overlay passes, downsampling, and
Python/NumPy glue.

## Responsibility Boundary

Optimizer can optimize render speed only when the emitted tensor remains
equivalent under the declared schema and render mode. Environment Reconstruction
owns source-fidelity semantics.

Optimizer-safe work:

- render microbenchmarks and timing buckets;
- buffer reuse and copy elimination;
- stack shift/copy optimization;
- lookup tables and precomputed masks/stamps;
- CPU vectorization or GPU rewrites that are exact-equivalence gated.

Needs Environment signoff:

- changing the default product mode away from `browser_lines`;
- using `body_circles_fast` as anything except an explicit approximation;
- promoting `fast_gray64_direct` from speed-first approximation to default
  training surface;
- changing trail continuity, gap, clear, wrap, radius, draw order, luma, or
  player perspective semantics;
- deriving one player's rendered view from another if sprites/HUD/effects or
  player-specific colors become training-visible;
- incremental rendering unless clear/reset/wrap/printing/bonus/head/radius/color
  cases prove exact equivalence;
- promoting bonus64/rich tensors to trainer input;
- adding sprite animation, HUD, explosions, idle arrows, or browser pixel parity
  claims.

## Measured Signal

Small no-death profiles are now long enough to expose render cost. In long
rollouts, `visual_stack_update_sec` grows with trail length and dominates the
trainer wall clock.

These measurements are pre-stabilization signals from before the full-resolution
render plus downsample path was active. They are kept only to explain why the
render lane became important.

Representative pre-stabilization profile, L4/T4, `browser_lines`, batch 16,
sim 8, 10 iterations, 32 collect steps per iteration, background eval/GIF off,
no optimizer step:

```text
iteration 10:
  visual_stack_update_sec = 59.057809
  policy_search_sec       = 2.343682
  env_step_sec            = 0.342577
  elapsed_sec total       = 361.448415
```

Same broad shape with cheaper `body_circles_fast`:

```text
iteration 10:
  visual_stack_update_sec = 27.729964
  policy_search_sec       = 1.707623
  env_step_sec            = 0.192395
  elapsed_sec total       = 181.684326
```

Batch 32 on H100 did not solve the render bottleneck:

```text
iteration 10:
  visual_stack_update_sec = 66.360183
  policy_search_sec       = 2.950907
  env_step_sec            = 0.380774
  elapsed_sec total       = 406.043263
```

Speculative experiment, not landed: deriving player 1's two-player perspective
from player 0's rendered gray frame by swapping self/other gray values reduced
late L4 `browser_lines` visual time from about `59.1s` to about `24.0s`.
That patch was reverted because richer render semantics are still moving.
The result is still useful evidence: duplicate per-seat redraw is a large
component of the current cost.

Toy direct-render experiment, 2026-05-12, read-only ephemeral script:

```text
B16/P2 median stack.update ms, no bonuses:
  browser_lines: L0 2.6, L64 47.1, L256 181.1, L1024 740.0
  body_circles_fast: L0 2.6, L64 34.1, L256 115.0, L1024 450.6
```

Plain read: current timing is basically `B * P * trail_length`. Direct render
and `SourceStateGray64Stack4.update` are close once trails are nontrivial, so
stack/copy overhead is secondary until render redraw is fixed. Four active
bonuses added about `1.8-1.9ms` per `B8/P2` loop in that small probe.

Local benchmark script, 2026-05-12, after adding
`scripts/benchmark_render_lane_microbench.py`:

```text
uv run python scripts/benchmark_render_lane_microbench.py \
  --iterations 3 --warmup-iterations 1 --format plain
```

Representative per-update values from that run:

```text
B16/P2 browser_lines stack.update:
  L0    2.9ms
  L64   160.0ms
  L256  218.2ms
  L1024 717.2ms
  L4096 2713.5ms

B16/P2 body_circles_fast stack.update:
  L64   28.2ms
  L256  104.3ms
  L1024 415.7ms
  L4096 1766.3ms

B16/P2 browser_lines gray64 render only:
  L1024 21.4ms per player-perspective gray64 render call
  L4096 85.7ms per player-perspective gray64 render call

B16/P2 stack-shift/insert/return-copy-only:
  0.26ms per stack update

B8/P2/L1024 browser_lines bonus delta:
  bonus0 336.5ms/update
  bonus4 339.9ms/update
```

Plain read: stack FIFO/copy cost is tiny in this synthetic probe. The hot path
is still redrawing long trails for every row and player perspective. Current
bonus circle overhead is small compared with long-trail redraw; real sprites
still need a separate benchmark after Environment finalizes their semantics.

Small bonus/render matrix, local-only synthetic probe, 2026-05-12:

```text
B16/P2 stack.update ms/update, iterations 3, warmup 1:

trail  bonus  browser_lines  body_circles_fast
0      0      2.5            2.6
0      4      3.4            3.5
0      20     5.9            5.7
1024   0      726.2          463.9
1024   4      709.0          434.7
1024   20     712.0          435.9
4096   0      2899.6         1800.6
4096   4      2813.9         1797.4
4096   20     2928.5         1806.7
```

Plain read: bonus circles matter when trails are empty, but they are not the
first-order cost in long-survival profiles. Long trail redraw dominates.

Active 704-to-64 probe, 2026-05-12, after renderer changed:

```text
uv run python scripts/benchmark_render_lane_microbench.py \
  --plan grid \
  --batch-sizes 1,8 \
  --player-counts 2 \
  --trail-lengths 0,1024 \
  --bonus-counts 0,4 \
  --trail-render-modes browser_lines,body_circles_fast \
  --cell-kinds full_stack_update,gray64_render_only,stack_shift_insert_return_copy_only \
  --iterations 2 \
  --warmup-iterations 1 \
  --allocation-mode reuse \
  --gpu-transfer off \
  --format plain
```

Representative per-update values:

```text
B1/P2/L0:
  browser_lines stack.update 6.6ms
  body_circles_fast stack.update 5.5ms

B1/P2/L1024:
  browser_lines stack.update 61.0ms
  body_circles_fast stack.update 41.4ms

B8/P2/L0:
  browser_lines stack.update 42.5ms
  body_circles_fast stack.update 44.2ms

B8/P2/L1024:
  browser_lines stack.update 480.8ms
  body_circles_fast stack.update 334.1ms

B8/P2/L1024 gray64 render only:
  browser_lines 30.2ms per player-perspective call
  body_circles_fast 20.8ms per player-perspective call

B8/P2 stack-shift/insert/return-copy-only:
  about 0.09-0.11ms per update
```

Plain read: on the active 704-to-64 path, stack copy is still irrelevant. The
dominant cost is gray64 rendering itself: full-resolution trail draw plus luma
and downsample. `browser_lines` remains about 1.4-1.5x slower than
`body_circles_fast` in long-trail synthetic rows, but `body_circles_fast`
remains an approximation, not a training recommendation.

Granular active-path probe after perspective reuse landed, 2026-05-12:

```text
uv run python scripts/benchmark_render_lane_microbench.py \
  --plan grid \
  --batch-sizes 8 \
  --player-counts 2 \
  --trail-lengths 0,1024,4096 \
  --bonus-counts 0 \
  --trail-render-modes browser_lines \
  --cell-kinds rgb_render_only,rgb_to_gray64_only,gray64_render_only,perspective_reuse_gray64,full_stack_update,stack_shift_insert_return_copy_only \
  --iterations 5 \
  --warmup-iterations 2 \
  --allocation-mode reuse \
  --gpu-transfer off \
  --format plain
```

Representative rows:

```text
B8/P2/L1024 browser_lines:
  rgb_render_only                  28.49ms per player-perspective RGB render
  rgb_to_gray64_only                1.05ms per frame
  gray64_render_only               29.94ms per independent player-perspective call
  perspective_reuse_gray64         23.83ms per player row
  full_stack_update               313.17ms per batch update
  stack_shift_insert_return_copy    0.08ms per batch update

B8/P2/L4096 browser_lines:
  rgb_render_only                 110.64ms per player-perspective RGB render
  rgb_to_gray64_only                1.06ms per frame
  gray64_render_only              113.40ms per independent player-perspective call
  perspective_reuse_gray64         63.00ms per player row
  full_stack_update               977.86ms per batch update
  stack_shift_insert_return_copy    0.10ms per batch update
```

Plain read: perspective reuse is real, especially as trails grow, but the
remaining big cost is still redrawing long trail history. Downsample is about
one millisecond per frame in this probe and stack movement is noise. The next
high-value target is a persistent/incremental trail layer or direct luma path,
not more FIFO-stack cleanup.

Prototype matrix after the cache/downsample/remap/library probes, 2026-05-12:

```text
incremental visual-trail cache, browser_lines, append step 16:
  L64:   parity true, full 5.03ms/frame, cached 15.29ms/frame, 0.33x
  L1024: parity true, full 22.38ms/frame, cached 17.83ms/frame, 1.26x
  L4096: parity true, full 80.62ms/frame, cached 20.69ms/frame, 3.90x

exact downsample scratch:
  baseline 1089.7us/call
  exact scratch 803.2us/call
  speedup 1.36x for the downsample bucket

safe P=2 remap/luma variants:
  best no-active-bonus exact variant:
    L0    3.20x
    L1024 1.23x
    L4096 1.04x

dependency/library feasibility:
  project uv runtime had no cv2/numba/skia/Pillow/scipy/cupy/triton/torch.
  system Pillow/SciPy probes were not production-exact or not faster.
```

Plain read: the cache is now the main promising production direction. It is
bad for very short trails because layer composition overhead is fixed, but it
improves the long-survival case we actually care about. The downsample scratch
is safe and small. The remap/luma variants are useful only as a later
low-risk short-frame optimization. Optional Modal packages should be treated as
a deliberate benchmark/deployment lane, not as a local blocker.

Tiny canonical Modal smoke on active path, 2026-05-12:

```text
run_id=opt-active-render-fullpath-wait-20260512
attempt_id=active-render-b2-sim2-it3-steps4-wait
compute=gpu-l4-t4
batch_size=2
num_simulations=2
outer_iterations=3
collect_steps_per_iteration=4
death_mode=profile_no_death
trail_render_mode=browser_lines
background_eval=false
background_gif=false
save_initial_checkpoint=false
```

Result:

```text
ok=true
model device=cuda:0
learner updated=true
surface rgb_source_frame_size=704
render_pipeline=source_state_rgb_canvas_like_raw_canvas_to_gray64
```

Instrumented collect timing:

```text
visual_stack_update_sec sum=0.411715, last=0.154739
policy_search_sec       sum=1.089014, last=0.046904
env_step_sec            sum=0.019989, last=0.006604
```

Plain read: the first policy/search call includes GPU/model warm-up
(`0.993s`), but by the final tiny iteration visual stack update is larger than
policy/search. This does not replace a steady-state profile. It does confirm
that the canonical GPU launcher is now exercising the active 704-to-64 render
surface. The run still saved a final checkpoint, so total wall time is not a
clean collect-loop timing.

Twenty-iteration no-death active-path profile, 2026-05-12, L4/T4, batch 8,
sim 4, 16 collect steps per iteration, 320 collected steps total, background
eval/GIF off, no initial checkpoint:

```text
browser_lines:
  elapsed_sec                         219.367262
  visual_stack_update_sec sum/last    191.643550 / 17.130539
  policy_search_sec sum/last            8.376071 / 0.371459
  env_step_sec sum/last                 0.961400 / 0.072292
  replay_row_build_sec sum/last         0.207597 / 0.009954

body_circles_fast:
  elapsed_sec                         177.744212
  visual_stack_update_sec sum/last    134.637206 / 11.349607
  policy_search_sec sum/last           11.985218 / 0.544842
  env_step_sec sum/last                 1.437765 / 0.105232
  replay_row_build_sec sum/last         0.296626 / 0.015302
```

Plain read: after warm-up and with death disabled, the current product
`browser_lines` visual path is the bottleneck by Amdahl's law. In this profile
visual stack update is about `95%` of the named instrumented time for
`browser_lines` and about `91%` for `body_circles_fast`. The faster trail mode
is only a comparison knob because it changes rendering semantics.

Artifact hygiene: these two profile runs were launched before marker suppression
landed, so their `show_in_gif_browser.flag` files were removed manually from the
Modal volume. Future profiling runs with `--no-background-gif-enabled` do not
write the GIF-browser marker. Default Coach runs still have background GIFs and
the marker enabled.

## Cost Model

The expensive path is roughly:

```text
batch rows * players * collect steps * active trail bodies
```

For each env row and player perspective, the current stack:

1. shifts the `[4,64,64]` FIFO stack;
2. clears/renders a full 704x704 RGB frame;
3. redraws trails from source-state body arrays;
4. draws heads and bonuses;
5. converts RGB to gray64;
6. normalizes to float32;
7. copies the full stack out.

In no-death profiles, trail bodies keep accumulating, so later iterations are
harder than early iterations. GPU warm-up is not the main story once long trails
exist.

More precise shape:

```text
B env rows * P player-perspective frames * active persisted body/trail slots
```

`trail_length` is effectively the active prefix of the vector state body arrays.
The stack updater also shifts the FIFO and returns a full stack copy. Secondary
costs to measure separately are RGB-to-gray temporaries, normalization,
policy-row observation packing, replay observation copies, and reset-row refresh
in normal-death runs.

## Optimization Menu

Do not land semantic-changing render rewrites without Environment signoff.
Exact-equivalence optimizer changes are allowed if they have parity tests and
clear fallback behavior.

1. Reduce duplicate perspective renders.
   - Landed for strict two-seat RGB palettes: render the shared trail layer
     once, remap palette pixels per perspective, then draw heads/bonuses.
   - Guarded by tests against independent renders and fallback for ambiguous
     palettes.
   - Further gray-only swaps or sprite/HUD/effect shortcuts need Environment
     signoff if they change training-visible semantics.

2. Incremental frame buffers.
   - First guarded cache has landed for `browser_lines` + non-empty
     `visual_trail_*` rows.
   - Keep per-owner trail layers, draw only appended trail segments, compose in
     browser owner draw order, then draw moving heads/bonuses fresh.
   - Dirty-block production rendering now reuses previous RGB/gray frames and
     recomposes only dirty 11x11 source blocks when cache state is supported.
   - Trail layer append now refreshes only the dirty bbox instead of rescanning
     the full 704 mask.
   - Keep expanding reset, clear, wrap, palette, active-bonus, and unsupported
     state parity coverage around the exact full-render fallback.

Fresh benchmark correction, 2026-05-12: `scripts/benchmark_render_lane_microbench.py`
now has `--trail-source visual` so it can synthesize the current
`visual_trail_*` fields instead of only the older `body_*` fallback. The helper
cells also pass the production trail cache and exact downsample scratch. Current
local visual-trail grid:

```text
full_stack_update, B8/P2/browser_lines/visual, after copy/recolor patch
  L64   110.71ms/update
  L256  162.52ms/update
  L1024 109.38ms/update
  L4096  99.25ms/update

full_stack_update, B16/P2/browser_lines/visual, after copy/recolor patch
  L64   220.26ms/update
  L256  195.29ms/update
  L1024 195.28ms/update
  L4096 199.80ms/update

isolated rgb_to_gray64 with exact scratch: about 0.80ms per call
isolated perspective reuse with cache: about 6-7ms per policy row after warmup
```

Plain read: once the cache is warm, long-trail cost is no longer proportional
to all historical trail slots in this synthetic visual-trail path. The
remaining target is fixed per-row work: layer composition, remap, head/bonus
draw, downsample, normalization, and stack copy.

Dirty-block rendering is landed and validated. It keeps the full renderer as
the exact fallback path, reuses previous RGB/gray frames for supported cache
state, and refreshes trail-layer appends by dirty bbox. Focused validation:

```text
ruff passed
uv run pytest tests/test_curvytron_two_seat_render_mode.py \
  tests/test_vector_visual_observation.py \
  tests/test_benchmark_render_lane_microbench.py -q
60 passed
```

Code files touched by the production landing:

```text
  src/curvyzero/env/vector_visual_observation.py
  src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py
  tests/test_curvytron_two_seat_render_mode.py
```

Modal profile launch correction: use `modal run --detach` for background
matrices, or `--wait-for-train` when the local session should stream the
summary. The first non-wait `.spawn` matrix printed function-call IDs but did
not produce attempt/progress files after the local app exited.

Current attached wait-mode matrix:

```text
opt-render-cache-wait-l4-b16-sim16-20260512a
opt-render-cache-wait-l4-b64-sim16-20260512a
opt-render-cache-wait-l4-b128-sim16-20260512a
opt-render-cache-wait-h100-b128-sim16-20260512a
opt-render-cache-wait-l4-b64-sim32-20260512a
```

Completed matrix summary:

```text
run                                      B    sim  wall     visual   search  replay_rows
opt-render-cache-wait-l4-b16-sim16      16   16   198.6s   136.2s   14.0s   2356
opt-render-cache-wait-l4-b64-sim16      64   16   559.4s   494.9s   21.2s   7957
opt-render-cache-wait-l4-b64-sim32      64   32   562.1s   493.5s   28.4s   7922
opt-render-cache-wait-l4-b128-sim16     128  16   1112.7s  990.8s   61.0s   15623
opt-render-cache-wait-h100-b128-sim16   128  16   978.9s   853.9s   57.8s   15611
```

Plain read: batch width improves replay rows per iteration, but render grows
enough that wall-clock replay-row throughput only improves slightly, and
physical env-step throughput gets worse. H100 at B128 is faster than L4 at B128
but still render-bound. Sim32 at B64 costs little extra wall time only because
render is already dominating; it is not evidence that search is free.

Dirty-block render landed after the matrix above. It reuses previous RGB/gray
frames, refreshes only dirty source bboxes, and redownsamples only touched
11x11 source blocks. Prototype B16/P2/L1024 geometry dirty redownsample measured
`3.59x` versus full downsample with no parity failures.

Local CPU dynamic stack profile, `browser_lines`, active production path:

```text
B16/P2/init_trails1024/bonus0: full 194.773ms, dirty  45.451ms, 4.285x
B16/P2/init_trails1024/bonus4: full 201.876ms, dirty  74.415ms, 2.713x
B32/P2/init_trails1024/bonus4: full 396.173ms, dirty 144.196ms, 2.747x
B32/P2/init_trails4096/bonus4: full 415.198ms, dirty 122.943ms, 3.377x
```

Static microbench after patch:

```text
B16/P2/L1024/bonus4 stack  56.135ms/update
B32/P2/L1024/bonus4 stack 112.522ms/update
B32/P2/L4096/bonus4 stack 140.105ms/update
```

Short canonical smoke in flight:
`opt-dirty-render-smoke-20260512 / b16-sim8-no-death`, canonical launcher,
`gpu-l4-t4`, B16, 4 iterations, collect32, updates2, sim8,
`profile_no_death`, background eval/GIF off, `--wait-for-train`.

3. Vectorized CPU renderer.
   - Batch rows and players; avoid Python loops over env rows and body slots
     where possible.
   - Add a `render_many(state, rows, players, out)` style API after semantics
     settle.
   - Consider NumPy chunking, Numba, Cython, or OpenCV/skia drawing calls.
   - Numba CPU is likely the first compiler spike to try; Cython is the more
     conservative deployment path if JIT warm-up is annoying in Modal.

4. GPU tensor renderer.
   - Keep source-state tensors, render masks/sprites into `[B,P,64,64]` on GPU,
     and feed policy/search without host round trips.
   - Highest ceiling, also highest contract and parity cost.
   - Prefer a few fused Triton/custom CUDA kernels over many tiny PyTorch scatter
     ops. Small 64x64 work is launch-overhead sensitive.
   - Only useful if CPU state copies do not force a sync every tick.

5. Sprite stamping/compositing.
   - If bonuses/effects become sprites, precompute small grayscale/RGB stamps
     and composite them with vectorized scatter/blend.
   - Needs exact rules for sprite animation, priority, and whether these are
     training-visible or inspection-only.

6. Stack and copy hygiene.
   - Consider a `uint8` ring buffer for `[B,P,4,64,64]` and normalize at the
     model boundary or on GPU.
   - Avoid materializing oldest-to-newest order except for active policy/replay
     rows.
   - Use pinned CPU batches and non-blocking transfer if observations stay CPU
     before LightZero policy/search.

8. Dependency-enabled / Modal renderer lane.
   - Local package absence is not a blocker; Modal can install benchmark images.
   - Treat OpenCV/Pillow/Skia/Numba/Triton as experiments with parity gates.
   - Web/literature read supports persistent offscreen/layer rendering first:
     MDN Canvas optimization recommends pre-rendering repeated primitives and
     layered canvases, and Pygame `LayeredDirty` is the same dirty-surface idea.
     GPU env systems such as CuLE and Isaac Gym matter if we move the whole
     sim/render/obs path onto GPU, not just downsample.

7. Static trail plus dynamic overlays.
   - Split persistent trail buffers from moving heads/bonuses.
   - Long games should update new trail segments and small dirty overlays, not
     replay thousands of historical bodies each frame.
   - This needs explicit clear/gap/wrap/reset invalidation semantics.

## Benchmark Tool

Full trainer profiles are too slow for renderer iteration. The current local
microbenchmark lives at:

```text
scripts/benchmark_render_lane_microbench.py
```

Focused test:

```text
uv run pytest tests/test_benchmark_render_lane_microbench.py -q
```

Latest focused render validation:

```text
uv run pytest tests/test_benchmark_render_lane_microbench.py \
  tests/test_curvytron_two_seat_render_mode.py \
  tests/test_vector_visual_observation.py -q
60 passed
```

The script controls:

- batch size;
- player count;
- trail length/body slots;
- render mode;
- bonus count and sprite/effect flags;
- stack depth;
- CPU/GPU transfer mode.
- float32 stack vs uint8 ring-buffer bandwidth;
- current rerender vs render-once-plus-LUT;
- full rerender vs incremental trail buffer;
- OpenCV/Pillow/Skia/Numba/Cython spikes against current NumPy;
- CPU render plus pinned async transfer into a dummy Torch conv/search;
- GPU stamp kernels, including kernel count and policy-consumption latency;
- exact-output diff on reset, clear, wrap, bonus, death, and terminal rows.

Required timing buckets:

- source-state RGB-to-gray64 render;
- RGB-to-gray;
- normalization;
- stack shift/insert/return copy;
- per-seat perspective work;
- total `SourceStateGray64Stack4.update`.

Output should be JSON so profile docs can compare runs without scraping Modal
logs.

Current gaps:

```text
GPU transfer is explicit but not implemented yet.
There is no Modal wrapper yet.
It is a synthetic render benchmark, not an Environment fidelity test.
It is not a full LightZero training-loop profile.
It now splits raw RGB draw, RGB-to-gray64, independent gray64 render, guarded
perspective reuse, full stack update, and stack-copy-only buckets.
It benchmarks the production trail cache/dirty path through stack updates, but
does not yet isolate direct-luma render, Numba/Cython/OpenCV kernels, or GPU
kernels.
```

The script should avoid LightZero/DI-engine entirely and benchmark direct
renderer APIs:

- `render_source_state_canvas_gray64`;
- `normalize_source_state_gray64`;
- `SourceStateGray64Stack4.update`;
- stack FIFO shift/copy cost;
- optional NumPy-to-Torch and host-to-GPU copy.

Core sweep inputs:

```text
--batch-sizes 1,8,32,128
--player-counts 2,3,4
--trail-lengths 0,64,256,1024,4096
--bonus-counts 0,4,20
--trail-render-modes browser_lines,body_circles_fast
--iterations 200
--warmup-iterations 20
--allocation-mode reuse|allocate
--gpu-transfer off
```

Synthetic states should be vector-runtime-shaped state dicts, not stepped envs,
so the benchmark can isolate render cost from physics/search.

## Current Recommendation

For the current stock fixed-opponent lane, use `browser_lines` as the trusted
visual surface and `body_circles_fast` as the explicit fast comparison. Use
`profile_no_death` for optimizer timing, but train with normal death and do not
use no-death runs for learning claims.

Historical custom-adapter speed guidance used `fast_gray64_direct`, B64 as the
main baseline, and small `browser_lines` sentinels. Do not copy that command
surface into stock fixed-opponent runs; translate the render comparison to
`browser_lines` versus `body_circles_fast`.

For bonus visuals: product gray64 should expose bonus type with stable per-type
grayscale circles if `bonus_type` is available. Generic bonus circles hide
policy-critical type information. `bonus64`/rich status planes are useful
diagnostics but should not become trainer input without an explicit schema/model
decision. Browser sprite art is lower priority for training than location,
radius, type, and status.

If Environment ports real bonus sprites, port the JS sprite facts, not the
browser renderer. Training render should stay deterministic and source-state
driven:

- use `images/bonus.png` as a 3x4 RGBA sheet with the JS type order;
- preserve map bonus center/radius and layer order: above trails, below heads;
- precompute grayscale stamps plus masks once, cached by `(bonus_type, radius_px)`;
- add coarse age buckets only if spawn animation is intentionally train-visible;
- avoid DOM/headless-browser rendering, per-frame resizing, and HUD stack icons
  in board pixels;
- keep type/status semantics in luma or structured planes, not only in tiny
  sprite glyph recognition at 64x64.
