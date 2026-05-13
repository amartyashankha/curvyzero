# Optimizer Current Status

Date: 2026-05-09

Optimizer scope: synthesize speed and training-loop setup. Do not claim
environment fidelity or policy quality from this lane.

Active operating memory now lives in
[continuous optimization loop](continuous_optimization_loop_2026-05-12.md).
The standing rule is to keep measuring and optimizing: reorient, state the
Amdahl picture, run isolated experiments, integrate only when a whole-loop win
is plausible, reprofile, and update docs.

Current reset, 2026-05-12: the trusted CurvyTron lane is stock LightZero
`train_muzero` with `env_variant=source_state_fixed_opponent`,
`opponent_policy_kind=frozen_lightzero_checkpoint`, and the env-owned frozen
opponent on CPU by default. The custom `--mode two-seat-selfplay` path below is
historical/postmortem until replay and target semantics are trusted. Start from
[stock frozen optimizer pivot](stock_frozen_optimizer_pivot_2026-05-12.md).

Fresh stock-vs-custom correction, 2026-05-12: the clean stock LightZero
`train_muzero` paths are healthy controls, but they are not clearly faster in a
matched tiny profile. Stock fixed-opponent took `21.689s` for `818` roots and
4 learner updates; stock centralized joint action took `19.261s` for `929`
roots and 4 learner updates; custom two-seat took `19.674s` for `1024`
policy/search rows and 4 learner updates. Do not compare raw `steps/s`: custom
steps are physical CurvyTron ticks. Current read: speed alone does not force a
switch to stock. The custom two-seat risk is replay/target correctness, not an
obvious throughput collapse. Details:
[stock train-MuZero vs two-seat profile plan](train_muzero_stock_vs_two_seat_profile_plan_2026-05-12.md).

Superseded pre-reset Coach handoff: the old instruction to use
`--mode two-seat-selfplay`, including the custom two-seat overnight matrix in
`coach_next_training_run_recommendations_2026-05-12.md`, is
historical/postmortem only. Do not use it as current trusted training guidance.

Fresh read-only live-run check, 2026-05-12: do not mutate the overnight Coach
runs. Read-only progress from the running fast-direct rows says the immediate
Amdahl target has moved away from rendering and toward policy/search/MCTS. In
`overnight40a` row 33, H100/B128/sim8/fast-direct at iteration 120,
`policy_search_sec` was `54.9s` versus `visual_stack_update_sec=2.25s`;
observation noise plus replay observation noise was `8.87s`, loop autoreset was
`4.63s`, and env step was `1.18s`. In row 37, H100/B256/sim8/fast-direct at
iteration 20, `policy_search_sec` was `323.5s` versus
`visual_stack_update_sec=5.95s`. In `mixpast` row 01, L4/T4/B64/sim8 with obs
noise disabled at iteration 40, `policy_search_sec` was `16.44s` versus
`visual_stack_update_sec=1.44s`. Plain read: keep render fidelity as a guardrail,
but the next speed work is search/MCTS batching and search scaling, with
observation noise/autoreset/env as secondary CPU terms.

Fresh isolated Amdahl matrix, 2026-05-12: the read-only live-run shape held up.
For the current `fast_gray64_direct` main surface, B64/L4/sim8 with learner on
spent `14.87s` in search, `9.52s` in observation plus replay noise, `4.16s` in
visual stack, and `2.79s` in learner. The matched no-death sentinel still says
rich `browser_lines` is render-bound: `31.37s` visual versus `9.01s` search.
Plain read: optimize fast-direct search/noise first; keep browser-lines render
optimization as a separate fidelity/sentinel lane.

Wave 2 collect-only scaling, 2026-05-12: default observation noise is now the
clearest low-risk Amdahl target. B64/L4/sim8 with noise on collected about `452`
replay rows/s; the matched no-noise bound collected about `753` replay rows/s.
H100/B128/sim8 improved search throughput, but noise (`58.11s`) was larger than
search (`48.22s`). Plain read: do not change MuZero or remove noise for Coach;
make the CPU augmentation cheaper, then reprofile.

Fresh replay-throughput read, 2026-05-12: compare replay rows per second, not
iteration count. Old-prefix row 01 (L4/T4, B64, sim8, fast-direct) reached
iteration 360 with `2.949M` replay rows in `10850s`, about `272` rows/s. Row 08
(L4/T4, B32, sim8) reached iteration 690 with `2.826M` replay rows in `10641s`,
about `266` rows/s. Row 09 (L4/T4, B128, sim8) reached iteration 90 with
`1.475M` replay rows in `10081s`, about `146` rows/s. Plain read: B32 and B64
are close on throughput, B64 remains the better main default because it does
more work per iteration without losing throughput, and B128 on L4 is not a good
speed default. The named late-iteration buckets still put policy/search above
render for fast-direct: row 01 had `policy_search_sec=16.41` vs
`visual_stack_update_sec=1.39`; row 09 had `79.57` vs `3.29`.

Fresh isolated long-survival render A/B, 2026-05-12: same B8/sim2 no-death
workload, `browser_lines` took `53.6s` wall with `31.2s` visual stack time;
`fast_gray64_direct` took `25.5s` wall with `2.4s` visual stack time. Plain
read: the user's reminder is right for rich long-survival rendering. Rendering
is still the Amdahl target for browser-lines/trained-long-game regimes, but not
for current fast-direct short random-policy rows.

Fresh optimizer instrumentation, 2026-05-12: future two-seat progress summaries
now split the old `policy_search_sec` bucket into `policy_tensor_prepare_sec`,
`policy_collect_forward_sec`, `policy_output_decode_sec`, and
`policy_batch_fallback_sec`. A later patch also adds
`learner_timing_summary` and `iteration_timing_summary`, and makes the expensive
per-update model hash opt-in with `--two-seat-verify-model-update-hash`. This is
profiling/telemetry only; it does not change MuZero search, action selection,
replay, or learner updates.

Fresh full-loop profile before the learner-timing/hash patch: B64/L4/sim8
fast-direct, normal death, 24 iterations, 4 learner updates/iteration took
`401.5s`. Collect timing per iteration was about `4.8s` policy/search, `2.9s`
observation plus replay noise, `1.4s` visual stack, `1.2s` autoreset, and
`0.4s` env step. The missing wall time is now the next measurement target.

Follow-up no-hash profile, same broad B64/L4/sim8 fast-direct shape, 12
iterations, took `289.4s`. This did not show a clean speedup; the codebase and
timing surface moved. It did show the next learner-side Amdahl target: replay
sampling plus learner batch/target construction cost about `2.9s/iteration`,
while actual learner forward was only about `0.36s/iteration`.

Replay-cache follow-up, same broad B64/L4/sim8 fast-direct shape, 12 iterations,
took `234.9s`. Learner-side context/sample/batch work is now about
`0.5s/iteration` instead of about `2.9s/iteration`, and iteration wall before
progress is about `18.5s/iteration`. This is a real cleanup of repeated Python
replay/target work, not an algorithm change. Current fast-direct Amdahl picture:
search/collect is the largest named term, observation noise is still material,
autoreset/env are secondary, visual stack is small. Keep browser-lines render
optimization separate because long-survival rich-render profiles are still
render-bound.

Tiny isolated Modal smoke also passed:
`opt-searchsplit-smoke-20260512 / searchsplit-smoke-20260512`, B4, sim2,
`fast_gray64_direct`, 2 iterations, no checkpoint, background eval/GIF off.
It reported `lightzero_policy_model_device=cuda:0` and the new split timers.
Warm-up was obvious: first `policy_collect_forward_sec=1.059181`, last
`policy_collect_forward_sec=0.040710`, last `policy_search_sec=0.046144`.
Therefore steady-state search profiles must ignore warm-up or run enough
iterations to amortize it.
Focused validation after the final wiring:

```text
uv run ruff check src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  src/curvyzero/env/vector_visual_observation.py \
  src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py \
  tests/test_curvytron_two_seat_render_mode.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py
All checks passed

uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_curvytron_two_seat_render_mode.py \
  tests/test_vector_visual_observation.py \
  tests/test_benchmark_render_lane_microbench.py -q
87 passed, 1 skipped
```

Superseded 2026-05-11 note: the old statement that Coach canonical CurvyZero
training used `--mode two-seat-selfplay` is no longer current. Treat that as
custom-adapter postmortem evidence; the trusted route is stock LightZero
`train_muzero` with the fixed-opponent frozen checkpoint on CPU by default.

Historical custom two-seat render-path note, 2026-05-12: custom two-seat
self-play has an explicit `two_seat_trail_render_mode` knob. Default is
`browser_lines`, and the two-seat stack now uses the full source-state
RGB-to-gray path
`render_source_state_canvas_gray64(...)` before stacking. `body_circles_fast`
is an explicit profiling comparison mode, not the default. The two-seat runner
also exposes `two_seat_death_mode`; use `profile_no_death` only for optimizer
long-survival profiling. This matters because the previous two-seat stack was
still using the older direct gray renderer while that path was still treated as
the handoff path.
Local focused validation:

```text
uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_benchmark_render_lane_microbench.py \
  tests/test_curvytron_two_seat_render_mode.py -q
33 passed, 1 skipped
```

Tiny custom two-seat Modal smokes also passed with background eval/GIF off and no
optimizer step:

```text
browser_lines: opt-two-seat-render-browser-smoke-20260512 /
  smoke-browser-sim2-c2-steps2, ok=true, cuda:0, 2 collect steps,
  visual_stack_update_sec=0.006174, policy_search_sec=0.996561
body_circles_fast: opt-two-seat-render-fast-smoke-20260512 /
  smoke-fast-sim2-c2-steps2, ok=true, cuda:0, 2 collect steps,
  visual_stack_update_sec=0.004412, policy_search_sec=0.984035
browser_lines profile_no_death: opt-two-seat-render-nodeath-smoke-20260512 /
  smoke-browser-nodeath-sim2-c2-steps2, ok=true, cuda:0, 2 collect steps,
  death_mode=profile_no_death, visual_stack_update_sec=0.006332,
  policy_search_sec=1.041592
```

Fresh 2026-05-12 long-render read: pre-stabilization long no-death two-seat
profiles now show render redraw as the clear bottleneck once trails grow. For L4/T4,
`browser_lines`, batch 16, sim 8, 10 outer iterations, 32 collect steps, the
last iteration spent about `59.1s` in `visual_stack_update_sec` versus `2.34s`
in policy/search. `body_circles_fast` reduced the same late visual bucket to
about `27.7s`, but remains an approximation mode. A speculative duplicate
two-seat render avoidance experiment reduced late L4 browser-lines visual time
to about `24.0s`; it was intentionally not kept because rich rendering is still
changing. See
[render optimization research](render_optimization_research_2026-05-12.md).
Reprofile after Environment finishes the current browser-lines trail updates.
Toy direct-render probes already confirm the shape: B16/P2 browser-lines
`stack.update` was about `2.6ms` at trail length 0, `47ms` at 64, `181ms` at
256, and `740ms` at 1024. Direct render and stack update were close at nonzero
trail lengths, so redraw dominates stack/copy for now.
The local render microbench now lives at
`scripts/benchmark_render_lane_microbench.py`; focused validation passed:

```text
uv run pytest tests/test_benchmark_render_lane_microbench.py \
  tests/test_curvytron_two_seat_render_mode.py -q
15 passed
```

Its latest small synthetic probe reports B16/P2 browser-lines stack update
around `717ms/update` at L1024 and `2714ms/update` at L4096, while
stack-shift/insert/return-copy-only is only about `0.26ms/update`.
These older local numbers are direct-64 synthetic shape evidence. They are not
current optimizer targets now that canvas-gray64 renders 704x704 RGB and
downsamples to 64x64; reprofile the active 704-to-64 surface before choosing
render optimizations.

Active 704-to-64 local probe, 2026-05-12: B8/P2/L1024 `browser_lines`
`stack.update` is about `481ms/update`; `body_circles_fast` is about
`334ms/update`; gray64 render-only is about `30.2ms` per player-perspective
call for browser-lines; stack-shift/insert/return-copy-only is only about
`0.1ms/update`. Plain read: the current bottleneck is full-resolution gray64
rendering, not FIFO stack copying.

Tiny custom two-seat Modal smoke on the same active path passed:
`opt-active-render-fullpath-wait-20260512 /
active-render-b2-sim2-it3-steps4-wait`. It used `gpu-l4-t4`, `cuda:0`,
`profile_no_death`, `browser_lines`, `batch_size=2`, `num_simulations=2`,
3 outer iterations, 4 collect steps per iteration, background eval/GIF off,
and no initial checkpoint. The surface reported `rgb_source_frame_size=704`.
Final-iteration timing was `visual_stack_update_sec=0.154739`,
`policy_search_sec=0.046904`, and `env_step_sec=0.006604`; first policy/search
included warm-up (`0.993303`). It saved a final checkpoint, so use the
instrumented buckets, not total wall time, for this smoke.

Active 20-iteration no-death profiles on the custom two-seat path also
completed, L4/T4, batch 8, sim 4, 16 collect steps per iteration, 320 collected
steps, background eval/GIF off. Product `browser_lines` took `219.37s` elapsed;
named buckets were `191.64s` visual stack update, `8.38s` policy/search,
`0.96s` env step, and `0.21s` replay row build. Approximate
`body_circles_fast` took `177.74s` elapsed; named buckets were `134.64s`
visual stack update, `11.99s` policy/search, `1.44s` env step, and `0.30s`
replay row build. Plain read: the active full-resolution render/downsample path
is currently the bottleneck; MCTS/search is not the limiter in this small sim4
long-trail profile.

2026-05-12 render optimization landing: the two-seat stack now uses
`render_source_state_canvas_gray64_player_perspectives` for `P=2`. It renders
the shared trail layer once, palette-remaps the full RGB frame for each player
perspective, draws heads/bonuses, then downsamples. Ambiguous palettes fall back
to independent renders. Focused validation passed:

```text
uv run pytest tests/test_benchmark_render_lane_microbench.py \
  tests/test_curvytron_two_seat_render_mode.py -q
11 passed
```

Fresh granular local read, B8/P2 `browser_lines`, active 704-to-64 path:

```text
L1024:
  independent gray64 render     29.94ms per player-perspective call
  perspective reuse             23.83ms per player row
  full stack update            313.17ms per batch update
  stack copy only                0.08ms per batch update

L4096:
  independent gray64 render    113.40ms per player-perspective call
  perspective reuse             63.00ms per player row
  full stack update            977.86ms per batch update
  stack copy only                0.10ms per batch update
```

Plain read: duplicate perspective redraw is partially fixed, but the remaining
Amdahl bottleneck is still long trail-history rendering. The next useful
optimizer target is an incremental/static trail layer or direct-luma renderer,
not stack FIFO/copy cleanup.

2026-05-12 follow-up render landing: the active two-seat stack now passes a
`SourceStateBrowserLineTrailLayerCache` into the `P=2` perspective renderer and
uses `SourceStateGray64DownsampleScratch` for exact luma/downsample. The cache
is conservative: only `browser_lines` rows with active `visual_trail_*` use it;
unsupported rows and unsafe changes fall back to the existing full renderer.
Production uses a minimum active-trail threshold so short early episodes do not
pay the cache overhead before it is likely to help.
Prototype timing with byte parity:

```text
visual_trail_append_L64:   0.33x, cached slower than full redraw
visual_trail_append_L1024: 1.26x
visual_trail_append_L4096: 3.90x

exact downsample scratch bucket: 1.36x
```

Focused validation after wiring:

```text
ruff passed
uv run pytest tests/test_curvytron_two_seat_render_mode.py \
  tests/test_vector_visual_observation.py \
  tests/test_benchmark_render_lane_microbench.py -q
60 passed
```

Plain read: this is not a blanket 10x yet, but it is the first production hook
that improves the long-survival render shape instead of polishing tiny buckets.
Next cache work should cut the fixed layer-composition overhead and expand
parity tests around reset, clear, wrap, palette changes, and active bonuses.

Fresh visual-trail microbench after fixing the benchmark to use current
`visual_trail_*` fields and the production cache/scratch path:

```text
B8/P2/browser_lines/visual trail full_stack_update, after copy/recolor patch
  L64:   110.71ms/update, 144.5 policy rows/s
  L256:  162.52ms/update, 98.4 policy rows/s
  L1024: 109.38ms/update, 146.2 policy rows/s
  L4096:  99.25ms/update, 161.2 policy rows/s

B16/P2/browser_lines/visual trail full_stack_update, after copy/recolor patch
  L64:   220.26ms/update, 145.2 policy rows/s
  L256:  195.29ms/update, 163.8 policy rows/s
  L1024: 195.28ms/update, 163.9 policy rows/s
  L4096: 199.80ms/update, 160.1 policy rows/s
```

Plain read: the cache changes the long-trail slope from explosive to mostly
fixed per-row composition/downsample cost in this synthetic visual-trail
benchmark. The follow-up copy/recolor patch helps the fixed cost, especially
at B16, but it does not change the high-level conclusion.

2026-05-12 dirty-block render landing: the active two-seat path now reuses
previous RGB/gray frames and recomposes only dirty 11x11 source blocks when the
cache state is supported. Reset or unsupported cache state falls back to full
render. Trail-layer append now refreshes only the dirty bbox instead of
rescanning the full 704 mask. This is exact-pixel intended and does not change
render semantics.

Prototype B16/P2/L1024 geometry dirty redownsample measured `3.59x` versus full
downsample with no parity failures. Local CPU dynamic stack profile:

```text
B16/P2/init1024/bonus0: full 194.773ms, dirty  45.451ms, 4.285x
B16/P2/init1024/bonus4: full 201.876ms, dirty  74.415ms, 2.713x
B32/P2/init1024/bonus4: full 396.173ms, dirty 144.196ms, 2.747x
B32/P2/init4096/bonus4: full 415.198ms, dirty 122.943ms, 3.377x
static microbench: B16/L1024/b4 56.135ms, B32/L1024/b4 112.522ms,
                   B32/L4096/b4 140.105ms per update
```

Short custom two-seat smoke is running:
`opt-dirty-render-smoke-20260512 / b16-sim8-no-death`, custom bridge path on
`gpu-l4-t4`, B16, 4 iterations, collect32, updates2, sim8,
`profile_no_death`, background eval/GIF off, `--wait-for-train`.

Fresh custom two-seat wait-mode matrix, 2026-05-12, `browser_lines`,
`profile_no_death`, 20 iterations, 8 collect steps per iteration, 4 learner
updates, background eval/GIF off:

```text
run                                      B    sim  wall     replay_rows  visual   search
opt-render-cache-wait-l4-b16-sim16      16   16   198.6s   2356         136.2s   14.0s
opt-render-cache-wait-l4-b64-sim16      64   16   559.4s   7957         494.9s   21.2s
opt-render-cache-wait-l4-b64-sim32      64   32   562.1s   7922         493.5s   28.4s
opt-render-cache-wait-l4-b128-sim16     128  16   1112.7s  15623        990.8s   61.0s
opt-render-cache-wait-h100-b128-sim16   128  16   978.9s   15611        853.9s   57.8s
```

Plain read: this pre-dirty matrix showed larger batches produce more replay
rows per iteration, but wall clock was still mostly render. B64 is the next
large self-play test after dirty render; B128 should wait until render is no
longer dominant. H100 and multi-GPU are not defaults until search/model
dominates.

Profiling artifact hygiene: default Coach runs still enable background GIFs and
write the `show_in_gif_browser.flag` marker. Optimizer profiling runs with
`--no-background-gif-enabled` now suppress that marker too, so profile runs do
not clutter the GIF browser website. The two pre-fix profile markers above were
removed from the Modal volume.

Superseded custom-adapter overnight recommendation, 2026-05-12: the speed-first lane was
`fast_gray64_direct`, not `body_circles_fast`. This is a strong semantic visual
approximation, not browser pixel fidelity. It preserves trail/head positions,
self/other contrast, bonus presence, and bonus type luma, but drops connected
browser-line rasterization, sprite texture, antialiasing, and exact downsample
coverage. Focused validation passed:

```text
ruff passed
uv run pytest tests/test_curvytron_two_seat_render_mode.py \
  tests/test_vector_visual_observation.py \
  tests/test_benchmark_render_lane_microbench.py -q
62 passed
```

Speed signal from custom two-seat no-death full-loop profiles:

```text
B64/L4/sim8 browser_lines:        about 768s wall, visual about 40s/iteration
B64/L4/sim8 fast_gray64_direct:   about 203s wall, visual about 2s/iteration
B128/L4/sim8 fast_gray64_direct:  about 726s wall, worse per replay row
B128/H100/sim8 fast_gray64_direct about 429s wall, useful scale probe only
```

Historical recommendation at that time: the custom adapter matrix used L4/T4,
B64, sim8, collect64, updates4, learner sample 256, accumulated replay, normal
death, sparse checkpoints, and `fast_gray64_direct`. Do not use this as the
current trusted Coach launch plan. Details remain in
[coach next training run recommendations](coach_next_training_run_recommendations_2026-05-12.md).

Fresh 2026-05-10 read: active optimizer target is CurvyTron visual, non-ALE,
wrapper-stacked debug survival profiling. The missing artifact is a bounded
`[4,64,64]` collect -> MCTS/search -> replay -> sample -> learner profile.
Scalar/ray rows and old Pong profiles are diagnostics/history unless explicitly
reopened; Pong eval-speed work is a separate optimizer side task about runtime
architecture, not CurvyTron readiness.

Fresh 2026-05-11 read: active fixed/frozen-opponent stock profile target is now
the source-state visual native LightZero path, not the old debug visual surface:
`env_variant=source_state_fixed_opponent`, non-ALE `[4,64,64]`,
`train_muzero`, fixed/frozen opponent. The latest evidence is in
[CurvyTron native LightZero profile](curvytron_native_lightzero_profile_2026-05-11.md).
Renderer/stack/obs packing was a real long-trail bottleneck and has been
reduced sharply. GPU runs are genuinely CUDA-backed (`cuda:0` model samples),
but MCTS/model inference is underbatched. Simple speed knob: raise
`collector_env_num` and `n_episode` together. In short-episode profiles,
`c4/sim16 -> c32/sim16` improved collected-step throughput about `8x`, and
`c4/sim50 -> c16/sim50` improved about `4x`. Sparse telemetry is available
through `--env-telemetry-stride`; sampled summaries are labeled. This remains
setup/speed evidence, not a learning claim and not current-policy self-play.

2026-05-11 no-death/source-default profile rerun: the long-survival profile is
now unblocked for optimizer timing. Environment fixed the old `BonusAllColor` /
`BonusSelfMaster` source-default catch/effect gap. Optimizer raised the vector
natural-bonus placement retry slab from `16` to `256` because scalar source
placement retries until it finds a free spot. Matched no-death profiles now
reach collector, LightZero MCTS/search, replay, learner, evaluator, and sparse
checkpoint hooks:

```text
c16/sim16: 3840 collected env steps, 56.86s wall, 67.5 steps/s, 5 learner calls
c32/sim16: 7680 collected env steps, 74.76s wall, 102.7 steps/s, 5 learner calls
```

Plain read: wider collectors improve search batching, but long-survival env
step/runtime work is now visible and grows almost linearly with decisions. Keep
death suppression labeled `profile_only_not_source_fidelity`; this is not a
source-fidelity claim or a learning-quality claim. The old blocker handoff is
now historical:
[environment_handoff_bonus_runtime_blocker_2026-05-11.md](environment_handoff_bonus_runtime_blocker_2026-05-11.md).

2026-05-11 env-manager optimization: CurvyTron trainer now exposes
`--env-manager-type` and defaults to `subprocess`, matching the stock
LightZero/Pong pattern. Long no-death A/B profiles:

```text
base c16: 56.86s, 3840 steps, 67.5 steps/s
sub  c16: 48.20s, 3840 steps, 79.7 steps/s
base c32: 74.76s, 7680 steps, 102.7 steps/s
sub  c32: 52.50s, 7680 steps, 146.3 steps/s
base c64: 109.47s, 15360 steps, 140.3 steps/s
sub  c64: 68.05s, 15360 steps, 225.7 steps/s
```

Plain read: subprocess is a real wall-clock win, up to about `1.6x` in the
c64/sim16 long profile. It hides detailed env method timers because envs run in
worker processes, so use `--env-manager-type base` when timing env internals.
A tiny normal train smoke with subprocess returned `ok=true` and copied
checkpoints; ignored `BrokenPipeError` lines may appear during DI-engine
subprocess teardown.

2026-05-11 train smoke: the native source-state trainer completed a tiny
`--wait-for-train` Modal run on `gpu-l4-t4`
(`opt-native-train-smoke-c16-s1121-wait`,
`train-smoke-c16-sim16-sparse-wait`). It called `train_muzero`, returned
`ok=true`, kept profiler hooks disabled, and copied `iteration_0`,
`iteration_35`, and `ckpt_best`. This proves the normal trainer path runs; it
is not a learning-quality claim.

Superseded 2026-05-11 coach-usage correction: the recommendation to use
`--mode two-seat-selfplay` is now historical. Current trusted route is stock
LightZero `train_muzero` via `--mode train`,
`env_variant=source_state_fixed_opponent`,
`opponent_policy_kind=frozen_lightzero_checkpoint`, with
`opponent_use_cuda=false` by default. Historical results from deleted wrappers
remain smoke evidence only.

MCTS/collector clarity: native single-ego runs call stock LightZero
`train_muzero`, so collector, GameBuffer, learner loop, and MuZero MCTS/search
are LightZero internals with our env/config wrapper around them. The current
two-seat self-play bridge, reached through the custom bridge path, does
not use stock `train_muzero`, the LightZero Collector, or the upstream
GameBuffer. It does use installed LightZero `MuZeroPolicy`
`collect_mode.forward`/`eval_mode.forward` and `learn_mode.forward`, but action
selection is currently one active policy row at a time. If the two-seat lane
becomes speed-critical, first profile/batch that row-wise policy/search call;
do not assume it has the same batching behavior as stock LightZero collection.

2026-05-11 MCTS/GPU recheck: current native CurvyTron profiles are
GPU-configured when launched with `--compute gpu-l4-t4` (`policy.cuda=True`),
and profiler samples see model parameters on `cuda:0`. Observed L4 utilization
is still low/sampled noisily because root/model batches are still small
relative to GPU capacity. The profiler wraps
LightZero `MuZeroMCTSCtree` / `MuZeroMCTSPtree`, so the search bucket is
LightZero-native, not a repo-owned MCTS loop. Do not jump straight to a GPU env
rewrite or framework migration. First use subprocess env manager and
`128/128` collection with `64/64` as fallback, then consider actor/search
fanout if the single-process loop remains underfed.

2026-05-11 current-lane correction: reward-credit risk does not stop optimizer
profiling. It only limits learning claims. I reran both relevant stock
LightZero paths after the coach/environment churn:

```text
fixed-opponent no-death profile:
  run_id=opt-fixed-current-reprofile-s20260511f
  attempt_id=fixed-sub-c16-sim16-steps240-matched
  env_variant=source_state_fixed_opponent
  called_train_muzero=true
  ok=true
  env_steps_collected=3840
  wall=25.21s
  MCTS=10.87s
  policy_forward_collect=13.93s
  learner=1.75s
  replay_sample=0.10s

turn-commit no-death profile:
  run_id=opt-turncommit-mainlane-reprofile-s20260511f
  attempt_id=mainlane-profile-c16-sim16-steps128
  env_variant=source_state_turn_commit
  called_train_muzero=true
  ok=true
  env_steps_collected=4096 scalar LightZero steps
  physical sampled rows=2176
  pending sampled rows=102
  wall=46.58s
  MCTS=28.66s
  learner=1.73s
  replay_sample=0.14s
```

Plain CPU/GPU split: model inference and learner are on CUDA; MCTS uses
LightZero's C++/CPU tree with CUDA neural-network calls; replay and env
internals are CPU/NumPy; subprocess env workers hide detailed env timers. The
small base-manager turn-commit profile measured env vector step at `0.63s`,
runtime `0.40s`, and render `0.24s` versus `6.25s` MCTS and `5.45s` eval, so
today's biggest immediate target is still search/collector batching, not a GPU
env rewrite. Keep no-death mode as the default optimizer profile shape for long
survival cost.

Fresh subprocess width sweep on the same current fixed-opponent path:

```text
collectors  steps   wall    steps/s  root batch  MCTS    policy collect  learner  replay
16          3840    25.21s  152.35   16.0        10.87s  13.93s          1.75s    0.10s
32          7680    39.90s  192.46   16.5        19.92s  14.20s          1.78s    0.09s
64          15360   67.51s  227.53   32.5        31.63s  25.82s          1.95s    0.16s
```

Plain read: widening still helps, but not linearly. Learner and replay stay
small. The dominant buckets are search/model/collector. The next serious
scale experiment should be searched actor chunks from a frozen checkpoint,
merged into a learner step, not more single-process micro-polish unless a finer
profile points at one exact hot path.

2026-05-11 contract-fix scale update: source-state env now emits the
LightZero-required terminal `eval_episode_return` key, so stock collector runs
reach replay buffer push/sample and learner calls cleanly with stock in-loop eval
skipped for profile. Corrected no-death dense profiles on `gpu-l4-t4-cpu40`:

```text
L4/T4+CPU40 c32/sim16:   7680 steps, 38.09s wall, 201.60 steps/s
L4/T4+CPU40 c64/sim16:  15360 steps, 50.78s wall, 302.46 steps/s
L4/T4+CPU40 c128/sim16: 30720 steps, 77.05s wall, 398.72 steps/s
L4/T4+CPU40 c256/sim16: 61440 steps, 151.74s wall, 404.91 steps/s
L4/T4+CPU40 c32/sim50:   7680 steps, 72.25s wall, 106.30 steps/s
L4/T4+CPU40 c64/sim50:  15360 steps, 90.95s wall, 168.88 steps/s
L4/T4+CPU40 c128/sim50: 30720 steps, 136.75s wall, 224.65 steps/s
CPU64 c32/sim16:         7680 steps, 74.93s wall, 102.49 steps/s
H100+CPU40 c128/sim16:  30720 steps, 56.78s wall, 540.99 steps/s
H100+CPU40 c128/sim50:  30720 steps, 127.22s wall, 241.46 steps/s
```

Plain read: the user's batch intuition is correct for this stock loop. Larger
self-play batches take longer in total but improve searched self-play
throughput by feeding larger batches into LightZero MCTS/model calls. c128 is
the single-container sweet spot so far; c256/sim16 is basically plateaued
against c128/sim16. c128/sim50 is the best tested L4/T4 serious-search
throughput. Cheap GPU beats CPU64. H100 helps c128/sim16 by about `1.36x`, but
only about `1.07x` at c128/sim50, so serious sim50 is host/search-orchestration
dominated enough that H100 should not be required by default. Use H100 when it
is convenient or for sim16 fast sweeps; L4/T4+CPU40 is fine for serious sim50
unless capacity says otherwise. Search/model/collector remain the Amdahl
target; replay buffer sampling and learner updates are small in these profiles.
These are `mode=profile`, no-death, sparse-telemetry, eval-skipped speed
profiles, not learning-quality evidence.

2026-05-11 eval-cadence correction: `evaluator_eval_sec` in the phase profile
is stock LightZero's in-loop evaluator, not the checkpoint-triggered
eval/inspection/GIF path we added. Background checkpoint eval/GIF is controlled
by `background_eval_enabled`, `background_gif_enabled`, and checkpoint saves.
Stock LightZero eval is controlled by `policy.eval_freq` and also has an
initial eval call. The trainer now exposes `--lightzero-eval-freq` and profile
mode can skip stock LightZero eval with `--skip-lightzero-eval-in-profile`
while leaving checkpoint eval/GIF as a separate sparse Coach artifact path.

Distributed-loop correction: current stock `train_muzero` is synchronous, so
there is no actor-fleet policy-staleness issue inside that loop. If we later
build coarse Modal actor fanout or continuous actors, every chunk should carry
checkpoint/policy-version metadata. That metadata is a freshness metric and an
audit guard, not a reason to avoid parallel self-play.

## 2026-05-10 Runtime Verdict

See [runtime verdict](runtime_verdict_2026-05-10.md) for the compact source of
truth.

- Observation is optimizable; it is not a dead end. First pass vectorized the
  wall-hit and hit-normalization helpers in
  `vector_trainer_observation.py` while leaving circle-hit semantics unchanged.
  Strict native `B=32,T=64` improved from `2985/s` to `5046/s`; source-backed
  circle-ray `B=8,T=64` improved from `1087/s` to `1748/s`.
- Current source-backed circle-ray path is stacked and cursor-bounded:
  benchmark source rows are padded into one vector-trainer batch and
  `source_snapshot_to_vector_trainer_state` exposes `body_write_cursor`.
  Current source matrix: `B=8,T=64` loop `0.252s`, obs `0.174s`, ray
  `0.144s`, `2035/s`; `B=16,T=64` loop `0.467s`, obs `0.324s`, ray
  `0.277s`, `2194/s`; `B=32,T=32` loop `0.403s`, obs `0.298s`, ray
  `0.255s`, `2540/s`. Observation is still the largest bucket, so the next
  useful work is dense/chunked exact circle-ray math or a compiled CPU kernel,
  not replay.
- Fresh local optimizer refresh after reviewing the latest env/coach docs:
  source-backed circle-ray rows with observation probes still put observation
  first: `B=8,T=64` loop `0.471s`, obs `0.381s`, ray `0.322s`, `1087/s`;
  `B=16,T=64` loop `0.875s`, obs `0.708s`, ray `0.603s`, `1170/s`;
  `B=32,T=32` loop `0.831s`, obs `0.706s`, ray `0.595s`, `1233/s`.
  Strict native vector no-event rows: `B=8,T=64` `2470/s`,
  `B=16,T=64` `2529/s`, `B=32,T=64` `2985/s`, `B=128,T=16`
  `3360/s`, all still dominated by public env.step/observation;
  synthetic Modal Mctx `B=64,P=2,sim=8` measured steady search `2.454ms`,
  steady H2D `0.511ms`, action D2H `0.0147ms`, app
  `ap-EkNEv5A3xDRj7QxZbmeTFe`; new native-observation Modal Mctx sample
  `curvytron_vector_trainer_sample` on app `ap-ZkCdPu0mPNrniXaQAgxDjv`
  used real strict native `[64,2,106]` observations and masks, took `0.206s`
  for env init/reset/two env steps/mapping, then steady synthetic Mctx search
  `2.330ms`, steady H2D `0.536ms`, action D2H `0.0157ms`.
  Read: source/native observation remains the
  measured CurvyTron tax; synthetic GPU search is not enough evidence for a
  full GPU env rewrite.
- Lane split: Optimizer owns setup, profiling, Amdahl, CPU/GPU, Modal, and
  process architecture; Coach owns learning/checkpoint/eval quality and
  LightZero replication status; Environment/RAM owns source truth, fidelity,
  parity, reset/final-observation contracts, and reward semantics.
- Environment reorientation: the strict public `VectorTrainerEnv1v1NoBonus`
  path is usable for `1v1/no_bonus/P=2` profiling and replay plumbing, but it
  is not full CurvyTron. Source fidelity still flows through JS/source claims,
  `CurvyTronSourceEnv`, and promoted bridge tests. Open optimizer-relevant gaps
  are row-local RNG/seed history, broad lifecycle/3P/4P runtime coverage,
  bonuses, source-faithful visual truth, and whole-loop
  policy/search/replay measurement.
- Coach reorientation: official/control LightZero Atari Pong is the current
  training-pipeline reference lane. It uses installed `LightZero==0.2.0`,
  ALE/Pong visuals, stock-ish `8` collectors, `3` evaluators, `50` MCTS sims,
  and same-run checkpoint evals. Its speed evidence points at
  evaluator/collector/env/MCTS wall time, not learner GPU bulk. Do not use Pong
  scores as CurvyTron readiness.
- Historical diagnostic: scalar-ray sidecar observation is a compact vector, not the primary visual
  target:
  `24` ray directions * `4` channels plus `10` scalars = `106` `float32`
  values per ego.
- Historical diagnostic: the old visual smoke target was non-ALE
  `debug_visual_tensor` / `curvyzero_debug_occupancy_gray64/v0`. The active
  visual trainer/profile target is the source-state stack
  `curvyzero_source_state_gray64_stack4_player_perspective/v1`.
- Historical diagnostic: the source scalar/ray path is CPU/Python/NumPy:
  `CurvyTronSourceEnv` snapshots ->
  `source_snapshot_to_vector_trainer_state` ->
  `observe_vector_1v1_egocentric_rays_v0` -> `[B,2,106]` observations plus
  `[B,2,3]` masks -> policy/search/replay profile.
- Latest source profile refresh: env step is tiny and observation/raycast
  dominates. `B=8,T=64` loop `0.392s`, env step `0.011s`, source adapter
  `0.058s`, obs `0.318s`, ray cast `0.272s`, `1306.6/s`; `B=16,T=64` loop
  `0.783s`, obs `0.632s`, ray cast `0.540s`, `1307.3/s`; `B=32,T=32` loop
  `0.680s`, obs `0.577s`, ray cast `0.490s`, `1506.7/s`. Ray casting is about
  `69-72%` of loop time.
- Native vector trainer profile now exists for strict
  `VectorTrainerEnv1v1NoBonus` plumbing only. Corrected results:
  `B=8,T=64` loop `0.259s`, public env.step `0.252s`, throughput `1980/s`;
  `B=16,T=64` loop `0.582s`, step `0.572s`, `1760/s`;
  `B=32,T=32` loop `0.640s`, step `0.635s`, `1600/s`;
  `B=128,T=64` loop `6.846s`, step `6.830s`, `1197/s`.
  Separate one-pass ray probes were `0.0035s`, `0.0069s`, `0.0124s`, and
  `0.0561s` for `B=8/16/32/128`. Read: native vector removes source adapter
  cost but remains observation/ray-bound; bigger CPU batch is not automatically
  better.
- First native batch-array observation writer is now wired into
  `VectorTrainerEnv1v1NoBonus._observe_arrays`. It validates once per batch and
  avoids per-row trainer dataclass construction, but keeps the scalar ray
  kernel. Post-patch profiles with the corrected phase probe:
  `B=8,T=64` `1993/s`, `B=16,T=64` `2375/s`, `B=32,T=64` `2241/s`,
  `B=128,T=16` `2493/s`. Read: this helps medium/large batch overhead but
  does not solve the ray-bound observation path. The `B=128` run is `T=16`
  because the longer straight-action run hit a terminal row.
- Second native ray cleanup slices body arrays by `body_write_cursor[row]`
  before trail/ray work, avoiding scans over unused fixed body-buffer tail.
  Post-slice profiles: `B=8,T=64` `3038/s`, `B=16,T=64` `2599/s`,
  `B=32,T=64` `3108/s`, `B=128,T=16` `3399/s`. Read: real speed win; still
  strict native plumbing evidence, not source fidelity.
- LightZero is not all GPU: subprocess envs, ALE/preprocessing, replay,
  checkpoint/eval, artifacts, and MCTS tree/control are CPU/host-side, while
  Torch model/learner calls use GPU when CUDA is active.
- Coach/resource context: official LightZero Pong looks slow in
  collect/eval/env/MCTS rather than learner GPU. Keep that for later resource
  profiling; it is not the current CurvyTron optimizer task.
- No fundamental blocker is known for full GPU env/obs/model/search, but the
  current source env is a CPU object graph. Full GPU means a new tensor runtime
  and parity tests. Near-term: CPU env/obs producers feeding GPU model/search,
  larger batches, and process sharding before a GPU raycaster rewrite.
- Toy bridge refresh supports process sharding as a near-term systems probe:
  serial `22063.6` env steps/s, threads `18950.4/s` (`0.813x`, bad), process
  shards `51859.3/s` (`3.579x`, `0.895` efficiency). Caveat: toy object env and
  synthetic CPU policy only, not source/Modal/MCTS production evidence.
- Retimed Modal Mctx check on L4, app `ap-u3YpTqQcqArxzFk5PI6ZbH`,
  `curvytron_trainer_flat B=64,P=2,obs=106,sim=8,hidden=64,depth=8` reported
  compile+first `5.0005s`, steady Mctx median `2.904ms`, host obs setup
  `0.622ms`, steady H2D `0.545ms`, selected-action D2H median `0.0471ms`, and
  action-weights D2H median `0.0055ms`. First action conversion was `19.14ms`,
  likely first-use/sync overhead. This synthetic boundary does not measure CPU
  ray generation or source fidelity.
- Superseded next-action note: the old instruction to keep the two-seat
  self-play launcher as the Coach canonical path is obsolete. Current trusted
  route is stock LightZero `train_muzero` via `--mode train`,
  `env_variant=source_state_fixed_opponent`,
  `opponent_policy_kind=frozen_lightzero_checkpoint`, with
  `opponent_use_cuda=false` by default. Keep scalar-ray policy/search and ray
  work as diagnostics, not the main path.

## Current Read

- Coach now reports weak movement on the stock-ish LightZero Atari Pong control:
  the completed `8192` faithful-short run moved from roughly `-13` at
  `iteration_0` to `-8` stock-ish / `-5` manual at final `iteration_3697`.
  Optimizer should treat this as a reason to profile LightZero seriously, not
  as a learning-quality claim.
- The active concrete slow loop is the installed-package LightZero Atari Pong
  faithful-short `32768` scale/accounting run on Modal:
  `train-faithful-short-installed-0.2.0-s0-32768-relpath`, app
  `ap-xiGLACKHPZLvL1eYgygqvm`. It is `gpu-l4-t4`, one Modal training function,
  one mounted `curvyzero-runs` Volume, and stock-ish settings except the
  shortened `train_muzero(max_env_step=32768)` argument.
- Local process check immediately after stopping the optimizer profile still
  showed coach-owned Modal commands for `32768-relpath` and
  `32768-ckpt1000-relpath`; optimizer did not touch them. A later local
  `pgrep` showed no matching Modal commands, so do not infer current remote job
  status from the local process table without checking Modal/artifacts.
- Full actor-loop measurement comes before isolated simulator optimization.
  See [Amdahl's law](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md).
- For the current LightZero lane, the first optimizer question is not "make the
  env faster." It is: how much wall time is setup/eval, collect/search,
  replay push/sample/target construction, learner update, checkpointing,
  artifact scan/Volume commit, and CPU wait versus GPU work?
- The current CPU/vector speed evidence is useful but narrow: fixture-backed
  rows, debug/no-event splits, debug obs/reward packing, synthetic feedback,
  and in-memory replay staging. See
  [self-play speed lane](../environment/selfplay_speed_lane_2026-05-09.md).
- Modal/JAX/Mctx runs prove dependency and boundary timing on small synthetic or
  debug-shaped roots. They are not real CurvyTron rollout throughput.
- Environment owns source parity and unsupported-case labels. See
  [environment active lanes](../environment/active_lanes.md).
- Training owns Pong, dummy Pong, eval, and checkpoint-quality claims. See
  [training state index](../training_state_index_2026-05-09.md).
- Setup synthesis now keeps LightZero as a serious replication/control lane
  while the optimizer lane prototypes an owned CurvyTron runner path. See
  [setup synthesis](setup_synthesis_2026-05-09.md).
- Historical CurvyTron interface verdict: the old primary visual hook was
  non-ALE `debug_visual_tensor` smoke/profiling. The active visual hook is now
  the source-state `[4,64,64]` native LightZero trainer. The repo-native
  source/trainer scalar-ray path remains diagnostic:
  `CurvyTronSourceEnv` snapshots ->
  `source_snapshot_to_vector_trainer_state` ->
  `observe_vector_1v1_egocentric_rays_v0` -> policy-row mapping -> replay-v0.
  That trainer row is flat rays/scalars `float32[106]` per ego plus a `bool[3]`
  action mask, arranged as all-player wrapper `[B, P]` rows with `P=2`.
  ALE is only for the official Atari Pong control lane.
- Native vector trainer verdict: `scripts/benchmark_vector_trainer_actor_loop_profile.py`
  profiles the strict `1v1/no_bonus/P=2` path with public `[B,2,106]`
  observations/masks, policy-row mapping, a tiny policy/search stand-in, and a
  replay-v0 chunk. Treat it as plumbing speed evidence; source-backed
  JS/`CurvyTronSourceEnv` remains the oracle for environment semantics.
- Framework stance is explicit but not final: owned PPO/IPPO-style runner is the
  leading repo-native optimizer bench hypothesis; LightZero remains a serious
  MuZero replication/control lane; Mctx is a later search-module hypothesis.
  See [framework working hypotheses](framework_decision_2026-05-09.md).
- The current bottleneck ranking is not knowable yet. See the
  [MuZero loop bottleneck map](muzero_loop_bottleneck_map_2026-05-09.md).

## Working Stance

- Keep the real environment on CPU first.
- Treat training quality as an input from the coach lane, not an optimizer
  deliverable.
- Measure env step, observation packing, real policy/search, transfer,
  action-unmap, replay stage/write, reset/autoreset, actor idle, learner idle,
  and policy staleness in one report.
- Use a transparent project-owned PPO/CleanRL-style baseline as a leading
  measurement hypothesis alongside the LightZero replication/control lane.
- Optimize the largest comparison-valid bucket after debug events are off and
  calibrated model/search timing is included.
- Treat replay JSON-per-row, Modal hot-loop calls, and GPU env rewrites as
  blocked until evidence changes the premise.
- Latest tiny local actor-loop scouts reinforce the same rule: with light fake
  search the P2 fixture loop is env/autoreset-heavy, but heavier fake search
  quickly becomes the top bucket. Real policy/search timing is the next needed
  measurement before env-only optimization.
- The current actor-loop bridge can write a replay-v0-shaped file, but it is
  explicitly blocked for training because it still carries debug obs/reward
  payloads (`obs_dim=9`) instead of trainer rays (`obs_dim=106`).
- The repo-native dry run does exercise trainer-shaped arrays
  (`obs[T,B,P,106]`, `mask[T,B,P,3]`), but it uses toy scalar env rows, masked
  uniform policy, and no learner.
- The profile report shape has been cut back to the useful minimum: run
  provenance, schema IDs, shapes/dtypes/checksums, denominators, timings,
  latency, integrity checks, artifacts, and caveats.
- Live optimizer-owned reports now use the same lean key names:
  `policy_search`, `latency_sec.policy_action`, `env_transitions_per_sec`,
  `ego_decisions_per_sec`, and `artifacts.report_json`.
- A narrow source snapshot adapter now exists for the next profile:
  `source_trainer_adapter.py` maps source positions/headings/alive into
  trainer-shaped `EnvState` with empty or coarse center-cell occupancy. It
  supports shape/timing probes, not source-faithful trail/body observation
  claims.
- `benchmark_source_trainer_actor_loop_profile.py` now runs a tiny source-stepped
  `[B,2,106]` profile, writes replay-v0, and read-validates replay schema. Its
  caveat is central: occupancy is approximate or empty depending on mode, and
  replay semantic validation is not done.
- The source/trainer profile is hookable today for optimizer profiling, but not
  yet for a final training claim. It covers real source stepping, trainer-shaped
  observations, action masks, rewards, policy-row mapping, and replay-v0
  shape. It still lacks exact source trail/body geometry, broad lifecycle and
  bonus coverage, a production replay handoff, real policy/search, and a
  learner.
- Source profile reports now record `occupancy_policy`,
  `occupancy_source_fields`, and `approximate_fields` so center-cell body
  occupancy does not get mistaken for exact source-faithful observation.
- In the first small sweeps, Python observation packing dominates the
  source-stepped trainer profile; increasing the tiny NumPy hidden size did not
  materially move the needle. This is a clue about the current helper, not a
  final production bottleneck verdict.
- A center-cell source body occupancy mode now exists. It is better than empty
  occupancy for profile plumbing, but still not exact source circle geometry,
  visible trail history, or own-body latency semantics.
- Base local dependencies are NumPy-only. Torch, JAX, Mctx, and LightZero are
  not importable in the default local environment, so any real framework timing
  must be a deliberate optional/Modal run or a project-owned NumPy/Torch stub
  with dependencies pinned.
- Latest mini probes: local Torch PPO learner smoke skipped because Torch is not
  importable; synthetic policy/search stand-in became search-dominated as
  simulations rose; scalar source-env scout was fast on one narrow 1v1 lifecycle
  but is not production vector-loop throughput.
- Latest cleanup validation: focused optimizer/runtime tests passed
  (`72 passed`), dry-run JSON emitted the lean report keys, source trainer
  profile emitted center-cell occupancy caveats, and the optional Torch learner
  smoke still wrote a skipped report because Torch is not importable in `uv`.
- Latest wrapper sanity from a prior sub-agent: the exact reproduction wrapper
  locally `py_compile`s, imports with Modal extras, and passes `ruff`; no source
  edits were made. That clears wrapper syntax/lint as a blocker for profiling.
- The exact reproduction wrapper now exposes opt-in phase profiling
  (`--profile-phases`, `--gpu-sample-interval-sec`) plus optimizer-only
  iteration caps (`--max-train-iter-override`) for tiny profile/control runs.
  Defaults remain off. The profiler was hardened after review: in-place method
  patching, installed-hook reporting, partial-install restore, broader
  GameBuffer class coverage, inheritance-aware method owner patching, and
  profiler-only count extraction guarded away from training failures. Local
  `py_compile` and `ruff` pass. A local fake-LightZero smoke confirms
  collector/evaluator/learner/replay hooks fire and restore cleanly, including
  inherited GameBuffer methods.
- Sub-agent wave requested on 2026-05-10: deep Modal disaggregation review,
  LightZero internals map, CPU/GPU Amdahl map, CurvyTron transfer critique,
  Modal systems critique, async/overlap critique, and profiler patch review.
  First completed reviews agree: keep the hot loop one Modal function for now,
  split only coarse eval/probe/artifact jobs, and profile before actor/learner
  or replay disaggregation.
- Optimizer-owned profile run launched then stopped:
  `train-faithful-short-installed-0.2.0-s0-2048-profile-v0`, app
  `ap-CLIw2m3bXwNbKVHItDQP33`. It reached learner iteration `1300` while still
  running, proving that `max_env_step=2048` was still too training-shaped for
  cheap phase profiling. The local Modal process was killed and the specific
  profile app was stopped. Treat this as an aborted profiler-design lesson, not
  a usable final profile report.
- Correction from that abort: for optimizer profiling, a few learner train
  calls are enough if the goal is phase shares, but stock `max_env_step` and
  `max_train_iter` controls are not reliable tiny-run caps in this path. Use
  either the explicit learner-train hook stop or a separate direct profile
  harness.
- Second correction: `max_train_iter_override=5` still did not cap the actual
  hot learner loop; the run reached `Training Iteration 600` and app
  `ap-xkTDXj5wNV8DiwVvLFpYJ9` was stopped. The wrapper now needs/has an
  explicit profiler stop in the `BaseLearner.train` hook
  (`profile_stop_after_learner_train_calls`) so the next run can produce a
  summary after a fixed number of learner train calls.
- Local fake-LightZero test now verifies that the profiler stop hook raises
  after the requested number of `BaseLearner.train` calls and restores the
  original method. Focused validation: `29 passed`; `ruff` passes.
- LightZero bypass scout verdict: the best optimizer profile path is a direct
  one-collect + replay sample + learner train harness that copies
  `train_muzero` setup but omits `Evaluator` entirely. Stock `train_muzero`
  always pays evaluator setup and an unconditional initial eval before first
  collect, so even a learner-hook stop cap still includes that startup tax.
  A bounded worker spike to implement this directly was stopped after it ran
  too long without producing a harness file; keep the task, but decompose it
  into smaller local inspection/implementation steps.
- Profile interpretation guard: do not optimize from inclusive bucket names
  alone. `collector_collect_sec` may hide Atari env stepping, preprocessing,
  policy/search, and segment construction. Branch only after comparing phase
  time with denominators, GPU samples, envstep deltas, train-iter deltas, and
  checkpoint/artifact bytes.
- LightZero Pong is now the concrete slow control loop to profile, not a reason
  to chase scores from the optimizer lane. The current stock-ish wrapper runs
  one Modal training function with one `L4`/`T4` GPU allocation, `8` CPUs,
  `32GB` memory, one mounted `curvyzero-runs` Volume, and LightZero-owned
  collector/replay/learner/evaluator internals in that container. See
  [LightZero Modal loop](lightzero_modal_loop_2026-05-09.md).
- Current LightZero disaggregation stance: keep train hot loops inside one
  container until timers prove otherwise. Split only coarse train/eval/probe/
  artifact jobs. Do not stream env steps, MCTS nodes, replay rows, or opponent
  inference through Modal Queue/Dict/function boundaries.
- Fresh Modal disaggregation review agrees with the stance: one hot Modal
  training function for now; split whole train attempts, checkpoint evals,
  checkpoint probes/diffs, artifact summaries, and manifest repair only.
  Candidate experiments: coarse split A/B for eval/artifact jobs, and an
  in-container actor/search handoff microbench before trying a Modal boundary.
- Fresh LightZero internals review says the next fine hooks, if collect/search
  dominates, are `MuZeroPolicy._forward_collect`, `_forward_eval`, runtime
  model `initial_inference`/`recurrent_inference`, `MuZeroMCTSCtree.search`,
  env-manager `step`, and selected `MuZeroGameBuffer.sample` helpers. Exact
  class owners should be discovered in-container with `inspect` because
  `lzero`/`ding` are not importable locally.
- Discovery-only metadata for those LightZero deep-hook candidates is now
  implemented behind `--profile-phases`: `phase_profile.candidate_hooks` and
  `phase_profile.deep_hook_discovery_notes`. It imports/inspects candidate
  owners but does not patch or instantiate them. The already-running
  `iter5` profile was launched before this patch, so it will not include these
  fields.
- CurvyTron no-train hookup scout passed locally with center-cell source body
  occupancy: `[T,B,P,106]` trainer observations, `[T,B,P]` actions/rewards,
  replay-v0 write/read, no terminal-mask integrity failures, and about `1,037`
  env transitions/sec on a tiny `B=2,T=8` run. Observation packing dominated the
  useful loop time. This proves plumbing, not source-faithful training.
- Larger local CurvyTron no-train profile `B=16,T=64` reinforced the same
  bottleneck shape: `1024` env transitions, loop `0.778s`, observation packing
  `0.733s`, source env step `0.013s`, policy forward `0.0028s`. In this
  plumbing lane, optimize/split trainer ray observation packing before env-step
  speed, unless real policy/search later changes the Amdahl picture.
- First optimizer patch landed on that bottleneck: empty-occupancy fast path in
  `_cast_rays` skips trail occupancy lookups when occupancy is all zero. Focused
  tests passed. Worker timing: empty observation call about `237us`; local
  `B=16,T=64` center-cell/warmup-0 profile improved to observation packing
  `0.604s`, loop `0.649s`. This helps the current empty/no-body plumbing case;
  exact trail/body geometry still needs separate work.
- Second narrow observation patch precomputes the fixed ray angle sin/cos table
  and builds all ray directions per ego heading at once. Focused validation
  still passes (`29 passed`, `ruff`, `py_compile`). A fresh local
  `B=16,T=64` source/trainer no-train profile reported loop `0.641s`,
  observation packing `0.594s`, env step `0.013s`, and policy `0.004s`
  (`/private/tmp/curvy-source-trainer-b16-t64-raydir-scout`). This is a small
  improvement, not a final bottleneck verdict.
- Self-critique/caveat: current `source_world_bodies_center_cell_v0` profiles
  are still mostly empty-body plumbing. A direct check after source warmup and
  a few steps showed `world_bodies_snapshot()` length `0`, so these numbers
  prove the current observation helper dominates the current profile, not that
  source-faithful trail/body observation is cheap. Exact trail/body occupancy
  may make observation packing worse and must be profiled separately.
- The source/trainer profile report now records `source_body_trail` counts:
  source world body count, adapted occupancy nonzero cell count, per-player
  occupied-cell counts, and nonempty sample counts. Smoke run
  `/private/tmp/curvy-source-trainer-body-count-smoke` confirmed the current
  center-cell profile had `0` nonempty body/occupancy samples, so the caveat is
  machine-visible instead of buried in prose.
- Root cause found: the default benchmark reached `game:start` but did not fire
  the delayed source PrintManager trail-start timer, so the rare nonempty
  samples were death artifacts, not steady trail bodies. The profile now exposes
  `source_setup_mode=controlled_trail`, which force-places two live avatars in
  safe lanes and fires `trail_start_delay_ms` before the measured loop. This is
  explicitly a body/trail observation benchmark, not natural reset/spawn
  evidence.
- First controlled trail/body profile:
  `/private/tmp/curvy-source-trainer-b2-t64-controlled-trail`,
  `B=2,T=64`, center-cell body occupancy. All `256` pre/post samples had
  nonempty source bodies/occupancy; mean `world_bodies_count=22.67`, mean
  adapted occupied cells `20.38`, max occupied cells `38`. Loop `0.351s`,
  observation packing `0.331s`, env step `0.0034s`, policy `0.0017s`.
  This is the strongest current CurvyTron optimizer signal: with nonempty
  approximate body/trail occupancy and no real search/learner, observation
  packing dominates.
- Observation phase profiling confirmed the hot sub-bucket: `ray_cast_sec`
  was `0.318s` out of `0.330s` observation packing on the controlled trail
  run. A narrow vectorized center-hit patch replaced the per-cell Python loop in
  trail/body ray hits; a second patch batches all rays against center arrays at
  once. Clean post-patch run
  `/private/tmp/curvy-source-trainer-b2-t64-controlled-trail-raybatch-clean`
  improved loop `0.351s -> 0.105s`, observation packing `0.331s -> 0.086s`,
  and throughput `364 -> 1,220` env transitions/sec. Phase timing after the
  patch reported `ray_cast_sec=0.078s`. This is a real local speed win on the
  nonempty center-cell trail/body profile, but still not a production training
  verdict because geometry is approximate and policy/search/learner are absent.
- Larger controlled-trail check `B=8,T=64` kept the same shape:
  `/private/tmp/curvy-source-trainer-b8-t64-controlled-trail-raybatch-clean`
  had all `1024` body/occupancy samples nonempty, loop `0.432s`, observation
  packing `0.363s`, env step `0.012s`, and throughput `1,186` env
  transitions/sec. Phase run showed `ray_cast_sec=0.296s` and
  `source_adapter_sec=0.049s`. Read: the local bottleneck still scales with the
  scalar observation path, so the next optimizer branch should be a batched
  two-ego/source-observation writer or calibrated real policy/search timing,
  not more fake environment work.
- The batched two-ego observation writer landed in
  `trainer_observation.py`. It validates once, shares the simple ray context,
  writes both ego rows directly, and keeps scalar parity/copy semantics. Focused
  validation passed (`31 passed`, `ruff`, `py_compile`). It cleared the stop
  rule on larger controlled-trail profiles: `B=8,T=64` improved loop
  `0.432s -> 0.357s` and observation packing `0.363s -> 0.296s`; `B=32,T=16`
  improved loop `0.410s -> 0.328s` and observation packing
  `0.367s -> 0.288s`. This is worth keeping, but observation packing remains
  the top no-train bucket.
- Matched CPU policy/search overlay did not overturn that priority. For the
  same `1024` policy rows, fake NumPy search at `32` simulations took
  `0.036s` at `B=8,T=64` and `0.0146s` at `B=32,T=16`, while source/trainer
  observation packing was `0.327s` and `0.274s` respectively in phase runs.
  This is only a CPU proxy, not Mctx/LightZero/GPU, but it says current
  source/trainer profiling should still prioritize observation geometry and
  source-faithful body/trail representation before env-step or replay polish.
- Separation of responsibility is now explicit in
  [optimizer lane contract](lane_contract_2026-05-10.md): Optimizer owns setup,
  measurement, Amdahl reads, and speed architecture; Coach owns learning and
  checkpoint/eval claims; Environment/RAM reconstruction owns source truth and
  fidelity labels.
- Source body-circle CurvyTron profiling is now the preferred local optimizer
  bench over center-cell occupancy. The path is
  `CurvyTronSourceEnv` snapshots plus `world_bodies_snapshot()` and
  `avatar_body_metadata_snapshot()` ->
  `source_snapshot_to_vector_trainer_state(...)` ->
  `observe_vector_1v1_egocentric_rays_v0(...)` -> `[B,2,106]` trainer rows.
  It is still not Atari/ALE, not a real LightZero env, and not a
  browser-visible-trail or bonus-geometry claim.
- Circle-ray profiling now exposes observation sub-timers. Latest short source
  refresh: `B=8,T=64` loop `0.392s`, observation `0.318s`, ray cast `0.272s`,
  throughput `1,306.6/s`; `B=16,T=64` loop `0.783s`, observation `0.632s`,
  ray cast `0.540s`, throughput `1,307.3/s`; `B=32,T=32` loop `0.680s`,
  observation `0.577s`, ray cast `0.490s`, throughput `1,506.7/s`. Ray casting
  remains the main measured sub-bucket at about `69-72%` of loop time.
- Native vector trainer path validation landed alongside the profile script:
  reset now scales the warmup timer callback cap for larger `B`, a `B=128`
  reset regression exists, and focused validation passed:
  `pytest tests/test_benchmark_vector_trainer_actor_loop_profile.py tests/test_vector_trainer_env.py -q`
  -> `14 passed`; `ruff` passed for the new script, tests, and env files.
- Post-review native vector trainer profile patch was report-shape/metadata
  correctness, not a new speed conclusion. The report now carries
  `optimizer_profile_schema/status`, `run.debug_event_mode`, explicit timing
  notes that `env_step_public` includes observation/mask/reward/done packing,
  real masked-action violations checked against pre-step legal masks, separate
  `selected_action_positive_weight_violations`, the straight-action mixed
  fallback fix, and nonnegative seed validation. Focused validation passed:
  `ruff` for the script/test, `pytest tests/test_benchmark_vector_trainer_actor_loop_profile.py -q`
  -> `3 passed`, and `py_compile`.
- Batch-array observation writer patch landed after that report cleanup:
  `observe_vector_1v1_egocentric_rays_batch_arrays_v0` now builds
  `[B,2,106]` observations and masks for the public env path while the scalar
  observer remains the oracle. Parity tests cover mixed live/terminal,
  borderless, and own-trail-latency cases. Focused validation:
  `ruff` passed for touched env/script/tests and
  `pytest tests/test_vector_trainer_observation.py tests/test_vector_trainer_env.py tests/test_benchmark_vector_trainer_actor_loop_profile.py -q`
  -> `31 passed`.
- Body cursor slicing patch landed next: observation ray code now uses
  `body_write_cursor[row]` as the native fixed-buffer bound, with fallback for
  source-adapter states lacking that key. Added a regression that active junk
  beyond the cursor is ignored. Focused validation:
  `ruff` passed for touched files and the same focused pytest target returned
  `32 passed`.

## Caveats

- No full CurvyTron training-speed proof yet.
- No GPU-env recommendation yet.
- No training-quality claim from any speed scout.
- No environment-fidelity claim from optimizer notes.
- No claim that current CurvyTron is using Atari/ALE. Current primary visual
  optimizer work is the debug occupancy tensor; source/trainer `[B,2,106]`
  remains scalar-ray diagnostic work. Official Atari Pong remains a separate
  LightZero control.
- No claim that strict native vector `1v1/no_bonus` is the environment oracle.
  It is optimizer plumbing evidence; source-backed JS/`CurvyTronSourceEnv`
  remains the oracle boundary.
- No claim that LightZero is rejected. It remains a serious
  replication/control lane unless the coach lane records a credible
  reproduction outcome or a decisive blocker.
