# Optimizer Working Memory

Date: 2026-05-10

Status: active optimizer lane front door.

Active working memory:
[optimizer active working memory](active_working_memory_2026-05-14.md). Read
this first for the current optimizer plates, to-do list, and operating pattern.

2026-05-15 strict correction: the target launch policy observation surface is
`browser_lines + simple_symbols`. That is the semantic surface, not a hardware
claim. The current reliable backend is CPU `cpu_oracle`. The scalar
`jax_gpu` backend now reaches stock `train_muzero`, but it is slower than CPU
and fails in subprocess workers, so it is a canary only. The production
direction is a **batched** GPU observation backend or render service for this
same surface. The only deliberate policy-observation approximation is replacing
original bonus sprite art with the 12 designed simple symbols.
`body_circles_fast`, `fast_gray64_direct`, `browser_sprites`, and legacy
policy-surface bypasses are not current training/tournament surfaces. H100
learner/search compute is not the same thing as GPU rendering.

Current GPU-observation promotion gates:
[GPU observation next gates](gpu_observation_next_gates_2026-05-15.md). Plain
read: the isolated H100 renderer has exact smoke-row parity now, but stock
training stays on `cpu_oracle` until adversarial parity and the profile-only
batched observation facade prove the whole trainer-visible contract.

Current Coach-facing stock-path recommendation:
[fast stock recommendation](coach_handoff_fast_stock_recommendation_2026-05-14.md).
Superseded read: keep this as historical compute-speed advice only. Do not copy
its `body_circles_fast` launch commands into new runs.

Current plate map:
[current optimizer plate map](current_plate_map_2026-05-13.md). Read this
first when old docs disagree; it pins the current stock LightZero lane,
CPU-reference visual target, downsample questions, GPU side lane, and browser
golden-frame side lane.

Current system map:
[optimizer system architecture map](system_architecture_map_2026-05-13.md).
This is the short source for the latest Coach refactor context, live-run
boundaries, stock `train_muzero` ownership, artifact/GIF boundaries, and
optimizer questions.

Fresh stock full-loop speed picture:
[stock full-loop profile](stock_full_loop_profile_2026-05-13.md). Plain read:
the biggest measured full-loop speedup so far is wider stock LightZero
collection, not the renderer alone. C1 did about `10.8` env steps/sec, C32 did
about `153.6`, C64 did about `408.4`, and C96 did about `487.3` on the current
one-frame source-state fixed-opponent profile. The C64 H100 sim8 row was slower
than the C64 L4/T4 row, so the current sim8 shape is not worth moving to H100.
The C64 L4/T4 sim16 row did about `366.7` env steps/sec, so doubling MCTS sims
only cost about `10%` throughput in that shape. The paired C64 H100 sim16 row
did about `488.4`, so H100 may be useful when search pressure is higher even
though it was worse at sim8. Drop `body_circles_fast` as an optimizer decision
proxy; the relevant render lane is CPU `browser_lines` versus exact
GPU/compiled `browser_lines`.
The paired sim32 rows were L4/T4 `255.6` steps/sec and H100 `370.2`, so H100
helps once search is heavy, but sim32 is slower overall than sim16 in the rows
we have.
Render work remains important for long trajectories, but current full-loop
Amdahl points at collection/search/process overhead first. The CPU renderer is
the production oracle today; the intended replacement is a future batched GPU
renderer once it matches the same `browser_lines + simple_symbols` contract in
the trainer.

2026-05-15 render target truth: optimizer recommendations now target
`browser_lines + simple_symbols` policy observations in the stock LightZero
path. The current reliable backend is CPU. The GPU path worth building is
batched, not the measured scalar `jax_gpu` hook. This is not a browser-canvas
pixel claim. Browser sprites are display/reference artifacts unless a test
explicitly asks for original-sprite parity. The trusted Coach lane is stock
LightZero `--mode train` with
`env_variant=source_state_fixed_opponent`. `body_circles_fast` is historical
control/ablation evidence only, not the current target. The old
`fast_gray64_direct` name belongs to the superseded custom `two-seat-selfplay`
adapter and should not be copied into current stock-path commands. For
fixed-length local no-death render tables, use
`scripts/profile_curvytron_render_trajectory_lengths.py`. First fixed-length
tables live in
[render trajectory profile](render_trajectory_profile_2026-05-13.md). The exact
dirty/incremental cache is now wired into the stock fixed-opponent
`browser_lines` path. In local no-death env-only rollouts, cached
`browser_lines` improved from `39.1s -> 10.5s` at 500 steps and
`175.9s -> 46.9s` at 2000 steps. Keep `body_circles_fast` numbers only to
explain old comparisons or controls; do not recommend new body-circles work
unless the production policy contract explicitly changes away from
`browser_lines + simple_symbols`.
Current downsample and parity contract:
[downsample/reference fidelity](downsample_reference_fidelity_2026-05-13.md).
Plain read: the 704 RGB -> BT.601 luma -> 11x11 area average is a sane
anti-aliased path, but one grayscale channel cannot preserve arbitrary color
identity. Current 2P self/other colors are separated; future 3P/4P needs a
palette/channel decision because red and blue collapse to nearly the same luma.

2026-05-13 Amdahl guardrail: after the dirty-cache and bonus dirty-block fixes,
do not assume render is the whole bottleneck. Render remains important for
long-survival/no-death env profiles, but full stock LightZero profiles must be
used before assigning the next production bottleneck. Every optimizer profile
must name whether it includes env step, observation/render, frozen opponent,
MCTS/search, replay/sample, learner, checkpoint/eval/GIF, and artifact I/O.

2026-05-13 GPU render exploration: two isolated Modal GPU probes now exist:
`src/curvyzero/infra/modal/curvytron_gpu_render_probe.py` and
`src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`. They do not
touch live training runs or Modal volumes. Plain read: L4 device-side direct64
rendering is very fast, but host-to-device transfer is often the larger bucket.
For example, the synthetic source-state direct-gray64 probe at
`browser_lines`, `B=128`, `trail_slots=2000` measured about `5.18ms` device
render and `3.47ms` host->device transfer. A closer timing-only
`block_704_gray64` probe at `browser_lines`, `B=16`, `trail_slots=100` measured
about `6.37ms` device render and `3.17ms` host->device transfer. This is not a
fidelity claim because it still does not prove exact RGB browser-line parity.
The benchmark now has a tiny production-render comparison for the 704-style
surface. With bonuses off, `B=1`, `trail_slots=64` now differs from production
`render_source_state_canvas_gray64` on `0/4096` pixels after the
production-like pixel-coordinate and float-luma fixes. With `8` bonuses active,
it differs on `51/4096` pixels because bonus sprites are not implemented in the
prototype. `B=16`, `trail_slots=64` takes about `4.74ms` device render plus
`3.34ms` host-to-device copy on L4. Plain read: promising, but not trusted yet.
At `B=16`, `trail_slots=500`, naive full GPU redraw takes about `46.7ms` device
render plus `3.2ms` copy, so long trails still need dirty/incremental thinking.
Next step is broader production-shaped parity, bonus sprites, or a CPU-compiled
dirty-kernel route that keeps the current cache semantics. Current Amdahl read:
production should first instrument and optimize the CPU dirty/cache renderer;
GPU remains research until device-resident policy handoff or dirty GPU rendering
is proven.
Current GPU notes live in
[GPU render exploration](gpu_render_exploration_2026-05-13.md).
Current GPU backend implementation plan lives in
[GPU observation backend plan](gpu_observation_backend_plan_2026-05-15.md).
Current GPU parity-gap notes live in
[GPU render parity gap](gpu_render_parity_gap_2026-05-13.md).
Current dirty-cache component notes live in
[dirty cache component profile](render_dirty_cache_component_profile_2026-05-13.md).
Current GPU sprite research lives in
[GPU sprite render research](gpu_sprite_render_research_2026-05-13.md).

2026-05-14 bonus-symbol lane: there are exactly 12 active source-game bonus
sprites, all roughly diamond-framed icons with one internal mark. The current
policy surface uses the simplified `simple_symbols` renderer for those bonuses;
that is the one deliberate visual approximation. Keep original
`browser_sprites` for display/reference checks, not as the policy default.
Current plan:
[bonus symbol render plan](bonus_symbol_render_plan_2026-05-14.md).

2026-05-13 moving-target warning: Environment Reconstruction is still landing
source-state and renderer fidelity changes. Optimizer profiles are comparable
only when they name the code state and render semantics. If a speed number moves
unexpectedly, first check whether the environment changed underneath the
profile. Every new profile should record render mode, bonus render mode,
natural bonus spawning, death mode, trajectory length, warmup, whether
search/learner are included, and dirty-cache hit/fallback/component timing.

2026-05-13 natural-bonus clarification: `natural_bonus_spawn` means the real
source-game pickup system is active. It schedules bonus timers, samples bonus
types and positions, inserts active bonus objects, and later expires or applies
them. In the trusted visual path, active bonuses are browser-like sprite atlas
tiles alpha-blended onto the RGB canvas. A tiny local component smoke showed
about `0.13s` render/observation time over 20 no-death env steps with natural
bonuses on versus about `0.05s` with natural bonuses off. That is roughly
`6.5ms/step` vs `2.5ms/step` for local env observation rendering only, not the
full MuZero training loop. The hot bucket in that smoke was `draw_bonuses_sec`,
so active bonus sprite drawing is the next suspect to profile and optimize.

2026-05-13 bonus dirty-cache fix: the real bug was bigger than sprite drawing.
Stationary active bonuses were forcing their sprite boxes dirty every step. A
500-step local no-death `browser_lines` profile with natural bonuses went from
`162,185` dirty blocks and `4.43s` render time to `12,153` dirty blocks and
`1.54s` render time after tracking bonus slot/id/type/position/radius and only
dirtying changed bonus snapshots. The cache still expands dirty blocks when a
trail/head touches a stationary sprite, so the draw order remains correct:
trail, sprite, head. Focused RGB parity tests cover far stationary sprites,
trail-under-sprite redraw, and bonus type-only changes.

2026-05-13 renderer architecture read: the next serious renderer should be
block-local, not a full 704-frame redraw. For each dirty 64x64 output cell,
compose the exact 11x11 RGB tile in draw order, then immediately do BT.601 luma
and area average. That is the shared CPU and GPU target. A trusted GPU version
should be a Torch/CUDA custom op only if the observation can stay on GPU into
policy/search; otherwise CPU dirty/block-local work is the nearer production
win.

2026-05-12 current optimizer pivot: the trusted CurvyTron lane is stock
LightZero `train_muzero` with
`env_variant=source_state_fixed_opponent`,
`opponent_policy_kind=frozen_lightzero_checkpoint`, and the frozen opponent on
CPU by default. The custom `--mode two-seat-selfplay` path is
postmortem/profiling only until it feeds native LightZero replay/targets or
passes target parity. Start with
[stock frozen optimizer pivot](stock_frozen_optimizer_pivot_2026-05-12.md).

2026-05-12 architecture re-exploration: the current speed question is broader
than renderer micro-optimization. The live research folder is
[architecture re-exploration](architecture_reexploration_2026-05-12/README.md).
Plain read: stock LightZero is the trusted proof path, but large speedups likely
need actor/search fanout, replay chunking, and cleaner learner/checkpoint
handoff. MCTX is a possible search primitive; EfficientZero and MiniZero are
architecture references, not current migrations.

2026-05-12 latest stock profile tensor read: use `gpu-l4-t4-cpu40`,
`env_manager_type=subprocess`, and start wide runs around `collector_env_num=96`
for the stock fixed-opponent lane. C128/C160 only barely improved over C96.
MCTS is not dominant at sim8-sim16. Long trajectories are still collection/render
bound. The old C32 no-death comparison where `body_circles_fast` was about
1.75x faster than `browser_lines` is control evidence only now; it is not a
recommendation lane because it changes fidelity. Frozen checkpoint opponent
inference is a real fixed-opponent lane cost.
Current tables and Coach-facing speed recommendations live in
[profile validation results](architecture_reexploration_2026-05-12/profile_validation_results.md)
and
[Coach speed recommendations](architecture_reexploration_2026-05-12/coach_speed_recommendations.md).

2026-05-12 fresh stock-frozen profile read: learner/search can run on GPU while
the frozen checkpoint opponent stays on CPU, and subprocess collection now works
with that split. Short normal-death profiles keep scaling through C64 on the
40-CPU L4 shape. Long no-death profiles are different: rich `browser_lines`
rendering dominates env/collector time. The matched 256-step and 1,024-step
`body_circles_fast` wins are historical control rows, not current advice.

2026-05-12 fresh subprocess observability: profile-mode stock envs now emit
worker-side timing in env telemetry. A C4 browser-lines no-death validation row
spent about `20.6` worker CPU-seconds in observation/render, `9.2` in frozen
opponent action selection, and `4.6` in vector stepping over 256 env steps.
Those are worker CPU-seconds, not wall seconds; use them to explain subprocess
collector time without pretending they are serial wall time.

2026-05-12 postmortem correction: the custom `--mode two-seat-selfplay` path is
not the trusted Coach learning baseline. It is an experimental/profiling
adapter until it feeds native LightZero replay/targets or passes target parity.
Current training architecture research lives in
`docs/working/training/curvytron_architecture_research_2026-05-12/`.

Fresh stock-vs-custom control read, 2026-05-12: stock `train_muzero` runs are
healthy for `source_state_fixed_opponent` and `source_state_joint_action`.
Current trusted baseline is stock fixed-opponent with a frozen checkpoint
opponent. The matched tiny profile did not prove stock is faster than custom
two-seat, but the custom path is not trusted for learning until replay/target
semantics are fixed.

Superseded launcher truth: the old note saying Coach canonical training used
`--mode two-seat-selfplay` is no longer true for learning claims. That path is
now custom-adapter evidence. The current trusted route is stock LightZero
`train_muzero` via `--mode train` with
`env_variant=source_state_fixed_opponent`,
`opponent_policy_kind=frozen_lightzero_checkpoint`, and the frozen opponent on
CPU by default.

Historical custom two-seat render note, 2026-05-12: custom two-seat self-play
defaults to `two_seat_trail_render_mode=browser_lines`. That route renders
source-state RGB browser-style lines at 704x704, converts/downsamples to gray64,
and stacks the 64x64 policy tensor. `body_circles_fast` was an explicit speed
comparison mode; keep it historical/control only. The same two-seat runner exposes
`two_seat_death_mode=profile_no_death` for optimizer long-survival profiles
only.

Fresh render bottleneck read, 2026-05-12: long no-death profiles show render
redraw dominating policy/search once trails grow. Current research lives in
[CurvyTron render optimization research](render_optimization_research_2026-05-12.md).
The local render microbench is `scripts/benchmark_render_lane_microbench.py`;
use it for renderer iteration before launching full training profiles.
The active surface now renders 704x704 RGB and downsamples to 64x64, so older
direct-64 numbers are stale shape evidence only. Reprofile this active surface
before choosing render optimizations.

Fresh render optimization landing, 2026-05-12: the two-seat stack now reuses
one shared trail render for safe `P=2` controlled-player views and falls back
to independent renders for unsafe palettes. Focused tests pass. The new granular
microbench says stack copy and downsample are small compared with long trail
redraw; next renderer work should target incremental/static trail rendering or
direct-luma drawing.

Fresh render cache landing, 2026-05-12: the active two-seat stack now uses a
conservative `browser_lines` visual-trail cache and exact downsample scratch.
The cache is gated behind a minimum active-trail threshold because it is slower
for short trails but shows parity-preserving speedups in long synthetic append
profiles: about `1.26x` at L1024 and `3.90x` at L4096. Keep optimizing this
cache before chasing stack-copy or local-package trivia.

Historical custom-adapter notes follow. They explain the old two-seat
`fast_gray64_direct` work, not the current stock fixed-opponent render knob.

Fresh fast-direct live read, 2026-05-12: in running `fast_gray64_direct` rows,
render is small and `policy_search_sec` is the largest named bucket. Do not
confuse that with the `browser_lines` or long-survival case, where rendering can
return as the limiter. The current optimizer question is always: which render
surface and which survival length are being timed?

Fresh isolated Amdahl matrix, 2026-05-12: the current main
`fast_gray64_direct` profile is search/noise first, not render first. B64/L4/sim8
with learner on spent `14.87s` in search, `9.52s` in observation plus replay
noise, `4.16s` in visual stack, and `2.79s` in learner. Turning observation
noise off cut wall from `58.69s` to `45.61s`, which prices the CPU noise term
but is not a training recommendation. Matched no-death render sentinels still
show `browser_lines` is render-bound: `31.37s` visual for browser-lines versus
`2.59s` visual for fast-direct. Full table:
[fresh Amdahl re-exploration](fresh_amdahl_reexploration_2026-05-12.md).

Fresh isolated no-death render A/B, 2026-05-12: same B8/sim2 workload,
`browser_lines` took `53.6s` wall with `31.2s` visual stack time, while
`fast_gray64_direct` took `25.5s` wall with `2.4s` visual stack time. So yes:
rendering is still the main bottleneck in the rich long-survival regime.
Fast-direct moves the current main-training bottleneck back toward
search/noise/reset.

Fresh search instrumentation, 2026-05-12: future two-seat progress summaries can
split `policy_search_sec` into `policy_tensor_prepare_sec`,
`policy_collect_forward_sec`, `policy_output_decode_sec`, and
`policy_batch_fallback_sec`. This is profiling only; it does not change MuZero
search or training behavior.

Fresh full-loop timing correction, 2026-05-12: the collect buckets were useful
but not the whole iteration. A B64/L4/sim8 fast-direct learner-on profile took
`401.5s` for 24 iterations, and the missing wall time was likely learner/sample,
checkpoint/progress, and the old per-update model hash. The two-seat path now
adds `learner_timing_summary` and `iteration_timing_summary`. Per-update model
hash verification is opt-in with `--two-seat-verify-model-update-hash`; leave it
off for speed runs.

Fresh no-hash profile read, 2026-05-12: turning off model hashes did not by
itself show a wall-clock win. The measured learner-side cost is replay sampling
plus learner batch/target construction, about `2.9s/iteration` in the B64/L4/sim8
follow-up; actual learner forward is small, about `0.36s/iteration`. That run
identified replay/sample/target building as the next low-risk target at the
time; the replay-cache profile below supersedes it.

Fresh replay-cache profile read, 2026-05-12: the replay return-context cache is
a real low-risk win. In the same broad B64/L4/sim8 fast-direct full-loop shape,
learner-side context/sample/batch work fell to about `0.5s/iteration`, and
iteration wall before progress fell to about `18.5s` from the prior no-hash
attribution run's `22.8s`. Keep the patch. Do not claim a 10x win. The next
fast-direct bottleneck is collect/search; the next low-risk CPU target is
observation noise, then autoreset/env and visual stack. Browser-lines
long-survival render remains a separate bottleneck story.

Fresh search hot-path fix, 2026-05-12: the batched two-seat collector no longer
converts the whole LightZero output once per row. In a matched B64/sim8
collect-only profile, `policy_output_decode_sec` fell from about `143s` to about
`4s` over 12 iterations after the full lean decode patch. The hot path now keeps
only the learner-needed search record fields: selected action, visit-count
policy target, and root value. See
[continuous optimization loop](continuous_optimization_loop_2026-05-12.md).

Fresh observation-noise read, 2026-05-12: default two-seat training uses
`observation_noise_std=0.10`. In the lean B64/sim8 profile, current-observation
noise plus replay-next-observation noise cost about `65s` over 12 iterations,
nearly as much as LightZero collect forward. The implementation now generates
float32 noise directly and clips in-place. Matched B64/sim8 collect-only wall
time fell to about `170s` with default noise. A no-noise Amdahl bound was about
`129s`, but disabling noise is a Coach/training-quality decision.

Fresh reset refresh fix, 2026-05-12: short-survival random policies pay heavy
autoreset cost. The two-seat stack now has `reset_rows(...)`, so reset refresh
renders only reset rows instead of building a full-batch stack and copying a
slice. No-noise isolation cut `loop_autoreset_sec` from about `31s` to about
`14s`, with wall time moving from about `129.5s` to `125.1s`. See
[continuous optimization loop](continuous_optimization_loop_2026-05-12.md).

Fresh stack-copy fix, 2026-05-12: public stack APIs remain copy-safe, but the
two-seat hot path now calls `update(..., copy=False)` and `reset_rows(...,
copy=False)`. The no-noise B64/sim8 collect-only profile moved from about
`125.1s` to `118.4s`; default-noise profile was about `154.4s`. Keep the patch,
but do not oversell it as the big Amdahl win.

Correctness caveat, 2026-05-12: the float32 observation-noise patch keeps the
same intended Gaussian augmentation and bounds, but it is not bit-identical to
the old float64-normal-then-cast stream for fixed seeds.

Profiling artifact hygiene, 2026-05-12: default Coach runs still create the GIF
browser marker and background GIF artifacts. Optimizer profiling commands should
use `--no-background-gif-enabled`; that now also suppresses the
`show_in_gif_browser.flag` marker so profile runs do not clutter the website.
For real wide Coach matrices, leave GIFs enabled but cap them explicitly; the
default GIF max-steps value means no GIF-specific cap.

Checkpoint GIF fidelity fix, 2026-05-13: checkpoint self-play GIF workers now
always save the full `704x704` source-state RGB canvas. Older `128`/`320` GIF
requests are recorded as `requested_frame_size` only; `effective_frame_size` is
`704`. This fixes the observability path only. The trainer was already using
full RGB render -> luma/area downsample -> `[4,64,64]` stack.

This lane owns speed/training-loop setup synthesis: how CurvyTron visual
LightZero-style stacked-frame rollouts, sidecar scalar diagnostics,
policy/search, replay, reset/autoreset, Modal jobs, and learner boundaries
should be measured together.

This lane does not own environment fidelity, vector parity, or training-quality
claims. Environment claims stay in [environment active lanes](../environment/active_lanes.md).
Training and policy-quality claims stay in the [training state index](../training_state_index_2026-05-09.md).

## Start Here

- [Continuous optimization loop](continuous_optimization_loop_2026-05-12.md) -
  the active operating rule: reorient, measure, isolate, integrate only when
  evidence predicts a whole-loop win, reprofile, and keep docs current.
- [Architecture re-exploration](architecture_reexploration_2026-05-12/README.md) -
  current high-level systems investigation: LightZero stock dataflow, external
  framework patterns, actor/search/replay/learner split, and next fanout/search
  experiments.
- [Stock frozen optimizer pivot](stock_frozen_optimizer_pivot_2026-05-12.md) -
  current postmortem correction and next optimizer gates for the trusted stock
  `train_muzero` frozen-opponent lane.
- [Lane contract](lane_contract_2026-05-10.md) - separation of responsibility
  between Optimizer, Coach, and Environment/RAM reconstruction.
- [Current status](current_status_2026-05-09.md) - short optimizer read.
- [CurvyTron native LightZero profile](curvytron_native_lightzero_profile_2026-05-11.md) -
  archived/current-control source-state visual `train_muzero` timing, renderer
  fix, telemetry stride, MCTS/search read, and next bottlenecks. It is not the
  current trusted Coach route.
- [CurvyTron render optimization research](render_optimization_research_2026-05-12.md) -
  current two-seat render bottleneck evidence, cost model, and optimization
  menu while Environment Reconstruction stabilizes richer visuals.
- [CurvyTron render trajectory profile](render_trajectory_profile_2026-05-13.md) -
  current stock fixed-opponent local no-death tables. Use the `browser_lines`
  rows for full-fidelity recommendations; `body_circles_fast` rows are
  historical/control comparisons.
- [Runtime verdict](runtime_verdict_2026-05-10.md) - compact CurvyTron source
  path, CPU/GPU boundary, current profile, Modal Mctx evidence, and near-term
  architecture stance.
- [World model](world_model_2026-05-09.md) - what this project is and where
  optimizer fits.
- [Setup synthesis](setup_synthesis_2026-05-09.md) - LightZero/custom env
  setup read.
- [Framework working hypotheses](framework_decision_2026-05-09.md) - current
  base-runner hypotheses, LightZero control stance, and evidence gates.
- [Actor-loop architecture](actor_loop_architecture_2026-05-09.md) -
  framework-agnostic training-loop pieces.
- [Measurement plan](measurement_plan_2026-05-09.md) - Amdahl-style buckets,
  reports, and optimization gates.
- [MuZero loop bottleneck map](muzero_loop_bottleneck_map_2026-05-09.md) -
  actor-loop bucket map for self-play, search, replay, learner, checkpoint/eval,
  and policy freshness.
- [LightZero Modal loop](lightzero_modal_loop_2026-05-09.md) - current
  stock-ish Pong control loop, CPU/GPU split, Amdahl buckets, and disaggregation
  gates.
- [Profile report contract](profile_report_contract_2026-05-09.md) - shared
  timing and metadata report shape for repo-native and LightZero lanes.
- [Profile next steps](profile_next_steps_2026-05-09.md) - immediate report
  sequence and microbench matrix.
- [Blockers](blockers_2026-05-09.md) - what prevents production-speed or setup
  claims.
- [Profiling log](profiling_log_2026-05-09.md) - compact timing evidence and
  next measurement placeholders.
- [Questions](questions.md) - decisions this lane should answer.
- [Backlog](backlog.md) - small synthesis tasks.

## Key Inputs

- [Documentation map](../../README.md)
- [Training state index](../training_state_index_2026-05-09.md)
- [Environment active lanes](../environment/active_lanes.md)
- [Training-loop bottlenecks and Amdahl's law](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md)
- [Self-play speed lane](../environment/selfplay_speed_lane_2026-05-09.md)

## Guardrails

- Primary CurvyTron training target is visual LightZero-style stacked frames.
  Do not treat scalar-ray `[B,2,106]` rows as the main coach-facing optimizer
  target unless new evidence explicitly justifies that switch.
- Current render recommendations target full-fidelity `browser_lines`: source-state
  RGB at the 704-style canvas, BT.601 luma, and 11x11 downsample to 64x64.
  Treat `body_circles_fast` as historical/control only, not an optimization
  lane.
- CurvyTron visual profiling is non-ALE. The current trusted Coach-facing route
  is stock LightZero `train_muzero` via `--mode train` with
  `env_variant=source_state_fixed_opponent`,
  `opponent_policy_kind=frozen_lightzero_checkpoint`, and the frozen opponent
  on CPU by default. Custom `--mode two-seat-selfplay` notes are
  postmortem/profiling evidence until replay/target semantics are trusted.
  The source-state visual stack is not an ALE path and not a browser/canvas
  pixel claim. The old `debug_visual_tensor` /
  `curvyzero_debug_occupancy_gray64/v0` surface is historical smoke plumbing.
- Current scalar-ray profiling is diagnostic sidecar evidence:
  `CurvyTronSourceEnv -> [B,2,106]` rows, not Atari/ALE and not a real
  LightZero visual CurvyTron env. The rows are trainer-wrapper diagnostics, not
  native CurvyTron source objects and not the primary observation contract.
- Scalar-ray sidecar observation shape: `24` ray directions * `4` channels plus
  `10` scalars equals `106` `float32` values per ego.
- Treat speed numbers as setup/runtime evidence unless they include the actor
  loop and comparison-valid payloads.
- Every speed number must say whether it includes env step, render,
  stack/normalize, policy/search, replay, and reset.
- Treat return curves, checkpoint quality, and reproduction status as coach-lane
  inputs, not optimizer outputs.
- Treat source-fidelity claims as Environment/RAM-lane inputs, not optimizer
  outputs.
- Treat Modal/JAX/Mctx runs as boundary evidence until real CurvyTron rollouts,
  replay, final observations, and trainer contracts are wired.
- Full GPU env/obs/model/search has no known fundamental blocker, but the
  current source env is a CPU object graph; a GPU env/obs rewrite needs a new
  tensor runtime plus parity tests.
- Be precise about GPU claims. The current native LightZero GPU entrypoint puts
  model inference and learner tensors on CUDA. MCTS is LightZero-native, but it
  is a CPU/C++ tree with CUDA model calls, not a pure CUDA tree-search kernel.
  CurvyTron env reset/step/render/stack and replay storage are CPU/NumPy today.
- Do not promote GPU env work, native rewrites, distributed actors, or larger
  batches until wall-clock shares and p95/p99 action latency justify them.
- Link to source evidence; do not duplicate experiment logs here.
- For optimizer speed profiles, prefer `--mode profile` with
  `--disable-death-for-profile` when the question is steady-state long-game
  cost. Bad early policies otherwise hide env/search/replay work.
