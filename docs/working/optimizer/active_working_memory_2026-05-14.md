# Optimizer Active Working Memory

Date: 2026-05-14

Purpose: short live state for the optimizer lane. Read this before older
optimizer notes if the docs disagree.

## High-Level Goal

Make real CurvyTron training faster without changing what MuZero is learning.

That means:

- optimize the trusted stock LightZero training lane, not old custom adapters;
- preserve policy targets, value/reward targets, MCTS search semantics, player
  perspective, reset/death/autoreset behavior, RND semantics, and tournament
  observation contracts;
- use profile-only probes to find and price bottlenecks before changing
  training defaults;
- promote a speed path only after it passes semantic/parity gates and a matched
  full-loop profile.

Current optimizer role in one sentence:

```text
Find the next Amdahl wall, remove it with the smallest semantics-preserving
boundary change, then reconnect the result to stock train_muzero profiles.
```

2026-05-23 latest correction:

```text
The compact Torch search-service boundary now runs remotely and passes compact
replay proof, but the eager Torch tree body is not the big breakthrough.
```

Current numbers on H100 B512/A16/sim16:

```text
direct_ctree_gpu_latent:       about 4.0k-5.0k steps/sec in fresh pairs
compact_torch_search_service: about 5.1k-5.7k steps/sec
service_tax_probe:            about 5.9k steps/sec in the same short pair
```

Plain read:

```text
Keep Coach on stock train_muzero/source_state_fixed_opponent/cpu_oracle.
Keep compact_torch_search_service profile-only. The next optimizer move is not
more wrapper polishing; it is a genuinely compiled/fused fixed-shape search body
or an array-native MCTX/JAX-style comparator behind CompactSearchResultV1.
```

2026-05-21 latest correction:

```text
Renderer work is not the current main wall in the active profile shape.
The public LightZero collect/MCTS branch is.
```

2026-05-22 latest correction:

```text
More CPUs are not the fix. CPU64 made the current H100 search-boundary rows
slower, so park the CPU lane unless a future timer shows a different
CPU-parallel section.
```

The active next target is train-facing, not another sidecar record:

```text
collect_search_backend=direct_ctree_gpu_latent
```

This is now implemented as a profile-only hook in the stock CurvyTron launcher.
It patches LightZero `MuZeroPolicy._forward_collect` during `train_muzero`,
keeps stock collector/replay/target/learner ownership, and returns stock
per-env collect outputs. It is not enabled in train mode and not yet a Coach
speed claim. The next proof is a matched full-loop A/B against
`collect_search_backend=stock`.

2026-05-22 compact sidecar correction:

```text
The active architecture boundary is now HybridCompactBatch, not a naked
obs+mask toy probe.
```

The compact sidecar includes observation, action mask, row/player ids, reward,
target reward, done roots, final/autoreset/terminal facts, `to_play`, and
`active_root_mask`. The profile-only direct CTree arrays hook can now consume
that sidecar directly and passed a corrected Modal wiring smoke on L4/T4 with
the persistent policy framebuffer and `direct_gray64`. This is still not a
training default and not Coach advice. RND latest-frame extraction now has a
compact-sidecar input proof with scalar timestep materialization off.

2026-05-22 update: replay/target-shaped compact consumption now has a local
adapter proof. Active-root ordered `selected_action`, `visit_policy`, and
`root_value` arrays can become checked `PolicyRowRecordV0` records and pass
through `build_source_state_multiplayer_target_rows_v0`. The next proof is
matched full-loop A/B after the compact gates pass, with an optional combined
direct-CTree-output-to-target-row profile row if that edge becomes uncertain.

Follow-up: the combined local edge now passes too:

```text
HybridCompactBatch
-> direct CTree compact profile hook
-> compact action/visit/value arrays
-> target-row adapter
-> checked source-state target rows
```

The optimizer lane should now get more aggressive. Correctness scaffolding is
good enough for the next falsifier. Build a closed compact batch consumer with
search output, RND latest-frame input, and target rows in one compact profile
loop. If that cannot plausibly produce a `3x` class profile win over current
direct, stop polishing direct CTree wrappers and move to the search-service or
native/vector buffer architecture.

2026-05-22 follow-up: the first closed compact falsifier moved the local
denominator in the right direction. Compact RND latest-frame extraction now
slices the latest channel before normalization instead of normalizing the full
four-frame stack. B512/A16 closed compact arrays + native actor buffer improved
from about `26.5k` to about `57.9k-62.8k` timesteps/sec; B2048/A16 reached
about `71.6k` timesteps/sec versus an `80.4k` native-vector mock ceiling. Plain read:
compact sidecar/RND/target overhead is not the local toy wall anymore. The
real 5-10x work is still the LightZero collect/search/replay object boundary.

Fresh H100 rows say initial model inference is fast, pure-policy collect is much
faster than MCTS collect, and raw ctree traverse/backpropagate is small. The
large buckets are root setup, CPU/list conversion, MCTS wrapper/result handling,
and per-root output fanout around `collect_mode.forward`.

Completed ceiling task: profile-only replacement-ceiling toy:

```text
pre-scalar uint8 [B,2,4,64,64]
-> real scratch MuZero model calls
-> compact action/value/policy arrays out
```

This was not a trainer change, not real MCTS, and not Coach launch advice. It
was a ceiling test to decide whether replacing the public MCTS branch boundary
is worth designing.

Result:

```text
H100 public MCTS collect sim8:       ~2572 roots/sec
H100 public pure-policy collect:     ~6287 roots/sec
H100 array ceiling recurrent_toy:    ~8681 roots/sec
H100 array ceiling policy_arrays:    ~9958 roots/sec
```

Plain read: yes, this is worth pursuing. The active optimizer task is now
semantics-preserving compact arrays-in / arrays-out MCTS boundary design, plus
fixed-seed parity gates against stock LightZero. The toy is not MCTS. It proves
model-call pressure is not the main wall; it does not prove a replacement can
skip PUCT, legal-mask semantics, root noise, temperature, value/reward
transforms, visit counts, or replay target semantics.

2026-05-21 update: the first profile-only compact arrays facade is implemented
and smoke-tested in Modal. It still calls stock LightZero
`collect_mode.forward`, then decodes compact action, visit, and searched-value
arrays. This is a boundary/wiring proof, not a speed win yet. The local optional
dependency extra is now:

```text
uv run --extra lightzero ...
```

with `LightZero==0.2.0` and `torch==2.8.0`.

Medium facade rows also passed:

```text
L4/T4 B512/A16/sim8: 1421.28 scalar roots/sec
H100  B512/A16/sim8: 2319.65 scalar roots/sec
```

Plain read: this gives us a useful timing anchor for the compact boundary, but
because the facade still calls public LightZero MCTS, it confirms the same wall
rather than removing it.

2026-05-21 later update: the first direct compact arrays profile probe is now
implemented. It calls real LightZero MuZero `initial_inference`, real
`policy._mcts_collect.search` / CTree MCTS, then returns compact arrays without
calling public `collect_mode.forward`. It is still profile-only and does not
change trainer defaults.

Matched medium rows:

```text
H100 stock facade:                       2419.81 roots/sec
H100 direct_ctree_arrays:                2806.64 roots/sec
H100 direct_ctree_arrays old fastpath:   3859.44 roots/sec
H100 direct_ctree_arrays host_uint8:     5247.95 roots/sec
H100 direct_ctree_arrays pinned_uint8:   4678.23 roots/sec
H100 direct_ctree_arrays resident reuse: 5820.96 roots/sec
L4/T4 direct_ctree_arrays:               1460.41 roots/sec
```

Plain read: direct CTree arrays is a real speed signal, but not a production
change. The first direct row exposed output assembly as a silly local wall
(`4.71s` over 25 H100 steps). The all-actions-legal fast path cut that to
`0.027s` and moved the wall back to MCTS search/root prep/model/output plus
observation/stack. Input transfer is now priced: pinned lowered the H2D bucket
but did not beat host uint8 in the matched short total-wall row; resident reuse
is only an upper bound because it reuses stale input. Keep the exact parity
gates before this touches training.

Longer repeat correction:

```text
H100 direct_ctree_arrays host_uint8, 60/15:   4111.80 roots/sec
H100 direct_ctree_arrays pinned_uint8, 60/15: 4513.15 roots/sec
H100 direct_ctree_arrays resident reuse, 60/15: 5537.40 roots/sec
```

Plain read: the short host-vs-pinned row was noisy. In the longer row, pinned
input is a real modest total-wall win, about `1.10x` over host uint8, because
H2D falls from about `1.21s` to about `0.14s`. It is still not the main win:
search/root-prep/model/output and ordinary observation remain larger buckets.
Resident reuse is faster, but it deliberately reuses stale input and is only an
upper bound, never a training mode.

Fresh current-telemetry refresh:

```text
H100 stock facade host_uint8, 60/15:             2473.11 roots/sec
H100 direct_ctree_arrays fresh host_uint8, 60/15: 4564.03 roots/sec
H100 direct_ctree_arrays pinned_uint8, 60/15:    4113.52 roots/sec
H100 direct_ctree_arrays resident reuse, 60/15:  4884.69 roots/sec
```

Plain read: this refresh corrects the input-copy story. Direct CTree over the
stock facade is still a real profile-only win, about `1.85x`. But pinned input
did not win total wall here, and stale resident reuse is only modestly above
fresh host input. The current Amdahl wall is not mostly H2D; it is the remaining
real CTree search/root-prep/model-output path plus observation/stack work and
runtime variance.

Instrumentation correction now in code: future direct CTree rows report
`input_freshness`, `input_transfer_bytes`, and model-output GPU-to-CPU timing
(`lightzero_mcts_arrays_boundary_model_output_d2h_sec/bytes`). Older Modal rows
above predate some of those fields, so use them for throughput and broad splits,
not for a complete transfer accounting table.

Completed side lane: the toy H2D question has been priced. Pinned `uint8` is a
real win in the array-ceiling recurrent-toy row, and a modest win in the longer
direct CTree row. Resident Torch reuse is only a stale-input ceiling. The
immediate optimizer lane is not more input-mode guessing; it is direct CTree
fixed-seed parity plus a deeper split of the remaining direct path wall.

2026-05-21 parity correction: the first local direct-CTree parity gate is now
honest but incomplete. The stock-facade decoder now handles two important
LightZero output shapes: full action-space visits and legal-action-only visits.
It also zeroes illegal visit mass before normalizing. Real CPU LightZero tests
now compare stock facade versus direct CTree arrays for sim1/sim2/sim8 on
searched values, visit shape/normalization, and illegal-action count.
A separate biased-logit test makes action 1 the clear winner and verifies both
paths choose that top action at sim8. This is a simple tie-break sanity gate,
not full visit-distribution parity.
Additional deterministic-mask tests now pass: single-legal-action rows produce
exact one-hot visits in both paths, and masked biased-logit rows choose the best
legal fallback with zero illegal visit mass.
The checked multiplayer target-row bridge also accepts compact MCTS visit
distributions as non-one-hot `policy_target` rows and preserves root
value/source metadata. That proves the compact arrays output fits this
repo-owned replay target shape; it still does not prove native LightZero
GameSegment/full trainer integration.

Promotion caveat:

```text
Exact neutral/tie-heavy visit parity is not the production ruler.
```

Even with fixed Python/NumPy/Torch seeds and root noise disabled, repeated real
CTree calls can produce different action/visit tie-breaks in neutral-logit
rows. Clear-choice and forced-mask cases now pass, and those should stay exact.
Neutral rows are still useful to catch obvious bugs, but the current promotion
contract is: exact forced-mask and clear-preference checks, schema/target-row
compatibility, stochastic/statistical collect-row comparison, and a matched
full-loop profile. `direct_ctree_arrays` remains profile-only until that
contract passes.

2026-05-21 H2D split update: the profile-only array-ceiling input modes are
implemented and tested. H100/B512/A16/sim8 recurrent-toy rows:

```text
host_uint8:             10086.23 roots/sec
host_uint8_pinned:      12295.15 roots/sec
host_float32 corrected:  9641.80 roots/sec
resident_torch_reuse:   14414.56 roots/sec
```

Plain read: pinned `uint8` copy is a small real win, resident input is the
ceiling, and host-side float32 preprocessing is worse once its CPU time is
counted. This should inform the compact MCTS-boundary design; it is not trainer
launch advice by itself.

2026-05-20 current focus:
[batched GPU full-loop reorientation](batched_gpu_full_loop_reorientation_2026-05-20/README.md).
The optimizer lane is now split into two linked questions:

1. Can the profile-only batched GPU observation boundary become a real
   trainer-visible vector facade without breaking player view, stack, reset,
   final-observation, tournament, or RND semantics?
2. If render becomes cheaper, what host/process/search/replay/RND bucket becomes
   the next full-loop bottleneck?

Keep scalar `jax_gpu` out of defaults. Use `float64 + exact` as a debug guard
and `float32 + tolerant` as the aggressive learned-observation speed candidate.

Historical renderer note: the renderer work below was real and useful, but it
is no longer the main active Amdahl wall in the current profile shape. Keep it
as background unless fresh long-trajectory rows make observation construction
dominant again.

2026-05-20 latest result: the trainer-visible renderer-backed surface canary
now has an approximate `direct_gray64` GPU surface. It is profile-only, not
stock full-loop integration, but it is a major local speed signal. The direct
simple-symbol blocker is fixed: all 12 bonus symbols remain distinct, bonuses
overwrite trails, heads overwrite bonuses, and an H100 adversarial two-view
canary matched the CPU-direct oracle exactly.

```text
H100 B64 steps256 surface canary
CPU dirty-cache surface:      0.237s median step
GPU block_704_gray64 surface: 0.144s median step
GPU direct_gray64 surface:    0.0339s median step
```

Plain read: dense source-resolution GPU rendering was the local observation
wall. `direct_gray64` is not browser-pixel parity; it is an approximate
learned-observation candidate that now preserves the simple-symbol contract in
the direct path. A separate real stock full-loop RND profile also passed on
H100 C32/sim8 with `rnd_meter_v0`: `16,384` env steps, `512` MCTS searches,
`12` learner calls, about `457 steps/s`, and max GPU utilization around `17%`.
That full loop still uses `cpu_oracle` observations.

2026-05-20 follow-up: the next canary is now started locally, not promoted.
`BatchedSourceStateTrainerProfileLoop` keeps
`SourceStateMultiplayerTrainerSurface` batched through reset/step and only
materializes LightZero-shaped scalar rows at the outer profiling boundary. New
tests cover row/player order, both seats, stack FIFO, and fail-closed dynamic
renderer requests. Remaining gates before any trainer/default change:
terminal/final observation, partial autoreset, missing/extra action handling,
RND latest-frame extraction through this wrapper, and a matched full-loop A/B.
The next matched profile grid is recorded in
`batched_gpu_full_loop_reorientation_2026-05-20/next_experiment_grid.md`.

Critique fix in the Modal surface-facade profiler: payload timing now uses the
surface's live policy rows instead of flattening every seat, RND metrics are
returned, and terminal rows are reset after measurement. This makes the next
direct-GPU surface profile more trustworthy, but it still is not a stock
training default.

Direct scale rows changed the Amdahl read: H100 `direct_gray64` B512 has
surface step median `0.225s`, renderer median `0.0484s`, and device render
median `0.0141s` for `1024` policy rows. So the direct renderer itself is no
longer the only wall at large batch. Next local targets are non-render surface
work, compact packing, stack copy/update, payload/pickle/process overhead, and
then the real full-loop MCTS/RND split.

Instrumented B512 rerun sharpened that: surface total `0.254s`, renderer
`0.0523s`, env step `0.0211s`, stack update `0.104s`, surface package/copy
`0.126s`, payload pickle outside the surface step `0.0265s`, payload size
`67.1MB`. The optimizer should now look at host-side materialization and data
layout before spending more effort on the direct render kernel.

Host-copy fix result: renderer-backed B512 direct surface total dropped to
`0.143s` after avoiding redundant full-observation/final-observation copies in
the profile path. Package/copy fell from `0.126s` to `0.00038s`. This is a
real `~1.8x` local surface improvement. The payload is still `67.1MB` per
step, so dtype/layout/payload strategy remains the next big local question.
Follow-up stack `copy=False` in the renderer-backed profile path dropped the
B512 surface step again to `0.123s`. Current read: the cheap wins were host-side
copies, and the next profile targets are renderer/pack and the remaining stack
update path. `stack_update_sec` includes renderer time.

RND reorientation: keep it modular and separate from render optimization.
Direct-surface RND cadence microprofiles show the cadence dominates cost. On
H100 B128 direct-surface profile rows, CPU RND with `update_per_collect=100`
spent about `7.43s` training per step; CUDA RND with cuDNN disabled spent about
`2.39s` at update100, `0.258s` at update10, and `0.0267s` at update1. This is a
training/RND knob, not a renderer result. Positive-weight RND still needs the
normalization/cap decision before it becomes launch advice.

Current Amdahl read: direct GPU rendering is valuable, but after the latest
copy fixes the next likely 10x path is preserving large vector batches through
observation, payload, collection/search, and RND boundaries. Before another
deep kernel pass, run or design a fresh full-loop rebaseline that tells us
whether the next wall is observation, collection/search, RND, or payload.

## Role

Optimizer owns speed, profiling, setup advice, and renderer/search performance
experiments. Coach owns learning claims and which checkpoints are good. Do not
touch live overnight training runs except read-only.

## Current Trusted Path

Training/profiling source of truth is stock LightZero:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
```

Current runs storage default is the fresh Modal Volume `curvyzero-runs-v2`.
The shared source of truth is
`src/curvyzero/contracts/curvytron.py::DEFAULT_CURVYTRON_RUNS_VOLUME_NAME`.
The old `curvyzero-runs` volume still exists only as historical/read-only
material unless a separate exact-delete decision is made.

2026-05-15 strict contract: the target policy observation surface is
`browser_lines + simple_symbols` everywhere that trains, rates, or runs
policies. The source of truth is
`src/curvyzero/env/observation_surface_contract.py`. `browser_lines` names the
semantic trail surface; it does not mean CPU-only. GPU rendering must implement
that same surface through a separate backend flag/path, not by changing the
policy mode. CPU `browser_lines + simple_symbols` is the production backend and
parity oracle today. The intended replacement is a future batched GPU backend,
not scalar `jax_gpu`. The only deliberate policy-observation approximation is
replacing original bonus sprite art with the 12 designed simple symbols; trail
geometry, heads, controlled-player view, grayscale, and downsample semantics
should match the CPU oracle. The controlled-player/seat contract is Coach-owned
and summarized in
`docs/working/training/leaderboard_to_training_2026-05-13/policy_observation_perspective_contract_2026-05-15.md`;
Optimizer only implements faster backends for that requested view. Current
policy paths reject
`body_circles_fast`, `browser_sprites`, `fast_gray64_direct`, and any
`allow_legacy_policy_surface` escape hatch.

The CPU parity oracle is `browser_lines + simple_symbols`:

```text
source-state RGB 704-style canvas
-> simple-symbol bonuses
-> BT.601 luma
-> 11x11 area average
-> uint8[1,64,64]
-> [4,64,64] stack for LightZero
```

This is CPU-reference fidelity, not browser-pixel fidelity. The optimizer target
is to make a future batched GPU backend match this oracle closely enough, then
prove a trainer-visible vector facade before any training/tournament default
changes. Until then, production training/profile commands that represent the
current trusted lane should keep `policy_observation_backend=cpu_oracle`.

Training/profile commands should be explicit about the semantic surface, or
import the shared constants:

```text
--source-state-trail-render-mode browser_lines
--source-state-bonus-render-mode simple_symbols
```

The old `auto`/browser-sprite default was removed from the current policy
surface. Raw RGB/GIF artifacts may still use richer visual paths for display,
but policy observations and tournaments should name `simple_symbols`.

Backend truth: `compute=gpu-*` puts LightZero model/search/learner on GPU. It
does **not** by itself move the CurvyTron observation renderer to GPU. The
explicit backend flag is `policy_observation_backend`, with values
`cpu_oracle` and `jax_gpu`. The scalar `jax_gpu` canary is wired through the
canonical trainer and reaches `lzero.entry.train_muzero`, but it is currently
slower than CPU because it renders one env step at a time and copies frames back
to NumPy. Do not overload `source_state_trail_render_mode` to mean GPU.

2026-05-15 GPU lab update: the isolated H100 `block_704_gray64` renderer now
matches the CPU oracle exactly on the checked real-env smoke rows after the
same-owner, cursor, avatar-color reference, and owner-priority fixes. The latest
S1024 rows are still expensive: B64 is about `212ms` for one view and B256 is
about `735ms` for one view. The shorter S256/B64 row is about `59ms`. This
means the GPU path has become a correctness candidate, not a speed recommendation
yet. A profile-only batched observation facade now exists; use it for boundary
measurement while stock training remains on `cpu_oracle`. Gate list:
[GPU observation next gates](gpu_observation_next_gates_2026-05-15.md).
The first owner-ordered compact H100 row improves that S1024 B64 cost without
changing the policy surface: `208.9ms -> 135.5ms` device render and
`214.0ms -> 140.4ms` end-to-end, with exact checked CPU-oracle parity. Treat it
as the current best isolated GPU render candidate, still lab-only until the
batched trainer-visible boundary exists.
Follow-up isolated rows kept exact parity and improved B256/S1024
`729.6ms -> 391.4ms` device render and B64/S256 `54.2ms -> 36.0ms` device
render; B64/S2048 improved `412.6ms -> 265.8ms`. Plain read: the priority
buffer really was a bad cost center. Caveat: these are still isolated one-view
renderer rows, not a two-view trainer-boundary profile.
Two-view H100 renderer rows now exist too. They render both player perspectives
in one JIT and return `view_major` order: all player-0 rows, then all player-1
rows. B64/S1024 improved `494.18ms -> 251.51ms` for H2D + render + readback
with owner-ordered compact; B256/S1024 improved `1844.97ms -> 1142.06ms`.
Checked rows had exact CPU-oracle parity. This still excludes env stepping,
per-step packing, stack update, reset, and final observation handling.
Scalar two-view H100 profiles confirm that B1 GPU observation remains the wrong
shape: fused two-view exact render is about `28-29ms` per env step including
compact/H2D/render/readback, even though it is about `1.8x` faster than two
separate view renders. The next useful GPU test is batched rows and both views,
not scalar trainer integration.
Batched two-view boundary profiles now exist. Plain read: float32 geometry is
the aggressive GPU candidate. It failed exact CPU-oracle parity in a B64/S1024
later-step check by one luma at one edge pixel (`100` versus `101`), which is
acceptable for a learned policy observation unless future checks show missing
objects, wrong ordering, or unstable symbols. Float32 B64/S1024 measured
`255ms` candidate observation versus `379ms` for float64 (`~1.5x` faster);
B256/S1024 measured `1.14s` versus `1.38s` (`~1.2x` faster). Float64 remains
the exact-parity reference/debug mode: B64/S1024 exact boundary `379ms` versus
`1.09s` CPU reference render+stack; B128 `713ms` versus `1.43s`; B256 `1.38s`
versus `2.79s`. The first timeout/autoreset row also passes exact checked
parity in float64: B64/S1024 with `max_ticks=5` kept a `376ms` median candidate
observation and showed `920ms` p95 on terminal steps because they include
final-observation copy plus reset render/stack. This is the best GPU
observation lab candidate so far, but still not the current scalar trainer
backend; scalar `jax_gpu` is measured slower than CPU. The next job is wiring
the batched backend, not flipping the scalar backend default.
First adversarial renderer slice now passes too: `adversarial_fixture` on H100,
3 players, 12 symbols, non-identity/duplicate/high `avatar_color`, controlled
players `0/1/2`, exact `0`-mismatch parity against CPU oracle. This is still
frame parity, not trainer-contract parity.
2026-05-15 batched integration reorientation: batching is the right direction,
but the current trainer flag does not use it. Scalar
`policy_observation_backend=jax_gpu` is a diagnostic canary and stays out of
defaults. The profile-only batched GPU boundary thinks in full Curvy rows
`[B,2]` and can produce `[B,2,4,64,64]`; stock LightZero expects scalar ego
observations, one `BaseEnvTimestep` per env id. Therefore the next
implementation slice is a repo-owned vector facade first, not a direct
`train_muzero` claim: `VectorMultiplayerEnv(B,2)` -> compact source state ->
batched two-view GPU render -> row/player stacks and masks -> explicit
row/player-to-env-id materialization. Only after that should it be wrapped as a
single-process DI-engine env-manager canary; subprocess/CUDA/IPC is a separate
later proof. Planning doc:
[batched GPU observation trainer integration plan](batched_gpu_observation_trainer_integration_plan_2026-05-15.md).
CPU oracle cleanup also landed: simple-symbol RGB stamping now touches only the
local symbol crop instead of allocating a full-frame scratch canvas, and dirty
cache fallback reasons are recorded outside full timing profiles. This keeps the
trusted backend faster and easier to debug while the GPU path remains gated.
Tournament contract cleanup also landed: policy evaluation still selects
`observation[0, seat]` and `action_mask[0, seat]`, but LightZero `to_play` is
now `-1` to match the stock controlled-player training env. The seat is encoded
by the selected view/action mask and metadata, not by `to_play`.
Frozen checkpoint opponent inference has the same rule now: the opponent policy
gets its own controlled-player visual/action-mask slice, while LightZero
`to_play` is `[-1]`. Do not reintroduce player-id `to_play` for current
non-board-game source-state rows.

Latest critique synthesis:
[parallel critique synthesis](parallel_critique_synthesis_2026-05-15.md).

## Current Speed Read

- 2026-05-16 tooling update: use the simple manifest flow in
  [Modal profile tooling](modal_profile_tooling_2026-05-16.md). Direct blocking
  rows are now supported for small grids. Detached grids must use parent
  `--detach` plus child `--profile-spawn`; never use `--profile-spawn` from a
  short-lived non-detached Modal parent.
- 2026-05-20 current speed correction: the newest stock-path profile top point
  is H100, subprocess, C512, `num_simulations=4`, at about `1061 steps/s`.
  C512/sim8 dropped to about `826`, C768/sim4 nearly tied but did not improve,
  and C1024/sim4 regressed. Use L4/T4 only as the cheaper broad-run option; it
  is no longer the optimizer's top throughput recommendation.
- 2026-05-16 current-surface H100/L4 update, now historical: completed
  no-death512 grid with `browser_lines + simple_symbols`, `cpu_oracle`, sim8.
  Best L4 was C256/batch64 at `713.83` env steps/s. Best H100 was C256/batch32
  at `1001.94`. Batch64 helped L4/C256 but hurt H100, so it is not a universal
  default.
- Fresh stock LightZero full-loop rows on 2026-05-15 used the current trusted
  surface, `browser_lines + simple_symbols`, through
  `lzero.entry.train_muzero`. They are the current speed reference. C8/sim8
  collected 4,096 env steps in `36.60s` (`111.9` steps/s). C32/sim8 collected
  16,384 env steps in `50.34s` (`325.5` steps/s). See
  [full-loop Amdahl reorientation](full_loop_amdahl_reorientation_2026-05-15.md).
- Fresh env-only no-death profiles from 100, 200, 500, 1000, and 2000 steps
  say render is stable at about `77%` of a single-env rollout wall for the
  current `browser_lines + simple_symbols` surface. If render became free, that
  narrow loop could improve about `4.3x-4.5x`; if render became 10x faster, it
  would improve about `3.2x-3.3x`.
- Amdahl read: render still matters a lot for long single-env trajectories, but
  current stock LightZero full-loop wall is less renderer-pure because env
  workers run in parallel and collector/policy/MCTS/replay/learner work remains
  on the critical path. C32/sim8 spent `37.78s` in collector collect, with
  `22.43s` policy collect and `14.55s` MCTS/search buckets, inside a `50.34s`
  wall. Do not promise a 3x full-loop win from renderer work alone until the
  batched GPU backend is actually wired and profiled.
- Bigger GPUs are not automatically better. H100 is justified by the latest
  C512/sim4 profile, but C512/sim8, C768/sim4, C1024/sim4, and sim16 rows show
  that brute force hits collector/process/search overhead quickly.
- Historical stock fast-path grid, batch32/no-death/source512: C64/L4/sim8 fast V8
  measured `591.6` env steps/sec, versus a matched C64/L4/sim8 browser
  reference at `491.4` env steps/sec. C384/L4/sim8 fast V8 measured `946.1`
  env steps/sec and then wider L4 rows dropped. C256/H100/sim8 measured
  `1081.9` env steps/sec, and C768/H100/sim8 measured `1204.0`, the fastest
  speed-only row so far. This does not make C768 a learning default; it is a
  speed/aggression probe from the old `body_circles_fast` speed-only surface,
  not a current policy-surface recommendation. See
  [stock fast-path scaling grid](stock_fast_path_scaling_grid_2026-05-14.md).
- Superseded Coach-facing recommendation:
  [fast stock recommendation](coach_handoff_fast_stock_recommendation_2026-05-14.md).
  Plain read: keep its speed numbers as historical evidence, but do not copy
  its `body_circles_fast` launch commands into new runs.
- Scalar GPU observation canary, H100/base/C1/sim2/no-death:
  `cpu_oracle` at 512 steps was `15.54s` wall (`32.94` steps/s) with
  `4.42ms` observation/step; `jax_gpu` was `63.73s` wall (`8.03` steps/s) with
  `80.31ms` observation/step. The scalar GPU hook is a proof gate, not the
  production optimization. Full catalog:
  [GPU observation full-loop canary](gpu_observation_full_loop_canary_2026-05-15.md).
- Operational risk: the profile Modal container warned that `/runs` volume inode
  usage was about `97.7%`. Clean or route future artifacts before serious long
  runs.

## Active Plates

1. **Bonus symbol renderer**
   Implemented and wired as the current policy bonus encoding. The production
   lane is CPU `cpu_oracle` `browser_lines + simple_symbols`; the future GPU
   lane must preserve the same 3 outer shapes x 4 inner marks, high-contrast
   luma by shape, row-specific asymmetric marks, and minimum 7x7 footprint.
   Browser sprites are display/reference artifacts, not the default policy
   surface.
   Current artifact:
   `artifacts/local/curvytron_render_profiles/bonus_simple_symbols_actual_v8_20260514.png`.
   Independent visual and numeric critiques agree that the direct fast path has
   no class collisions in the tested center/offset/radius/edge/remap sweeps.
   V8 keeps thick horizontal/vertical marks and gives the three X variants
   different asymmetry. The raw 7x7 nearest-pair floor is pinned at L1 `>=1300`
   and mismatch pixels `>=10`. Caveat:
   `simple_symbols` must be tested in the same gray64 canvas/downsample path
   used by the future batched GPU backend and CPU oracle before it is treated
   as production equivalent.
   Trail overlay is tested: all 12 symbols overwrite both direct
   `visual_trail_*` and fallback `body_*` trail pixels while staying distinct.
   Do not alpha-blend the symbol body unless a future test proves it helps;
   blending makes the same bonus vary with whatever trail is underneath.
   Current design note:
   [bonus symbol render plan](bonus_symbol_render_plan_2026-05-14.md).

2. **GPU browser-lines renderer**
   The lab GPU renderer is promising but not a trainer backend. After the
   same-owner trail and owner-priority fixes, first H100 smoke rows match the
   CPU oracle exactly on checked rows; adversarial parity and a real batched
   LightZero boundary profile are still required for the current
   `simple_symbols` target. Current note:
   [GPU render parity gap](gpu_render_parity_gap_2026-05-13.md).
   Current implementation plan:
   [GPU observation backend plan](gpu_observation_backend_plan_2026-05-15.md).
   Correction as of 2026-05-15: the next practical GPU target is **not**
   original browser sprites. It is faithful GPU `browser_lines` trail/head
   geometry plus `simple_symbols` bonus encoding. This keeps the important
   browser-line trail surface while avoiding the harder RGBA sprite parity
   problem. This was implied by the GPU-render and simple-symbol docs but was
   not promoted into the active implementation list; treat that as an optimizer
   miss, not a settled decision against the lane.
   Fresh isolated benchmark: `block_704_gray64` still outputs final `64x64`;
   it just computes each output pixel from 11x11 high-resolution sample
   positions to match the CPU oracle. On 2026-05-15, `browser_lines +
   simple_symbols` hit exact checked parity on synthetic CPU-oracle rows. H100
   B64/S64 measured `9.85ms` device render versus L4 `81.7ms`; H100 B64/S256
   measured `31.0ms` versus L4 `291.0ms`. Host-to-device copy stayed around
   `3ms`. Plain read: GPU render is real, but this was pre owner-priority fix
   smoke evidence; LightZero boundary integration remains a blocker.

   Fresh real-env rollout smoke, 2026-05-15: L4, B8, trail_slots64,
   real_env_steps32, controlled_player0, one active simple-symbol bonus. JAX
   used GPU. Device render median was about `2.78ms` for 8 frames, end-to-end
   with host copies/readback was about `7.32ms`, and the CPU oracle for the one
   checked row took about `93ms`. Parity was `4088/4096` exact; the remaining 8
   pixels were all off by one gray value. Plain read: real-state GPU rendering
   works and is fast in isolation; the next questions are tiny rounding parity,
   both-seat coverage, and trainer backend integration.

   Follow-up real-env rows: L4 B64/S64 took about `74.18ms` device and
   `78.75ms` end-to-end; H100 B64/S64 took about `7.87ms` device and
   `10.88ms` end-to-end; H100 B64/S256 with controlled_player1 took about
   `28.74ms` device and `31.92ms` end-to-end. The checked rows still had only
   tiny parity gaps (`6` pixels max diff `1`, then `25` pixels max diff `2` for
   the longer trail row). Plain read: H100 is the serious hardware target for
   the GPU renderer, and the remaining blocker is integration/fidelity, not raw
   GPU math.

   Integration guardrail: do not put a B1 JAX call inside every scalar stock
   env. The stock LightZero scalar canary is now measured and is slower:
   `jax_gpu` was about `4.1x` slower than `cpu_oracle` at 512 steps. The
   promising shape is batched GPU rendering, not scalar GPU round trips.
   Stronger critique, 2026-05-15: gross parity now outranks batching, but exact
   one-luma equality does not. The old GPU
   `browser_lines` benchmark connected each raw trail slot to `slot-1`; the
   current lab renderer now carries the previous active same-owner point. It
   also carries a high-resolution owner-priority buffer so overlapping trails
   follow CPU owner draw order instead of slot order or max luma. Fresh H100
   B64/S1024 real-env rows now match the CPU oracle exactly on the checked
   rows, at about `212ms` end-to-end for 64 frames. Scalar fused two-view
   rendering also matches CPU exactly on the checked row and is about `1.8x`
   faster than two separate scalar JAX renders. Later profile-only batched
   boundary work added explicit exact/tolerant parity modes and showed that
   `float32 + tolerant` can verify real steps with only tiny luma drift. Keep
   `cpu_oracle` as the production backend until the vector facade and
   trainer-visible gates prove row order, player view, stack/reset,
   final-observation, RND, checkpoint, and tournament semantics.

3. **Full-loop scaling**
   Keep measuring stock LightZero with clear buckets. Relevant knobs are
   collector width, search sims, learner batch, CPU count, GPU class, telemetry
   stride, and trajectory length. Do not infer learning quality from speed
   profiles. Current C32/sim8 is much faster than C8/sim8 by throughput, but
   the next recommendation needs C64/C96 and sim16 rows before choosing a final
   Coach config.

4. **Moving environment**
   Environment Reconstruction is still changing source-state/render details.
   Every speed number must name code state, render mode, bonus mode, warmup,
   trajectory length, and whether search/learner are included.

## To Do

- [x] Harvest sprite inventory and practical symbol-design subagents.
- [x] Write down the 12-bonus symbol plan and Amdahl implication.
- [x] Harvest the running signature-probe subagent and fold results into the
  symbol plan.
- [x] Add an explicit opt-in `simple_symbols` bonus render mode for the direct
  fast path.
- [x] Gate the fast path with centered, offset, radius, remap, metadata, and
  render/wrapper tests.
- [x] Rerun local env-only profile after symbol implementation.
- [x] Harvest the independent actual-symbol visual and numeric critique agents.
- [x] Add explicit stock training flag/metadata:
  `source_state_bonus_render_mode`.
- [x] Collect the fresh waited stock full-loop profile pair.
- [x] Harvest the active wider-collector stock fast-path grid and write the
  Coach-facing recommendation. Current grid:
  [stock fast-path scaling grid](stock_fast_path_scaling_grid_2026-05-14.md).
- [x] Feed real env rollout rows into the isolated GPU render benchmark and
  verify controlled-player GPU output against the CPU oracle.
- [x] Run fresh stock LightZero full-loop C8/C32 profiles on the current
  `browser_lines + simple_symbols` surface and write the current Amdahl read:
  [full-loop Amdahl reorientation](full_loop_amdahl_reorientation_2026-05-15.md).
- [x] Wire an explicit scalar `policy_observation_backend=jax_gpu` canary into
  the canonical stock LightZero trainer and compare it to `cpu_oracle`.
- [ ] Active: design a semantics-preserving compact arrays-in / arrays-out MCTS
  boundary and its fixed-seed LightZero parity gates.
- [x] Done for now: price host-stack -> Torch input modes. Pinned helps
  modestly; resident reuse is stale ceiling only.
- [ ] Active: split/reduce the remaining direct CTree path after input-copy is
  priced, then run fixed-seed parity before any trainer advice.
- [ ] Background/superseded unless a fresh Amdahl row re-shows observation
  dominance: batched faithful-GPU-plus-simple-symbols trainer integration,
  older vector-facade expansion, and old C32/C64/C96/sim matrix rows.

## Operating Pattern

- Main thread is for planning, delegation, orchestration, and short synthesis.
- Use subagents for bounded code reading, toy probes, literature/research, and
  critique.
- Keep docs current before details fall out of context.
- Do not change live training runs. New renderer experiments must be opt-in.
