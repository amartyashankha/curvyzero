# Bottleneck Model

Date: 2026-05-23

Purpose: name the current walls in plain language and decide what to test next.

Status after OPT-108: this file contains useful historical bucket reads, but it
is not the active target selector. For active speed work, use `goal.md`,
`CURRENT_STATE.md`, and `TASK_BOARD.md`. Current baseline is OPT-104; OPT-105
through OPT-108 are measured no-wins. The next speed move must either remeasure
the current row or remove a complete remaining boundary.

## Full Loop Buckets

A real training iteration pays for several things:

1. Environment state update.
2. Observation/render/stack creation.
3. Policy inference for root positions.
4. MCTS search: root prep, tree traversal, recurrent inference, value/policy
   payloads, visit counts.
5. Action selection and environment feedback.
6. Replay insertion and sampling.
7. Learner update.
8. RND/reward/exploration bookkeeping when enabled.
9. Sparse side work: checkpoints, evals, GIFs, tournament exports.

If a profile excludes one of these buckets, it cannot claim full-loop speed.

## Current Read, 2026-05-31 After OPT-078

The current same-denominator compact speed-row evidence says replay-index/
commit was a real child, but no longer the selected next target after OPT-077b.
OPT-076 measured the envelope on H100:

```text
compact_torch_steps_per_sec=12846.8
fixed_floor_steps_per_sec=13759.0
compact_vs_floor_wall_ratio=1.071
dispatch_residual_delta_sec=0.005
action_residual_sec=0.004
action_readback_sec=0.011
core_residual_sec=0.152
initial_cuda_event_sec=1.510
tree_total_sec=0.368
slab_replay_index_rows_build_delta_sec=2.457
slab_commit_previous_delta_sec=2.491
search_delta_sec=2.072
```

Plain read: dispatch accounting, action readback, root decode, and tree residual
are not the next wall. Initial model execution is still real, but replay-index
rows build and commit-previous now dominate the target-selection decision. The
next bounded artifact should decompose/reduce replay-index, commit, and sample
gate costs under the same denominator. GPU mechanics remains deferred: actor
wall is not dominant in the OPT-076 bundle, and selected-action D2H is only the
expected int16 action transfer.

OPT-077/077b then split that replay-index bucket. Scalar transfer was not the
wall: the packed-scalar row had only `0.067s` scalar transfer, while metadata
list materialization took `1.651s`. Replacing full terminal/death metadata lists
with counts/checksums reduced replay-index build to `0.145s` and commit to
`0.185s`.

Plain read now: replay-index/commit was worth fixing as a bucket, but it did not
produce a top-level speed win. OPT-077b is `11735.2` steps/sec, below OPT-076
`12846.8` and fixed floor `13759.0`. The formal decision returns to guarded
initial-model-forward/service work. GPU mechanics remains deferred because actor
wall has not repeatedly dominated under a stabilized row.

OPT-078 then removed the recurrent nested `no_grad` under the existing eval/
inference guard and made guard timing visible:

```text
compact_torch_steps_per_sec=10740.8
fixed_floor_steps_per_sec=13759.0
compact_vs_floor_wall_ratio=1.281
initial_cuda_event_sec=1.546
recurrent_cuda_event_sec=0.336
guard_total_sec=0.208
search_delta_sec=2.234
learner_sample_delta_sec=1.610
actor_wall_delta_sec=1.360
replay_index_rows_build_sec=0.182
commit_previous_sec=0.234
```

Plain read after OPT-078: the cleanup is safe and the guard timing is useful,
but it is not a speed win and it does not change the selected target. Continue
guarded initial-model-forward/service work. Treat learner/sample or actor-wall
movement in this one row as noise until it repeats after the service/model path
is sharpened; do not jump to GPU mechanics.

## Historical Current Read Before OPT-064/065

Render was a real problem earlier. It is no longer the only credible wall.

2026-05-26 reorientation: "observation path" means rendering plus frame-stack
ownership plus host/device copies plus feeding the policy/search consumer. The
best resident compact row already made the render/search-input handoff much
cheaper, but it did not enter the stock Coach trainer.

Best current resident trainer-like row:

```text
path                  optimizer profile-only, not train_muzero
measured_sec          15.010
steps_per_sec         16373.5
actor/env wall         5.354s
compact search slab    4.775s
observation/stack      2.288s
sample gate            1.855s
learner/RND gate       0.637s
search obs H2D bytes   0
resident fallback      0
```

Plain Amdahl read: after resident GPU observation, optimizing rendering alone
cannot produce a huge end-to-end gain. The remaining large costs are the
batched actor/env loop, compact search, replay/sample materialization, and the
gap between this profile-only loop and stock `train_muzero`.

The immediate bottleneck is therefore not a single function. It is the
training-path boundary: stock LightZero owns scalar env timesteps and replay
objects, while the optimized candidate owns compact arrays and resident device
observations.

As of 2026-05-26, do not start a new search/actor/observation optimization
lane until the compact trainer-like profile row is clean. The active proof row
must include search feedback, bounded replay-ring sampling, learner gate, and
RND-style input with scalar timestep materialization off. That row is the next
denominator for Amdahl, not another isolated renderer or search microbench.

That proof row has now passed on H100 at B1024/A16/sim8 with 100 warmup and 120
measured steps:

```text
measured_sec        22.262
search/probe_sec     6.934
observation_sec      6.674
actor_wall_sec       6.221
sample_gate_sec      1.702
learner_gate_sec     0.532
scalar_mat_sec       0.000
python_rows          0.000
```

Plain Amdahl read: replay sample/learner accounting is real and visible, but it
is not the only wall. The next bounded comparison should isolate search backend
cost with controlled actions, then return to observation/actor ownership if the
search comparison is not decisive.

The controlled-action comparison now says:

```text
direct CTree measured_sec   47.864
compact Torch measured_sec  32.553
fixed-shape floor sec       22.766
compiled compact Torch      24.107  # root_noise_weight=0 support row
compiled compact Torch      29.530  # default root noise still on
```

All three rows used the same scripted trajectory. Compact Torch beats direct
CTree by `1.47x`. The no-real-search floor beats compact Torch by `1.43x`.
That means search remains meaningful, but after search the loop immediately
lands on actor/env and observation. The actual compiled-helper support row
moved compact Torch close to the no-real-search floor, so current priority
should be:

1. keep the compiled compact Torch helper path, but do not call root-noise-zero
   a training default until pre-sampled root noise is wired;
2. promote native actor buffering through stronger gates;
3. design the resident observation/search contract so the search consumer reads
   fresh device-owned observations without host stack refresh;
4. revisit replay/sample/learner only after the first three move again.

Latest concrete rows:

```text
trainer-like warm baseline       22.262s
trainer-like native actor        18.459s  # 1.21x
native actor + compiled helpers  17.127s  # 1.30x, default root noise on
fixed-shape floor baseline       22.766s
device-only observation ceiling  14.433s  # 1.58x over floor baseline
no-refresh observation ceiling   11.344s  # 2.01x over floor baseline
```

Plain read: the next practical profile-only setup is native actor buffering
plus compiled compact Torch helpers. The next bigger architectural win is
resident observation ownership. A pure search-only rewrite is no longer the
best bet for a large loop speedup on this denominator.

The current profile evidence points at these bigger costs:

- repeated scalar object creation around LightZero env timesteps;
- CPU/list-shaped root prep and CTree payloads;
- per-simulation CPU/GPU traffic around recurrent outputs;
- replay and RND materialization that is not fully compact-owned;
- learner/update cadence that may become visible once collection/search is
  faster;
- under-filled GPU work when root batches are too small or too fragmented.

## Amdahl Read

If render is only a small slice now, a perfect render gives a small full-loop
gain. That does not mean the render work was useless. It means the bottleneck
moved.

The speed target now needs to attack the biggest remaining repeated costs. The
current candidate is compact ownership because it removes repeated scalar
materialization and gives search/replay/learner a batch-shaped contract.

The 2026-05-24 compact two-phase profile sharpens this:

- compact Torch two-phase beats direct CTree by `1.44x` at B512 and `1.38x` at
  B1024 on the compact profile denominator;
- a no-real-search fixed-shape floor is still faster, but only `1.37x` faster
  than compact Torch at B1024;
- therefore the remaining B1024 profile wall is not just "copy back the full
  search payload." It is observation transfer/staging, actor/env loop time,
  compact batch construction, and the remaining search/model work.

Plain implication: compact two-phase is worth keeping, but the next large move
has to own more of the loop, not just make the search result object smaller.

The first learner-edge honesty row found a sharper wall. With compact Torch,
search feedback, replay payload flush, sample gate every 8 commits, sample
batch 256, and the old `toy_probe` tiny CUDA learner/RND consumer:

```text
measured_sec       24.260
sample_gate_sec    12.432
learner_gate_sec    0.343
```

So the next Amdahl target is compact replay/sample materialization. In plain
terms: turning compact index rows into learner-shaped NumPy batches is now much
more expensive than the `toy_probe` learner consumer. A faster learner does not
help much until this sample/materialization path is made array-native or moved
out of the hot path.

That target produced a real win. The fast sample-batch path reduced the same
B512 learner-edge row from `24.260s` to `11.520s`; sample-gate time fell from
`12.432s` to `0.096s`.

New Amdahl read for that row:

- search/probe: `4.777s`;
- actor/env: `3.401s`;
- observation: `2.483s`;
- learner gate: `0.355s`;
- sample gate: `0.096s`.

Plain implication: compact replay materialization is no longer the first wall
in this profile. The next wall is again the live loop itself: search, actor/env
stepping, and observation movement.

2026-05-26 latest strict compact MuZero rows make this sharper. The best B1024
row so far is H100 B1024/A16/sim8 with sample B512 every 8:

```text
measured_sec       14.371
steps_per_sec      17100.7
search/probe       ~39%
actor/env          ~36%
observation        ~15%
learner             ~5%
sample              ~4%
```

The B2048 row reaches `19811.4` steps/sec but has the same shape: search/probe
and actor/env dominate. Learner/sample are now too small to explain the missing
speedup. The concrete next wall is the actor/search boundary:

```text
GPU search -> selected actions copied to CPU -> CPU env step -> CPU masks/state
sidecars -> GPU search again
```

So the next speed cut should remove public env packaging and repeated
Python/NumPy merging from compact profile rows before chasing learner/sample
again. A perfect sample/learner optimization cannot move the full loop much on
this denominator.

That first actor-boundary cut is now local. It does not move physics to GPU;
it removes public env packaging from the compact actor path. The expected gain
is therefore modest, not `10x`: it can only attack the part of actor/env time
spent building public observations/info/batches. Local B512/A8 smoke says it
removes about `0.19s` actor wall on a `4.30s` public-payload profile and makes
public info/batch-pack timing exactly `0.0` in native compact mode. The H100
profile rerun is the useful denominator for whether this matters in the strict
compact MuZero row.

H100 result: it matters, but it is still bounded. Exact B1024 improved
`14.371s -> 12.935s` (`1.11x`), and B2048 same-cadence improved
`24.810s -> 18.303s` (`1.36x`). Public packaging is gone from the compact actor
path; the remaining wall is now actual search/probe and actual env/runtime
work. The next Amdahl target should not be another public-packaging patch.

The no-death collision diagnostic skip is a good example of why we need
matched rows. It reduced the actor/runtime bucket on B2048 (`3.815s -> 2.580s`
runtime), but the first total row was `19.907s`, slower than the `18.303s`
actor-cut comparison because search/probe, observation, and sample varied. This
is worth keeping as a runtime cleanup, but it does not change the priority:
search/probe is still the largest bucket in the latest B2048 row, followed by
actual actor/env runtime.

The active search experiment is intentionally narrow:

```text
auto recurrent action shape
flat recurrent action shape
flat recurrent action shape + opt-in model compile
```

The rows answered the narrow question:

```text
auto                         17.609s, effective flat, fallback 0
forced flat                  18.678s, slower
forced flat + model compile  failed remotely after Modal retries
root noise weight 0          17.841s, not faster and trajectory changed
```

So the next real move is not more telemetry around the same Python Torch tree.
It is a stronger compact search backend behind `CompactSearchServiceV1`,
keeping the resident observation, selected-action-only sync, and device replay
contracts. `auto` can stay as the default because it already takes the flat
action path without fallback in the current model.

The B1024 confirmation row includes replay flush, sampling, and a tiny CUDA
learner/RND gate:

```text
measured_sec       22.771
probe_sec           8.359
actor_wall_sec      5.982
observation_sec     5.556
sample_gate_sec     0.366
learner_gate_sec    0.476
```

So the current compact-profile priority is:

1. search/probe cost;
2. actor/env stepping and parent merge;
3. observation movement/staging;
4. learner/sample edge only after those move again.

2026-05-26 update: workspace reuse in compact Torch search did not help on the
strict B2048 H100 row (`19.601s` versus `17.609s` default-auto). Node/latent-id
and `leaf_depth` helper rewrites also failed to beat the baseline (`19.430s`
and `17.723s`). This weakens the hypothesis that tiny Torch helper bookkeeping
is the main search wall. The next search work needs a larger backend/ownership
change, not more local reshuffling. The low-risk action boundary cleanup stays:
row-major selected actions now bypass the per-root Python scatter back to joint
action form.

Follow-up copy-cleanup row, same broad B1024/A16/sim8 compact Torch shape but
direct64/persistent renderer and `search_feedback`, confirmed that the copy
cleanup worked but also exposed the action-confounding problem again:

```text
measured_sec                 51.524
actor_env_runtime_sec        39.259
observation_sec               4.217
compact_rollout_slab_sec      5.794
sample_gate_sec               0.310
learner_gate_sec              0.380
root_observation_copy_bytes   0
env_action_checksum_total     0
```

Plain implication: the next big speed decision cannot be based on that row's
wall time. It proves the duplicate-copy tax is gone. It does not prove the
candidate loop is slow, because all-zero selected actions changed the trajectory
and made env runtime dominate. Matched speed rows need either controlled actions
or a replay/learner test that explicitly accepts policy-action trajectory
differences.

Later OPT-064 decomposed the same-denominator H100 speed-row floor. The first
slab-timer rerun showed the apparent slab residual was dispatch-wall, not
commit/flush/replay-index materialization. Service-envelope timers then found a
pure metadata tax: repeated policy-refresh model digest computation spent
`9.769s` in `metadata_build_sec` on the pre-cache compact Torch row. Caching
policy-refresh digest/metadata at `refresh_model_state` improved compact Torch
from `4797.9` to `9608.9` compact trainer env steps/sec and dropped metadata
build to `0.0034s`.

OPT-065 then split the current compact Torch service wall. The canonical H100
timing row reads: service `2.808s`, model `2.249s`, tree `0.506s`, initial
enqueue `0.344s`, initial sync `1.905s`, recurrent enqueue `0.186s`, and final
tree sync `0.0048s`. Warm model compile failed closed with a Torch CUDAGraph
overwritten-output error. The timing-mode surface now has a host-phase
CUDA-event baseline too: `host_phase_sync_cuda_event` records initial CUDA
event `2.232s`, initial host sync `1.751s`, recurrent CUDA event `0.471s`, and
tree CUDA event `0.536s` while preserving `initial_sync_enabled=true`.
Final-sync mode then disabled the explicit initial host sync but did not improve
wall speed: `9041.2` steps/sec, wall `20.387s`, initial sync `0.0s`, initial
CUDA event `2.249s`, recurrent CUDA event `0.475s`, tree CUDA event `0.543s`,
service `2.871s`, action wall `2.946s`, replay D2H `0`, and fast-path/recurrent
calls `180/180`. The current read is that the sync wall was mostly queued
initial model GPU execution, not a removable CPU barrier.

The safe compile branch now has first evidence. A separate
`model_compile_mode` knob lets model inference use `default` while helper
compile behavior remains unchanged. The event row
`opt065-h100-modelcompile-default-sim1-20260531` reduced initial CUDA event
`2.232s -> 1.391s`, recurrent CUDA event `0.471s -> 0.333s`, and service
`2.862s -> 1.916s` without replay D2H or CUDAGraph failure. The canonical
compile row lowered service to `1.806s` and model to `1.324s`, but wall stayed
worse than eager canonical: `10258.6` versus `12335.7` steps/sec. The current
fresh paired repeats now favor compile: r1 `12450.3` compiled versus `9840.1`
eager, r2 `12563.9` compiled versus `11520.7` eager. The floor bundle now also
derives dispatch residual accounting, and the rebuilt r1/r2 bundles show signed
dispatch residual deltas of only about `0.003s`. So the dispatch headline is
not unmeasured slab wrapper overhead; it is the compact Torch action-wall
envelope. The hash-bound OPT-065 compile/eager comparator blocked approval on
action/trajectory mismatch. OPT-066 then added the fixed-root model-compile
gate and eval/inference search guard: fixed-root parity passed, and post-guard
r1/r2 action/trajectory checksums match. The formal decision is now
`park_model_compile_default_speed_unapproved`, because compile improves
service/model buckets without a repeatable end-to-end wall win. OPT-075 then
removed the redundant inner `no_grad` around initial inference under the
existing eval/inference guard. That cleanup is safe and measured, but its H100
initial CUDA event is still `1.562s`, so it does not prove an
initial-forward-speed win. The current speed target is therefore compact Torch
search dispatch/service-envelope decomposition, not game mechanics or slab
commit/flush in the pre-OPT-064 sense.

## What Could Falsify Compact Ownership

Compact ownership is not a religion. It should be killed or demoted if matched
tests show:

- learner/RND dominates even after compact collection/search;
- LightZero scalar materialization is not a meaningful cost at Coach scale;
- GPU work is already saturated and compact batches only add overhead;
- semantic gates make the fast path too different from trusted LightZero;
- larger collector batches improve speed but hurt learning enough to lose wall
  clock progress.

## Tests Needed

OPT-076 tests needed now:

- local measurement surface is landed: service/slab/profile/bundle telemetry
  now exposes action accounted/residual, root-output decode, tree root-prior
  build, tree total/accounted/residual, action readback, and core
  accounted/residual buckets;
- same-denominator H100 row or decision packet that decomposes the compact
  Torch search dispatch/service envelope after OPT-075;
- refreshed floor bundle or equivalent residual accounting against the
  accepted/fixed siblings;
- same-root or same-trajectory fidelity before using any compiled/warmed model
  path as evidence;
- service-bucket decision refresh if the decomposition moves dominance away
  from initial model forward;
- timer comparison for search-service total, model, tree search, H2D, action
  wall, initial enqueue/sync, CUDA-event elapsed time, recurrent enqueue,
  recurrent CUDA-event time, tree sync, metadata build, fast-path count,
  recurrent calls, action-wall-over-total, dispatch residual, and non-service
  residual;
- fixed-root-tape fidelity if replacing compact Torch semantics or backend;
- recurrent-deferral refresh-hazard proof before any pending-payload deferral
  implementation;
- no promotion, live-run, stock-speedup, or `train_muzero` claim.

We need matched rows that isolate one wall at a time:

- stock full-loop profile vs direct output-fast, same RND/reward/death/hardware;
- CPU-oracle observation vs batched GPU observation, same search and learner;
- search-only direct CTree vs MCTX/JAX vs mock, but labeled as search-only;
- compact replay/RND gate with the learner attached or explicitly mocked;
- scalar-output on/off as a ceiling, not a launch setting;
- normal death and no-death rows kept separate.
