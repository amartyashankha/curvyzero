# World Model

Date: 2026-05-20

## 2026-05-23c High-Level Reorientation

Plain current answer:

```text
We should not call the current work "ready for Coach training speed".
We should take a bigger optimizer move, but keep it behind profile and
validation gates.
```

What happened:

```text
The render work was real, but it did not create a huge Coach-loop speedup.
The reason is simple: once rendering became cheaper, the wall moved inward to
the collect/search/dataflow path.
```

Current hot shape:

```text
CPU env/compact rows
-> GPU model/root work
-> LightZero CTree or profile search service
-> CPU/list/object/search/replay materialization
-> replay/RND/sample edges
```

The strongest current evidence:

```text
H100 B512/A16, 80 measured + 20 warmup, root_noise_weight=0.0:

sim16:
  direct CTree:   5.47k steps/sec
  compact Torch:  4.05k steps/sec
  service-tax:    7.81k steps/sec
  mock ceiling:   7.46k steps/sec

sim32:
  direct CTree:   3.14k steps/sec
  compact Torch:  2.67k steps/sec
  service-tax:    5.19k steps/sec
  mock ceiling:   9.17k steps/sec
```

Interpretation:

```text
The current eager compact Torch search is not the move. It loses to direct
CTree on H100.

The compact service/replay contract is still valuable. It gives us a clean
place to swap in a better backend and test whether the whole chain still works.

The next big move is search/dataflow ownership, not another small render patch.
```

Best next candidate after the fixed-A3 report:

```text
Use CompactSearchServiceV1 as the boundary.
Do not treat flat-A3 as the next main move: it already exists in the
train_muzero profile shell and the matched row was flat/slightly worse
than direct CTree.
Move to a stronger fixed-shape search/dataflow backend behind the same
contract.
Keep MCTX/JAX as a side comparator for the all-device ceiling.
Keep Puffer-style contiguous buffers as the broader architecture direction.
```

Fixed-A3 correction:

```text
opt-flat-a3-ab-20260522a, H100 C64/sim16/3 learner:
  direct LightZero CTree: 516.55 steps/sec
  flat-A3 CTree:          509.69 steps/sec

Meaning: list/backprop ABI cleanup is real engineering, but it is not the
current 5x-10x lever. The remaining wall is search/control/dataflow ownership.
```

Fast falsifier:

```text
Same-denominator H100 profile-only rows:
  B512/P2/A3, sim16 and sim32, root noise off, replay proof on.

Compare:
  direct_ctree_gpu_latent
  stronger fixed-shape compiled/search-service candidate
  service-tax/mock ceiling
  MCTX side comparator if available

Kill the candidate as a main lane if it cannot beat direct CTree after warmup
while passing action feedback, legal mask, replay, terminal, perspective, and
RND latest-frame gates.
```

Promotion rule:

```text
Profile-only speed is not Coach speed.
Coach-facing advice needs a capped stock/full-loop smoke that proves the row
called the trusted entrypoint, avoided fallbacks, and produced valid replay/RND
sample payloads.
```

Tooling rule:

```text
Hybrid optimizer sidecars:
  build_curvytron_hybrid_observation_profile_grid.py
  run_curvytron_hybrid_observation_profile_manifest.py
  profile_only=true, calls_train_muzero=false, touches_live_runs=false

Stock full-loop profiles:
  build_curvytron_profile_grid.py
  run_curvytron_optimizer_profile_manifest.py
  called_train_muzero=true, mode=profile

MCTX sidecars:
  mctx_synthetic_benchmark.py / compact visual root comparator
  profile_only=true, not_lightzero_ctree=true, not_train_muzero=true

Do not compare these as the same speed currency unless the summary says exactly
what loop each row measured.
```

2026-05-23c implementation update:

```text
The profile-only CompactRolloutSlab is now reachable from the hybrid Modal
profile tooling.

It proves this wiring:
  compact batch
  -> CompactRootBatchV1
  -> CompactSearchServiceV1
  -> selected joint action
  -> next env step
  -> CompactReplayIndexRowsV1

Tiny L4 Modal smoke proved wiring only:
  ok=true
  profile_only=true
  compact_rollout_slab_enabled=true
  compact_rollout_slab_calls=1
  compact_rollout_slab_total_roots=4
  compact_rollout_slab_committed_index_row_count=4
  compact_rollout_slab_search_impl=mock_search_service

This is still not a Coach training backend and not a speed claim.
Next useful experiment is the same slab with real direct CTree or a new backend
candidate on a warm H100 denominator.
```

2026-05-23d correction:

```text
Do not use last slab search-call time as the denominator for all slab roots.
That overstates throughput.

Use:
  probe_total_sec = aggregate compact_rollout_slab_sec

Keep:
  compact_rollout_slab_last_* = one last-call diagnostic
```

Corrected warm H100 B64/A8/sim8 smoke:

```text
direct_ctree_arrays:
  full profile: 1560 steps/sec
  slab only:    2270 roots/sec
  last service: 0.0505 sec for 128 roots

direct_ctree_gpu_latent:
  full profile: 2204 steps/sec
  slab only:    3726 roots/sec
  last service: 0.0305 sec for 128 roots
```

Plain read:

```text
GPU-latent direct CTree is still the real-search row to carry forward.
But the honest denominator says this is not a huge win yet.
The active H100 grid is meant to find whether larger batches/sim counts expose
real headroom, or whether the next move must be a stronger backend behind
CompactSearchServiceV1.
```

2026-05-23i real-checkpoint MCTX update:

```text
The toy MCTX result was not enough because it did not use real CurvyTron policy
weights.

Now the profile-only bridge can run:
  immutable LightZero checkpoint
  -> JAX shadow model
  -> MCTX compact search
  -> compact rollout slab

The tiny L4 smoke passed. The shadow model consumed all required inference keys.
The checkpoint SHA is recorded in the profile telemetry.
```

Current matched H100 rows:

```text
B64/A4/sim8, scalar rows on:
  MCTX real checkpoint: 3817 steps/sec
  direct CTree:         2813 steps/sec
  speedup:              1.36x

B512/A8/sim8, scalar rows on:
  MCTX real checkpoint: 10037 steps/sec
  direct CTree:          4233 steps/sec
  speedup:               2.37x

B512/A8/sim8, scalar rows off:
  MCTX real checkpoint: 14257 steps/sec
  direct CTree:          8999 steps/sec
  speedup:               1.58x

B1024/A16/sim8, scalar rows off:
  MCTX real checkpoint: 19334 steps/sec
  direct CTree:          8792 steps/sec
  speedup:               2.20x
```

Plain Amdahl read:

```text
MCTX/JAX is worth keeping as an optimizer lane.
It is not enough by itself.

The cleanest current optimizer row is B1024 scalar-off:
  real-checkpoint MCTX/JAX is 2.20x faster than direct CTree
  on the same profile-only denominator.

At larger batches the wall splits between:
  search backend
  scalar LightZero timestep materialization
  observation stack movement
  CPU env/public info/payload work

The strongest next architectural idea is therefore not "swap MCTS only".
It is:
  compact rows on the hot path
  compiled/device search where useful
  delayed scalar/replay materialization only where the trusted trainer actually
  needs it
```

Coach-facing rule:

```text
Do not tell Coach to use MCTX yet.
Do tell Coach that profile-only evidence says compact replay/dataflow plus a
compiled search backend is the next large optimization direction.
```

2026-05-23j same-root comparator update:

```text
We now have a direct semantic comparator:
  same CompactRootBatchV1
  same immutable checkpoint
  MCTX/JAX primary
  direct LightZero CTree reference

Tiny H100 sim2 smoke:
  identity match: true
  selected-action match: 50%
  visit L1 mean: 1.46
  root-value abs diff mean: 57.4

Larger H100 sim8 rows:
  direct_ctree_gpu_latent reference:
    selected-action match: 34.4%
    visit L1 mean: 1.28
    root-value abs diff mean: 15.97
  direct_ctree_arrays reference:
    selected-action match: 37.5%
    visit L1 mean: 1.29
    root-value abs diff mean: 15.97

Plain meaning:
  The MCTX speed lane is not dead.
  The same-root wiring is correct.
  The two direct CTree references agree with each other enough that the mismatch
  is probably MCTX-vs-LightZero semantics or value scale, not root ordering.

Next proof:
  Pre-search values/logits now say the model bridge is close:
    predicted policy logits mean/max diff: 0.0000084 / 0.0000203
    predicted root value mean/max diff:    0.0090 / 0.0219

  So stop blaming checkpoint loading/root order for the big post-search delta.
  Treat it as MCTX/Gumbel search semantics, value-backup rules, or root-value
  summary differences. Keep MCTX profile-only and do not call it a drop-in
  replacement for LightZero CTree.
```

2026-05-23d H100 grid result:

```text
direct_ctree_gpu_latent is the real-search baseline:
  B1024/A16/sim8:  8291 steps/sec
  B1024/A16/sim16: 6992 steps/sec
  B1024/A16/sim32: 5314 steps/sec

direct_ctree_arrays is only a control:
  B1024/A16/sim16: 2613 steps/sec
  B1024/A16/sim32:  947 steps/sec

Ceilings show real but bounded headroom:
  service-tax vs direct at B1024/sim16: 1.73x
  mock vs direct at B1024/sim16:        2.23x
  service-tax vs direct at B1024/sim32: 2.00x
  mock vs direct at B1024/sim32:        2.84x
```

Current Amdahl read:

```text
The compact slab itself is no longer the proof problem. The hot part is real
search/control. In the patched B512/A16/sim16 direct row, aggregate slab time
was 2.076s over 20 steps; search was 1.288s, model 0.427s, H2D 0.362s. Search
is the largest bucket, but model/H2D and env/observation still matter.
```

Next move:

```text
Build or test a stronger CompactSearchServiceV1 backend. The falsifier is now
clear: beat direct_ctree_gpu_latent on B512/B1024 at sim16/sim32 while keeping
the same action-feedback and replay-index gates.
```

2026-05-23d MCTX compact service update:

```text
MctxCompactSearchServiceV1 now exists as a profile-only backend behind
CompactSearchServiceV1.

Tiny H100 smoke passed:
  ok=true
  jax backend=gpu
  LightZero=missing
  torch=missing
  compact_rollout_slab_total_roots=16
  compact_rollout_slab_committed_index_row_count=16
  search_impl=mctx_compact_search_service_profile_only_v0
  calls_train_muzero=false

This is not a speed claim and not a Coach path. It is the next falsifier for
whether replacing the CTree/list/control loop with a compiled JAX/MCTX search
body actually matters on the same compact slab denominator.
```

Next MCTX grid:

```text
H100, B512/B1024, A16, sim16/sim32, materialize_scalar_timestep=false,
compact_rollout_slab=true, hidden_dim=64, visual_channels=8.
```

2026-05-23d MCTX grid result:

```text
Same compact slab denominator, H100, A16, scalar materialization off:

shape              MCTX steps/sec   direct CTree baseline   speedup
B512/sim16         16,250           6,522                   2.49x
B512/sim32         14,306           4,177                   3.43x
B1024/sim16        20,557           6,992                   2.94x
B1024/sim32        16,255           5,314                   3.06x
```

Plain read:

```text
The old "maybe compiled/device search matters" hypothesis is now supported.
The correct next optimizer move is to widen this comparator and then decide
whether to prototype a real training-compatible compiled search path.

Do not promote this directly to Coach. It uses MCTX/JAX and a toy model, so it
answers an architecture-speed question, not a learning-quality question.
```

2026-05-23e matched direct check:

```text
Same 40 measured / 10 warmup shape:

shape              direct CTree   MCTX      speedup
H100 B512/sim16    5,109          12,886    2.52x
H100 B512/sim32    3,827          9,523     2.49x
H100 B1024/sim16   6,261          14,803    2.36x
H100 B1024/sim32   5,145          15,639    3.04x
L4 B512/sim16      4,370          13,802    3.16x
L4 B512/sim32      3,124          9,818     3.14x
L4 B1024/sim16     4,802          16,581    3.45x
L4 B1024/sim32     2,399          13,315    5.55x
```

Interpretation:

```text
The MCTX speed signal survives a matched direct-baseline rerun.
The next proof is longer H100 rows: 80 measured / 20 warmup.
If that still holds, the main implementation question becomes model parity:
can a JAX shadow of the real LightZero MuZero model match PyTorch outputs well
enough to use MCTX for collector search while keeping the PyTorch learner?
```

2026-05-23f strict H100 80/20 check:

```text
shape              direct CTree   MCTX      speedup
B512/sim16         5,864          11,826    2.02x
B512/sim32         4,781          8,667     1.81x
B1024/sim16        4,947          11,700    2.36x
B1024/sim32        4,400          13,964    3.17x
```

Updated optimizer read:

```text
The search-backend optimization is real enough to continue.
Use `1.8x-3.2x` as the current strict full-profile speedup range, not 10x.
The search sub-bucket itself is much faster, but Amdahl has moved the wall to
observation/env/handoff plus the unresolved model-ownership bridge.
```

Real model inventory:

```text
Current CurvyTron LightZero model class:
  lzero.model.muzero_model.MuZeroModel

Patched model surface:
  model_type=conv
  observation_shape=[4,64,64]
  action_space_size=3
  downsample=true
  norm_type=BN
  use_sim_norm=true
  self_supervised_learning_loss=true
  reward_support_size=601
  value_support_size=601
  support_scale=300

Parameter count from a local instantiation:
  7,995,781
```

Precomputed recurrent check:

```text
direct_ctree_gpu_latent_precomputed_recurrent is not a real training backend.
It deletes recurrent model calls with synthetic resident outputs.

Current H100 slab result:
  B1024/A16/sim16: 8875 steps/sec, 1.27x over real direct
  B1024/A16/sim32: 8055 steps/sec, 1.52x over real direct
```

Meaning:

```text
Recurrent model work matters, but deleting it alone does not unlock 5x-10x.
The search/control/list/CTree path remains the bigger architectural target.
```

## 2026-05-23b Strategy Reset

Current plain answer:

```text
We are not ready to recommend a new faster Coach training backend.
We are ready to take a bigger optimizer move inside the profile/validation lane.
```

The next move is not another small render tweak and not polishing the current
eager compact Torch search service. The fresh same-denominator H100 rows say
that service loses to direct CTree at both sim16 and sim32. The useful part is
the compact service/replay contract, not that backend implementation.

Current direction:

```text
build a compact search/dataflow slab that proves
observation -> search -> action -> env -> replay/sample
while keeping stock LightZero as the semantic oracle
```

Working note:

```text
docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/strategy_reorientation_20260523b.md
```

## 2026-05-23 Full Dataflow Reorientation

Current optimizer target:

```text
remove high-frequency CPU/GPU/control crossings from the collect/search path,
while keeping stock LightZero and compact replay/RND parity as validation edges
```

The latest compact Torch search-service row is useful but not promoted:

```text
B512/A16/sim16 H100:
  direct_ctree_gpu_latent:        ~4,966 steps/sec
  compact_torch_search_service:  ~5,575 steps/sec
  service_tax_probe:             ~5,853 steps/sec
```

Plain meaning:

```text
The compact service boundary is real and replay-proofed, but the eager Torch
search body is not the big win. It mostly relocates work into a tree/recurrent
loop. The next search move needs a genuinely compiled/fused fixed-shape search
body, an array-native CTree, or an MCTX/JAX-style comparator behind the same
CompactRootBatchV1 -> CompactSearchResultV1 contract.
```

Sync budget:

```text
Accept one selected-action readback per env tick while the environment remains
CPU. Do not accept repeated full observation copies, replay payload readbacks,
Python row/list materialization, or per-simulation recurrent output readback as
unmeasured "small" costs. Tiny payloads can still be expensive when they force
GPU synchronization at high frequency.
```

Scale reminder:

```text
At B512/P2, there are 1024 policy roots. A [4,64,64] uint8 stack is about
16 MiB for those roots; float32 is about 64 MiB. Actions, values, masks, and
visit policies are tiny by bytes. The hard cost comes from where those bytes
force synchronization and object materialization.
```

Current Amdahl warning:

```text
Buckets labeled "env" often contain observation/search-input handoff, stack
ownership, final-observation packing, and metadata/object packaging. Pure
CurvyTron mechanics are not proven to be the main wall. Likewise, compact
roots/sec can look good while stock replay, learner samples, and RND still
materialize full observation rows.
```

Living planning doc:

```text
docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/full_iteration_dataflow_designs_20260523.md
```

2026-05-23 architecture critique update:

```text
The best current 5x+ hypothesis is not "render on GPU" or "use an H100" by
itself. It is ownership:

CPU batched env rows
-> compact slab / resident observation stack
-> device-resident batched search/model service
-> CPU gets only selected actions per tick
-> replay/RND payloads flush in chunks
-> scalar LightZero rows only at validation/sample/debug edges
```

The selected action readback is allowed because the CPU env needs it. The
dangerous costs are repeated full observation copies, scalar timestep/object
fanout, per-simulation model-output readback, target-row materialization,
replay chunk stacking, and RND per-frame tensor/list materialization.

Promotion rule:

```text
No speed path counts until it proves observation k -> search root k -> selected
action k -> env transition k+1 -> replay row k -> learner-visible sample.
```

This proof must carry `env_row`, `player`, `policy_env_id`, legal mask, player
perspective, terminal final observation, and RND latest-frame identity through
the whole chain.

2026-05-23 wave-2 profile update:

```text
Same denominator, B512/A16, 80 measured + 20 warmup, compact replay proof on,
root_noise_weight=0.0:

H100 sim16:
  direct CTree:   5.47k steps/sec
  compact Torch:  4.05k steps/sec
  service-tax:    7.81k steps/sec
  mock ceiling:   7.46k steps/sec

H100 sim32:
  direct CTree:   3.14k steps/sec
  compact Torch:  2.67k steps/sec
  service-tax:    5.19k steps/sec
  mock ceiling:   9.17k steps/sec
```

Plain meaning:

```text
Do not spend the next main patch polishing the current eager compact Torch
service. It loses to direct CTree in this refresh. The headroom is still in the
service/data ownership shape: remove high-frequency CTree/list/control
boundaries, keep replay compact until sample/validation, and test fixed-shape
compiled search or array-native CTree behind the existing compact service
contract.
```

The smallest clean insertion point from the wave-2 code audit is:

```text
HybridBatchedObservationProfileManager.step
after actor payload merge, stack/final-observation capture, and root sidecar
construction
before materialize_lightzero_scalar_timestep
```

Use the existing chain:

```text
HybridCompactBatch -> CompactRootBatchV1 -> CompactSearchServiceV1
-> CompactSearchResultV1 -> CompactReplayIndexRowsV1
```

Do this first in the profile path, not the live Coach lane.

## 2026-05-23 Batch Scaling Update

Fresh H100 refresh-on rows:

```text
B512  sim16: 29.1k roots/sec, env 71.8%, search 18.7%
B512  sim32: 24.5k roots/sec, env 57.2%, search 35.8%
B1024 sim16: 53.9k roots/sec, env 75.2%, search 16.9%
B1024 sim32: 36.4k roots/sec, env 65.0%, search 28.6%
B2048 sim16: 57.4k roots/sec, env 83.6%, search 10.5%
B2048 sim32: 40.3k roots/sec, env 76.6%, search 18.0%
```

Plain meaning:

```text
Bigger batches improve aggregate throughput, but the improvement is not
linear. B2048 only modestly beats B1024 on this shape, because the env and
observation handoff grow too. B1024 remains the clean current profiling
denominator; B2048 is a stress test for compact-state ownership.
```

New validation state:

```text
CompactReplayIndexRowsV1 now composes into the repo learner-facing sample batch
and matches the immediate replay path under the same sample seed. The remaining
trainer-facing proof is stock LightZero GameSegment/GameBuffer parity or a
matched stock-vs-candidate full-loop smoke once the compact service backend is
real enough.
```

2026-05-23 validation update:

```text
Compact index rows now also pass an opt-in stock LightZero target-hook parity
gate when `lzero` is installed locally. The test materializes compact rows,
builds real `GameSegment`s, pushes them into `MuZeroGameBuffer`, then compares
the stock private reward/value/policy target hooks to the materialized compact
rows.
```

What this does and does not prove:

```text
Proves: compact materialized rows are compatible with stock LightZero target
construction at the native segment/buffer hook level.

Does not yet prove: full trainer-facing compact service safety.

Later update: public `MuZeroGameBuffer.sample(...)` parity is now covered by an
opt-in local canary when `lzero` is installed.
```

2026-05-23 speed-path wiring update:

```text
The existing dense Torch fixed-shape MCTS profile probe now emits compact search
arrays and validates through `CompactSearchServiceV1`, the same boundary used
by direct CTree, service-tax, and mock modes. This makes the next H100 profile
row a real same-boundary speed falsifier rather than another separate side
channel.
```

## 2026-05-23 Compact Render-State Buffer Read

The opt-in compact render-state buffer proved one thing and falsified another:

```text
proved:   renderer production-to-compact conversion can be removed cleanly
falsified: parent-side compact buffer writes are an automatic win
```

Fresh no-borrow B1024 rows:

```text
sim16 copied production state: 26.3k roots/sec
sim16 compact state buffer:    35.8k roots/sec
sim16 borrowed production:     48.6k roots/sec

sim32 copied production state: 29.5k roots/sec
sim32 compact state buffer:    25.3k roots/sec
sim32 borrowed production:     37.9k roots/sec
```

Plain meaning:

```text
The compact buffer path moves work from renderer conversion into actor render
state writes. It can help when the renderer conversion/H2D/delta side is the
larger tax, but it can lose when full parent-side trail writes dominate. The
borrowed single-actor path is still faster because it does no parent trail copy.
The compact buffer is therefore a diagnostic, not a promoted speed path.
```

Next architecture read:

```text
Do not optimize by adding another parent-side full trail copy. The next real
render-state move must avoid both conversion and parent copying: actor-owned
compact x/y state, renderer-side direct visual_trail_pos consumption, or a
small event/delta log from the env runtime.
```

Ramanujan sidecar follow-up, 2026-05-23:

```text
The compact buffer is expensive because `_write_persistent_compact_render_state_rows`
writes the full visual trail capacity from actor env state into parent compact
buffers every actor step. That includes x/y split, dtype conversion, masks,
cursor, heads, colors, and bonuses. The env append path already knows the tiny
incremental event, but the diagnostic path expands that O(new points) event into
an O(full trail history) parent copy.
```

Ranked implication:

```text
P0 if observation handoff stays hot: env-emitted render delta/event log from
the append/reset sites.

Fallback: actor-owned compact x/y render state borrowed by the renderer.

Small falsifier only: renderer direct from production visual_trail_pos.

Not a main lane: more patches to the parent compact buffer, except for quick
live-prefix trimming measurements.
```

## 2026-05-23 Current Dataflow Read

Fresh current-code closed-loop H100 matrix:

```text
H100, B1024/P2, body4096, h64/depth16, loop96,
native actor buffer, resident GPU stack, replay-index on,
resident sync off, no root observation copy.

refresh-on sim16: 53.9k roots/sec, env 75.2%, search 16.9%
refresh-on sim32: 36.4k roots/sec, env 65.0%, search 28.6%
refresh-off sim16: 91.8k roots/sec, env 56.2%, search 29.6%
refresh-off sim32: 57.9k roots/sec, env 39.5%, search 49.2%
```

Plain meaning:

```text
Observation refresh is still expensive, but deleting it is only a 1.6-1.7x
ceiling here. Raw GPU draw is tiny. The expensive refresh-on leaves are
production-to-compact conversion, delta pack, renderer H2D/update, public
packaging, and search. At sim32, search is already a first-class wall.
```

The newest durable same-denominator compact search grid is:

```text
H100, B512/A16, 80 measured steps, 20 warmup, compact replay proof on,
host_uint8_pinned input, no scalar timestep materialization.
```

| sims | mode | measured sec | steps/sec | probe sec | probe roots/sec | model sec | search sec | compact proof sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `direct_ctree_gpu_latent` | `18.82` | `4,353` | `7.734` | `10,592` | `2.047` | `5.924` | `5.577` |
| 16 | `dense_torch_mcts` | `15.23` | `5,380` | `3.687` | `22,219` | `1.576` | `1.942` | `4.086` |
| 16 | `service_tax_probe` | `10.94` | `7,487` | `2.041` | `40,135` | `1.642` | `0.228` | `4.082` |
| 16 | `mock_search_service` | `11.78` | `6,955` | `0.615` | `133,107` | `0.362` | `0.000` | `5.240` |
| 32 | `direct_ctree_gpu_latent` | `30.15` | `2,717` | `14.096` | `5,812` | `3.836` | `11.523` | `8.738` |
| 32 | `dense_torch_mcts` | `31.75` | `2,580` | `9.230` | `8,875` | `2.980` | `6.058` | `6.535` |
| 32 | `service_tax_probe` | `11.97` | `6,847` | `3.478` | `23,553` | `2.911` | `0.422` | `4.087` |
| 32 | `mock_search_service` | `14.38` | `5,696` | `0.519` | `157,708` | `0.360` | `0.000` | `9.011` |

Durable local results:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-direct-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-dense-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-service-tax-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-mock-20260523b
```

Current Amdahl read:

```text
Direct LightZero CTree is the slow real search boundary. Dense Torch is not a
promotion candidate: measured steps/sec is 1.24x at sim16 but 0.95x at sim32,
and it is not LightZero CTree semantics.

The service-tax row is the useful signal. It is 1.72x measured over direct CTree
at sim16 and 2.52x at sim32 while still paying real model calls. That points at
the current direct CTree control/list/object boundary.

The mock row is the ceiling. It is not search, but it shows the rest of the
compact plumbing can be much cheaper than direct CTree if the per-simulation
CPU/GPU/object loop is removed.

`compact_service_replay_proof_sec` is validation overhead in this profile. It
is useful for trust, but it is not a promoted trainer hot-path claim.
```

MCTX/JAX sidecar refresh:

```text
Profile-only, not LightZero-equivalent:
  curvytron_hybrid_compact_visual_sample, H100, B512/P2/body1024,
  closed_loop_steps=24, native_actor_buffer=true,
  compact_root_copy_observation=false, compact_visual_resident_sync=false.

sim16: 27.6k active roots/sec, search 0.149s, env_step 0.647s
sim32: 22.2k active roots/sec, search 0.362s, env_step 0.650s
```

Plain MCTX read:

```text
The JAX/MCTX sidecar confirms the bigger architecture path is plausible. It is
much faster than direct LightZero CTree on similar compact visual roots, but it
uses a toy JAX model/search and is not training advice.

Once search gets fast, Amdahl shifts back to env/observation. In these sidecar
rows, env_step_sec is the slowest bucket. The real 5x+ path must pair
array-native search with compact env/observation ownership.

Implementation note: the MCTX benchmark now emits a
`compact_search_service_profile` row labeled
`mctx_hybrid_compact_visual_search_service` with explicit `profile_only`,
`not_lightzero_ctree`, and `not_train_muzero` flags. This gives us a comparable
profile row without pretending MCTX is a stock LightZero backend.
```

Validation read:

```text
The compact service boundary now has local replay/action-feedback tests, but
promotion still needs RND and player-perspective coverage. Until then, these are
profile-only rows, not Coach training recommendations.
```

Validation update:

```text
The first combined identity/RND/perspective tripwire now exists in the compact
replay proof. It verifies that compact replay rows keep the same env row,
player, policy_env_id, compact root row, and RND latest-frame slice as the
search result. This catches the main "search one root, train another root" class
of bug.
```

2026-05-23 proof update:

```text
The combined RND/terminal compact replay canary now exists. It builds compact
service index rows, materializes them, verifies terminal rows use the final
observation instead of the latest live observation, feeds the materialized
observations through the actual CurvyRNDRewardModel collect/train/estimate
path, checks that predictor weights move, checks that the target network hash
stays fixed, and checks that positive intrinsic reward changes the target
reward.

Validation:
  ruff passed on the touched optimizer/test files
  compact/hybrid/RND focused pytest set: 180 passed

The next proof then added an opt-in public LightZero sampler canary. It pushes
materialized compact rows into real `MuZeroGameBuffer`, calls public
`buffer.sample(...)`, maps sampled transition ids back to compact row ids, and
compares sampled observation, target policy, target reward, and zero-model value
targets. This closes the first public sampler-edge proof when `lzero` is
available locally.

The next proof added a real direct-CTree compact-service closed-loop smoke. It
builds roots from a hybrid profile step, calls the compact service, converts the
selected actions into the next joint action, steps the env, and compares compact
deferred rows plus learner sample batches against the trusted immediate path.

Validation:
  tests/test_compact_search_replay_contract.py -k public_sample: 1 passed
  tests/test_compact_search_replay_contract.py: 15 passed
  tests/test_source_state_batched_observation_boundary_profile.py
    -k real_direct_ctree_compact_service_drives_next_step_and_matches_rows: 1 passed

Remaining promotion gap: matched stock-vs-candidate full-loop smoke with the
new candidate compact backend. The direct CTree compact-service smoke is green,
but it is still not a Coach launch recommendation.
```

Side-agent synthesis:

```text
Faraday: RND and compact search each had useful separate coverage; promotion
needed a combined proof. We added the first identity/RND latest-frame piece.

Hegel: the practical backend ladder is fixed-A=3 CPU CTree SoA compatibility,
then fixed-shape Torch device-tree search, then JAX/MCTX sidecar. The Torch
device-tree lane is the first real speed attempt if parity gates stay green.

Kierkegaard: large payloads are visual stacks, but the current latency wall is
small payloads forced through repeated per-simulation CPU/GPU syncs. Accept one
action sync per env step; avoid sync/listification inside every MCTS sim.
```

## 2026-05-22 Current Dataflow Read

The newest compact sidecar comparison is now:

```text
H100, B512/A16, 60 measured steps, 15 warmup, compact replay proof on:
  mock_search_service:       17,711.9 steps/sec
  service_tax_probe:         12,461.6 steps/sec
  direct_ctree_gpu_latent:    7,155.7 steps/sec
```

Plain meaning:

```text
mock_search_service is a fake-search ceiling.
service_tax_probe pays real initial and recurrent model calls but no real CTree.
direct_ctree_gpu_latent is the current real LightZero CTree comparator.
```

Ratios:

```text
mock / direct:        2.48x
service_tax / direct: 1.74x
mock / service_tax:   1.42x
```

Current Amdahl read:

```text
Raw GPU drawing is not the wall. Direct CTree spent about 5.05s of 8.59s in
the LightZero MCTS arrays boundary on the fresh row. Observation/renderer stack
update was about 1.26s, actor step wall was about 1.55s, and compact replay
proof was only about 0.17s.

The next large target is the search/dataflow boundary:
  compact roots -> model/search -> compact action/visit/value -> replay rows.
The service-tax row says model calls matter, but the remaining CTree/list/control
path is also material. A 5-10x result still requires changing ownership across
search, replay, RND, and scalar LightZero object boundaries; it will not come
from another renderer-only patch.
```

Validation read:

```text
The compact service path is profile-only until a closed compact-loop parity
test proves that search actions, visit policies, root values, replay rows, RND
inputs/rewards, terminal final observations, and player views all attach to the
same record as the current trusted path.
```

## 2026-05-22 Direct Root Extraction Update

Latest plain finding:

```text
The expensive "search payload" wall was mostly our root-value extractor using
MCTX's summary/materialization path. Reading the root node value directly from
the search tree makes root-value extraction small.
```

Same profile-only denominator unless noted:

```text
H100, B1024/P2, loop24, no-death compact loop,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack, no root observation copy,
explicit resident-stack sync off,
vectorized delta pack off.
```

Before the fix:

```text
sim16 full materialization, replay off:
  44.8k roots/sec, root_value_extract 0.310s, total 1.097s
sim32 full materialization, replay off:
  39.8k roots/sec, root_value_extract 0.266s, total 1.236s
```

After the fix:

```text
sim16 full materialization, replay off:
  71.2k roots/sec, root_value_extract 0.014s, total 0.690s
sim32 full materialization, replay off:
  43.5k roots/sec, root_value_extract 0.021s, total 1.131s

sim16 replay-valid, replay index on:
  55.0k roots/sec, replay_index 0.010s, root_value_extract 0.019s,
  env_step 0.656s, search 0.157s, total 0.894s
sim32 replay-valid, replay index on:
  38.1k roots/sec, replay_index 0.012s, root_value_extract 0.024s,
  env_step 0.771s, search 0.418s, total 1.289s

Longer stability rows:
  loop48 sim16 replay on/off: 44.0k / 70.4k roots/sec
    This mismatch came from env/render time variance, not replay_index time.
  loop96 sim16 replay on/off: 50.6k / 53.6k roots/sec
    This is the cleaner current read: replay-valid row construction is not the
    big wall on this shape.
  loop48 sim32 replay on/off: 46.9k / 45.5k roots/sec
    Also consistent with replay rows being small.
```

Run ids:

```text
replay off, sim16 full: ap-h8eXVxgdCZN2LIopHF7B2u
replay off, sim16 deferred: ap-6NR7HJCrkjYufLOIKfCs2t
replay off, sim32 full: ap-Oq9wRkcR6m9g8TpDVCoKzM
replay off, sim32 deferred: ap-6UK2GEsLAxna5qU7ZmSrVy
replay on, sim16: ap-Y6LC6pbtD7ZbiIvXPb7d3D
replay on, sim32: ap-QDsbC1GNnAiyGnsrIQRXU6
loop48 replay on, sim16: ap-yqMpUJbZwFMhYourDP65u5
loop48 replay off, sim16: ap-n6voAA6ytAJTwv2x5H6d4f
loop48 replay on, sim32: ap-x6Fy3jNx0jfJWkucx7utG0
loop48 replay off, sim32: ap-jyGhPww8F3PgQfhnIf6jkQ
loop96 replay on, sim16: ap-X4qFnFS9XbkuHKIO6vmwci
loop96 replay off, sim16: ap-snG5Z3yGKzGWo8hdTIKqfp
```

Current Amdahl read:

```text
Payload extraction is no longer the main wall. Replay row construction is also
small in the replay-valid rows; the loop96 replay-on/off pair is the best
sanity check for that. The biggest wall is env_step_sec, but that name mostly
means observation/search-input handoff: compact render-state work, delta pack,
H2D/update, resident stack/root ownership, and public packaging.

At sim16, env/observation handoff dominates. At sim32, MCTS search is also a
large wall. This means the next useful work should target the next-search-input
boundary and search scaling, not serial deferred payload flushing.
```

Operational decision:

```text
Keep action-only, deferred-payload, and overlap-payload modes as diagnostics.
Do not quote them as training speed. The default replay-valid profile should
use direct root-node extraction and normal compact replay rows.

The thread-overlap canary is not recommended: it hid wait time but inflated
env/render time through contention.
```

Full dataflow/sync subagent read:

```text
The selected-action readback is mandatory while the env is CPU-owned, but it is
tiny. At B1024/P2 the selected action payload is about 8 KiB, the invalid mask
is about 6 KiB, visit policy is about 24 KiB, and root values are about 8 KiB.

The large objects are the latest frame (~8 MiB), resident stack (~32 MiB), root
observation if copied (~32 MiB), and visual trail arrays if copied. Current
best rows already avoid the full root observation copy and parent visual-trail
copy. The remaining repeated work is mostly CPU compact-state build, CPU delta
pack, H2D delta/compose payload, resident stack/root input ownership, and
public packaging.
```

Validation subagent read:

```text
The row-level compact replay tests are strong. The remaining validation gap is
an end-to-end multi-record canary that proves a faster closed-loop profile row
materializes exactly the same replay rows as the immediate replay-valid path.
This matters most for any future deferred/overlap/action-only variant. The
current default replay-valid path is covered by focused local tests, including
the new direct root-node extractor test.
```

External/framework read:

```text
Fast systems keep hot state in contiguous owners, batch model/search work, and
delay scalar framework objects to validation/logging edges. For CurvyTron now,
the practical version is not "rewrite everything" immediately; it is to make a
compact state/search-input owner so each CPU env tick emits the smallest delta
needed for the next GPU search input.
```

Host-overhead audit follow-up:

```text
We added `compact_batch_build_sec` and `batched_stack_probe_wall_sec` so the
capture/probe path no longer hides inside env_step_sec.

Fresh loop48 sim16 row after the timer patch:
  run: ap-XQkmXe2e1L18k2ehqphZIY
  roots/sec: 54.4k
  compact_batch_build_sec: 0.0016s over 48 steps
  batched_stack_probe_wall_sec: 0.0006s over 48 steps

Conclusion: compact batch construction is not the current wall in the repeated
closed-loop denominator. The visible repeated renderer/search-input leaves are
still production-to-compact, delta pack, H2D, resident stack update, public
packaging, and actual CPU mechanics.
```

Async H2D canary:

```text
We changed the existing async device-only profile flag so it also defers
`jax.device_put(...).block_until_ready()` inside the persistent renderer.

Paired loop96 H100 rows:
  sim16 baseline: 50.4k roots/sec, ap-72TeEccePu9pjaGgaD35lj
  sim16 async H2D: 53.2k roots/sec, ap-sDRS3TY7mm4EWhXBeWN9Qv
  sim32 baseline: 40.0k roots/sec, ap-o3ZC9X3cX4AdC6WN2wYNbV
  sim32 async H2D: 41.9k roots/sec, ap-WPk47lTw1F6FCIaOCCSORz

Decision: keep this as an opt-in profile flag. It is a small win, not the big
architecture move.
```

## 2026-05-22 Search Payload Readback Update

Superseded by the direct root extraction update above. This section records the
first read before we found that `_extract_mctx_root_values` was taking the
expensive MCTX summary/materialization path.

Latest profile-only denominator:

```text
H100, B1024/P2, loop24, no-death compact loop,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack, no root observation copy,
explicit resident-stack sync off,
vectorized delta pack off.
```

New plain finding:

```text
The CPU env only needs selected actions each step. Replay/training also needs
visit policies and root values, but forcing those payloads onto CPU inside the
action loop is expensive.
```

Matched rows:

```text
sim16:
  full materialization, replay off: 44.8k roots/sec
    root_value_extract: 0.310s, total: 1.097s
  action-only ceiling: 68.8k roots/sec
    total: 0.714s
  deferred payload flush: 42.9k roots/sec
    action loop: 0.834s, flush: 0.313s, total: 1.146s

sim32:
  full materialization, replay off: 39.8k roots/sec
    root_value_extract: 0.266s, total: 1.236s
  action-only ceiling: 53.3k roots/sec
    total: 0.923s
  deferred payload flush: 39.3k roots/sec
    action loop: 1.010s, flush: 0.240s, total: 1.250s
```

Interpretation:

```text
Action-only is real as a ceiling, but not replay-valid. The deferred flush row
proves the cost comes back when visit policies/root values are actually copied.
So a simple "copy it later" implementation is not enough.

The real architecture target is overlap or ownership:
  - read selected actions immediately for the CPU env;
  - keep visit policies/root values resident or in a chunked payload owner;
  - commit replay rows only when the full payload is ready;
  - avoid exposing partial rows to the sampler.
```

Current Amdahl read after this split:

```text
At sim16, env/observation handoff is still the largest visible bucket, and
search-payload materialization is also large. At sim32, search itself becomes a
large wall, and payload extraction remains large enough to matter.

This is why the next big move is not another renderer micro-patch. It is a
clean collector/search/replay boundary where the action-critical path is small
and replay payloads are resident, chunked, or overlapped.
```

## 2026-05-22 Fresh Sync/Async Retest

Latest profile-only denominator:

```text
H100, B1024/P2, actor_count=1, loop24, no-death compact loop,
borrow_single_actor_render_state=True, resident GPU stack,
no root observation copy, replay-index on.
```

Fresh rows:

```text
borrowed + resident stack + explicit resident sync on:
  sim16: 45.4k roots/sec
  sim32: 35.7k roots/sec

borrowed + resident stack + explicit resident sync off:
  sim16: 50.4k roots/sec
  sim32: 43.3k roots/sec

borrowed + resident stack + explicit resident sync off
+ async internal renderer device-only profile:
  sim16: 50.2k roots/sec
  sim32: 40.9k roots/sec
```

Vectorized delta-pack A/B:

```text
Guarded vectorized CPU delta-pack fast path, current-code A/B:
  sim16 exact -> vectorized: 51.9k -> 53.1k roots/sec, about 1.02x
  sim32 exact -> vectorized: 45.1k -> 37.9k roots/sec, regression/noise

Decision:
  keep vectorized delta pack opt-in, not default.
  recommended profile path uses the exact delta pack.
```

Plain read:

```text
The current best simple profile mode is borrowed render state + resident GPU
stack + no explicit resident-stack sync. The extra async internal renderer flag
did not improve total wall time, so do not promote it. It is only a canary that
shows the raw renderer wait is not the next big wall.
```

Current Amdahl read:

```text
Raw GPU drawing is already small. Game mechanics is already small. The wall is
now preparing and handing off the next search input: production-to-compact,
delta pack, H2D/update, stack/root ownership, public packaging, MCTS search,
and residual Python/control overhead. The next architecture move should make a
compact render/search state owner more resident, not keep shaving renderer
block_until_ready calls.

After the A/B, delta pack is not a promising default speed lever. The visible
remaining leaves are renderer H2D, production-to-compact, public packaging,
MCTS search, and a large unlabeled residual/control bucket.
```

Local code read:

```text
The compact closed loop still has a CPU -> GPU -> CPU rhythm.

1. MCTX/search produces selected actions on the GPU.
2. Selected actions are read back because the CPU env must step.
3. CPU VectorMultiplayerEnv advances the game and writes small sidecars.
4. The renderer converts CPU production state into compact render state.
5. The renderer packs trail deltas and compose state on CPU.
6. Those deltas/compose arrays move to the GPU.
7. The GPU persistent framebuffer updates and produces the latest frame.
8. Resident stack shifts on the GPU.
9. Search consumes the resident stack plus a small legal-action mask.
```

Large data:

```text
At B1024/P2, the observation stack is about 32 MiB as uint8
([1024,2,4,64,64]). The latest frame is about 8 MiB. Legal masks and selected
actions are tiny. So the remaining target is not selected-action readback; it
is avoiding repeated large-ish render-state/materialization work and keeping
the search input resident in the shape MCTX already consumes.
```

Next architecture hypothesis:

```text
The actor/env should emit compact render/search deltas directly, or the
renderer should own a compact CPU/GPU state that is updated in place. Today the
profile still rebuilds compact_state and delta_state from production arrays
every step. That is the next real target if we want another material win.
```

## 2026-05-22 Borrowed-State Update

Note: this section records the first borrowed-state wave. The fresh sync/async
retest above supersedes the exact sync-off numbers and recommendation.

The current best profile-only compact visual row is now:

```text
H100, B1024/P2, actor_count=1, resident GPU stack,
borrow_single_actor_render_state=True, no root observation copy,
sim16: 51.8k roots/sec with explicit resident-stack sync on
sim16: 54.7k roots/sec with explicit resident-stack sync off
sim32: 38.5k roots/sec with explicit resident-stack sync on
```

What changed:

```text
The actor no longer copies visual_trail_* into parent render-state buffers.
For this no-death single-actor profile canary, the renderer borrows the actor's
env.state directly. If terminal/autoreset rows appear, the mode fails closed.
```

Plain read:

```text
This validates the state-ownership hypothesis. The biggest previously named
copy leaf, actor_render_state_write_sec, is now zero in the borrowed rows and
total roots/sec improves, not just the timer label.
```

Measured same-denominator gains:

```text
sim16 resident pre-borrow: 30.3k -> 45.4k roots/sec, about 1.50x
sim32 resident pre-borrow: 26.8k -> 35.7k roots/sec, about 1.33x
key-filtered retest sim16 copied -> borrowed: 34.1k -> 51.8k, about 1.52x
lazy-sync sim16 borrowed: 48.6k -> 54.7k, about 1.13x
lazy-sync sim32 borrowed: 36.0k -> 27.9k, a regression
```

Updated Amdahl read:

```text
Raw drawing is not the wall. Actual game mechanics is not the wall. The next
wall is still preparing the next search input: renderer delta-pack/H2D/update,
observation/stack ownership, public packaging, MCTS search, and the remaining
unlabeled profile residual. The next useful work is to make more of the compact
render/search state persistent and resident. Lazy resident-stack sync is only
an attribution switch for now because the first sim32 row regressed.
```

## 2026-05-22 Orientation Reset

### Late 2026-05-22 Amdahl Read

Current repeated compact-loop profile shape:

```text
H100, B1024/P2, sim16, loop16, native actor buffer,
real renderer-backed compact visual observations.
```

Fresh matched rows say:

```text
host stack + replay rows:      20.7k active roots/sec
host stack, no replay rows:    17.0k active roots/sec
resident GPU stack + replay:   16.2k active roots/sec
resident GPU stack, no replay: 17.9k active roots/sec
```

Plain read:

```text
Replay-index construction is not the wall. It is around 0.3% when enabled.
Resident stack removes the obvious observation H2D/readback slice, but in this
matched run it did not win overall because root build and production-to-compact
work still dominate. The current bottleneck is env_step_sec, but that name is
misleading: it mostly means render-state write, production-to-compact,
renderer/stack update, and root-batch handoff. Actual game physics is small.
```

The active ceiling test is now:

```text
step the compact env but skip observation refresh entirely
```

That is not a training lane. It prices the maximum possible speedup from
deleting the current observation/render-state wall. If that does not jump, the
project needs a broader architecture change than observation residency. If it
does jump, the next implementation target is a lower-copy compact state owner,
not another MCTS micro-patch.

Ceiling result:

```text
sim16: 20.7k -> 48.6k active roots/sec, about 2.35x
sim32: 17.9k -> 32.1k active roots/sec, about 1.80x
```

Interpretation:

```text
The observation/render-state wall is real. It is not 10x by itself. When it is
removed, root-batch construction, H2D, and search become the next visible
buckets. The next patch is lower-copy root-batch observation materialization;
the next architecture move is a compact resident state owner that feeds search
and replay without rebuilding large root observations every decision.
```

Patch result after that:

```text
Fast visual compact-state adapter:
  production-to-compact fell from roughly 0.37-0.52s to about 0.054-0.057s.

No-copy root observation:
  root-build fell from about 0.06-0.25s to about 0.009s in the sim16 rows.

Best refresh-on compact loop so far:
  20.7k -> 26.6k active roots/sec, about 1.29x.
```

Plain read:

```text
These are real fixes, but they do not change the final diagnosis. The current
wall has moved inward to actor render-state writes and observation/stack
ownership. Retest resident GPU stack after the no-copy patch, because the
earlier resident rows were measured while root-build copying was still hot.
```

Resident retest result:

```text
After root-copy was removed, resident stack wins in the matched profile rows.

sim16:
  host no-copy best:     26.6k active roots/sec
  resident no-copy:      31.6k active roots/sec
  profile speedup:       about 1.19x

sim32:
  host no-copy:          21.2k active roots/sec
  resident no-copy:      28.9k active roots/sec
  profile speedup:       about 1.36x
```

Updated Amdahl read:

```text
The phrase env_step_sec is too broad. In the current closed compact loop it
means "advance env and prepare the next search input." The true game-mechanics
leaf, actor_env_runtime_sec, is only about 0.055-0.089s in the latest B1024
rows. The larger leaves are actor_render_state_write_sec and observation_sec.

So the next target is not game mechanics alone and not replay-index. The target
is state ownership: avoid copying render/search state through parent buffers
and keep the latest stack in the resident layout that search consumes.
```

Fresh split after grouped render-state timers:

```text
H100, B1024/P2, sim16, loop16, native actor buffer, no root observation copy.

Resident stack row ap-tPtlc8abDmtUsjLuqYuNOi:
  37.9k active roots/sec
  env_step_sec fraction: 64.2%
  search_sec fraction:   11.8%
  actor_env_runtime_sec: 0.052s
  actor_render_state_write_sec: 0.245s
    visual_trail copy: 0.244s
    player copy:       0.001s
  observation_sec:      0.193s
  raw GPU draw:         0.003s

Host stack row ap-SKFkPMiLLBRws9KNyTDPyJ:
  25.7k active roots/sec
  env_step_sec fraction: 67.6%
  search_sec fraction:   7.9%
  actor_env_runtime_sec: 0.098s
  actor_render_state_write_sec: 0.323s
    visual_trail copy: 0.321s
    player copy:       0.002s
  observation_sec:      0.374s
```

Plain read:

```text
The renderer/kernel itself is not the wall in these rows. The biggest named
leaf is copying visual_trail_* arrays into parent render-state buffers, followed
by observation/stack handoff. Actual game mechanics is roughly 10% of the
inclusive env_step_sec bucket. If we optimize physics now, Amdahl says the
whole loop barely moves. If we remove or shrink full visual-trail copying, the
whole loop can move.
```

Borrowed render-state falsifier:

```text
The profile-only borrowed single-actor render-state mode is already wired and
locally tested. It skips the actor env.state -> parent native_render_state copy
when actor_count=1 and no terminal/autoreset row appears.

Matched H100 B1024/P2 loop24 rows:

host sim16:
  copied:   21.3k roots/sec
  borrowed: 35.7k roots/sec
  win:      1.67x

resident sim16:
  copied:   32.3k roots/sec
  borrowed: 44.8k roots/sec
  win:      1.39x

resident sim32:
  copied:   34.6k roots/sec
  borrowed: 36.1k roots/sec
  win:      1.04x

resident refresh-off ceiling:
  sim16:    57.1k roots/sec
  sim32:    37.7k roots/sec
```

Updated Amdahl read:

```text
Borrowed render state is a real sim16 win and should stay in the optimizer
profile lane. It also proves the earlier diagnosis was right: the
visual_trail_* parent-buffer copy was not fake. But at sim32 the same patch is
almost exhausted because search/control/residual overhead is now visible. The
next phase should measure and attack observation renderer handoff plus
search/control topology, not keep polishing the removed parent copy.
```

Current plain picture:

```text
The renderer was a real problem earlier, but it is no longer the only thing
worth staring at. The bigger wall is the shape of the collect/search/replay
boundary: batches are repeatedly turned into Python objects, LightZero public
collect/search owns CPU tree work, and compact arrays do not yet stay alive all
the way into replay/learner-facing rows.
```

Next ownership note:

```text
The next aggressive-but-clean patch should be profile-only compact render-state
ownership. The manager should feed the persistent GPU renderer from already
compact render buffers instead of rebuilding that compact state from a
production-state dict every hot step. This attacks actor_render_state_write plus
observation/stack handoff without changing MuZero search, targets, rewards, or
replay semantics.
```

Keep/kill rule:

```text
Best current refresh-on row: 26.6k active roots/sec.
Observation-refresh-off ceiling: 63.6k active roots/sec.
Keep the state-ownership patch only if matched sim16/sim32 closed-loop rows
move total roots/sec by at least ~1.2x. Timer movement alone is not enough.
```

See
[subagent_next_state_ownership_patch_20260522.md](subagent_next_state_ownership_patch_20260522.md).

Borrowed render-state result:

```text
The first ownership canary passed. In profile-only native_actor_buffer +
actor_count=1 rows, the manager can borrow the actor env.state directly for
the persistent renderer instead of copying visual_trail/player/bonus arrays to
parent render buffers.

H100 B1024/P2/loop24/no-copy/replay rows:
  host stack sim16:        26.6k -> 32.8k roots/sec, about 1.23x
  resident stack sim16:    32.9k -> 48.6k roots/sec, about 1.48x
  resident stack sim32:    24.0k -> 36.0k roots/sec, about 1.50x
```

Updated read:

```text
Actor render-state copy was a real wall. It is now removed in the clean
single-actor profile case. The best refresh-on compact row is close to the
old refresh-off ceiling, so the next small target is not another parent-copy
fix. It is renderer delta-pack/H2D/update plus resident stack synchronization
and sampled host mirrors.
```

Recent correction:

```text
The flat/vector sample had its own separate waste bug. Batched ray observation
scanned the entire allocated trail capacity instead of the live
body_write_cursor prefix. That made a B512 short sample spend seconds in ray
math over inactive slots. The code now trims to max(body_write_cursor) before
ray casting.
```

What not to confuse:

- Vector ray-trim speed is huge for the flat vector sample.
- Persistent GPU visual rendering and direct64 paths are still useful.
- Direct CTree train-profile hooks are useful but only gave about `1.28-1.31x`
  in matched train profiles.
- Guarded MCTX/JAX rows show 10x-class search-boundary headroom, but they are
  still profile-only and not Coach launch advice.

Next big test:

```text
Real compact CurvyTron visual roots -> CompactRootBatchV1 -> MCTX/JAX search
-> CompactSearchResultV1, with legal masks and row/player identity validated.
```

That test now exists and passed.

Latest correction:

```text
The closed-loop env_step_sec bucket is not mostly "game physics." A timing-split
smoke showed actual VectorMultiplayerEnv runtime around 0.085s while actor
render-state write was around 1.05s and observation/renderer/stack work was
around 0.98s. The smoke was not a matched speed row, but it is a strong
directional correction: the next wall is state handoff and observation
ownership, not another search-only patch.
```

Immediate optimizer posture:

- Keep MCTX/search rows only as guardrails; they are no longer the main
  denominator.
- Keep the persistent GPU renderer, live-prefix trim, and native actor buffer;
  they removed real waste but did not end the Amdahl wall.
- The next big move should make compact render/observation state live in the
  same shape the search consumes. Host materialization should become a sampled
  validation edge, not mandatory hot-loop work.
- Any claim must say which currency it is in: actual Coach training,
  stock/full-loop profile, or profile-only compact boundary.

Current read from the first rows:

```text
Persistent GPU render + compact visual MCTX is fast enough to keep pursuing.
B256/P2/sim16 produced about 70k fresh-boundary active decisions/sec and passed
CompactRootBatchV1/CompactSearchResultV1 validation. CPU oracle rendering was
the wrong speed surface; it made host setup dominate.
```

The next Amdahl question is the replay/learner edge:

```text
MCTX selected actions must drive the next compact env step and produce
CompactReplayIndexRowsV1 without turning back into scalar LightZero timesteps.
```

If that stays clean, the aggressive path is a compact search/replay service. If
it does not, we need to find exactly which replay/target/learner boundary kills
the speed before touching the trainer again.

## 2026-05-21 Current Optimizer Read

The persistent GPU framebuffer changed the bottleneck map.

Plain version:

```text
Before: long rows were mostly redraw old trail history.
After: long rows are split between env collision/body work and host stack work.
```

The important H100 profile-only numbers:

- B512/100 dynamic direct64 surface step: about `81ms`.
- B512/100 persistent direct64 surface step: about `45ms`.
- B512/500 persistent direct64 surface step: about `79ms`.
- In that B512/500 row, render was only about `16ms`; env step and stack
  update were each about `39ms`.
- After cursor-bound collision scanning, the same B512/500 row dropped to about
  `66ms`; env step dropped from about `39ms` to about `14ms`, while stack update
  stayed around `45ms`.

Current Amdahl read:

1. Persistent GPU rendering is a real win, but it already moved the wall.
2. Cursor-bound body collision scanning is a real small win and should stay if
   the broader fidelity slice keeps passing.
3. The next stack lane is not "just use a ring buffer" unless the downstream
   policy/search path can consume it without immediately materializing the same
   float32 chronological stack on host.
4. A larger win still needs a batched boundary that keeps observation/model
   work batched longer; the current profile paths still scalarize for
   LightZero-shaped timesteps.

What this means for priorities:

- Keep the persistent renderer profile-only until final-observation,
  autoreset/partial-row, map-clear/prefix-mutation, and backend-label gates are
  closed.
- Treat host stack movement/device handoff as the next architecture lane.
- The stack lane needs an explicit profile-only contract. A naive ring buffer
  likely moves the copy into packaging unless policy/search consumes device or
  ring metadata directly.
- Do not report this as production trainer speed. It is profile evidence for
  the next bridge shape.

## 2026-05-21 Device-Stack Probe Read

The first hybrid H100 rows answered the stack/materialization question more
cleanly than another trainer-surface patch would have.

Matched profile-only rows at B512/A16, persistent direct64 renderer, synthetic
batched stack probe:

```text
uint8 stack, no scalar materialization:    ~16.3k scalar steps/sec
uint8 stack, scalar materialization on:    ~9.8k scalar steps/sec
float32 stack, no scalar materialization:  ~9.2k scalar steps/sec
```

Interpretation:

- Compact `uint8` stack matters because it cuts stack bytes by about `4x`.
- Scalar LightZero materialization matters because turning it on cost about
  `3.59s` across 100 measured B512 steps in this profile.
- A naive host-only ring buffer is still not the right first move; the high
  value shape is compact batched stack -> GPU normalization/consumer -> scalar
  only at the edge.

The next architecture proof should therefore target a real consumer:

```text
batched uint8 stack
-> GPU normalize/model/search-like consumer
-> only then decide how much scalar LightZero payload is still necessary
```

This is still not a production trainer recommendation. It is the clearest
profile evidence so far that the 5-10x-class path requires preserving compact
batched observations across the policy/search boundary, not just optimizing
the renderer kernel.

Device-latest correction:

- A first explicit device-latest probe reduced probe H2D time, but throughput
  dropped to about `11.6k` scalar steps/sec because the host stack was still
  maintained and a second device stack was added.
- So the next implementation must remove or bypass host stack update in the
  no-scalar profile lane. A parallel device stack is measurement evidence, not
  the win itself.

Sim8 repeat update:

- H100 B512/A16, persistent direct64, uint8 stack, synthetic stack probe sim8:
  scalar-off measured about `13.8k` scalar roots/sec.
- L4/T4 with the same shape measured about `9.0k` scalar roots/sec.
- Turning scalar materialization on dropped both GPUs by roughly half
  (`6.5k` H100, `4.2k` L4/T4).
- H100 device-latest still regressed (`9.8k` roots/sec) despite lower probe
  H2D, because the host stack is still built and a second device stack is
  maintained.

Plain read:

```text
The batch-resident shape is real. The expensive part is re-entering scalar
LightZero-shaped Python/NumPy rows, not drawing the latest frame by itself.
```

This changes the next priority from "try another renderer kernel" to "test how
much of this resident batch can survive contact with real policy/search/replay
work."

Guardrail:

```text
Do not call this a training speedup yet.
```

The clean resident rows are no-RND/no-death and mostly synthetic-consumer
profile rows. The real training world adds RND meter/bonus work, natural
death/autoreset, real LightZero MCTS/search, replay sampling, learner updates,
and sidecar cadence. Any Coach-facing recommendation must include matched
no-RND, `rnd_meter_v0`, and normal-death/autoreset gates.

## 2026-05-21 Real LightZero Consumer Read

The real-consumer canary changed the bottleneck map again.

Matched B512/A16/sim8 rows with actual `MuZeroPolicy.collect_mode.forward`:

| compute | scalar edge | roots/sec | render sec | probe sec | scalar materialization sec |
| --- | --- | ---: | ---: | ---: | ---: |
| H100 | off | `2669.32` | `0.26` | `6.13` | `0.00` |
| H100 | on | `2100.31` | `0.21` | `7.77` | `0.79` |
| L4/T4 | off | `2159.35` | `0.17` | `8.50` | `0.00` |
| L4/T4 | on | `2053.57` | `0.19` | `8.36` | `0.58` |

Plain read:

```text
The batch reaches real LightZero collect/search, but the public collect/search
path is much slower than the synthetic resident probe. Rendering is no longer
the main Amdahl wall in this profile shape.
```

The important ratios:

- H100 scalar-off real collect-forward is only about `24%` of the synthetic
  resident probe throughput.
- L4 scalar-off real collect-forward is about `37%` of the synthetic resident
  probe throughput.
- H100 only beats L4 by about `1.24x` scalar-off and basically does not beat it
  scalar-on in this row, which points at CPU/tree/Python pressure rather than
  pure GPU rendering pressure.
- The scalar LightZero timestep edge is real but not the largest bucket in the
  real-consumer rows: about `0.79s` on H100 and `0.58s` on L4 across the
  measured B512/A16/sim8 window.

Current priority:

```text
Stop treating render-only work as the 10x lane for this shape. The next
falsifier should separate model initial inference from LightZero CPU tree/MCTS
inside collect-forward, then decide whether the next serious lane is a deeper
batched search replacement, a custom collector boundary, or topology tuning.
```

Corrected canary contract after critique:

```text
This canary is intentionally narrow:
uint8 pre-scalar stack -> real LightZero collect-forward -> CPU tree/search.
```

It now refuses accidental float32/block-surface rows, uses `to_play=-1` to match
the fixed-opponent scalar env convention, and filters zero-action-mask roots.
That makes future rows cleaner, but it does not change the high-level result:
the real public LightZero collect/search boundary is much slower than the
synthetic resident probe.

## 2026-05-22 Radical Architecture Read

The current small patch lane has a clear ceiling.

Matched train-profile rows:

```text
no-RND:     stock 433.17 steps/sec -> direct output-fast 566.19, about 1.31x
rnd_meter: stock 351.02 steps/sec -> direct output-fast 448.52, about 1.28x
```

Those are real wins, but they are the size of a boundary cleanup.

Fresh durable H100 profile-only falsifier, B512/A16, 60 measured steps,
15 warmup steps, scalar materialization off:

| row | sim8 roots/sec | sim16 roots/sec |
| --- | ---: | ---: |
| `mock_search_service` | `11978.11` | `11648.29` |
| `direct_ctree_gpu_latent` | `7233.81` | `5303.97` |
| `recurrent_toy` | `9623.42` | `8512.57` |

Plain read:

```text
mock_search_service sim16 is about 2.20x faster than direct_ctree_gpu_latent.
recurrent_toy sim16 is about 1.60x faster than direct_ctree_gpu_latent.
```

This does not prove a 10x search rewrite. It proves the current direct CTree
boundary still has meaningful headroom, and it says a bigger move must own more
than the CTree call itself.

Fresh local Puffer-style native-vector boundary probe,
B512/A16/steps100/zero-observation/uint8/no-pickle:

| row | timesteps/sec | measured sec over 102400 timesteps | read |
| --- | ---: | ---: | --- |
| no scalar + native probe | `23515` | `4.35s` | compact consumer is cheap, fastest local row |
| scalar-only | `18604` | `5.50s` | scalar LightZero object materialization costs real time |
| scalar + native probe | `17380` | `5.89s` | paying both boundaries is worse |

Component read:

```text
scalar materialization: about 2.07s, about 20.2 us/timestep
native compact probe:   about 0.62s, about  6.1 us/timestep
actor_step_wall:        about 3.42s, about 33.4 us/timestep
```

Plain read: the object edge matters, but actor/env scheduling is now visible
too. A compact native/vector design must not stop at replacing scalar timestep
materialization; it also has to make actor/env scheduling feed compact buffers
cheaply.

Wide-sidecar update:

```text
The compact probe boundary now has the right shape to test more than obs+mask.
```

`HybridCompactBatch` carries observation, action mask, reward, done, row/player
ids, target reward, done roots, terminal/autoreset row masks, final observation,
episode step, elapsed time, round id, alive flags, and joint action. Old probes
still work through `run(observation, action_mask)`, but real next probes should
use `run_compact_batch(batch)`.

Fresh local row after widening the sidecar:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
payload+merge:        ~22.1k timesteps/sec
native actor buffer:  ~30.5k timesteps/sec
```

Do not overfit the exact ratio. The important change is that terminal/autoreset
state is now visible to the compact consumer, and the direct-write path still
removes the parent merge as a meaningful wall. The remaining high-level wall is
broader than one function:

```text
actor/env scheduling + stack update + real search/replay/RND consumer boundary
```

Current priority:

1. Keep the compact row/player sidecar as the boundary of truth.
2. Attach a real search/replay/RND-shaped consumer to that sidecar.
3. Only promote a production rewrite after matched rows show a multiplier on
   the same denominator as Coach training.

Local CPU-oracle follow-up, B128/A8/steps40/warmup10, says:

```text
no scalar + native probe: 192.91 timesteps/sec, measured 53.08s
scalar-only:              192.17 timesteps/sec, measured 53.29s
renderer/observation:     about 52.2-52.4s in both rows
```

This is not a scalar-boundary verdict. It is a reminder that the old CPU
renderer can dominate so hard that topology changes disappear. For architecture
decisions, compare either zero-observation topology rows or the real batched
GPU observation path.

Fresh native actor-buffer probe:

```text
B512/A16/steps100/warmup20/zero-observation/uint8/no-scalar/native-probe
old actor payload + merge: 40477 timesteps/sec, measured 2.53s
native actor buffer:       67890 timesteps/sec, measured 1.51s
```

The win is mainly actor-wall/merge:

```text
actor wall:   1.879s -> 0.907s
gather merge: 0.0138s -> 0.000018s
native probe: 0.460s -> 0.411s
```

Plain read:

```text
The Puffer-style buffer idea has a real local signal. It is still profile-only,
but it shows the object/payload/merge boundary can be removed cleanly and can
matter even before real search is attached.
```

Current best architecture thesis:

```text
compact native/vector CurvyTron buffers
-> batched GPU observation/action-mask tensors
-> batched model/search service
-> compact replay/target materialization
-> scalar LightZero/Python objects only at compatibility edges
```

PufferLib is the closest external pattern now in scope. The relevant pieces are
not its exact algorithm, because it is not MuZero. The relevant pieces are its
systems choices: static memory, contiguous env buffers, no redundant
observation copies, pinned async transfer, CUDA graph replay, and native C/CUDA
where Python object churn would dominate.

Fresh PufferLib repo inspection sharpened the contract:

```text
StaticVec owns flat observation/action/reward/done/mask buffers.
Env instances write directly into their assigned buffer slices.
GPU mode uses pinned host buffers plus device buffers and per-buffer streams.
The native backend registers all tensor shapes, makes one allocation, and then
hands out pointers into that allocation.
```

CurvyTron translation:

```text
Do not build another scalar wrapper.
Build a compact batch owner that can feed search/replay-shaped consumers.
```

Boundary-test correction:

```text
Speed numbers are not advice unless the row says what it measured.
```

The summary tooling now enforces this at the table edge: profile rows must carry the
stock/profile identity, observation contract, render modes, death/RND mode,
denominator source, MCTS/root counts, and core timers. New compact outputs now
also carry a `semantic_identity` block with observation dtype,
scalar-materialization, `to_play`, zero-mask/action-mask, and consumer-semantics
labels. Full `--require-attestation` now requires that block. Old local
artifacts predate it, so the next fresh profile wave must verify it appears
before we rely on the row as launch advice.

2026-05-21 attestation update:

- The fresh stock smoke `opt-semantic-identity-smoke-20260521a` now passes
  `--require-attestation`.
- The first rerun exposed two real tooling issues, both fixed:
  - the local profile runner could parse a nested `semantic_identity` object
    instead of the top-level compact JSON;
  - the stock trainer image needed the same Torch `2.8.0` CUDA12 pin already
    used by the boundary profile image.
- Treat the green smoke as an artifact-contract proof, not as a speed
  recommendation. It used sim2/C64/no-death and measured about `326` collected
  env steps/sec.

2026-05-21 fresh split refresh:

| row | roots/sec | important read |
| --- | ---: | --- |
| H100 initial inference only, B512/A16/sim8 shape | `9465.80` | root neural net pass is fast enough to not be the wall |
| L4 initial inference only, same shape | `5560.31` | same read on cheaper GPU |
| H100 collect-forward sim8, same shape | `2304.28` | public LightZero collect/search collapses most of the model-only headroom |

## 2026-05-22 Train-Facing Search Hook Read

The sidecar search result now has a stock-loop reality check.

Plain version:

```text
direct_ctree_gpu_latent is real, but it is only a partial fix.
```

What it fixes:

- It avoids stock LightZero's root-latent CPU copy before CTree search.
- It keeps a latent pool on GPU through recurrent inference.
- It preserves stock `train_muzero` outside the collect hook: collector,
  replay, target builder, learner, RND hooks, and checkpoint machinery still run
  through the trusted path.

What it does not fix:

- CTree still receives reward/value/policy logits through CPU/list-shaped
  payloads on every simulation.
- The hook still returns one LightZero-style output dict per env id.
- Output assembly and model-output D2H/listifying are now visible costs.

Fresh H100 full-loop rows, no RND, no death, sparse telemetry:

| shape | stock steps/sec | direct steps/sec | read |
| --- | ---: | ---: | --- |
| C64/sim8/3 learner calls | `495.05` | `477.19` | direct improves search but loses wall |
| C64/sim16/1 learner call | `387.23` | `456.15` | direct wins about `1.18x` |
| C64/sim16/3 learner calls | `205.55` | `420.67` | direct wins, but stock row looks anomalously slow |
| C128/sim16/3 learner calls | `477.68` | `396.84` | direct loses wall despite faster search |

Current Amdahl read:

```text
The hook can matter when sim16 makes search a large enough denominator, but it
is not a broad 5-10x answer. The next useful work is not more CPUs. It is
reducing the remaining train-facing collect/search overhead: direct output
assembly, model-output D2H/listifying, and the CTree Python/list boundary.
```

Current caution:

```text
C64/sim16 is promising enough to repeat. C128 is not a Coach recommendation
from this wave. Sim8 sparse says telemetry alone did not make the old sim8 row
miss a huge direct-hook win.
```
| L4/T4 collect-forward sim8, same shape | `1250.21` | Modal placed this row on T4; still confirms the same qualitative wall |

Plain read:

```text
The current high-value wall is inside LightZero collect/search/output handling
after the model root pass, not renderer-only work.
```

The next probe now instruments model calls inside `collect_mode.forward` so the
remaining collect-forward bucket can be split into model-call time versus
non-model tree/output time.

## 2026-05-21 Direct CTree Arrays Read

The active optimizer lane is now the direct CTree arrays boundary:

```text
pre-scalar uint8 [B,2,4,64,64]
-> real LightZero initial_inference
-> real policy._mcts_collect.search / CTree MCTS
-> compact arrays out
```

Current H100 B512/A16/sim8 rows:

| row | run id | roots/sec |
| --- | --- | ---: |
| stock facade | `ap-HJk70PQP2iLAvA7mxxn99u` | `2419.81` |
| direct CTree, old host uint8 | `ap-XEB8GF9B2Gw5V600QVtu10` | `3859.44` |
| direct CTree, current host uint8 | `ap-DoCqvAulFMhZyoAcownQmn` | `5247.95` |
| direct CTree, current pinned uint8 | `ap-APSw7b1ZSJjSSuPtGEHO3w` | `4678.23` |
| direct CTree, resident reuse ceiling | `ap-KCtqhJDwTuLptLKd4XSv38` | `5820.96` |

Plain read:

```text
Direct CTree arrays is the only current near-term speed lane with a real
profile signal. Input transfer is priced now: pinned can cut H2D, but the
matched short row did not beat plain host uint8 overall. Resident reuse shows
the upper bound if H2D disappears, but it reuses stale input and cannot be used
for training. The next decision is parity first, then search/root-prep/output
split, not another renderer rabbit hole.
```

2026-05-21 late P2 refresh:

| row | roots/sec | read |
| --- | ---: | --- |
| stock facade, host uint8 | `2670.68` | current public LightZero facade anchor |
| direct CTree, fresh host uint8 | `4764.06` | about `1.78x` over stock facade |
| direct CTree, pinned uint8 | `3689.15` | H2D cut hard, but search/other buckets made total wall worse |
| direct CTree, resident stale ceiling | `3069.08` | no H2D, but slower total wall; stale-input ceiling is not useful here |

Plain update:

```text
The robust current signal is still direct CTree over the public collect facade.
The robust current non-signal is pinned/resident input. Input copy is not the
big remaining wall in this shape.
```

Longer repeat rows changed the input-mode read:

| row | run id | roots/sec | measured wall | boundary total | H2D | search | observation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host uint8 | `ap-QPLEHOs3dGrcs2tlRpbMge` | `4111.80` | `14.94s` | `10.24s` | `1.21s` | `7.86s` | `2.08s` |
| pinned uint8 | `ap-5F1tMU2HiuHXDcu4O1tGkw` | `4513.15` | `13.61s` | `8.87s` | `0.14s` | `7.54s` | `2.05s` |
| resident reuse | `ap-wsKyodSayU2KGsTgKKpAqc` | `5537.40` | `11.10s` | `6.95s` | `0.00s` | `5.83s` | `1.85s` |

Plain read: the short host-vs-pinned comparison was not stable enough. The
longer row says pinned input is a real modest total-wall win, but not a
phase-changing one. Resident reuse shows how much input transfer can matter if
it disappears, but it is stale-input by design and only bounds the possible
speedup. The direct path still needs fixed-seed stock parity before it becomes
anything except a profile probe.

Guardrail:

```text
This does not touch train_muzero, live runs, checkpoints, eval, GIFs, or
tournaments. It cannot become Coach advice until fixed-seed parity against
stock LightZero passes.
```

2026-05-21 deep H100 collect-forward split:

- Same B512/A16/sim8 shape, 120 measured calls, H100.
- Throughput with instrumentation: about `1444` roots/sec. This row is slower
  than the earlier collect-forward row, so use it for fractions, not as the
  fastest throughput estimate.
- Aggregate `batched_stack_probe_sec`: about `75.05s`.
- Aggregate LightZero collect-forward wall inside that: about `69.82s`.
- Timed model calls inside collect-forward:
  - `initial_inference`: 1 call per root batch, about `0.0046s` in the last
    call.
  - `recurrent_inference`: 8 calls per root batch, about `0.0177s` in the last
    call.
  - aggregate model-call time is roughly `2.7s` across the row.
- Non-model collect-forward residual is roughly `67s` across the row.

Plain read:

```text
The current collect-forward wall is not GPU neural-network inference.
It is mostly LightZero tree/search/output machinery around the model calls.
```

This makes the next optimization decision cleaner: either instrument/replace
the public LightZero MCTS/tree boundary, or prove that a different search
implementation can keep the batched roots resident without the same Python/CPU
cost. Bigger GPUs alone should not be expected to fix this wall.

2026-05-21 next split now in code:

- `collect_mode.forward` total is already measured.
- Model `initial_inference` and `recurrent_inference` are already measured.
- The hybrid collect-forward probe now also times:
  - `policy._mcts_collect.search`;
  - ctree `batch_traverse`, when the C++ binding allows monkey patching;
  - ctree `batch_backpropagate`, when the C++ binding allows monkey patching;
  - decode/output readback outside the forward call.

Plain expectation:

```text
If MCTS search ~= collect-forward residual:
  search/tree is the wall.
If ctree traverse/backpropagate is small but MCTS search is large:
  Python loops, tensor<->NumPy conversion, tolist(), and root/output handling
  around the C++ calls are the wall.
If MCTS search is small:
  policy output assembly or input/root setup is the wall.
```

This is the next useful Amdahl split. It is profile-only and does not touch
Coach training defaults.

2026-05-21 preliminary deeper-search read:

- H100 first row: `2360.52` roots/sec. Last call spent `0.304s` in
  collect-forward, `0.0155s` in model calls, `0.105s` in MCTS search,
  `0.0099s` in ctree traverse/backpropagate, `0.200s` outside MCTS, and
  `0.0145s` in output decode.
- L4 first row: `1291.41` roots/sec. Last call spent `0.599s` in
  collect-forward, `0.0501s` in model calls, `0.173s` in MCTS search,
  `0.0162s` in ctree traverse/backpropagate, `0.426s` outside MCTS, and
  `0.0258s` in output decode.

Plain read, pending the aggregate rerun:

```text
The wall is not just the neural net.
It is also not just the C++ tree traverse/backprop calls.
The next suspect is the public LightZero collect wrapper: root setup,
Torch/NumPy/list conversion, action-mask/to_play plumbing, action sampling,
and large per-root output dict assembly.
```

2026-05-21 direct arrays update:

```text
The direct CTree arrays probe confirms there is removable public-wrapper
overhead, but the first direct implementation also proved that our own output
packing can become the wall if it loops per root.
```

Measured H100 B512/A16/sim8 rows:

- stock facade: `2419.81` roots/sec;
- direct CTree arrays before output fast path: `2806.64` roots/sec;
- direct CTree arrays after all-actions-legal output fast path: `3859.44`
  roots/sec.

The fast path cut output assembly from `4.709s` to `0.027s` over 25 measured
steps. Current read: raw CTree math is still not the big visible wall, compact
output assembly is no longer the big wall, and the next direct-lane bottleneck
is MCTS search/root-prep/input handling around CTree plus ordinary
observation/H2D cost.

Aggregate H100 rerun settled the split:

- collect-forward total: `35.36s`;
- model calls: `1.81s` (`5%`);
- MCTS search: `10.97s` (`31%`);
- ctree traverse + backpropagate: `0.98s` (`3%`);
- outside MCTS: `24.40s` (`69%`);
- decode after forward: `2.30s` outside the forward bucket.

Updated Amdahl read:

```text
The next 2x-class win is unlikely to come from another renderer kernel or a
bigger GPU. It must reduce the public LightZero collect/search wrapper cost:
root setup, CPU/GPU conversion, per-root Python output fanout, or the way search
results are represented.
```

The next experiment should therefore inspect LightZero `_forward_collect` and
MCTS `search` enough to place one more split around root prep and output
construction. If that confirms Python/output fanout dominates, build a small
replacement-ceiling toy that keeps roots/results batched instead of producing
one Python dict per root.

Source read: LightZero `MuZeroPolicy._forward_collect` does the expensive
public-wrapper work in plain Python around the C++ tree:

1. runs `initial_inference`;
2. immediately moves root values, latent states, and policy logits to CPU
   NumPy/list form;
3. builds `legal_actions` with one Python list per root;
4. samples Dirichlet noise with one Python call per root;
5. builds ctree roots and calls `self._mcts_collect.search`;
6. fetches root distributions/values back as lists;
7. loops over every ready env id to select actions and construct one output
   dict per root.

Reference source: <https://github.com/opendilab/LightZero/blob/main/lzero/policy/muzero.py>

That source shape matches the measured split: raw ctree traverse/backprop is
small, while wrapper-side CPU/list/dict work around it is large.

Pure-policy H100 probe:

- Same public `collect_mode.forward` wrapper, but `collect_with_pure_policy=true`.
- Result: `6286.61` roots/sec versus `2572.12` roots/sec for MCTS collect sim8.
- Aggregate pure-policy collect-forward: `4.88s`.
- Aggregate MCTS collect-forward: `35.36s`.

Interpretation:

```text
The MCTS branch is the slowdown, but not because the raw ctree calls alone are
slow. Pure-policy still pays initial inference and per-root output dicts and is
much faster. The remaining target is MCTS-branch setup/conversion/result
handling around ctree, plus the public LightZero search wrapper.
```

2026-05-21 escape-plan correction:

```text
LightZero already has C++ tree kernels. The ceiling is that the kernels are
wrapped by a Python per-simulation search loop and list-shaped Cython APIs.
```

So "move it to C" means one of two different things:

1. small version: add array-native Cython APIs for roots, backprop inputs, and
   output visits/values. This can remove visible wrapper cost but probably not
   10x by itself.
2. big version: replace the Python search loop with a deeper C++/Cython or
   accelerator-native batched search architecture. This is the plausible path
   beyond the current `1.8x` direct-boundary win.

Detailed current plan: `search_boundary_escape_plan_20260521.md`.

Current best next implementation experiment:

```text
Build a profile-only replacement-ceiling toy for the MCTS branch:
batched root tensors -> simple batched search/result arrays -> compact action
array, without per-root Python dict/list output fanout. It does not have to be
algorithmically complete at first; it must bound the cost of the representation
and output path.
```

Subagent design sharpened this into two modes:

- `policy_arrays`: real initial inference, masked priors, compact arrays out.
  This is the "no MCTS representation overhead" floor.
- `recurrent_toy`: real initial inference plus `num_simulations` batched
  recurrent calls and vectorized visit/value updates, compact arrays out. This
  is the useful replacement ceiling.

Rules for the toy:

- do not call `collect_mode.forward`;
- do not call `_mcts_collect.search`;
- do not return one Python dict per root;
- do not build `legal_actions` lists per root;
- do not call `.tolist()` on logits in the hot path;
- do synchronize CUDA around timed model/search work;
- do consume outputs with checksum and illegal-action checks.

If `recurrent_toy` is near pure-policy speed, the public LightZero
root/list/dict path is the obvious target. If it is near real collect-forward,
then representation fanout is not the only problem.

## 2026-05-21 Where The Batch Dies

Current evidence from the local code audit:

- The stock env path still renders/updates one scalar env stack and then
  returns one `BaseEnvTimestep`.
- The batched profile bridge can form flat `[N,4,64,64]` timesteps, but then
  `_ready_obs_by_env_id`, `_split_timestep_by_env_id`, and stock-shaped
  conversion loop over scalar env ids.
- LightZero policy/search appears to batch roots after the env boundary, and
  our profiler records MCTS root counts and model inference batch sizes.

Plain read:

```text
LightZero can batch some policy/search work, but our env boundary still turns
the resident batch into scalar Python/NumPy objects too early.
```

The next prototype should therefore price a resident replay/search-like
consumer before scalar materialization. If that wins, a real custom collector
or central batching boundary may be worth building. If it fails, the practical
work is stock-path manager/search cleanup rather than a larger GPU observation
rewrite.

## Full Loop Shape

Current stock LightZero flow, simplified:

```text
Modal trainer
-> build LightZero config
-> stock train_muzero or train_muzero_with_reward_model
-> collector asks env manager for ready observations
-> policy/search chooses actions
-> env manager steps scalar CurvyTron env wrappers
-> each wrapper returns one LightZero timestep with [4,64,64] obs
-> collector writes GameSegments
-> replay samples batches
-> learner updates model
-> checkpoint/eval/GIF side work runs on cadence/background jobs
```

The current CurvyTron wrapper owns an underlying `VectorMultiplayerEnv` with
`batch_size=1`. That means many collector envs do not automatically create one
large render batch. They create many scalar env steps.

## Where A 10x Win Could Come From

Renderer-only speed is useful but not enough unless render dominates full-loop
wall time. The likely order-of-magnitude path needs at least one architecture
change:

- collect many env rows behind one vector facade;
- batch policy observations before returning scalar LightZero timesteps;
- reduce per-step Python info/timestep payload work;
- reduce subprocess IPC or keep the batched owner in one process first;
- keep search/model calls batched over many roots;
- avoid extra host/device round trips where possible.

## Current Bottleneck Hypotheses

1. Long single-env trajectories: observation/render can dominate.
2. Wide stock LightZero training: collector/search/process overhead can dominate
   after env workers run in parallel.
3. Scalar GPU observation: launch/copy/readback and fixed trail capacity
   dominate, so it is slower than CPU.
4. Batched GPU boundary: device render is still the largest local bucket, but
   the whole-loop value is unknown.
5. RND: adds reward-model estimate/train work and extra data movement; it must
   be measured separately from render/backend changes.

## Current Best Interpretation

Batching is the right idea. The missing piece is not "can JAX render a frame on
GPU." It can. The missing piece is a trainer-visible vector boundary that
preserves LightZero's timestep contract while actually forming a useful batch.

## 2026-05-21 Amdahl Check

The newest full-loop profile rows say the same thing in simpler language:

- Batched GPU observation can beat fresh subprocess CPU-oracle controls at
  C128/C256/C512 in this no-death, no-RND, sim2 profile shape.
- It is not a 10x story right now. C256 real batched GPU post-patch measured
  about `1096 steps/s`; the same one-process manager with zero-filled
  observations measured about `1735 steps/s`. That bounds the current
  observation tax at roughly `1.6x` for that row.
- Once observation work is removed, policy forward, MCTS/search, and
  manager/env-step remain large. This is why "make the render kernel faster"
  is useful but cannot be the only optimizer plan.
- The current batched GPU manager is not end-to-end GPU RL. It still returns
  scalar Python/NumPy LightZero timesteps. The outside systems that go much
  faster either keep the whole env on accelerator, or keep many actors/workers
  alive and batch the GPU-heavy pieces.
- The small post-patch cleanup is a modest C512 win, not a magic fix:
  `1439.84 steps/s` versus the prior C512 real-render anchor `1352.47`.
  Against the C512 zero-observation ceiling `1805.22`, the remaining pure
  observation headroom is about `1.25x`.

Practical read: keep optimizing the batched observation path, but do it with
clear stopping criteria. If real rows approach the zero-observation ceiling,
move to policy/search/manager architecture. If real rows stall far below the
ceiling, focus on render/stack/pack and possibly a multi-worker manager that
keeps subprocess-style parallelism.

## 2026-05-21 Gate Update And Priority Shift

The latest gates changed the shape of the work:

- Normal-death/autoreset now passes through stock `train_muzero` in the batched
  GPU profile lane. The row collected `36014` raw env steps at about
  `485 steps/s`. That number is not comparable to no-death rows because live
  roots collapse after agents die; the pass matters because partial row/player
  render requests and complete physical-row omissions are now handled.
- RND meter mode now passes through the same profile lane. Predictor weights
  change, target weights stay frozen, and target rewards stay unchanged. It
  costs about `10-12%` in the C512 meter rows, which is meaningful but not the
  main wall.
- C768 is not a clean win. C768 real render measured about `1420 steps/s`,
  basically flat versus C512 real render at `1440 steps/s`. C768 zero
  observation was slower at about `1192 steps/s`, so high-width zero rows are
  no longer a simple monotonic ceiling. Treat this as topology/scheduling
  saturation until repeated.

Current bottleneck read:

```text
C512 best real render:         ~1439.84 steps/s
C512 zero-observation ceiling: ~1805.22 steps/s
render-only remaining upside:  ~1.25x
```

So the next large gain is not "make the current render kernel a bit faster."
The render path still matters, but the bigger question is whether we can keep
large batches alive across env step, observation, policy/search, and replay
without paying scalar Python/NumPy LightZero overhead on every row.

Architecture direction:

1. Keep the current batched GPU manager as a profile lane, not a trainer
   default.
2. Repeat C512/C768 real-vs-zero only to confirm saturation and variance.
3. Prototype the next shape that could actually give multi-x speedup:
   actor-parallel env stepping plus a central batched GPU observation/render
   service, or a deeper device-resident env/search path.
4. Treat MCTS/search batching, policy forward timing, and manager scalarization
   as first-class walls now. At C512/C768 they are no longer background noise.

## 2026-05-21 GPU Architecture Research Reset

Research and fresh profiles now agree:

```text
The render win is real, but the next big speedup is a batching architecture
problem, not a single-frame rendering problem.
```

Fresh no-RND C256/sim sweep:

- L4 sim4 real batched obs: about `996 steps/s`; H100 sim4: about
  `1341 steps/s`.
- L4 sim8 real batched obs: about `627 steps/s`; H100 sim8: about
  `659 steps/s`.
- L4 sim16 real batched obs: about `619 steps/s`; H100 sim16: about
  `711 steps/s`.
- H100 zero-observation sim8 still spends about `62s` in policy collect and
  about `34s` in MCTS across a `93s` wall. Rendering is basically zero there.

Interpretation:

1. H100 helps when the row has enough GPU-benefiting work and not too much
   stock-loop overhead.
2. Increasing search simulations makes collect/search dominate quickly.
3. Zero-observation rows prove that deleting render does not delete the wall.
4. `policy_forward_collect` contains nested MCTS/model work, so timers are
   inclusive and must not be summed.

External systems point to the same answer:

- OpenSpiel AlphaZero C++ uses actors/threads, shared cache, batched inference,
  and GPU inference/training.
- Mctx batches MCTS inputs and JIT-compiles search for accelerators.
- CuLE gets huge Atari throughput by running thousands of envs and rendering
  directly on GPU.
- EnvPool gets large CPU-side env speedups by removing Python subprocess/GIL
  overhead.
- Brax keeps env and learning on accelerators.

Current optimizer priority:

1. Keep the current batched GPU profile path as evidence, not production
   default.
2. Add split timers for scalar bridge and surface packaging so zero-observation
   rows explain their host floor.
3. Prototype a profile-only resident chunk / central batching lane:
   compact state batch -> GPU observation -> batched policy/search pressure ->
   materialize LightZero-shaped rows only at the edge.
4. Use H100 selectively for sim/search sweeps; do not assume bigger GPU alone
   fixes the stock LightZero scalar boundary.

## 2026-05-21 Split-Timer Read

The split-timer grid answered the immediate Amdahl question.

At the most useful current shape, C512/sim8:

```text
L4 real batched obs:   ~1077 steps/s
L4 zero observation:   ~1263 steps/s
H100 real batched obs: ~1378 steps/s
H100 zero observation: ~1608 steps/s
```

So deleting observation entirely is only about `1.17x` faster in both L4 and
H100 C512/sim8 rows.

The new split timers also show that scalar bridge object churn is not currently
the big prize:

- ready-obs construction, timestep split, and stock timestep conversion are
  visible but single-digit seconds at C512;
- surface package pieces like policy-row selection, action-mask copy, and info
  dict construction are tiny;
- policy collect and MCTS/search remain large even when observations are zero;
- real-render rows still pay stack/render/manager work, but that remaining
  pure-observation headroom is bounded.

This changes the priority:

```text
Do not spend the next phase on another renderer-only rewrite.
Build or profile a batch-resident actor/search path.
```

The web and local code review agree on one simple point: fast GPU RL systems do
not win by putting one small render call on a GPU. They win by keeping a large
batch alive across the hot loop, or by keeping many CPU actors busy while GPU
work is batched centrally.

Useful outside patterns:

- CuLE/JAXAtari: env simulation and frame generation are GPU-native and highly
  batched.
- Isaac Gym/Brax/PixelBrax: env state, observation/reward, policy, and rollout
  tensors stay on the accelerator.
- EnvPool/Sample Factory: if envs stay CPU-side, use many workers, shared
  buffers, batched inference, and avoid per-step Python serialization.
- MCTX-style search: search is useful on accelerators only when roots/model
  calls are batched.

Current CurvyTron is in the middle: the batched GPU profile lane batches
rendering, but it still copies back to host, updates host stacks, and returns
scalar LightZero timestep objects. That explains why it can help and still not
produce a 5-10x speedup.

The newest local hybrid scaffold is a topology probe, not training:

```text
in-process CPU actors step compact CurvyTron rows
-> parent merges compact metadata
-> parent fills zero [B,2,4,64,64] stacks
-> scalar LightZero-shaped rows only at the edge
```

It reached about `15k-25k` scalar timesteps/sec locally with zero observations
at B64/B256/B512. That is far above the current stock-loop profile rows, so the
hybrid lane has enough headroom to keep exploring. The next proof must add real
direct GPU rendering and eventually subprocess/IPC behavior; do not confuse the
local zero scaffold with a trainer recommendation.

Near-term priority:

1. Keep Coach/live training on the trusted stock path.
2. Use C512 as the clean one-process profile anchor; stop widening to C768
   unless testing a specific search/topology hypothesis.
3. Replace the hybrid scaffold's zero stack with the direct batched GPU
   renderer in a profile-only harness.
4. Measure compact payload bytes, H2D/render/D2H, host stack update, scalar
   materialization, policy/search root batch, and RND separately.
5. If the hybrid real-render harness beats the one-process zero ceiling, then
   design a stock LightZero bridge. If it does not, consider deeper
   device-resident env/search work instead of more renderer-only tuning.

## 2026-05-20 Reorientation

Plain current read:

- `direct_gray64 + simple_symbols` made the local observation surface much
  faster, but only in profile-only paths so far.
- After the direct renderer and host-copy fixes, the local B512 surface wall is
  no longer just the GPU render kernel. The measured surface step is about
  `0.123s` for `1024` policy rows; device render is about `0.014s`, while
  renderer/pack/stack/payload movement make up most of the rest.
- In real stock full-loop rows, the latest trusted path still uses CPU-oracle
  observations. Those rows show low GPU utilization and most wall time in
  collection/search/RND/worker-side observation work, not learner replay.
- RND is a separate lane. CUDA RND helps, but cadence dominates: `100` updates
  per collect is expensive, `10` is much cheaper, and `1` is a smoke-only
  minimal cadence. Do not mix RND cadence claims into renderer speed claims.

Current priority:

1. Keep the direct GPU observation path profile-only until row/player/reset,
   final-observation, partial autoreset, RND latest-frame, and metadata gates
   are all covered.
2. Rebaseline a real stock/full-loop profile before deeper render-kernel work;
   otherwise Amdahl may point at the wrong bucket.
3. Keep RND modular: profile RND collect/train/estimate separately, decide
   positive-reward normalization separately, and only then combine it with
   observation-backend changes.
4. If a 10x win exists, it probably comes from preserving large vector batches
   across env/observation/policy/search boundaries, not from another small
   scalar render tweak.

Critique correction:

- Promotion should fail closed on the exact renderer backend identity, not just
  "some renderer exists." Otherwise a future profile could silently time the
  wrong backend.
- Positive RND is not truly blocked by the core spec; it is only blocked if the
  active launcher/process refuses it. Treat that as an open launch-gate task.

## Self-Critique: Do Not Chase Seed Noise

The H100 C512/sim4 stock rebaseline split into two slow rows and one fast row
with identical workload counts. That matters because it makes a single
throughput number unsafe.

But it is not the main optimizer lane.

The duplicate-seed run is only a background diagnostic for runtime variance. Do
not spend the next phase trying to explain every Modal/H100 timing wobble unless
it blocks an A/B comparison. The main optimization question is still:

```text
Can we preserve a large batched CurvyTron observation surface through the
LightZero collection/search/replay/learner path without collapsing back into
scalar env workers?
```

If yes, direct GPU observation can matter to full-loop speed. If no, more render
micro-optimization is mostly local theater.

## External Architecture Research Update

The useful outside pattern is not "put one small render on GPU." It is one of
these:

- **End-to-end GPU env:** Isaac Gym/Brax-style systems keep env state,
  observation, reward, reset, and policy tensors on the accelerator. That avoids
  per-step CPU/GPU copies and Python synchronization.
- **Very parallel CPU env:** EnvPool/Puffer/Sample Factory-style systems keep
  CPU simulation highly parallel and reduce Python/message overhead with worker
  pools, shared memory, static buffers, pinned transfers, and batched inference.
- **Zero-style self-play:** AlphaZero/MuZero/KataGo-style systems run many
  actors/search workers and batch neural inference/search work on GPU/TPU
  rather than collapsing the actor side into one blocking loop.

References read during this pass:

- NVIDIA Isaac Gym: GPU physics plus GPU observation/reward tensors, avoiding
  CPU-GPU transfers:
  <https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/>
- Brax paper: JAX env and learning code can compile and run on accelerators:
  <https://arxiv.org/abs/2106.13281>
- EnvPool: high-throughput C++ vector env pool and Atari/MuJoCo FPS examples:
  <https://github.com/sail-sg/envpool>
- Sample Factory architecture: rollout workers, inference workers, batcher, and
  learner are separate components, using shared memory to reduce message cost:
  <https://www.samplefactory.dev/06-architecture/overview/>
- Sample Factory sync/async note: for GPU-vectorized envs, synchronous mode can
  be enough because the work is already on the same device:
  <https://www.samplefactory.dev/07-advanced-topics/sync-async/>
- EfficientZero implementation notes: Ray workers, CPU/GPU actors, and parallel
  MCTS env count are explicit tuning knobs:
  <https://github.com/YeWR/EfficientZero>
- CuLE: Atari frames rendered directly on GPU to avoid CPU-GPU bandwidth
  bottlenecks:
  <https://research.nvidia.com/publication/2020-12_accelerating-reinforcement-learning-through-gpu-atari-emulation>
- DeepMind Mctx: JAX-native MCTS algorithms:
  <https://github.com/google-deepmind/mctx>

Current read from that research:

The current CurvyZero batched GPU manager is in the middle. It is not an
end-to-end GPU env, and it is not a highly parallel CPU pool. It proves the
render kernel can be moved, but it still returns scalar Python/NumPy LightZero
timesteps. The zero-observation row says this boundary has headroom, but the
real render/stack path still consumes too much of it.

Next architecture candidates:

1. Keep stock subprocess CPU-oracle as the safe Coach path until a real
   full-loop GPU-render row wins.
2. Optimize the batched render/stack path further because zero-observation
   proves that is still the immediate wall inside the batched manager.
3. Prototype a hybrid architecture later: subprocess-style actors step compact
   CurvyTron state, while a central batched GPU observation service renders
   larger batches. That tries to preserve CPU parallelism and GPU render
   batching at the same time.
4. Treat a full JAX/Torch env rewrite as the clean but larger Isaac/Brax-style
   path. It is not the next quick patch unless the hybrid boundary fails.

## 2026-05-21 Hybrid GPU Architecture Probe Update

New facts:

- The profile-only hybrid actor plus central renderer path now runs on Modal
  H100 with the real dynamic JAX renderer injected from infra code.
- It does not call `train_muzero`, touch live runs, change trainer defaults, or
  change tournament/checkpoint/eval behavior.
- B256/A8 reached about `4496` scalar steps/s.
- B512/A16 reached about `5447` scalar steps/s.
- B512/A16 with compact-payload pickle enabled reached about `6201` scalar
  steps/s, but the wall-time improvement is likely runtime noise; the useful
  fact is that compact payload pickle was tiny.
- B1024/A16 reached about `6662` scalar steps/s. Wider batch still helps in the
  profile-only harness, but the row is still observation-heavy.

Plain interpretation:

- This is the first real signal that preserving CPU actor fan-in while using a
  central batched GPU observation service is worth pursuing.
- It is still not a full training-loop result. Policy/search/replay/learner/RND
  are absent.
- The current "actors" are in-process partitions stepped sequentially. That
  means these rows do not prove subprocess/actor parallelism; they prove that
  the central renderer boundary and row partitioning can run at larger batch
  shapes.
- Amdahl has moved but not disappeared: at B512/A16, observation work is still
  about `3.04s` of `3.76s` measured wall. Inside that, device render is about
  `1.18s`, so render remains important, but host stack/update/render-service
  overhead is also real.
- The H100 is not saturated in this row. The bigger opportunity is probably
  preserving batch through policy/search or reducing host round trips, not just
  buying a bigger GPU.
- B1024 starts to show the next boundary: scalar materialization grew to about
  `0.088s` over 20 profile steps. That is still smaller than observation, but
  it will matter if render/stack drops further.

Current best next move:

1. Keep this lane profile-only.
2. Run a small Modal GPU terminal row now that local terminal
   `final_observation` support exists.
3. Measure compact payload cost with paired pickle-on/off rows.
4. Add a clearly labeled synthetic policy/search pressure probe before
   considering a stock LightZero bridge.

Latest local guardrail update:

- The scalar timestep materializer now accepts either `[B,P,3]` or `[B*P,3]`
  action masks. The hybrid profile passes the merged actor mask through instead
  of silently replacing it with all-true masks.
- This is still a profile-only seam. It improves the pressure test, but does
  not make the synthetic probe a LightZero MCTS or learning claim.

First synthetic policy/search pressure read:

- B512/A16 with sim2 reached about `5553` scalar steps/s; B1024/A16 with sim2
  reached about `7257` scalar steps/s.
- In both rows, observation/stack remained the dominant bucket. At B1024,
  observation was about `4.44s` of `5.64s`, while the synthetic probe was about
  `0.35s`.
- This does not prove real MCTS/search is cheap. It only says the current
  synthetic sim2 probe is too light to become the wall. The next pressure row
  should turn the synthetic simulation knob up before reprioritizing away from
  observation/host-boundary work.

Heavier synthetic pressure read:

- Sim16 still does not dominate. B512/A16 spent about `0.40s` in the synthetic
  probe versus `3.44s` in observation. B1024/A16 spent about `0.64s` in the
  synthetic probe versus `4.22s` in observation.
- Current Amdahl read for the profile-only hybrid lane: the observation boundary
  is still the next target. That bucket is no longer just "drawing trails"; it
  includes renderer service time, device render, GPU output transfer, and host
  stack update.
- A real MCTS integration could still change this. The safe next optimization
  work is to split and reduce observation-boundary costs while keeping a
  separate real-policy/search gate.

No-copy observation-boundary result:

- Removing the non-terminal full-stack return copy improved B512 no-probe from
  about `5447` to `6822` scalar steps/s and B1024 no-probe from about `6662` to
  `9495` scalar steps/s.
- That is a real local optimization, but it did not change the main bottleneck:
  B1024 no-probe still spends about `80%` of measured wall in observation.
- The next target is narrower now: renderer service/device render/output
  transfer and stack shift/update. The old Python return-copy was low-hanging
  fruit and is gone.

Fine-grain observation split:

- B2048 float32 stack timing: observation `6.43s`, renderer render `3.08s`,
  device render `1.50s`, stack shift `2.16s`, latest update `0.25s`.
- B4096 did not clearly improve throughput, so wider is not automatically
  better. The profile has enough batch to expose memory movement now.
- The next concrete experiment is uint8 stack storage with normalized float32
  scalarization. If it wins, it suggests the future trainer should avoid moving
  float32 image stacks through host memory. If it loses, the cost just moved to
  scalarization and we should focus on device-resident policy handoff instead.

Uint8 stack result:

- The cost moved to scalarization. B2048 uint8 stack shift fell to about
  `0.60s`, but scalar materialization grew to about `3.98s`.
- This means the current LightZero-shaped CPU row boundary is now the problem
  for byte stacks. Storing bytes is good only if the next consumer can read
  bytes or normalize them on the accelerator.
- Current practical Amdahl read: the largest remaining optimizer lever is not
  "make the byte stack copy faster." It is to avoid converting every batched
  observation into scalar host float32 rows before policy/search work.
- Next useful profile should compare:
  1. current host-scalarized policy/search pressure, and
  2. a device-resident or batched-stack policy/search pressure probe that
     consumes `[B,2,4,64,64]` before host scalarization.

## 2026-05-20 Pre-Scalarization Probe Gate

The new profile-only gate is intentionally narrow:

```text
CPU actors step compact CurvyTron rows
-> central GPU renderer updates [B,2,4,64,64] stack
-> synthetic batched-stack consumer reads that stack before CPU scalar rows
-> optional scalar LightZero-shaped materialization
```

This is not LightZero MCTS and not training. It answers one Amdahl question:
if the next consumer can read the batched stack before scalarization, how much
of the stack/scalarization tax can we avoid?

Why this matters:

- Previous host-scalarized probes consumed `[B*P,4,64,64]` after
  `materialize_lightzero_scalar_timestep`, so they could not distinguish GPU
  model/search cost from CPU row-materialization cost.
- Uint8 stack storage reduced memory movement in the stack itself, but lost at
  the scalar boundary. A pre-scalarization consumer is the direct falsifier for
  whether uint8/device ownership is still promising.
- The decision rule is simple: if pre-scalarization rows recover most of
  `stack_shift + scalar_materialization`, keep moving toward a batched/device
  policy/search handoff. If not, the current profile lane is mostly exposing
  renderer/service overhead and should not distract from stock full-loop work.

Guardrails:

- Terminal rows must use the protected pre-autoreset observation view.
- Uint8 terminal final observations must normalize the same way as live rows.
- Action masks must be passed as `[B,2,3]`; a consumer that ignores masks is
  only a rough pressure probe.
- The current implementation may skip scalar materialization only when a
  batched-stack probe is present. That prevents meaningless "no consumer"
  throughput numbers.

First result:

- Fixed `trail_slots=1024` was the wrong default for short profiles. It forced
  the renderer to draw empty trail capacity and made B2048 rows about `10x`
  slower than dynamic-slot rows. The Coach training launcher already uses
  dynamic slots; the profile wrapper now defaults to the same.
- With dynamic slots, B2048 no-probe float32 measured about `4703` scalar
  steps/s. B512 no-probe measured about `3976`, so widening from 512 to 2048
  helps, but only modestly.
- Pre-scalarization probing works, but skipping scalar materialization only
  saves about `0.35s` on the B2048 float32 sim16 row. That says scalar row
  creation is not the current big wall in this profile shape.
- Uint8 is still strategically interesting, but only if scalarization is
  skipped. With scalarization on, uint8 pays about `2.8s` converting back to
  float32 on CPU. With scalarization off, uint8 plus the synthetic device
  consumer is close to the float32 row and has lower H2D cost.
- Current Amdahl read: dynamic renderer cost and renderer/stack service
  overhead dominate the profile lane. The next useful question is trajectory
  length, because dynamic slots will become less cheap as alive trails grow.

Trajectory-length result:

- B512/A16 no-probe dynamic slots measured about `3976` scalar steps/s at
  20 profile steps, `1780` at 100 steps, `1149` at 200 steps, and `598` at
  500 steps.
- At 200 steps, observation was about `168s` of a `178s` measured row; renderer
  render alone was about `162s`. At 500 steps, observation was about `822s` of
  an `856s` measured row; renderer render alone was about `802s`.
- The 500-step row makes the Amdahl picture plain: for long no-death profiles,
  observation is about `96%` of wall and renderer render alone is about `94%`.
- Plain read: for bad short-lived policies, renderer speed is less urgent; for
  useful longer-lived policies, renderer cost becomes the wall again. The
  optimizer should not abandon renderer work just because short profiles look
  tolerable.
- Dynamic trail slots are mandatory for profiling and training-like runs. They
  avoid rendering empty trail capacity, but as trails become real the cost comes
  back. The future fix is not another scalar row tweak; it is a faster
  long-trail renderer/representation or a more compact observation path that
  preserves enough signal.

Dirty/incremental renderer status:

- There is already an exact CPU dirty-block cache in
  `SourceStateCanvasGray64DirtyRenderCache`, plus tests that compare it against
  the full browser-lines render.
- The separate `prototype_incremental_trail_layer_bench.py` local bench passed
  parity but only beat full redraw at 1000 trail points, by about `1.25x`. It
  was slower at 100, 200, and 500 trail points.
- Plain read: incremental rendering is likely the right shape, but not as a
  hand-written Python loop. The serious options are to use the existing dirty
  cache where it already works, port the dirty-block idea to a batched/GPU
  renderer, or deliberately choose a cheaper training observation that avoids
  long-trail redraw.
- The current local fixed-opponent CPU dirty-cache profile is the strongest
  practical clue so far: 100/500/1000-step no-death rows stay around
  `425-433` env steps/s with no fallbacks after cold start. Render remains
  about `77%` of local wall, but it no longer grows with total trail history.
- Therefore the high-level recommendation is not "full redraw on GPU forever."
  It is "keep the observation exact or semantically adequate while changing the
  cost model from redraw-all-history to update-only-what-changed." GPU only
  helps if it implements that cost model or batches enough work to beat the
  current dirty cache.

External-pattern research agrees:

- The strongest practical GPU lane is a persistent policy-space trail
  framebuffer plus transient head/bonus overlays. That is the same cost model
  as the exact CPU dirty cache, but aimed at batched/device execution.
- Scatter/raster writes must use explicit priority or ordered composition;
  racing duplicate writes are not a valid CurvyTron renderer.
- OpenGL/EGL, sprite atlases, nvdiffrast, and full graphics pipelines are
  probably over-scoped for the policy observation. They matter only if we need
  display fidelity, not if we need a learnable `[4,64,64]` policy tensor.
- Next profile-only gate: synthetic persistent policy framebuffer versus
  stateless redraw, with readback included. Device-only wins are not enough for
  the current loop if readback/stack ownership erases them.
- First synthetic gate result: persistent policy-space framebuffer is exact
  against the stateless synthetic target and gives real readback-included wins:
  `3.67x` at H100 B128/S64, `8.48x` at H100 B512/S512, `5.06x` at H100
  B2048/S256, and `10.86x` at L4 B512/S512. Device-only speedups were roughly
  `8x-53x`.
- No-readback H100 B512/S512 is `38.57x` faster end-to-end, while the same
  readback-enabled row is `8.48x`. That is the cleanest current proof that
  persistent rendering is powerful but frame ownership/readback still matters.
- H100 B512/S256 at `128x128` is `4.60x` faster end-to-end and `40.21x`
  faster device-side than synthetic stateless redraw. Larger observations are
  possible from the renderer side, but the cost moves toward readback/model
  input size.
- The right interpretation is not "this is production ready." It is "the
  update-only cost model can plausibly deliver the 10x-class renderer win if
  we can feed it real append/reset/clear/bonus events and preserve observation
  contracts."
- Independent local CPU/NumPy toybench found the same algorithmic shape:
  direct-64x64 append-only incremental rendering widened from about `10x` to
  `210x` versus replaying all old trail pixels as batch/trajectory pressure
  increased. It is not production timing, but it strengthens the cost-model
  conclusion.
- Current phase conclusion: stop spending main effort on full-redraw GPU
  kernel tuning. Build a real profile-only persistent renderer gate with
  explicit parity/fallback telemetry, then compare it against the GPU
  full-redraw ladder and CPU dirty-cache path.

## 2026-05-21 Fidelity Reorientation

The optimizer cannot recommend a faster observation path from speed alone.
The correct proof shape is:

```text
same source-state rollout
-> candidate fast renderer/stack
-> CPU oracle renderer/stack
-> compare latest frames, full stacks, terminal final_observation, reset rows
```

Current truth:

- Production/trusted policy observation is still CPU oracle
  `browser_lines + simple_symbols`, full 704-to-64 downsample.
- Persistent GPU `direct_gray64` is a policy-space approximation. It is not
  browser-pixel exact and should not be described as the same surface without a
  divergence gate.
- The first new divergence gate proves the harness works:
  CPU candidate versus CPU oracle is exact over reset plus four steps.
- The first persistent GPU `direct_gray64` smoke passed a deliberately loose
  tolerance over 32 steps, but it was not exact: max uint8 difference `61` and
  about `0.49%` aggregate mismatches by value.
- Longer follow-ups passed the same semantic gate:
  - timeout/autoreset row: max diff `64`, mismatch fraction `0.17%`, terminal
    rows observed;
  - 256-step no-death row: max diff `67`, mismatch fraction `2.71%`, no render
    truncation, active trail median `291`.

Interpretation:

- This does not kill the fast path. For policy learning, exact pixels are less
  important than stable, distinguishable game signal.
- It does change the burden of proof. Before Coach uses this as a default, we
  still need a component/visual diff summary and a small learning canary.
- Amdahl still matters: for long trajectories, renderer cost can dominate, so
  an approximate fast surface may be worthwhile. But it must be named as an
  approximation and tested as one.

## 2026-05-21 GPU / Host Boundary Reorientation

The current bottleneck question is no longer simply:

```text
Can the GPU render CurvyTron quickly?
```

The better question is:

```text
Can we keep a large CurvyTron batch alive across observation, policy/search,
and replay-shaped work without turning it into thousands of Python rows first?
```

Current evidence:

- The GPU resident profile canary works.
- H100 B512/A16/sim8 with replay/search-shaped synthetic pressure produced
  about `10.98k` scalar roots/sec before scalar materialization and about
  `7.62k` after scalar materialization.
- L4/T4 produced about `5.84k` before scalar materialization and about `4.13k`
  after scalar materialization on the same shape.
- H100 B1024 did not improve scalar-off throughput: it stayed around
  `11.07k` roots/sec and scalar-on fell to `6.75k`. Current profile-only
  scaling is therefore not "make B huge and win"; B512 is a better default
  shape until real consumer pressure changes the curve.
- These are profile-only numbers, not training speed.
- The stock LightZero-shaped path still has scalar env-manager/timestep
  boundaries. That is the likely place where the batch win gets lost.
- The stock-boundary audit found an even deeper issue: the trusted LightZero
  collect path rebuilds a batch from scalar env rows, then MuZero collect
  forward detaches latent roots/logits to CPU NumPy before MCTS. So the search
  boundary can also kill device residency, not just the env observation
  boundary.

Working model:

- One large `uint8 [B,2,4,64,64]` stack copy can be reasonable.
- Repeated tiny copies plus Python timestep objects are the danger.
- Readback is fine for coarse metrics/checksums; readback inside every env row
  is not.
- Bigger GPUs help only when the work reaches them as real batches.

Dedicated doc:

- [GPU / Host Overhead World Model](gpu_host_overhead_world_model_20260521.md)
- [Subagent GPU Host-Overhead World Model](subagent_gpu_host_overhead_world_model_20260521.md)
- [Stock Boundary Batch Death Audit](subagent_stock_boundary_batch_death_20260521.md)
- [Real-Consumer Canary Plan](subagent_real_consumer_canary_plan_20260521.md)
- [JAX/MCTX Spike Critique](subagent_jax_mctx_spike_critique_20260521.md)

## 2026-05-21 Real Consumer Canary Status

The smallest real-consumer gate now exists:

```text
uint8 [B,2,4,64,64]
-> flatten to [B*2,4,64,64]
-> normalize on the policy device
-> MuZeroPolicy.collect_mode.forward(...)
-> decode every root
-> optional scalar timestep materialization remains off for the primary row
```

The first remote smoke passed on L4/T4 with a scratch CurvyTron MuZero policy
on `cuda:0`, `8` roots per collect-forward call, zero illegal actions, and zero
materialized scalar timesteps.

The important wording:

```text
This proves the real public LightZero collect-forward boundary can consume the
pre-scalar batch. It does not prove device-resident MCTS, because this
LightZero path still crosses into CPU tree/search internals.
```

So the next Amdahl question is sharper:

```text
Does real collect-forward/search keep enough of the resident-batch win at
B512/A16/sim8, or does CPU MCTS/tree work collapse it back near stock/scalar
throughput?
```

Do not promote this to Coach launch advice until those medium rows are run and
matched against the scalar-edge rows.

## 2026-05-21 Corrected Split Interpretation

The corrected split rows are now measured.

Short version:

```text
Model inference is fast enough. LightZero collect/search is the current wall.
```

Concrete numbers:

| compute | model-only roots/sec | collect sim1 roots/sec | collect sim8 roots/sec |
| --- | ---: | ---: | ---: |
| H100 | `9238.85` | `3296.02` | `2693.10` |
| L4/T4 | `6790.63` | `1687.81` | `1381.35` |

What this means:

- The pre-scalar `[B,2,4,64,64]` stack can reach real LightZero APIs.
- The root model pass is not the blocker in this shape.
- Even sim1 collect-forward is much slower than model-only inference, so the
  cost is not only "more neural net calls from more simulations."
- H100 is better than L4 in collect-forward, but not enough to turn this into a
  pure bigger-GPU problem.
- Rendering is now a small fraction of these rows. Keep validating the
  approximation, but stop treating render-only work as the main 10x lane for
  this profile shape.

Current best bottleneck hypothesis:

```text
Public LightZero collect_mode.forward includes CPU tree/search work, Python
fanout, policy output decoding, and probably device/host synchronization that
our synthetic resident probe did not model.
```

Next useful falsifiers:

1. Add/inspect finer timing inside the collect-forward call if practical.
2. Compare a tiny direct model+toy search path against public collect-forward.
3. Keep the stock-loop validation gates in view, but do not re-open closed
   gates: normal death/autoreset and `rnd_meter_v0` have already passed as
   profile gates. The remaining validation work is semantic row labels and
   LightZero decode edge cases.
4. Keep L4/H100 comparisons, but only after the consumer path is clear enough
   that the GPU has real work to do.

Do not forget the validation lesson:

```text
Fast rows are only useful if they say exactly what they measured.
Every speed artifact needs backend, surface, death mode, RND mode, to_play,
mask filtering, scalar materialization, and consumer semantics.
```

## 2026-05-22 Search Boundary Update

The current best profile-only search boundary is now:

```text
direct_ctree_gpu_latent
```

It keeps the real LightZero CTree tree decisions, but keeps MuZero latent
states on GPU inside the simulation loop.

Fresh H100 B512/A16/sim8 fixed-denominator result:

| impl | scalar steps/sec | measured sec |
| --- | ---: | ---: |
| stock public facade | `2276.71` | `26.99` |
| direct CTree arrays | `4568.28` | `13.45` |
| direct CTree GPU-latent | `6580.32` | `9.34` |

Interpretation:

```text
The old "only 1.8x" boundary was incomplete. Avoiding repeated latent
CPU/GPU round trips makes the profile-only boundary about 2.9x faster than the
stock public facade on this row.
```

Amdahl now says:

- rendering is not the main wall in this denominator;
- public LightZero collect/search is the broad wall;
- GPU-latent CTree removes one real part of that wall;
- remaining walls are CPU CTree/list boundary, Python per-simulation loop,
  reward/value/policy CPU copies, root prep/output extraction, and host stack
  packaging.

Next architectural decision:

```text
If sim16 repeats the same pattern, finish GPU-latent gates as the tactical
bridge. For a larger jump, prototype dense GPU MCTS or a Cython array-native
CTree boundary rather than going back to renderer-only work.
```

The sim16 follow-up did repeat the pattern:

| impl | scalar steps/sec | measured sec |
| --- | ---: | ---: |
| stock public facade | `1734.05` | `35.43` |
| direct CTree arrays | `3083.58` | `19.92` |
| direct CTree GPU-latent | `4874.42` | `12.60` |

Current sharp version:

```text
LightZero's raw CTree kernels are not the only wall, and rendering is not the
main wall in this denominator. The wall is the whole collect/search boundary:
Python simulation loop, CPU CTree calls, latent/model-output crossings,
Python lists, root prep, output extraction, and compact packaging.
```

Decision:

```text
Keep GPU-latent as the best tactical profile boundary. Do not polish it forever.
The next high-upside work is either:

1. dense GPU search for the tiny action space; or
2. array-native Cython/C++ CTree APIs that remove Python list fanout.

Moving only the renderer or only the raw CTree kernels is no longer enough.
```

## 2026-05-22 Dense Search Reality Check

The dense GPU MCTS prototype is now real enough to profile, but not yet real
enough to recommend for training.

Current H100 sim8 read:

```text
direct_ctree_gpu_latent repeat: 6001 roots/sec
dense_torch_mcts v0:           6418 roots/sec
recurrent_toy ceiling:         7467 roots/sec
dense_torch_mcts cleanup v1:   7720 roots/sec
same-run recurrent ceiling:    9097 roots/sec
```

This changes the worldview slightly:

- Moving tree arrays to GPU is directionally right.
- The first dense implementation did not create a large jump because it still
  had Python control flow, tensor allocation, and synchronization inside the
  search loop.
- Removing the most obvious sync/allocation overhead did create a meaningful
  jump: about `1.30x` over same-run GPU-latent CTree.
- "Rewrite it in C" is too vague. LightZero already has C++ CTree kernels.
  The useful C/Cython version would be array-native CTree APIs and less Python
  list fanout, or a real compiled/batched search service.
- The immediate local patch is to reduce dense mode's remaining sync/kernel
  launch overhead and reprofile. The bigger architecture patch is batched GPU
  search with strong validation gates.

Current priority order:

1. Reprofile cleaned `dense_torch_mcts`.
2. If it moves, continue reducing GPU search sync/kernel-launch overhead.
3. If it does not move, prototype array-native CTree APIs or a more radical
   batched search service.
4. Keep renderer work as background only for long-trajectory rows where it
   reappears in the denominator.

## 2026-05-22 Sim16 Search Boundary Correction

The latest same-denominator sim16 rows changed the interpretation:

| impl | scalar roots/sec | measured sec |
| --- | ---: | ---: |
| `dense_torch_mcts` cleanup v2 | `4135.37` | `14.857` |
| `direct_ctree_gpu_latent` | `5010.39` | `12.263` |
| `recurrent_toy` ceiling | `9133.74` | `6.727` |

Plain read:

```text
GPU tree arrays alone are not the fix. If the GPU version is written as many
small eager Torch operations under Python control, sim count and depth can make
it slower than LightZero's CTree kernels with GPU-resident latents.
```

This means the next optimization target is more precise:

- keep `direct_ctree_gpu_latent` as the tactical profile baseline;
- test the fixed-shape dense rewrite because it removes one obvious dynamic
  indexing wall;
- if fixed-shape dense still loses at sim16, stop polishing eager Torch and move
  to compiled/fused search or array-native CTree APIs.

## 2026-05-22 Fresh Search-Boundary Worldview

The fresh H100 ladder is now the cleanest denominator for this lane:

```text
B512 physical rows, 1024 player-view roots per step, A16,
60 measured steps, 15 warmup, uint8 stack, scalar materialization off,
profile no-death, root noise 0.0
```

Key rows:

| impl | sim8 roots/sec | sim16 roots/sec | read |
| --- | ---: | ---: | --- |
| stock facade | `2430.21` | `2094.27` | public LightZero collect/search wrapper |
| direct CTree arrays | `5008.97` | `3448.25` | real CTree, compact arrays, CPU latent handoff |
| direct CTree GPU-latent | `7547.12` | `6145.25` | real CTree, hidden states stay on GPU |
| dense Torch MCTS after semantic fix | `8288.37` | `4293.88` | profile-only GPU tensor tree, not CTree |
| recurrent toy ceiling | `12834.00` | `9191.12` | fake search ceiling; sim8 row used H100 NVL |

Plain version:

```text
GPU-latent CTree is the best practical LightZero-shaped profile boundary.
Dense Torch proves that device-resident search can be fast at shallow sim8, but
eager Torch search does not scale cleanly to sim16 after the semantic fix.
```

What this does to priorities:

1. Keep `direct_ctree_gpu_latent` as the tactical baseline for any
   LightZero-shaped bridge recommendation.
2. Do not promote `dense_torch_mcts` to training. It is profile-only and fails
   the sim16 gate.
3. The next high-upside implementation is not "more renderer." It is either a
   compiled/fused fixed-shape search prototype or an array-native CTree boundary
   that removes Python/list fanout.
4. The larger production architecture question remains actor/search batching:
   many roots alive, batched recurrent model calls, compact tree state, and
   minimal host synchronization.

Current honest speed claim:

```text
Against the stock public facade in this profile denominator, GPU-latent CTree is
about 3x faster. Against the current best practical boundary, the remaining
credible headroom before a bigger architecture rewrite is closer to 1.5x-2x,
not 10x.
```

Sim32 update:

```text
direct_ctree_gpu_latent: 4127 roots/sec
dense_torch_mcts:       2007 roots/sec
recurrent_toy ceiling:  6162 roots/sec
```

This makes the priority sharper:

- eager dense Torch is not the route for deeper search;
- more local Python/Torch cleanup is not enough;
- if we want a GPU-tree lane, it must become compiled/fused/static-shape;
- if we want a LightZero-compatible lane, the array-native CTree boundary is
  the cleaner next design.

CPU-scaling question:

```text
We had not tested 64/128 CPU allocations in the current ladder. The sidecar now
has a profile-only `gpu-h100-cpu64` route for LightZero boundary probes. Modal
rejected `cpu=128` because the function CPU request limit is 64 cores. This is
only for falsifying whether more CPU helps the CTree boundary; it does not
change trainer defaults.
```

CPU-scaling result:

```text
direct_ctree_gpu_latent sim16:
  H100+4 CPU:  6145 roots/sec
  H100+64 CPU: 5119 roots/sec

stock_facade sim16:
  H100+4 CPU:  2094 roots/sec
  H100+64 CPU: 1776 roots/sec
```

Plain version:

```text
More CPUs are not the fix for this profile shape. The wall is not just "not
enough CPU cores." It is the boundary design: Python/list/object control,
per-simulation handoff, and CPU/GPU synchronization.
```

Train-facing correction:

```text
The profile-only fast path now has a first bridge into the trusted stock loop:
collect_search_backend=direct_ctree_gpu_latent.
```

What changed:

- The launcher can patch `MuZeroPolicy._forward_collect` during `mode=profile`
  only.
- Stock `train_muzero` still owns collector, GameSegment/replay, target build,
  learner, checkpoints, RND hooks, and eval/GIF sidecars.
- The patched collect method returns the stock per-env dict:
  action, raw legal-action visit counts, visit entropy, searched value,
  predicted value, and full-action policy logits.
- The hook keeps root latent tensors on CUDA during search and still uses
  LightZero CTree for traversal/backprop.
- It is not a train default and not a Coach speed claim yet.

Plain current blocker:

```text
We have turned the search-boundary idea into a real full-loop profile candidate.
We still need the matched train_muzero A/B before saying actual training is
faster.
```

2026-05-22 train-facing repeat correction:

```text
Matched stock train_muzero C64/sim16/no-RND/no-death/H100, three learner calls:
  stock:  445.19 collected env steps/sec, wall 36.80s
  direct: 438.56 collected env steps/sec, wall 37.36s
```

Plain version:

```text
The direct hook is not a stable full-loop speedup yet. It does reduce the
search-ish buckets, but it gives the time back at the boundary.
```

The remaining measured direct boundary costs in the repeat:

- model-output device-to-host/list conversion: `3.41s`;
- stock per-env output assembly: `2.87s`;
- CTree still receives CPU/list reward, value, and policy-logit payloads each
  simulation.

Current Amdahl read:

```text
The bottleneck is the shape of the LightZero collect/search boundary. More CPU
cores did not help. The next useful work is either a small output/list cleanup
inside the direct hook, or a larger array-native CTree / compiled tree design.
Do not sell direct_ctree_gpu_latent to Coach as a training-speed recommendation
until a matched full-loop repeat wins.
```

Output-fast correction:

```text
The small output/list cleanup did work. Matched H100 C64/sim16/3-learner
profile rows:
  stock:              433.17 steps/sec, wall 37.82s
  direct output-fast: 566.19 steps/sec, wall 28.94s
```

Plain version:

```text
Historical direct-CTree probe:
  direct_ctree_gpu_latent plus all-actions-legal output fast path.

Current clean profile-only optimizer headline:
  real-checkpoint MCTX/JAX shadow search at H100 B1024/A16/sim8 scalar-off
  MCTX:        19,334 active steps/sec
  direct CTree: 8,792 active steps/sec
  speedup:      2.20x

This is still not Coach-facing. The same-root comparator now proves root
identity, but it also shows MCTX and LightZero CTree search outputs diverge
enough that this is not a drop-in semantic replacement yet.
```

New bottleneck map:

- output assembly dropped to `0.077s`, so stop optimizing that for now;
- model-output D2H/list conversion is still `2.47s`;
- recurrent inference is `4.28s`;
- MCTS/search is `8.06s`;
- stock collector/env/replay shell still surrounds the direct hook and consumes
  the rest of the `28.94s` wall.

Priority:

```text
Stay off the CPU-count rabbit hole. The next serious speed lane is an
array-native CTree / collect-search boundary that removes per-simulation
D2H/list payloads. The smaller local lane is to see if any D2H/list conversion
can be cut without changing CTree semantics.
```

RND-meter check:

```text
H100 C64/sim16/3-learner with rnd_meter_v0:
  stock:              342.33 steps/sec, wall 47.86s
  direct output-fast: 410.55 steps/sec, wall 39.91s
```

Plain version:

```text
The output-fast direct hook survives the current RND meter path, but the win
shrinks to about 1.20x because RND adds separate work. The new RND-side wall is
not search: rnd_train_with_data is about 3.5s and rnd_state_hash is about 3.0s
in this short profile.
```

RND hash-fix update:

```text
The RND-side wall was real and easy to remove. We now hash predictor/target
state once before the whole RND update batch and once after it. That preserves
the proof that the predictor changed and the target stayed frozen, but avoids
hundreds of full model hashes during one collect.
```

After the fix:

```text
stock rnd_meter_v0:  351.02 steps/sec, wall 46.68s
direct rnd_meter_v0: 448.52 steps/sec, wall 36.53s
```

The relevant RND timers changed from roughly:

```text
rnd_train_with_data: ~3.5s -> ~0.6s
rnd_state_hash:      ~3.0s -> ~0.14s
```

Packed-transfer update:

```text
Packing reward/value/policy logits into one CPU transfer is safe and keeps the
sidecar and train-facing hook aligned, but it is not a large win. Listifying is
only about 0.08s. The D2H-labelled bucket is mostly synchronization/wait plus
boundary shape, not raw transfer bandwidth.
```

Current optimizer answer:

```text
What are we fixing?
  The stock LightZero collect/search boundary, plus measured side overhead like
  RND diagnostics when it shows up.

What is the best thing to optimize next?
  The per-simulation CTree boundary: CPU/list payloads and object-shaped API
  fanout. The realistic next lanes are array-native CTree for fixed A=3 or a
  compiled/fused batched search spike with parity gates.

Why are we blocked?
  The current direct hook still uses CPU CTree traversal/backprop and converts
  model outputs into CPU/list payloads every simulation. More CPU cores do not
  fix that topology, and tiny D2H/output cleanups cannot produce 10x against
  the whole train_muzero denominator.
```

## 2026-05-22 Radical Architecture Correction

The current `direct_ctree_gpu_latent` hook is a useful tactical bridge, not the
final architecture.

Current matched full-loop profile currency:

```text
no-RND:     stock 433 steps/sec -> direct output-fast 566 steps/sec, about 1.31x
rnd_meter: stock 351 steps/sec -> direct output-fast 449 steps/sec, about 1.28x
```

Plain read:

```text
This proves the collect/search boundary matters.
It also proves the current patch is too small to be the 5-10x move.
```

The next bottleneck is the topology around search:

- CTree traversal/backprop core is already C++.
- The loop still crosses Python, NumPy, lists, CPU, and GPU every simulation.
- The public LightZero output shape still wants per-env dicts and scalar
  objects.
- More CPU cores made this boundary slower, so this is not a simple CPU
  capacity shortage.

External systems agree with this read. OpenSpiel's faster AlphaZero path uses
threads/cache/batched inference/GPU; MiniZero keeps multiple MCTS instances
alive and batches GPU leaf evaluation; KataGo's analysis engine is fast because
of cross-position batching; MCTX is batch-first and JIT-oriented.

The updated optimizer thesis:

```text
Small wrappers can give 1-2x.
A 5-10x path needs compact batched search ownership: many roots alive, batched
model calls, compact tree/search state, and scalar replay objects only at the
compatibility edge.
```

Current ordered falsifier:

1. Finish train-facing direct hook validation.
2. Run one fixed-shape compiled/fused dense search spike.
3. If dense compile fails sim16, switch to array-native fixed-A=3 CTree.
4. If both are capped, sketch the MiniZero/KataGo-style search service.

## 2026-05-22 Compile Falsifier And Bigger-Move Read

The fixed-shape compile spike has now been measured on the intended H100
denominator:

```text
H100, B512, actor_count 16, 60 measured, 15 warmup,
root-noise0, all actions legal, scalar materialization off
```

| row | sim8 roots/sec | sim16 roots/sec | read |
| --- | ---: | ---: | --- |
| `dense_torch_mcts_compile_spike` | `10298.01` | `4872.70` | good sim8, fails sim16 |
| `direct_ctree_gpu_latent` | `7567.35` | `6153.95` | practical baseline still wins sim16 |
| `recurrent_toy` ceiling | `9524.57` | `8969.89` | model-call ceiling, not real MCTS |

Plain read:

```text
The exact compiled helper is not the next 5-10x path.
```

Why:

- sim16 is the real gate, and compiled dense search lost to
  `direct_ctree_gpu_latent`;
- Torch reported skipped CUDA graphs because the helper mutates tree tensors;
- sim16 hit Dynamo recompiles tied to the triangular `simulation_index` loop;
- recurrent inference and search update are still many small pieces rather than
  one stable service-owned loop.

This changes the next action:

```text
Stop polishing this compile helper.
Keep direct_ctree_gpu_latent + output-fast as the near-term practical profile
baseline.
Build the next falsifier around a compact search-service ceiling and an
array-native fixed-A=3 CTree critique.
```

Current architecture thesis:

```text
1.3x is not mysterious. It is the expected size of a boundary cleanup.
5-10x probably needs the batch to survive across search ownership, not just
across rendering or one LightZero hook.
```

## 2026-05-22 Compact LightZero Hook Update

The compact sidecar now reaches a real LightZero/search-shaped consumer in the
profile harness.

Plain version:

```text
Before: compact profiles could prove obs+mask and metadata could be batched.
Now: the direct CTree profile boundary can consume the compact sidecar itself.
```

What changed:

- `HybridCompactBatch` now includes `to_play[M]` and `active_root_mask[M]`.
- The fixed-opponent profile convention is `to_play=-1`.
- Active roots are roots with at least one legal action and `done_root=false`.
- `_LightZeroCollectForwardStackProbe.run_compact_batch(batch)` validates
  row/player ids, active masks, `to_play`, and `target_reward`.
- Inactive roots are filtered before entering the existing direct CTree arrays
  code path, so terminal rows with stale legal masks are not searched.

Validation so far:

```text
Focused compact sidecar + direct CTree hook tests:
11 passed, 100 deselected

Full boundary profile suite:
92 passed

After malformed-sidecar hardening:
113 passed across hybrid + boundary suites
```

The first remote attempts failed in useful ways:

```text
direct CTree arrays refused the old renderer backend.
The persistent policy framebuffer refused the wrong render surface.
The config validator refused trail_slots below min_render_trail_slots.
```

Those failures are evidence that the guard is doing its job. The corrected
remote smoke then passed:

```text
ap-RztU5jMKmKBpXuaDY3vZB0
L4/T4, batch_size=2, actor_count=1, sim1
jax_gpu_persistent_policy_framebuffer_profile + direct_gray64
compact_row_player_sidecar_v1
materialized_timestep_count=0
lightzero_illegal_action_count=0
```

The hook also now validates binary compact action masks before bool coercion,
`done_root == repeat(done, player_count)`, terminal/autoreset row masks,
final-observation masks, and basic sidecar shapes before search.

RND sidecar proof:

```text
HybridCompactBatch.observation [B,P,4,64,64]
-> uint8 or normalized float32 stack
-> [B*P,4,64,64]
-> existing RND latest-frame input [B*P,1,64,64]
```

This proof runs with `materialize_scalar_timestep=false`, so it shows the RND
input shape can come from the compact sidecar rather than from scalar
LightZero timestep objects. It is still an input-shape proof, not a full RND
training-throughput solution.

Post-guard local refresh:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
payload+merge:       40471.44 timesteps/sec
native actor buffer: 66136.26 timesteps/sec
```

The exact ratio is local and noisy, but the direction survived the correctness
guards: direct actor writes into parent-owned compact arrays still remove a
meaningful object/merge wall in the profile harness.

Current read:

```text
This is still profile-only.
It is not Coach training advice yet.
The important gain is architectural: the real direct-CTree profile hook can now
be driven from a compact row/player batch instead of scalar timestep objects.
```

Compact target-row proof:

```text
HybridCompactBatch sidecar
+ active-root ordered selected_action / visit_policy / root_value
-> PolicyRowRecordV0
-> existing checked source-state target rows
```

Validation:

```text
tests/test_multiplayer_source_state_target_rows.py -> 18 passed
```

Plain read:

```text
The compact sidecar can now cross the replay/target compatibility edge in a
checked way. This is still a profile-only adapter proof, not native LightZero
GameSegment support and not a learner/full-loop speedup claim.
```

Combined edge proof:

```text
HybridCompactBatch
-> direct CTree compact profile hook
-> action/visit/value arrays
-> compact target-row adapter
-> checked source-state target rows
```

Validation:

```text
tests/test_source_state_batched_observation_boundary_profile.py
tests/test_multiplayer_source_state_target_rows.py
-> 112 passed

Focused compact-sidecar/RND/boundary/target sweep
-> 156 passed
```

Plain read:

```text
Correctness scaffolding is no longer the main blocker. The aggressive next
step is a closed compact batch consumer or service-shaped falsifier that can
change the denominator by multiple times, not another small adapter polish.
```

Next gates:

1. Direct CTree compact parity/statistical gates must stay green.
2. Build and time a closed compact consumer:
   search output, RND input, and target-row materialization all from the same
   compact sidecar.
3. If that cannot plausibly reach a `3x` class profile win, escalate to a
   MiniZero/KataGo-style batched search service or native/vector buffer
   prototype.
4. Only after those gates should we run matched full-loop A/B rows and talk
   about actual Coach-facing speed.

## Current Optimizer Read, 2026-05-22

Separate the currencies:

```text
Coach iters/hour is the only production currency.
Train-facing steps/sec is the closest local proxy.
Profile-only roots/sec is a falsifier for one architecture slice.
```

The renderer/env wall is mostly not the main blocker anymore. The current wall
is the LightZero-shaped collect/search/replay topology: Python simulation
control, CTree/list ABI, recurrent-output handling, D2H/list conversion,
object-shaped replay/RND/target edges, and stock collector shell costs.

Important falsifications:

```text
flat-A3:
  no-model microbench won, matched full-loop lost/slightly worse.

dense_torch_mcts:
  fresh sim16 was close, fresh sim32 lost badly.

dense_torch_mcts_compile_spike:
  worse than direct on the fresh denominator.

precomputed recurrent:
  useful 1.15-1.32x slice, not the whole wall.
```

Most important active bet:

```text
Own the compact collect/search/replay boundary.

HybridCompactBatch
-> CompactRootBatchV1
-> compact/fixed-shape search service
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

The active `service_tax_probe` is not real MCTS. It is a bounded falsifier:
after paying real model/recurrent/packing/replay tax, how much of the
mock-search ceiling survives? If little survives, we need a more radical
device-resident search body, not more wrapper cleanup. If a lot survives, the
compact service contract is the right architecture lane.

Closed-service result:

```text
sim16 service_tax_probe: 2.10x direct loop throughput.
sim32 service_tax_probe: 1.29x direct loop throughput.
mock_search_service: about 1.8x direct at both sim16 and sim32.
```

Interpretation:

```text
The compact boundary is worth doing, but it is not enough by itself. At sim32,
real model/recurrent tax eats most of the mock ceiling. The next high-upside
work should test a real device-resident/fixed-shape search body, or an MCTX/JAX
visual-root scratch benchmark, because another Python wrapper cleanup is
unlikely to deliver a 5-10x result.
```

MCTX scratch update:

```text
The guarded H100 visual-root MCTX scratch rows passed the search-architecture
gate. B512/P2/sim16 reached 88.5k fresh-boundary roots/sec, and B512/P2/sim32
reached 45.5k-47.5k. The same direct_ctree_gpu_latent denominators were about
5.2k at sim16 and 4.4k at sim32 for the direct loop.

This older scratch result was not production training and not a learning proof.
It showed 10x-class search-boundary headroom with a toy model.

Updated read: the current real-shadow lane uses an immutable LightZero
checkpoint translated into JAX, but it remains profile-only. MCTX/Gumbel
semantics differ from LightZero CTree, same-root smoke proves wiring/identity
only, and selected actions/visit distributions/root values still diverge.
```

Guardrail:

```text
Never compare Amdahl slices from runner printed last-step telemetry. Use
aggregate compact.timings from row result JSONs.
```

Real compact visual-root update:

```text
The next H100 gate moved from synthetic visual roots to real renderer-backed
HybridCompactBatch roots. B512/P2/sim16 reached 124.1k fresh-boundary roots/sec;
B512/P2/sim32 reached 51.5k. Both rows validated legal actions,
CompactSearchResultV1, and CompactReplayIndexRowsV1 after stepping the compact
env once with selected MCTX actions.

This strengthens the device-resident compact-search bet. It still does not
settle the trainer question because the MCTX model is a toy JAX model, not the
current LightZero PyTorch network, and learner/RND ownership is not integrated.
```

Current priority:

```text
Stop polishing small wrapper lanes unless they protect the compact path.
The next big proof is current-model/device-resident search realism plus a
compact learner/replay/RND edge. Native/vector buffers are support work to keep
B512+ batches compact; they are not the headline by themselves.
```

Newest Amdahl correction:

```text
MCTX search is now fast enough that the next measured edge becomes the wall.
The replay-index contract rows step the compact env once after search. That
edge costs about 0.23-0.52s in the B512/B1024 rows, while the MCTX
fresh-boundary pass costs milliseconds. So the next proof must put repeated
env/observation/replay/RND edges in the denominator.

Do not sell the 50k-168k roots/sec rows as full training speed. Sell them as
evidence that search can stop being the wall if we own the compact loop.
```

Current highest-leverage question:

```text
Can we build a repeated closed compact loop that keeps observations, search
results, replay indices, and RND/target inputs compact/resident enough that the
MCTX search win survives? If yes, this is the 5-10x lane. If no, the next wall
is actor/env/observation/replay synchronization, and that is where Amdahl says
to spend effort.
```

Closed compact loop update:

```text
The first H100 repeated-loop rows answered part of that question.

B256/P2/sim16/loop4: 3.25k active roots/sec.
B512/P2/sim16/loop8: 5.06k active roots/sec.
B512/P2/sim32/loop8: 4.87k active roots/sec.
B1024/P2/sim16/loop8: 6.41k active roots/sec.
B1024/P2/sim32/loop8: 5.03k active roots/sec.

Search-boundary roots/sec in the same rows was tens or hundreds of thousands,
so the search win is real but does not survive the current closed-loop edge.
Bigger batches amortize the edge a little. They do not give the 10x full-loop
win by themselves.
```

New priority:

```text
Keep MCTX/JAX as the strongest search architecture signal, but move the
immediate optimizer work to the closed compact edge. Break down compact env
step, observation/stack update, replay-index row construction, RND latest-frame
input, target materialization, and synchronization. The next large win has to
remove or batch those costs, not just make search faster.
```

Native actor-buffer correction:

```text
The first renderer-backed native actor-buffer rows helped but did not change
the big picture.

B512/P2/sim16/loop8:  5.79k -> 6.82k active roots/sec.
B1024/P2/sim16/loop8: 6.25k -> 8.92k active roots/sec.

This removes one payload/merge layer, but env_step_sec is still the slowest
bucket. In the native rows, env_step_sec is about 74% of B512 closed-loop wall
and about 81% of B1024 closed-loop wall. Search is only about 3-5%.
```

Current Amdahl read:

```text
The bottleneck is not MCTX search right now. It is the compact env/observation
step around it. More exactly: actor step plus observation/renderer/stack update.

The next useful big change is therefore a device-resident or lower-copy
observation/update path. We need to stop rebuilding/reading/moving the compact
stack in the current parent loop if the next search call will consume the same
shape. A small native-buffer cleanup is worth keeping, but the 5-10x path needs
to cut the repeated env_step_sec bucket itself.
```

2026-05-22 live-prefix trim update:

```text
The newest rows found and fixed one concrete env-step waste:
production-state -> compact-render-state conversion was still walking/copying
inactive trail capacity. The live-prefix trim cuts that bucket strongly at
B1024 and is validated locally.

Representative repeated closed-loop H100 rows after the trim:
  B1024/sim16/loop16: 15.26k active roots/sec.
  B1024/sim16/loop32: 12.38k active roots/sec.
  B1024/sim32/loop32: 11.48k active roots/sec.
  B2048/sim16/loop16: 13.55k active roots/sec.

The improvement is real, but it is not the architecture break. The repeated
loop still spends roughly 71-82% of measured wall in env_step_sec. The search
bucket is usually single-digit percent, except sim32 where it rises to about
10%. That means a perfect search win would not give the next large full-loop
win in this denominator.
```

Actor-count correction:

```text
In-process actor sharding is not the answer in the current manager.

B1024/sim16/loop16 after the live-prefix trim:
  actor_count=1:  16.42k active roots/sec.
  actor_count=4:  13.15k active roots/sec.
  actor_count=16: 11.92k active roots/sec.

This does not prove subprocess/native parallel actors are bad. It only proves
that more in-process shards in this profile manager add overhead and should not
be the next "aggressive" move.
```

Current plain priority:

```text
The next big optimizer question is state residency, not another narrow search
kernel polish:

1. Split VectorMultiplayerEnv.step and observation update enough to see the
   remaining actor/runtime/package/copy buckets.
2. Prototype a profile-only resident compact observation loop: latest compact
   frame/stack stays in the device or compact array layout that MCTX consumes.
3. Keep strict compact replay/RND/target validation at the edge so speed work
   does not silently change training semantics.

This is the larger change if we want more than 1-2x. The small copy trims stay,
but they are support work.
```

2026-05-22 late H100 grid correction:

```text
The fresh grouped profile settles the "is it game mechanics?" question:
mostly no.

B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay rows:
  host sim16:     23.1k roots/sec, env 68.1%, search 7.5%.
  resident sim16: 30.3k roots/sec, env 68.3%, search 9.6%.
  host sim32:     19.5k roots/sec, env 63.0%, search 15.2%.
  resident sim32: 26.8k roots/sec, env 59.8%, search 19.7%.
  refresh-off ceiling sim16: 57.9k roots/sec.

Inside env_step_sec on refresh-on rows:
  actual game mechanics: about 8-11% of env_step_sec.
  observation/search-input handoff: about 76-80% of env_step_sec.
  GPU draw: about 5-7ms over the measured loop.
```

Plain read:

```text
GPU rendering work was not dropped. It worked enough that raw drawing is now
small. The active wall is the handoff around the renderer: copying actor visual
state into parent buffers, packing compact state/deltas, H2D for renderer
state, host stack update in host mode, and resident stack ownership in resident
mode.

Resident stack is now a real win after root-copy removal, but it does not solve
actor render-state copying. The next high-upside patch should move from
"copy production state into renderer input every step" to "own/update compact
renderer state in the layout the persistent renderer consumes."
```

2026-05-22 borrowed-state update:

```text
The first state-ownership canary passed strongly.

Latest matched H100 B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay:
  resident sim16 copied:    34.1k active roots/sec
  resident sim16 borrowed:  51.8k active roots/sec
  speedup:                  about 1.52x

Fresh no-refresh ceiling:
  resident sim16 refresh off: 61.9k active roots/sec
```

Plain read:

```text
Actor render-state copying was a real wall. Removing it did not just move
timer labels; it improved total closed-loop throughput.

The Amdahl picture changed after this fix. The best refresh-on row is now
within about 1.2x of the no-observation-refresh ceiling at sim16. At sim32,
search is already about 30% of measured wall. So the next phase should not
hammer raw rendering. The useful next split is:

1. remaining observation/state handoff: production-to-compact, delta pack,
   renderer H2D/update, public info/batch packaging;
2. search/service boundary as sim count rises;
3. normal-death/RND/replay edge validation so the compact path does not only
   work in no-death profile rows.
```

2026-05-22 current-code refresh ceiling:

```text
Fresh H100 B1024/P2/body4096/h64/depth16/loop48/native/no-copy/replay,
actor_count=1, resident GPU stack, explicit resident sync off.

refresh-on, borrowed render state:
  sim16: 62.7k active roots/sec, total 1.569s,
         env 71.2%, search 19.8%.
  sim32: 49.1k active roots/sec, total 2.002s,
         env 54.1%, search 38.9%.

refresh-off ceiling, borrowed flag disabled because refresh is off:
  sim16: 98.5k active roots/sec, total 0.998s,
         env 54.3%, search 29.8%.
  sim32: 74.9k active roots/sec, total 1.313s,
         env 31.9%, search 58.6%.
```

Plain read:

```text
The current renderer/observation refresh gap is real but bounded. On this
denominator, deleting refresh entirely is worth only about 1.5-1.6x. That is
useful, but not the 5-10x architecture break.

The refresh-off rows still pay CPU game mechanics and public packaging, and at
sim32 search becomes the largest bucket. So the next serious speed plan must
include compact-buffer/search-service ownership, not only another renderer
canary.
```

Subagent synthesis update:

```text
Full dataflow map:
  selected-action readback is tiny and required while the env is CPU-owned.
  Large objects are the latest frame, visual stack, root observation, and
  visual trail arrays. Keep those resident or borrowed.

GPU sync model:
  JAX/CUDA guidance matches the measurements: keep latest frame, stack,
  framebuffer, and MCTX tensors resident; read back only selected actions on
  the action-critical path; commit replay/RND/search payloads in compact
  chunks with parity gates.

Architecture critique:
  direct visual-delta is feasible but narrow. The top higher-upside tests are
  a mock batched search-service ceiling, Puffer-style contiguous compact
  buffers, and a fixed-shape GPU search core behind the same compact API.
```

## 2026-05-23 Full Dataflow Reorientation

Same-shape compact service comparator:

```text
H100, B512/A16, sim16:
  direct_ctree_gpu_latent:  7,155.7 steps/sec
  service_tax_probe:      12,461.6 steps/sec
  mock_search_service:    17,711.9 steps/sec
```

The important correction:

```text
The current LightZero service-comparator lane is not fully GPU-resident.
The JAX renderer may keep its framebuffer/latest frame on GPU, but the
comparator can still return frames to a host stack and then send that stack to
Torch/LightZero.

Do not compare that row directly with older resident MCTX rows as if they were
the same currency.
```

Data movement read:

```text
Actions, legal masks, visit policies, and root values are small. At B512 they
are KiB-scale payloads. Reading selected actions once per env step is fine and
required while the env is CPU-owned.

The bad shape is per-simulation synchronization:
  CPU CTree traverse
  -> small action payload to GPU
  -> GPU recurrent model
  -> small model output back to CPU
  -> Python listification
  -> CPU CTree backprop

The bytes are small, but the control loop is expensive and serial. That is the
main reason direct LightZero CTree GPU-latent can still be slow on an H100.
```

Architecture read:

```text
2x class:
  Make the compact service real and replay-valid.

5x class:
  Compact service + fixed-shape search + compact env/replay ownership.

10x class:
  Service/device-resident architecture with many roots in flight.
```

Validation read:

```text
Current local compact/search/replay tests are useful, but promotion is blocked
on one real closed-loop proof:

  actual search-selected actions
  -> drive the next env step
  -> land in the same replay, RND, and player-perspective rows

Until that passes, fast rows are profile-only evidence.
```

2026-05-23g model-bridge update:

```text
MCTX/JAX is no longer only a toy-search idea. We now have the first raw-model
bridge gate:

  LightZero PyTorch MuZeroModel weights
  -> JAX shadow model
  -> initial_inference / recurrent_inference parity

Fresh-model Modal L4 smoke passed on the real LightZero model class with
nonzero final heads and JAX backend=gpu.

This is still not Coach-ready:
  - no checkpoint parity has passed yet;
  - no MCTX search uses the real model yet;
  - no train_muzero path has been changed.

The next correct gate is current immutable checkpoint parity. If that passes,
the next optimizer move is to plug the JAX shadow model into the profile-only
MCTX compact search service.
```

Important numeric detail:

```text
Torch GPU and JAX GPU latent tensors are not bit-exact. The fresh smoke saw
max latent abs error around 2e-4. Use explicit GPU tolerance 5e-4 for this
bridge unless a stricter CPU-only gate says otherwise.
```

Implementation update:

```text
CompactSearchServiceV1 exists.
Direct CTree adapter exists.
Array-ceiling adapter exists.
compact_search_result_v1_from_arrays now validates arrays from an already-run
probe, so profile code can avoid double-running search.
The hybrid compact replay proof now uses that same helper.
The compact replay proof now emits explicit action-feedback verification:
expected joint-action checksum from search must equal the applied joint-action
checksum from the next env step.
Mock/service-tax array-ceiling probes now consume `HybridCompactBatch` directly
when available and emit common `compact_service_*` telemetry without a second
probe call.

Validation:
  tests/test_compact_search_replay_contract.py -> 10 passed
  tests/test_source_state_batched_observation_boundary_profile.py -> 108 passed
  tests/test_source_state_hybrid_observation_profile.py -> 35 passed
```

2026-05-23 compact sample-edge update:

```text
The profile-only compact slab now has a sample gate:

  compact search result
  -> selected action drives next env step
  -> compact replay index rows
  -> target rows
  -> learner-shaped sample batch

The important part is what is absent: this gate runs with
`materialize_scalar_timestep=False` and reports zero `MockBaseEnvTimestep`
rows. That means the optimizer profile can now price the compact collection
and sample boundary without routing through the old scalar LightZero timestep
objects.

Cadence matters. Sampling every environment step is an intentionally harsh
stress test and can dominate the tiny profile row. Sampling once per collected
chunk with a fixed learner-like batch is the better performance read. On the
first matched H100 row, that realistic cadence was only a small tax:
`3660` steps/sec baseline vs `3428` steps/sec with one batch64 sample gate
after 20 opportunities.

Still not proven:
  - no stock train_muzero integration has changed;
  - no Coach run should consume this as launch advice;
  - the sample gate is a profile proof, not a learner update proof.
```
