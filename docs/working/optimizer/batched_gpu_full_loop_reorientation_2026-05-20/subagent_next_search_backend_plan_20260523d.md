# Next Search Backend Plan, 2026-05-23d

Scope: design critique only. I inspected the compact search contracts, compact
slab profile summary, MCTX/JAX benchmark path, compact Torch service, and active
optimizer docs. I did not touch live Coach runs, trainer defaults, checkpoints,
evals, tournaments, GIFs, Modal volumes, or source code.

## Short Answer

The smallest next backend worth prototyping is a profile-only MCTX/JAX compact
search service behind `CompactSearchServiceV1`.

Plain shape:

```text
CompactRootBatchV1
-> fixed-shape JAX/MCTX search
-> CompactSearchResultV1
-> CompactRolloutSlab selected-action feedback
-> CompactReplayIndexRowsV1
```

This should not be sold as a Coach training backend. It is the cleanest next
falsifier for the question:

```text
If the CTree/list/control loop is the wall, how fast is the same compact slab
when search is a compiled batched array program instead?
```

## Current Facts I Trust

Latest H100 slab profile:

```text
direct_ctree_gpu_latent is the real-search baseline.
B1024/A16/sim16: 6992 steps/sec, 9668 slab roots/sec
B1024/A16/sim32: 5314 steps/sec, 6542 slab roots/sec
```

Ceiling rows on the same slab shape show real but bounded headroom:

```text
service-tax/mock are about 1.7x-2.8x faster than direct_ctree_gpu_latent
depending on batch and simulation count.
```

The timing split says the biggest real slab bucket is search:

```text
B512/A16/sim16 direct GPU-latent:
  measured total: 2.948s
  slab total:     2.076s
  slab search:    1.288s
  slab model:     0.427s
  slab H2D:       0.362s
```

Precomputed recurrent is only a synthetic falsifier:

```text
It gives about 1.1x-1.5x over direct_ctree_gpu_latent.
So deleting recurrent model calls alone is not the whole answer.
The remaining CTree/list/control path is still large.
```

## What Direct CTree Still Pays

`direct_ctree_gpu_latent` is useful because it keeps latent storage on the GPU,
but it does not make search fully GPU-native.

The current direct path still does this inside each MCTS simulation:

```text
CTree batch_traverse on CPU/list-shaped state
-> gather leaf latents on GPU
-> copy last actions to GPU
-> run recurrent model on GPU
-> copy reward/value/policy logits back to CPU
-> convert arrays to Python lists
-> CTree batch_backpropagate on CPU/list-shaped state
```

Relevant implementation:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
  _run_direct_ctree_gpu_latent_search(...)
```

That explains why wrapper cleanup is not enough. The expensive part is the
search body shape.

## Candidate Comparison

| Candidate | Why it could help | Why it may fail | Verdict |
| --- | --- | --- | --- |
| Compile current compact Torch service | Small code surface, already speaks `CompactSearchServiceV1` | Current service still has a Python loop around recurrent calls; previous compact Torch rows were not a clear win | Not first |
| Patch LightZero CTree to accept fixed `A=3` arrays | Best chance to preserve LightZero semantics | Requires CTree/Cython/API changes; hard to make quickly and safely | Worth later |
| More direct CTree wrapper cleanup | Low risk | Ceilings say wrapper cleanup alone cannot give a big jump | Not enough |
| MCTX/JAX compact service | Existing benchmark already runs real compact visual roots through JAX/MCTX; removes CTree/list/control loop | Not LightZero semantics; JAX model ownership problem; profile-only | Best next prototype |

## Recommended Prototype

Build a profile-only `MctxCompactSearchServiceV1` or equivalent adapter.

Start from donor code in:

```text
src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
```

Reuse the parts that already exist:

- real `curvytron_hybrid_compact_visual_sample` roots;
- fixed-shape `[R,4,64,64]` root observations;
- MCTX `gumbel_muzero_policy`;
- legality checks;
- root-value extraction;
- `CompactSearchResultV1` validation;
- closed compact loop selected-action feedback;
- compact replay-index proof.

Wire it behind:

```text
src/curvyzero/training/compact_search_service.py
  CompactSearchServiceV1.run(root_batch) -> CompactSearchResultV1
```

Keep it profile-only with hard labels:

```text
profile_only=true
not_lightzero_ctree=true
not_train_muzero=true
touches_live_runs=false
trainer_defaults_changed=false
```

The first version can use the tiny JAX model already in the MCTX benchmark. That
is enough to test the backend shape. It is not enough to claim learning parity.

## First Shape

Use the same denominator as the current slab rows:

```text
H100
B512 and B1024
A16 actors
sim16 and sim32
root_noise_weight=0.0
materialize_scalar_timestep=false
compact_rollout_slab=true
```

Compare four rows:

```text
direct_ctree_gpu_latent
service_tax_probe
mock_search_service
mctx_compact_search_service
```

Compile time must be reported separately and excluded from warm timing. Shape
must be fixed or padded so JAX does not recompile during the measured loop.

## Validation Gates

P0 local contract gates:

- `CompactRootBatchV1` row/player/policy id order is preserved.
- Legal-mask polarity is correct: no illegal selected action and no illegal
  visit-policy mass.
- Active-root compaction is correct for non-prefix active roots.
- Terminal/autoreset rows keep final observations.
- Selected actions from search are applied to the next env step.
- Replay-index rows match root ids, selected actions, rewards, done flags, and
  final-observation masks.
- RND latest-frame attachment remains tied to the same compact root record if
  RND is enabled in that proof.

P1 profile gates:

- No live Coach run touched.
- `calls_train_muzero=false`.
- `promotion_eligible=false`.
- Aggregate slab timing is used, not last-call telemetry.
- Compile time, H2D, search, D2H action-only, D2H replay payload, env step, and
  replay-index timing are separated.
- Warm `mctx_compact_search_service` beats `direct_ctree_gpu_latent` on
  B512/sim16 and does not collapse on sim32.

P2 semantic gates before any future promotion:

- Decide whether a JAX-native model is acceptable, or build a real model bridge
  without host callbacks.
- Compare target rows and sampled learner batches against the stock path.
- Decide whether independent `A=3` per-seat roots are acceptable or whether a
  joint `A=9` control is needed.
- Prove RND cadence and player perspective in a matched full-loop smoke.

## Keep / Kill Rules

Keep the MCTX compact service lane if:

```text
warm measured slab steps/sec is at least about 1.5x over direct_ctree_gpu_latent
on B512/sim16, and sim32 still wins or is close enough to justify more work.
```

Promote it only as an optimizer probe, not as Coach advice.

Kill or demote it if:

- JAX recompiles during the measured loop;
- H2D or env step dominates so much that search speed does not matter;
- MCTX cannot emit stable root values and visit policies for replay proof;
- legality or identity checks fail;
- it needs PyTorch calls inside JAX search;
- the speedup is under about 1.2x on the same H100 denominator.

## Why This Might Be Worth It

This is the smallest move that attacks the actual wall:

```text
CPU/list CTree control and per-simulation model-output readback.
```

The MCTX benchmark already showed that a JAX-native compact visual closed loop
can run much faster than current direct CTree shapes, but it used toy model
semantics. Turning that into a `CompactSearchServiceV1` backend makes the
comparison fairer because it pays the same compact slab boundary and replay
proof edges.

If it wins, we learn that the next serious architecture should be compiled,
fixed-shape, and device-resident.

If it loses, we stop fantasizing about pure search kernels and move to the next
wall: env/observation/replay/RND ownership.

## Why This Might Not Be Worth It

It may be a sidecar forever.

MCTX wants pure JAX model code inside a JIT-compiled recurrent function. The
current trusted Coach lane is PyTorch/LightZero. Bridging PyTorch into JAX would
likely recreate the host sync we are trying to remove.

So the useful claim is narrow:

```text
MCTX can tell us the speed of the architecture we want.
It cannot by itself make the current LightZero trainer faster.
```

## Concrete Next Step

Do not start with a trainer rewrite.

Implement the smallest profile-only adapter:

```text
CompactRootBatchV1
-> padded/static JAX root tensors
-> jitted MCTX search with tiny JAX model
-> selected_action, visit_policy, root_value
-> validate_compact_search_result_v1(...)
```

Then run the H100 same-denominator slab grid above. If that row does not beat
`direct_ctree_gpu_latent`, the next backend should not be MCTX. If it does beat
it, the next design question is model ownership, not more wrapper optimization.
