# Compact Slab Architecture Critique, 2026-05-23d

Scope: read-only architecture critique plus this note. I did not modify source
code, live Coach runs, checkpoints, evals, tournaments, GIFs, or Modal volumes.

## Short Answer

`CompactRolloutSlab` is a good boundary. It proves a cleaner loop:

```text
compact env batch
-> compact root batch
-> compact search service
-> selected joint action
-> next env step
-> compact replay-index rows
```

But it is still mostly a profile-only shell around the same expensive pieces.
It is not yet the real speed win. The speed win only becomes real if the backend
behind `CompactSearchServiceV1` stops doing high-frequency CPU/list/host sync
work and if replay/RND/sample materialization stops dragging the compact path
back into full Python rows.

## What Still Lives On CPU

These pieces are still CPU-owned or Python-owned in the current slab path:

- CurvyTron env stepping. The next action comes back to CPU, then the CPU env
  advances.
- Compact batch construction. `HybridBatchedObservationProfileManager.step()`
  still builds `HybridCompactBatch` from NumPy/Python-side payloads.
- Root-batch sidecars. `CompactRootBatchV1` is built in Python from the compact
  batch.
- Direct LightZero CTree search. `direct_ctree_arrays` and
  `direct_ctree_gpu_latent` still use LightZero CTree traversal/backprop and
  list/array APIs. GPU latent mode keeps more model state on the GPU, but it
  does not make the tree a fully device-resident search.
- Replay-index commit. `CompactRolloutSlab._commit_previous()` builds
  `CompactReplayIndexRowsV1` on the CPU side after the next batch arrives.
- Final trainer objects. Stock LightZero replay rows, target rows, learner
  sample tensors, RND buffers, checkpoint/eval objects, and tournament-facing
  policy setup are not part of this slab speed denominator.

Plain read: the slab removes some old scalar timestep work from this profile
shape, but it does not yet remove the real LightZero search/control boundary.

## What Syncs Remain

The unavoidable sync, while the env remains CPU, is:

```text
selected actions [B, 2] -> CPU env
```

That sync is small and probably acceptable.

The suspicious syncs are:

- fresh observation stack sent from host to GPU each search call;
- model output/root data read back for LightZero root setup;
- recurrent output or search payload crossing the CPU/GPU boundary inside MCTS;
- CTree traversal/backprop using CPU/list-shaped control;
- search result arrays converted back to compact replay fields;
- replay/RND/sample payloads materialized in Python instead of staying as
  compact ids and slabs until they are truly needed.

The current telemetry has one trap: the new flattened slab fields in the runner
look like last-call service telemetry, not an aggregate over all measured slab
calls. The aggregate wall-clock field to trust for the slab row is still the
measured loop time plus aggregate `compact_rollout_slab_sec`, not only
`probe_total_sec` from the last search call.

## Current Slab Evidence

The slab now runs from the hybrid Modal profile tooling and commits index rows.
That is a real plumbing win.

The current warm H100 slab rows are still small-denominator probes:

```text
B64, actor_count=8, sim8, 20 measured steps, 8 warmup, profile-only

direct_ctree_arrays:
  committed rows: 2560
  steps/sec: about 1560-2251 across the two recent summaries

direct_ctree_gpu_latent:
  committed rows: 2560
  steps/sec: about 2204-3680 across the two recent summaries
```

The range moved because the second run added telemetry and the summaries mix
aggregate loop time with last-call search telemetry. Do not use these rows as a
Coach-speed claim. Use them as proof that the slab can run a real direct CTree
search service, feed selected actions back into the next env step, and commit
compact replay-index rows.

## Is This A Profile-Only Trap?

It becomes a trap if we stop here.

The slab by itself mostly rearranges the boundary. It does not automatically
make search, replay, RND, or learner dataflow fast. A profile row can look clean
while still excluding:

- stock `train_muzero`;
- real replay buffer sampling;
- target building;
- learner update;
- RND training/estimation cadence;
- checkpoint/eval/tournament policy setup;
- subprocess env-manager costs.

The slab is valuable only if it becomes the narrow contract for a stronger
backend and for validation gates.

## Realistic Next Big Backend

The next backend needs to change ownership, not just wrap the existing call.
The target should be:

```text
CPU batched env
-> stable compact ids and sidecars
-> resident or chunked observation stack
-> device/native search backend
-> only selected actions return on the hot path
-> replay/RND payload flushes later, before sample visibility
```

A realistic backend has to provide:

- fixed-shape or padded root tensors, probably fixed `A=3`;
- preallocated search/tree arrays;
- no per-simulation Python list conversion;
- no per-simulation recurrent output readback;
- selected-action-only hot sync;
- delayed visit-policy/root-value payload flush;
- stable handles so out-of-order payload completion cannot attach to the wrong
  env row/player/perspective;
- exact gates for legal masks, terminal final observations, RND latest frame,
  replay-index materialization, and learner-visible sample rows.

Best candidates:

1. array-native/fixed-`A=3` CTree if we can preserve LightZero semantics while
   killing list/object overhead;
2. fixed-shape compiled Torch/Triton/CUDA search behind `CompactSearchServiceV1`;
3. MCTX/JAX as a side comparator for the all-device ceiling, not as an immediate
   Coach path unless model/replay move with it.

The current eager compact Torch service should not be polished as the main lane.
The docs already show it loses to direct CTree on the latest H100 sim16/sim32
same-denominator rows.

## Likely Amdahl Bottleneck After Current Slab

If the slab keeps using direct LightZero CTree, the bottleneck is still the
search/control boundary:

```text
GPU model work
-> CPU/list CTree control
-> recurrent/model payload crossings
-> compact result materialization
```

If a better search backend lands and becomes much faster, the wall probably
moves to:

- CPU env step plus observation/stack handoff;
- compact-to-public packaging;
- replay/RND/sample materialization;
- learner-edge full observation tensors.

That is why the slab needs both search timing and materialization timing. If the
search backend gets fast, the next wall is not mystery rendering. It is the
remaining CPU env/observation/replay/RND ownership.

## Recommended Next Falsifier

Run one honest same-denominator profile-only grid:

```text
H100
B512 or B1024
sim16 and sim32
root_noise_weight=0
materialize_scalar_timestep=false
compact_rollout_slab=true

rows:
  direct_ctree_gpu_latent slab
  service-tax slab, if supported
  mock slab, if supported
  next real backend candidate slab
```

Required summary fields:

- aggregate measured seconds;
- aggregate `compact_rollout_slab_sec`;
- aggregate model/search/H2D/D2H timing, not just last-call timing;
- committed index-row count;
- selected-action feedback verified;
- `python_rows_materialized`;
- replay/RND materialized rows;
- promotion lock saying this is profile-only.

Kill the slab as a speed claim if it cannot beat direct CTree on the same
denominator after paying the selected-action and compact replay-index edge.
Keep the slab as a validation boundary even if the current backend loses.

## Bottom Line

The slab is the right shape. The current backend is not enough.

The next real speed attempt should be a stronger search/dataflow backend behind
the slab contract, with one selected-action sync and delayed replay payloads.
Until that passes same-denominator speed plus replay/RND/sample gates, this is
optimizer evidence only, not Coach launch advice.
