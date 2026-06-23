# Fixed-Shape Compact Search Backend Designs, 2026-05-23

Role: parallel optimizer side-agent. Scope is architecture only. I read the
working docs and the current profile/service code, and did not edit source,
touch live Coach training runs, or touch Modal volumes.

## Starting Point

Current profile-only compact service rows:

```text
H100, B512/A16, sim16, 80 measured / 20 warmup:
  direct_ctree_gpu_latent:  5,965 steps/sec
  service_tax_probe:      11,855 steps/sec
  mock_search_service:    14,970 steps/sec
```

Interpretation:

```text
direct_ctree_gpu_latent is the real LightZero CTree control.
service_tax_probe pays real initial/recurrent model calls but avoids CPU CTree/list/control.
mock_search_service is a fake-search ceiling.
```

So the practical target is not renderer work. It is one real backend behind:

```text
CompactRootBatchV1
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

The service contract is intentionally narrow in
`src/curvyzero/training/compact_search_service.py`: `run(root_batch)` returns a
validated compact search result. That is the right place to hide a new backend
while keeping replay/RND/perspective validation outside the search engine.

## Recommendation Summary

| Rank | Backend | Expected Speedup Class vs direct_ctree_gpu_latent | Difficulty | Why This Order |
| ---: | --- | --- | --- | --- |
| 1 | Fixed-`A=3` CPU CTree SoA compatibility backend | `1.1-1.6x`; kill if `<10%` | Medium | Lowest semantic risk; gives a clean control and removes obvious list/object fanout without changing tree semantics. |
| 2 | Fixed-shape Torch device-tree backend | `1.5-2.5x`; possible `~2x` if it reaches service-tax class | Medium-high | Best near-term chance to actually remove the CTree/list wall while reusing the existing PyTorch model. |
| 3 | Fixed-shape JAX/MCTX sidecar backend | `2-5x` search-sidecar class if model/search stay JAX-resident; not trainer-facing initially | High | Cleanest all-device reference, but framework ownership makes it a scratch/backend-candidate lane, not the first trainer path. |

The first implementation should not choose a 5-month rewrite. Build the CPU SoA
compatibility backend as a semantic baseline, then push the Torch device-tree
backend until it either beats direct by `>=1.5x` with parity or clearly stalls.
Run MCTX as a separate scratch comparator only after the compact service parity
gates are already green.

## Design 1: Fixed-`A=3` CPU CTree SoA Compatibility Backend

### Shape

Keep the LightZero CTree algorithm as the semantic owner, but remove avoidable
dynamic surfaces around it:

```text
CompactRootBatchV1
-> flatten active roots once, fixed A=3
-> initial_inference on GPU
-> CPU CTree arrays/lists in a preallocated SoA facade
-> recurrent_inference batched on GPU per sim
-> compact arrays out
-> CompactSearchResultV1
```

This is not the final speed answer. It is a compatibility bridge that proves the
compact service can own search outputs without public collect dicts, root object
fanout, or per-mode output glue.

### Expected Speedup Class

`1.1-1.6x` over current `direct_ctree_gpu_latent`.

It cannot reasonably reach `service_tax_probe` because it still pays CPU CTree
traverse/backprop and CPU-driven simulation control. Its value is risk reduction
and denominator cleanup. Kill it as an optimization lane if a fair same-shape
row improves by less than `10%`.

### Implementation Difficulty

Medium.

Most of the pieces already exist in
`source_state_batched_observation_boundary_profile.py`: direct CTree compact
arrays, `direct_ctree_gpu_latent`, compact service validation, and compact
replay index rows. The work is mostly reshaping and ownership, not new
algorithm design.

### GPU Data

Keep on GPU:

```text
obs_tensor or resident root stack         [R,4,64,64]
legal_mask tensor if needed by model      [R,3]
initial latent_state                      model-shaped
root policy_logits/value before minimal readback
recurrent parent latent batch per sim
recurrent policy/value/reward output until the CTree update needs it
```

Read back only the minimum CTree needs:

```text
root priors/value                         [R,3], [R]
per-sim recurrent priors/value/reward     [R,3], [R], [R]
```

The existing `direct_ctree_gpu_latent` already keeps latent indices/device
latents in play; this design makes the surrounding compact arrays stricter and
removes remaining public output/list churn where possible.

### CPU Data

Keep on CPU:

```text
CompactRootBatchV1 identity sidecars:
  root_index, env_row, player, policy_env_id, active_root_mask

Fixed-action CTree state:
  visit counts/value sums/priors/rewards/children for A=3
  legal mask as bool [R,3]
  selected action/visit_policy/root_value compact outputs

CompactReplayIndexRowsV1:
  action, visit_policy, value, reward/done/final-observation indices
```

Avoid CPU ownership of full observation stacks after the root tensor is prepared.

### Sync Points

Required syncs:

```text
1. H2D for observation/legal mask unless resident input mode is reused.
2. Initial inference completion before root priors/value feed CTree.
3. Per-simulation recurrent output availability before CPU CTree backprop.
4. Final compact output assembly/readback.
```

The key improvement is not deleting all sync. It is making sync explicit and
bounded, with no extra public `collect_mode.forward` materialization and no
double-run adapter path.

### Replay/RND/Perspective Risk

Low-medium.

Algorithm semantics stay closest to the current direct CTree control, so the
main risks are attachment bugs:

```text
root k search output attached to transition k+1 incorrectly
player 0/player 1 sidecars swapped
policy_env_id assumed equal to compact row
terminal final observation index lost at replay materialization
RND latest-frame extraction reading a different frame than policy search saw
```

### Minimal Prototype Steps

1. Add a profile-only service class that consumes `CompactRootBatchV1` directly
   and owns exactly one direct CTree compact search call.
2. Preserve fixed `[R,3]` masks/priors/visits from input through result; only
   active roots may emit result rows.
3. Ensure `compact_search_result_v1_from_arrays()` is the only output path.
4. Run deterministic no-noise parity against current `direct_ctree_gpu_latent`
   for selected action, visit distribution, root value tolerance, and root ids.
5. Run the existing compact replay proof with non-prefix ids, non-identity
   `policy_env_id`, terminal/final observation, player sentinel, and RND
   latest-frame sentinel.
6. Profile same shape. Keep only if it is a measurable denominator cleanup or
   if it exposes direct CTree subphase facts needed by Design 2.

## Design 2: Fixed-Shape Torch Device-Tree Backend

### Shape

Replace CPU CTree with dense Torch tree tensors while keeping the current
PyTorch model:

```text
CompactRootBatchV1
-> fixed/padded R = B * P roots, A = 3
-> initial_inference
-> Torch PUCT tree tensors [R, N, A] and latent pool on device
-> recurrent_inference inside a fixed sim loop
-> final selected_action/visit_policy/root_value readback
-> CompactSearchResultV1
```

This is the most practical backend for the next real speed attempt because it
targets exactly what the service-tax row isolated: real model calls are not the
wall; CPU CTree/list/control is.

### Expected Speedup Class

`1.5-2.5x` over `direct_ctree_gpu_latent` if the tree update path is genuinely
device-resident and avoids per-simulation D2H/listification.

The near-term target is service-tax class:

```text
direct_ctree_gpu_latent:  5,965 steps/sec
service_tax_probe:      11,855 steps/sec
```

A robust Torch backend does not need to beat mock search immediately. It needs
to beat direct by `>=1.5x` with replay parity and explain any remaining gap to
mock.

### Implementation Difficulty

Medium-high.

There is already a `dense_torch_mcts` probe and compile spike in
`source_state_batched_observation_boundary_profile.py`. That gives a starting
point, but the current lane should be treated as a prototype, not a trusted
backend, until the semantic and fixed-shape gates are tightened.

### GPU Data

Keep on GPU:

```text
root observation tensor                   [R,4,64,64]
root legal mask                           [R,3] bool
root latent_state                         model latent shape
root priors/value                         [R,3], [R]

tree tensors:
  edge_child                              [R,N,3] int
  edge_visit                              [R,N,3] float/int
  edge_value_sum                          [R,N,3] float
  edge_reward                             [R,N,3] float
  edge_prior                              [R,N,3] float
  node_latent_slot                        [R,N] int
  latent_pool                             [N,R,...latent]
  min_value/max_value                     [R]
  path histories                          [N,R]

final outputs before readback:
  selected_action                         [R_active]
  visit_policy                            [R_active,3]
  root_value                              [R_active]
```

Use padded roots rather than active-root compaction for terminal/death-capable
profiles. Inactive roots get a safe dummy mask/action and are ignored in output.

### CPU Data

Keep on CPU:

```text
CompactRootBatchV1 identity fields and metadata
active_root_mask, env_row, player, policy_env_id
small final output arrays after readback
CompactReplayIndexRowsV1 sidecars
profile telemetry
```

CPU should not own tree state, recurrent outputs, per-node priors, or latent
movement.

### Sync Points

Target syncs:

```text
1. H2D root input unless resident tensor reuse is active.
2. Optional initial-inference CUDA event for timing, not required for control.
3. No per-simulation CPU sync.
4. One final sync/readback for action, visits, root value.
```

If Python exceptions or tensor shape fallbacks force a per-sim sync, the backend
has failed the point of the design. CUDA event telemetry should separate enqueue
time from actual drain time.

### Replay/RND/Perspective Risk

Medium-high.

This changes search mechanics, so it needs more than shape checks:

```text
PUCT score semantics can drift from LightZero.
reward + discount backup can be wrong under nonzero value_prefix.
root noise over partial legal masks can change effective noise weight.
tie-breaking can alter deterministic action parity.
inactive/padded roots can leak visit mass.
value transform/inverse scalar support can diverge from policy config.
```

Replay/RND risks are the same attachment risks as Design 1, plus an algorithmic
risk: the visit policy may be legal and finite but semantically different enough
to change training.

### Minimal Prototype Steps

1. Make a `CompactSearchServiceV1` adapter around `dense_torch_mcts` that
   returns only `CompactSearchResultV1`; no public output materialization.
2. Fix the backend to be fixed-shape over total roots, not active-root compacted
   roots, for the terminal/death validation row.
3. Keep root noise legal-mask-normalized and backup `reward + discount * value`;
   add fake-model tests with nonzero reward and partial masks.
4. Skip pre-dense `_masked_policy_arrays` work; dense needs logits, value, and
   latent, not a separate mock policy decode.
5. Add deterministic `root_noise_weight=0` parity against direct CTree on small
   fixtures:
   selected action, legal visit mass, root value tolerance, raw visit count
   shape, and tie-breaking policy.
6. Add stochastic/noise distribution checks separately; do not require exact
   action equality under root noise.
7. Run same-shape H100 B512/A16 sim16 profiles:

```text
direct_ctree_gpu_latent
service_tax_probe
dense_torch_device_tree_v1
mock_search_service
```

Kill if the backend cannot beat direct by `>=1.5x` after semantic fixes and
unused-decode removal. Keep if it lands between direct and service-tax with
clear remaining timing, because compile/fusion can then be justified.

## Design 3: Fixed-Shape JAX/MCTX Sidecar Backend

### Shape

Use MCTX as the clean all-device reference shape:

```text
CompactRootBatchV1
-> padded R = B * P roots, A = 3
-> JAX model representation/prediction
-> mctx.gumbel_muzero_policy or muzero_policy
-> JAX recurrent_fn inside search
-> action/action_weights/root values out
-> CompactSearchResultV1
```

This is not a drop-in backend for the current PyTorch LightZero model. It is a
scratch sidecar or future backend that proves whether a fully accelerator-owned
search body is fast enough to justify model/framework migration or distillation.

### Expected Speedup Class

`2-5x` search-sidecar class if all of these are true:

```text
model is JAX-native or tiny/distilled for the sidecar
tree and embeddings stay JAX-resident
R and sim count are fixed/padded
compile is excluded from steady rows
only compact action/weights/value are read back
```

If it bridges PyTorch model calls into JAX, expected speedup collapses and the
design should be stopped.

### Implementation Difficulty

High.

The repo already has MCTX benchmark precedent, and the docs identify the right
shape. The hard part is ownership: current trainer/model/search are PyTorch and
LightZero-shaped, while MCTX wants pure JAX functions.

### GPU Data

Keep on accelerator:

```text
obs_roots or compact feature roots        [R,4,64,64] or [R,H]
invalid_actions/legal mask                [R,3]
RootFnOutput:
  prior_logits                            [R,3]
  value                                   [R]
  embedding                               [R,H]

MCTX tree:
  node visits/values/raw values           [R,N]
  parents/action_from_parent              [R,N]
  children index/prior/value/visit        [R,N,3]
  children reward/discount                [R,N,3]
  embeddings                              [R,N,H]
  root invalid action masks               [R,3]

PolicyOutput:
  action                                  [R]
  action_weights                          [R,3]
  root value/search stats                 [R]
```

Prefer vector hidden state, e.g. `H=64`, for the first sidecar. Spatial
embeddings make `[R,N,...]` tree memory the likely bottleneck.

### CPU Data

Keep on CPU:

```text
CompactRootBatchV1 ids and active mask
JAX params/version metadata
small final output arrays
CompactReplayIndexRowsV1 validation/materialization sidecars
```

Do not keep CPU tree state. Do not call a PyTorch model from inside the JAX
search loop.

### Sync Points

Target syncs:

```text
1. H2D/JAX device_put for fixed roots unless roots are already JAX-resident.
2. JIT compile/warmup outside measured rows.
3. No sync inside MCTX simulations.
4. One D2H for action/action_weights/root_value.
```

Unexpected recompilation is a hard failure. Shape changes from live-root counts
must be handled with padding and masks, not new JIT signatures.

### Replay/RND/Perspective Risk

High.

This backend can be fast and still not be semantically trainer-compatible:

```text
JAX model may not match PyTorch policy logits/value/reward.
Gumbel MuZero action weights may not match LightZero visit-count semantics.
independent per-seat A=3 dynamics may mishandle simultaneous two-player action effects.
value/reward transforms can diverge.
policy versioning becomes a new concern if distilled/sidecar params are used.
RND reward shaping is outside the search engine and must attach later by root id.
player perspective must be proven with asymmetric fixtures.
```

Treat MCTX output as search-candidate data until compact replay parity and a
small behavioral gate pass.

### Minimal Prototype Steps

1. Create a profile-only sidecar, not a trainer path:

```text
CompactRootBatchV1
-> tiny JAX CNN/vector model
-> MCTX fixed R/A/sim search
-> CompactSearchResultV1
```

2. Start with synthetic or random params and validate only shape/legal/finite
   output. This measures framework/search feasibility, not training quality.
3. Add a deterministic tiny-model fixture where expected priors/values are
   controlled, including partial legal masks and inactive/padded roots.
4. Profile warmed steady rows at small then relevant shapes:

```text
R=128, sims=16, H=64
R=1024, sims=16, H=64  # if memory allows and this matches B512/P2
```

5. Compare only the search bucket first, then wire compact replay proof.
6. Stop immediately if it recompiles, spills memory, or requires PyTorch host
   callbacks in `recurrent_fn`.

## Shared Validation Gates

Any backend behind `CompactSearchServiceV1` must pass the same gates before it
is more than a profile row.

P0 search/result gates:

```text
binary legal masks only
selected_action legal for every emitted root
visit_policy finite, nonnegative, sums to 1
zero visit mass on illegal actions
root_index/env_row/player/policy_env_id round-trip to CompactRootBatchV1
no output for inactive roots
deterministic no-noise mode is stable
```

P0 compact replay gates:

```text
search consumes observation at record k
selected_action[k] equals the joint action used to produce record k+1
reward/done/final reward come from record k+1
terminal next observation uses final_observation before autoreset
non-prefix active roots work
non-identity policy_env_id works
player-perspective sentinel cannot be swapped
RND latest frame equals the policy-visible latest frame
```

P0 profiling gates:

```text
one search call per measured step, no adapter double-run
same B/A/sim/warmup/measured denominator as direct/service-tax/mock rows
compile excluded or reported separately
CUDA sync policy stated
compact replay proof enabled or explicitly marked absent
```

## Concrete Next Architecture

Build the next backend ladder like this:

```text
Step 1: direct_ctree_fixed_a3_soa_service
  Purpose: semantic control and compact denominator cleanup.
  Promotion: replay/RND/perspective gates pass; same-shape speed is not worse.

Step 2: torch_device_tree_fixed_shape_service
  Purpose: real attempt to remove the CPU CTree/list wall.
  Promotion: >=1.5x over direct_ctree_gpu_latent, legal/replay gates pass,
             no per-simulation D2H sync.

Step 3: mctx_jax_fixed_shape_sidecar
  Purpose: all-device reference and possible future backend.
  Promotion: warmed fixed-shape search beats direct search bucket by >=2x,
             no recompiles/host callbacks, compact service result validates.
```

The near-term "real" architecture should be Step 2, but Step 1 should exist
first as the compatibility rail. Step 3 is worth running as a sharp falsifier,
not as a replacement for fixing the current compact service path.

Short version:

```text
Use CompactSearchServiceV1 as the stable boundary.
Keep LightZero CTree only as the compatibility/control backend.
Prototype Torch device-tree search as the practical speed backend.
Use MCTX/JAX to test the all-device ceiling, not to sneak in a trainer rewrite.
```
