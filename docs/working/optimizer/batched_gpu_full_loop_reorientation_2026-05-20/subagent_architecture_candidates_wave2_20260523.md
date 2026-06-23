# CurvyTron Collect/Search/Replay Architecture Candidates, Wave 2

Date: 2026-05-23

Role: parallel optimizer research/critique sidecar. Scope is architecture and
falsification only. I did not touch source, live Coach training runs,
checkpoints, evals, tournaments, GIFs, or Modal volumes.

## Read First

The current useful target is still:

```text
CPU batched CurvyTron env rows
-> compact row/player sidecars
-> resident or compact visual stack
-> batched device-resident search/model tensors
-> CPU receives only selected actions on the action-critical path
-> replay/RND payloads flush later, before sample visibility
```

The repo already has the right contract vocabulary:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> materialized target rows only at validation/sample edges
```

Important local anchors:

- `direct_ctree_gpu_latent` is the practical real LightZero CTree comparator.
- `service_tax_probe` and `mock_search_service` are ceilings/falsifiers, not
  training algorithms.
- `compact_torch_search_service` proves the common compact service boundary,
  but its eager fixed-shape body is still profile-only and not trainer-ready.
- Native actor buffers and resident stacks have real profile signals, but the
  closed-loop wall moves between observation handoff and search depending on
  sim count and input mode.
- Compact replay index rows are the right hot-path shape; full materialized
  replay chunks are validation/sample edges.

Speed ranges below are approximate and are meant as same-denominator optimizer
profile ranges versus the current compact/direct LightZero-shaped boundary,
unless a candidate says otherwise. Coach-training speedup will be lower until
the trainer/replay/RND path is actually integrated and validated.

## Common Sync Budget

Acceptable per env tick while mechanics stay CPU-owned:

```text
selected_action[B,P]        # required to step the next CPU env state
small sidecar summaries     # ids, masks, reward/done/final flags as needed
```

Avoid on the action-critical path:

```text
visit_policy[B,P,3]
raw_visit_counts[B,P,3]
root_value[B,P]
predicted logits/value
RND latest-frame payloads
full observation stacks
stock GameSegment / BaseEnvTimestep objects
```

Keep device-resident or compact when possible:

```text
observation stack [B,P,4,64,64], preferably uint8 until model input prep
root logits/value/hidden state
tree tensors: visits, priors, values, rewards, child links, masks
recurrent outputs inside the search loop
visit policy/root value until chunk flush
RND latest-frame ring if RND is enabled
compact replay indices and observation handles
```

The core rule:

```text
One selected-action CPU sync is acceptable.
Per-simulation reward/value/policy readback for tree backprop is the bad sync.
Replay/RND payload readback should be chunked or delayed, not action-gating.
```

## Candidate Matrix

| # | Candidate | Expected speedup | Main bet |
| ---: | --- | ---: | --- |
| 1 | Conservative compact CTree control loop | `1.1-1.5x` vs current stock-shaped loop; small vs direct compact | Make the compact contract real without changing CTree semantics. |
| 2 | Two-phase action-now, payload-later service | `1.1-1.6x` alone; enables larger wins | Read only actions before env step; flush visits/value/RND later. |
| 3 | Fixed-shape Torch device tree | `1.5-2.5x`; maybe `~3x` search-only | Keep PyTorch model, move tree state/control to device tensors. |
| 4 | Fixed-`A=3` array-native CTree SoA | CPU `1.1-1.6x`; CUDA `2-4x` if promoted | Preserve CTree semantics while killing list/root object ABI. |
| 5 | MCTX/JAX resident visual sidecar | `2-5x` search-sidecar; `1.5-3x` closed-loop if input stays resident | Use accelerator-native search as the clean comparator. |
| 6 | Puffer/EnvPool-style static actor slab | `2-5x`; platform for `10x` with search/replay | CPU env rows write contiguous buffers, scalar objects are edge adapters. |
| 7 | Sample Factory-style batched search service | `2-6x` if GPU/search underfill is real | Many producers fill one ordered batched search/model service. |
| 8 | Delta renderer plus resident observation owner | `1.2-1.8x`; not a 10x lane alone | Stop bouncing frames through host stacks for search input. |
| 9 | Compact replay/RND tensor store | `1.1-1.6x` collect; `1.5-2.5x` if replay/learner objects are hot | Replay owns indices/handles now, materializes tensors later. |
| 10 | Single-framework accelerator loop | `5-10x+` endpoint; highest rewrite risk | Env/render/search/replay stay in one compiled/device runtime. |

## 1. Conservative Compact CTree Control Loop

Architecture:

```text
CPU VectorMultiplayerEnv / HybridCompactBatch
-> CompactRootBatchV1(copy_observation=False)
-> direct_ctree_gpu_latent wrapped as CompactSearchServiceV1
-> CompactSearchResultV1
-> selected actions drive next joint_action[B,P]
-> CompactReplayIndexRowsV1
-> materialized target rows only for validation/sample checks
```

Expected speedup range:

- `1.1-1.5x` versus stock-shaped collect/search/replay profiles when scalar
  public output and full replay materialization are removed.
- Only `1.0-1.2x` versus the current best direct compact comparator if the
  CTree CPU/list boundary remains the wall.

What must sync to CPU:

- Selected actions before the next CPU env step.
- CTree root priors/value and per-simulation recurrent reward/value/policy,
  because this candidate deliberately keeps CPU CTree semantics.
- Compact replay index rows and validation sidecars.

What stays device/compact:

- Root observation can be a no-copy compact view until tensor prep.
- Latent state/pool stay on GPU as in `direct_ctree_gpu_latent`.
- Public LightZero objects stay out of the measured hot path except at
  validation/materialization edges.

Key correctness risks:

- Search output does not actually drive the next env action.
- `root_index`, `env_row`, `player`, or `policy_env_id` drift across search and
  replay.
- Terminal final observations are replaced by autoreset observations.
- RND latest-frame extraction reads the wrong policy frame.

Smallest falsifying experiment:

```text
Closed compact loop, B512/P2/sim16, root_noise=0:
direct_ctree_gpu_latent through CompactSearchServiceV1,
service-selected actions drive next joint_action,
CompactReplayIndexRowsV1 materializes to trusted target rows.
```

Kill as an architecture lane if it cannot beat the current direct compact row
by at least `10%` while passing the closed-loop action-feedback and replay
identity gates. Keep it as the semantic control even if it is not the speed
answer.

## 2. Two-Phase Action-Now, Payload-Later Service

Architecture:

```text
search_service.step(root_batch_or_handle)
  -> selected_action_cpu[B,P]
  -> replay_payload_handle

search_service.flush_replay_payload(handle)
  -> visit_policy
  -> raw_visit_counts
  -> root_value
  -> predicted logits/value
  -> RND latest-frame payload if enabled
```

This is not a new search algorithm. It is a sync policy that every serious
search backend should support.

Expected speedup range:

- `1.1-1.6x` if visit policy/root value/RND payload readback is currently
  action-gating.
- Near zero if the final action readback already waits for the whole search and
  replay payload readback is tiny or fully overlapped.
- Enables larger `2-5x` lanes by preventing replay/RND from reintroducing a
  full-payload per-step fence.

What must sync to CPU:

- Selected joint actions only, before the CPU env step.
- A small handle/sequence number so replay rows can attach later payloads.
- Optional failure counters that gate promotion.

What stays device/compact:

- Visit policy, raw visit counts, root value, logits, and predicted value until
  chunk flush.
- RND latest-frame ring and bonus payload until the replay/RND cadence needs
  them.
- Compact observation references, not copied observation tensors.

Key correctness risks:

- Replay row `k` receives payload from root `k+1`.
- Flush happens after the learner samples the row.
- Terminal rows lose final observation or RND terminal frame before payload
  materialization.
- Metrics summarize a row before illegal-action/fallback checks arrive.

Smallest falsifying experiment:

```text
Fake delayed service:
return selected_action immediately,
store visit_policy/root_value in a delayed map keyed by root ids,
step real compact env once,
flush before materializing CompactReplayIndexRowsV1,
compare immediate vs delayed target rows for non-prefix ids, terminal rows,
player sentinels, and RND latest-frame sentinels.
```

Kill if delayed rows differ from immediate rows or if action-critical wall time
does not improve in a same-denominator profile with replay payloads enabled.

## 3. Fixed-Shape Torch Device Tree

Architecture:

```text
CompactRootBatchV1 or resident root tensor
-> PyTorch initial_inference
-> fixed padded roots R = B*P, A = 3
-> Torch tree tensors on device
-> recurrent_inference inside fixed sim loop
-> read back selected_action only for env step
-> flush visits/value/counts later
```

Expected speedup range:

- `1.5-2.5x` over `direct_ctree_gpu_latent` if the current wall is
  CTree/list/Python control and per-sim D2H.
- Search-only can approach the `service_tax_probe` class if tree update is
  genuinely device-resident.
- Full-loop speed will be lower if observation/replay/RND materialization is
  still hot.

What must sync to CPU:

- Selected actions once per env tick.
- Small metadata: illegal count, fallback count, active-root count, backend id,
  seed/noise/temperature.
- Delayed replay payload before the sampler can observe those rows.

What stays device/compact:

- Root observation tensor or resident stack.
- Root logits/value/hidden state.
- Tree arrays: child links, visit counts, value sums, priors, rewards,
  min/max stats, masks, path histories, latent slots.
- Recurrent outputs inside the sim loop.
- Visit policies/root values until flush.

Key correctness risks:

- PUCT/backprop math drifts from LightZero CTree in ways that change targets.
- Mask handling and single-legal rows break under padded roots.
- Root noise, tie-breaking, temperature, and epsilon semantics differ.
- Inactive or terminal roots leak dummy values into replay.
- Torch eager control is still too slow unless compiled/fused carefully.

Smallest falsifying experiment:

```text
No-noise deterministic fixture suite:
single-legal, biased-logit, mixed-mask, non-prefix active roots,
terminal/inactive roots, player swap sentinel.

Then H100 same-denominator B512/P2/sim16 and sim32:
direct_ctree_gpu_latent vs compact_torch_search_service,
action-feedback proof on,
CompactReplayIndexRowsV1 proof on,
selected-action-only critical sync if available.
```

Kill if sim16 cannot beat direct by `>=1.5x` with zero illegal/fallback rows, or
if no-noise fixtures cannot explain selected-action and visit-policy deltas.

## 4. Fixed-`A=3` Array-Native CTree SoA

Architecture:

```text
CompactRootBatchV1
-> initial_inference on GPU
-> root logits/value copied once to flat arrays
-> fixed-A3 SoA CTree:
     priors[R,3], visits[R,N,3], value_sum[R,N,3],
     reward[R,N,3], child[R,N,3], legal_mask[R,3]
-> recurrent inference still batched on GPU
-> flat selected_action/visit_policy/root_value arrays out
```

Stage it as CPU SoA first, then consider CUDA/Triton only if the CPU ABI proves
the list/root object boundary is the wall.

Expected speedup range:

- CPU SoA compatibility backend: `1.1-1.6x`.
- GPU/CUDA SoA backend: `2-4x` if it removes per-simulation CPU backprop and
  listification without semantic drift.

What must sync to CPU:

- CPU SoA version: root priors/value and recurrent reward/value/policy per sim.
- CUDA version: selected actions and delayed replay payload only.
- In both versions, compact identity sidecars stay CPU-visible for env/replay.

What stays device/compact:

- GPU model tensors and latent state.
- Fixed `[R,3]` masks/priors/visits instead of Python legal-action lists.
- Replay index rows, not full GameSegments in the collect hot path.

Key correctness risks:

- It looks safe because it is "same CTree," but fixed-A3 indexing can swap
  action meaning or legal mass.
- CPU SoA might only move overhead around and still sync every simulation.
- CUDA SoA becomes a second MCTS implementation with all the Torch-device-tree
  semantic risks.

Smallest falsifying experiment:

```text
No-model CTree ABI microbench:
feed synthetic reward/value/policy[R,3] into current CTree list API
and a flat SoA update path at R=1024, sims=16/32.

Then one full compact boundary row:
CPU SoA service vs direct_ctree_gpu_latent,
same model, same masks, no root noise.
```

Kill if the no-model ABI gap is not large, or if the full compact row improves
by less than `10%`. Promote only as a semantic bridge unless it beats the
direct compact comparator materially.

## 5. MCTX/JAX Resident Visual Sidecar

Architecture:

```text
CPU CurvyTron env rows
-> persistent JAX renderer last_output_device[B,2,1,64,64]
-> resident JAX FIFO stack[B,2,4,64,64]
-> JAX/MCTX root arrays and recurrent function
-> device tree/search arrays
-> selected_action D2H
-> delayed action_weights/root_value payload
```

Expected speedup range:

- `2-5x` for the search boundary when model/search stay JAX-resident.
- `1.5-3x` for a closed compact visual loop if resident input and compact
  replay stay in the same denominator.
- Much less if a PyTorch-to-JAX bridge or host stack rebuild reintroduces the
  old boundary.

What must sync to CPU:

- Selected actions for CPU env stepping.
- CPU sidecars for row/player ids, legal masks if masks remain CPU-owned,
  reward/done/final flags.
- Delayed replay payload before sampler visibility.

What stays device/compact:

- Latest frame, FIFO stack, root observations, model hidden state, MCTX tree,
  action weights, values.
- Host observation mirror only on sampled validation cadence and terminal rows.

Key correctness risks:

- MCTX semantics are not LightZero CTree semantics by default.
- JAX toy model/search results are not trainer-facing evidence.
- Resident stack order can silently swap `[env_row, player]`.
- Terminal/autoreset frames can come from reset state unless captured before
  autoreset.
- A framework bridge can erase the benefit while adding complexity.

Smallest falsifying experiment:

```text
B1024/P2/sim16 and sim32 closed compact visual profile:
host stack MCTX input vs resident JAX stack input,
native actor buffer on,
root no-copy on,
replay-index proof on,
sampled resident-vs-host stack parity,
selected-action feedback checksum.
```

Kill if resident input does not improve closed-loop throughput by `>=1.2x`, or
if stack/final-observation parity fails. Keep as a sidecar comparator until a
real trainer-compatible model/replay path exists.

## 6. Puffer/EnvPool-Style Static Actor Slab

Architecture:

```text
preallocated CPU slabs:
  action[B,P]
  reward[B,P]
  done[B]
  legal_mask[B,P,3]
  env_row/player/policy_env_id
  terminal/autoreset/final-observation sidecars
  render deltas or compact render state

CPU actor shards step_many(action[B,P]) into slabs
-> renderer/search consume slab handles
-> selected actions are written back into the action slab
-> replay stores indices/handles
```

This steals the useful PufferLib/EnvPool idea: contiguous ownership and batched
CPU env rows. It does not imply replacing MuZero with a Puffer trainer.

Expected speedup range:

- `2-5x` when scalar object materialization, actor payload/merge, and stack
  handoff are in the denominator.
- Platform for `10x` only when paired with device-resident search and compact
  replay/RND.
- Local zero-observation native actor-buffer probes already showed meaningful
  wins, but real visual/search rows must decide the actual range.

What must sync to CPU:

- Selected actions before each CPU `step_many`.
- Reward/done/legal/final sidecars are CPU-owned truth while mechanics remain
  CPU.
- Terminal final observation snapshot if the final frame cannot stay as a
  device handle.

What stays device/compact:

- Renderer framebuffer/stack if using persistent GPU renderer.
- Search tree/model tensors.
- Replay payload handles, not stock timestep objects.
- RND latest-frame handles/ring where possible.

Key correctness risks:

- Memory aliasing: replay rows reference slabs that later mutate.
- Actor shards write out of order or overwrite terminal/final rows.
- Simultaneous two-player joint action commit is broken by per-player queues.
- Borrowed render state is mutated by autoreset before final observation is
  captured.
- Debug adapters recreate scalar object fanout in the hot path.

Smallest falsifying experiment:

```text
In-process static-slab profile, no multiprocessing first:
B512 and B1024,
real GPU observation backend or resident stack,
mock_search_service and direct_ctree_gpu_latent comparators,
no scalar BaseEnvTimestep materialization in the hot path,
CompactReplayIndexRowsV1 proof on.
```

Kill if flat-slab collection plus mock search is not at least `1.5x` faster
than the current LightZero-shaped sidecar on the same visual denominator, or if
materialized target rows do not match the trusted compact/object oracle.

## 7. Sample Factory-Style Batched Search Service

Architecture:

```text
many CPU actor batches / slabs
-> ready-root queue with compact handles
-> central search/model service
   batches roots or leaves across producers
   owns model version and search config
   returns ordered selected actions and replay payload handles
-> actors continue after selected-action return
-> replay writer flushes payloads in chunks
```

This is the async/service topology version of the compact contract. It is most
useful if a single synchronous B512/B1024 loop underfills the GPU or spends too
much time alternating CPU env and GPU search.

Expected speedup range:

- `2-6x` if GPU/search underfill, Python scheduling, or per-batch setup are the
  wall.
- Near zero or negative if one large synchronous batch already fills the GPU
  and service latency/staleness dominates.

What must sync to CPU:

- Selected actions per actor batch, ordered by `(producer_id, sequence_id,
  env_row, player)`.
- Policy/model version and search config ids.
- Backpressure/failure metadata.

What stays device/compact:

- Model weights and search tree/workspace inside the service.
- Batched root tensors, recurrent leaf batches, and delayed replay payloads.
- Replay chunks as compact indices/handles until sampler materialization.

Key correctness risks:

- Result ordering bugs attach actions to the wrong producer/env/player.
- Policy staleness changes learning semantics without being recorded.
- Queue latency reduces sample efficiency or deadlocks terminal rows.
- Replay flush races with learner sampling.
- Determinism becomes much harder to reason about.

Smallest falsifying experiment:

```text
Local in-process service mock:
N producer loops submit compact fake or real root handles,
one service batches a real initial_inference plus mock/service-tax search,
returns ordered actions,
CompactReplayIndexRowsV1 verifies action/replay identity.

Sweep N=1,2,4,8 producer batches and report batch fill latency,
GPU model/search utilization proxy, actions/sec, and identity failures.
```

Kill if N>1 does not improve throughput after warmup, if batch fill latency
dominates env waiting, or if any ordered-action/replay identity gate fails.

## 8. Delta Renderer Plus Resident Observation Owner

Architecture:

```text
CPU env mechanics
-> actor emits compact visual deltas/reset masks/compose state
-> persistent GPU renderer updates framebuffer
-> resident device FIFO stack is authoritative for search input
-> host stack is sampled for parity and mandatory terminal rows only
-> search consumes device stack directly
```

Expected speedup range:

- `1.2-1.8x` in visual closed-loop profiles if observation handoff is hot.
- Smaller once search dominates at higher sim counts.
- Not a `10x` lane alone, but it removes the largest avoidable observation
  bounce before the search-service lanes are judged.

What must sync to CPU:

- Selected actions.
- CPU reward/done/legal/id sidecars.
- Terminal/final observation proof, either as a device handle plus checksum or
  a terminal-only D2H snapshot.

What stays device/compact:

- Persistent framebuffer/layer.
- Latest policy frame and FIFO stack.
- Search input root tensor.
- RND latest-frame ring if RND is enabled.

Key correctness risks:

- Trail delta wraparound/cursor bugs.
- Reset/autoreset rows mutate before final frame capture.
- Avatar color/player perspective drift.
- Host mirror sampled cadence misses rare terminal bugs.
- Observation timing buckets look faster only because waits moved into search.

Smallest falsifying experiment:

```text
B1024/P2/sim16 borrowed/native actor-buffer row:
current host-stack path vs resident/delta path,
obs_h2d_bytes must drop to zero for search input,
renderer_device_to_host_sec zero on non-terminal hot steps,
sampled host/device stack parity,
mandatory terminal/final-observation parity.
```

Kill if total roots/sec improves by less than `1.2x`, if wait time merely moves
into `search_sec` with no wall improvement, or if any terminal/reset parity
check fails.

## 9. Compact Replay/RND Tensor Store

Architecture:

```text
collect writes:
  CompactReplayIndexRowsV1
  observation handles or compact stack refs
  selected_action
  visit_policy/root_value payload handle
  reward/done/final flags
  RND latest-frame handle/bonus payload

sampler later materializes:
  learner tensors
  target rows for validation
  stock LightZero-compatible objects only if needed
```

Expected speedup range:

- `1.1-1.6x` in collection if replay/RND object materialization is on the hot
  path.
- `1.5-2.5x` in broader collect+sample profiles if stock GameSegment/target
  row construction becomes the next denominator after search improves.
- Prerequisite for larger service/device lanes because otherwise the win moves
  from search into replay.

What must sync to CPU:

- Selected actions for env step.
- Chunk ledger: record indices, root ids, policy ids, model/search version,
  reward/done/final masks.
- RND scalar metrics at coarse cadence, not every action step.

What stays device/compact:

- Visit policies/root values/counts until chunk flush.
- Observation stacks as refs/handles where possible.
- RND latest frames and predictor/target inputs in a tensor ring.
- Learner-ready tensors can be materialized directly to device.

Key correctness risks:

- Off-by-one replay rows: observation[k] paired with action[k+1].
- Final observation row lost before materialization.
- RND meter mode accidentally changes target rewards.
- RND reward mode changes target rewards but is mislabeled as meter mode.
- Sampler sees a row before delayed payload is flushed.
- Replay age/model-version metadata is missing.

Smallest falsifying experiment:

```text
Dry profile with learner disabled:
immediate compact rows vs deferred tensor-store rows
for non-prefix ids, mixed live/terminal rows, normal death, no-death,
RND meter zero-weight, player perspective sentinels.

Then sample K rows and materialize trainer-facing tensors/target rows;
compare digests against existing target-row builders.
```

Kill if materialized samples differ from the trusted oracle, if RND meter
changes target rewards, or if replay store overhead is not visible in a matched
ablation.

## 10. Single-Framework Accelerator Loop

Architecture:

```text
JAX or Torch owns:
  env state
  render/observation stack
  model/search tree
  replay/RND tensor ring

CPU owns only:
  coarse orchestration
  checkpoints/metrics
  validation sample mirrors
```

This is the Brax/MCTX endpoint, not the next safe patch. It becomes attractive
only after the compact contracts prove which pieces matter and after a smaller
device-resident search path works.

Expected speedup range:

- `5-10x+` is plausible as an endpoint because it removes CPU env/action sync,
  host stack bouncing, LightZero CTree CPU/list APIs, and replay object fanout.
- Near zero practical value if a small faithful env cannot be expressed without
  semantic drift, or if the trainer bridge materializes everything back to
  stock objects each step.

What must sync to CPU:

- Coarse metrics, checkpoints, sample manifests, validation mirrors.
- No per-step sync in the ideal version.
- Optional sampled action/replay checks for parity, outside the hot path.

What stays device/compact:

- Entire env state, random state, observation stack, search tree, model hidden
  state, replay/RND tensors, sampler batches.

Key correctness risks:

- This is a second CurvyTron implementation.
- Source-fidelity edge cases are broad: trails, gaps, collisions, bonuses,
  death, final observation, autoreset, action repeat, masks, stochasticity.
- JAX/Torch search may train different targets from LightZero CTree.
- Validation becomes hard because host mirrors are sampled, not primary.

Smallest falsifying experiment:

```text
No-bonus/no-death micro-CurvyTron accelerator loop:
B1024, fixed A=3, fixed sim count, 64x64 stack,
one-step env transition + render + toy search + replay tensor write.

Compare against CPU vector oracle for movement, masks, rewards, observations,
and replay row identities over seeded short rollouts.
```

Kill or postpone if even the no-bonus/no-death subset cannot maintain parity,
or if bridging the device loop back to existing replay/target rows erases the
speed advantage.

## Ranking And Recommendation

Near-term robust path:

```text
1. Candidate 1 as semantic control.
2. Candidate 2 as the service sync policy.
3. Candidate 3 or 4 as the first real search replacement.
4. Candidate 8 if the same-denominator row says observation handoff is hot.
5. Candidate 9 before any trainer-facing claim.
```

Parallel architecture probe:

```text
Candidate 6, Puffer/EnvPool-style static actor slab,
paired with mock/service-tax/direct search comparators.
```

This is the cleanest way to test whether the object/payload/merge boundary is
worth a larger collector rewrite without touching Coach training.

Search reference probe:

```text
Candidate 5, MCTX/JAX resident visual sidecar.
```

Keep it as a truth-serum benchmark for the device-resident shape. Do not call
it trainer-facing until model/replay semantics are owned end to end.

Defer:

```text
Candidate 7 until one actor slab plus one service batch works.
Candidate 10 until smaller compact/device lanes prove the multiplier and gates.
```

## Promotion Gates Shared By All Candidates

A fast row is not training advice unless it proves:

- selected search actions are exactly the next env `joint_action`;
- `root_index`, `env_row`, `player`, `policy_env_id`, `to_play`, and active-root
  order stay attached through root batch, search result, replay rows, and
  materialized samples;
- illegal selected actions and illegal visit mass fail closed;
- terminal rows use final observations, not autoreset observations;
- RND meter mode leaves target rewards unchanged;
- RND reward mode is explicitly labeled as objective-changing;
- root noise, temperature, epsilon, seed, tie policy, backend, sims, batch size,
  death mode, RND mode, input mode, and replay proof status are recorded;
- fallback counts are zero in promotion rows;
- speed tables do not mix Coach training speed, stock full-loop profile speed,
  and profile-only roots/sec as one number.

The smallest useful next combined grid:

```text
B512/P2 and B1024/P2
sim16 and sim32
root_noise=0 first
normal death and no-death
RND none and rnd_meter_v0 zero-weight
direct_ctree_gpu_latent, compact_torch_search_service, service_tax_probe,
mock_search_service
selected-action-feedback proof on
CompactReplayIndexRowsV1 proof on
resident stack / host stack split if testing observation ownership
```

If that grid cannot identify a candidate with both a real multiplier and clean
closed-loop identity, do not broaden into a service farm or full accelerator
rewrite yet.
