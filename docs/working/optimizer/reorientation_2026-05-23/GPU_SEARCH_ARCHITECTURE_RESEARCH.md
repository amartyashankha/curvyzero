# GPU Search Architecture Research

Date: 2026-05-23

Scope: architecture research and critique for CurvyTron optimizer work. No live
Modal runs, Coach runs, checkpoints, trainer defaults, or source code were
changed by this note.

Status: archive/research. Superseded as current work by OPT-132 direct trainer
repeatability. Use this only for external-pattern checks unless `TASK_BOARD.md`
reopens GPU search architecture work.

## Short Verdict

Sagan's recommendation is the right next probe.

If Gate A does not show a clean same-denominator `>2x` win for
`direct_ctree_gpu_latent`, stop shaving direct CTree. The plausible 5x-10x path
is not "MCTS on GPU" in the abstract. It is a fixed-shape search owner with
compact data ownership:

```text
CompactRootBatchV1
-> FixedShapeBatchedSearchOwnerV0 behind CompactSearchServiceV1
-> fixed R roots, A=3 actions, fixed S simulations
-> preallocated/padded device tensors and masks
-> selected action D2H on the env-critical path
-> visit/value/replay payload flushed separately
-> CompactSearchResultV1 / CompactSearchActionStepV1 / CompactSearchReplayPayloadV1
```

The external systems all point to the same thing: fast search systems either
batch many roots/leaves before neural inference, keep hot tensors in stable
contiguous memory, or split actor/search/replay/learner roles so no component
waits on tiny synchronous calls. CurvyZero's current bad shape is the opposite:
Python/list/root boundaries, recurrent calls inside a simulation loop, and
host-shaped payloads around LightZero CTree.

## Latest Critique Update

The next real search owner must be two-phase.

Plainly: the env-critical path needs only one small answer from search: the
selected action for each active row-seat root. The heavy replay payload
(`visit_policy`, root value, raw counts, debug arrays) should not be copied or
materialized before the env can step. It should flush later, before the replay
row becomes sample-visible.

Minimal V1 shape:

```text
CompactRootBatchV1
-> fixed device observation/action-mask buffers
-> initial inference over all R row-seat roots
-> preallocated tree tensors [R, S+1, A=3]
-> masked PUCT select / recurrent / expand / backup for fixed S
-> D2H selected_action[active_roots] only
-> delayed replay payload flush by compact identity
```

Do not sell this as LightZero CTree parity until it is proven. The comparator
must report action match, visit-distribution distance, root-value difference,
legality, inactive-root masking, and identity checks under no-noise
deterministic roots.

External research agrees with the same direction: PufferLib-style contiguous
buffers, EnvPool/native vector envs, Sample Factory shared-memory handles,
MiniZero/OpenSpiel batched inference workers, MCTX dense device arrays, and
GPU-resident env loops all remove hot Python/object/host-device boundaries.
They do not merely polish one renderer or one `tolist()` call.

5x-10x is plausible only as a compound win:

- `1.5x-3x` from removing scalar object/replay materialization in the hot path;
- `1.5x-4x` from batching roots/recurrent calls enough to feed the GPU;
- `1.2x-2x` from static shapes, preallocation, compile/CUDA graph style launch
  reduction;
- more only if actor/search/learner overlap or fanout removes a currently idle
  stage.

Those factors are not guaranteed to multiply. The falsifier grid has to keep
the speed currency honest.

## Local Constraints That Matter

- Trusted training remains stock LightZero `train_muzero`.
- The compact slab, direct CTree, dense Torch search, and MCTX/JAX paths are
  profile-only unless promoted later.
- CurvyTron action count is small: `A=3` per seat. This is good for tree memory
  but bad for naive GPU utilization.
- Root shape should be row-seat first: `R = B * P`, with inactive roots masked,
  not dynamically compacted every simulation.
- Current `CompactSearchServiceV1` already gives the right boundary. The new
  owner should consume `CompactRootBatchV1` and return a validated
  `CompactSearchResultV1`.
- The two-phase search payload in `compact_search_service.py` is important:
  return selected action first, and let replay-heavy visit/value payloads flush
  later.
- `CompactTorchSearchServiceV1` is a useful profile harness, but it is not yet
  the final fixed-shape owner: it still slices active roots, prepares tensors per
  call, runs an eager Python simulation loop, and reads replay payloads back
  immediately.
- MCTX/JAX rows are useful ceiling evidence, not production parity. A real MCTX
  backend needs a JAX model/checkpoint/learner path or an accepted shadow-model
  algorithm change.

## Design Pattern 1: Fixed-Shape Device-Owned Search

Pattern:

- Allocate all search tensors once per owner capacity:
  - `edge_child[R, N, A]`
  - `edge_visit[R, N, A]`
  - `edge_value_sum[R, N, A]`
  - `edge_reward[R, N, A]`
  - `edge_prior[R, N, A]`
  - `latent_pool[R, N, H...]`
  - `path_node[S, R]`, `path_action[S, R]`, `path_active[S, R]`
  - selected action, visit policy, root value outputs
- Use `N = S + 1` for a first dense owner.
- Keep `R`, `A=3`, `S`, observation shape, hidden shape, dtype, and device
  static for the profile.
- Run all roots every simulation and apply `active_root_mask`, `legal_mask`, and
  `node_active` masks instead of changing tensor shapes.
- Return only `selected_action[active_roots]` to host before env step.
- Keep visit counts, root values, predicted logits, and debug payload in a
  replay payload handle until the collector can advance.
- Start eager and preallocated. Only after the eager fixed-shape owner wins,
  test `torch.compile(mode="reduce-overhead")` or CUDA graph capture.

Why it can be 5x-relevant:

- It attacks the current repeated Python/list/root boundary directly.
- It removes per-simulation host-shaped recurrent payload traffic.
- It creates stable shapes and memory addresses, which are prerequisites for
  compile/CUDA graph benefits.
- It lets the env-critical path copy a tiny `[R]` action vector instead of
  hauling replay payloads back every step.

Known traps:

- `A=3` and small `S` can be too little GPU work. A dense GPU loop can lose to
  CTree if every simulation launches many tiny kernels.
- Dynamic `flatnonzero`, `nonzero`, `gather` on active roots, Python `if`
  branches, `.item()`, and shape-dependent allocation reintroduce CPU sync.
- CUDA graphs require stable shapes and stable memory addresses. A pretty
  wrapper around fresh tensors is not graph capture.
- `torch.compile(..., mode="reduce-overhead")` is not magic. It may use CUDA
  graphs for small batches, but it can fall back, use more memory, or be blocked
  by mutation/dynamic control.
- JAX/XLA has the same shape discipline problem in a different outfit: static
  args that vary cause recompilation.
- If the model call remains an ordinary PyTorch call inside a Python `for sim`
  loop, the owner may still be launch-bound.
- Padded inactive roots must not leak legal actions, rewards, RND state, or
  replay rows.

Toy falsifier:

```text
Inputs:
  R in {128, 256, 512, 1024}
  A = 3
  S in {4, 8, 16}
  hidden in {64 vector, current compact latent if cheap}
  observation off first; then real [R,4,64,64]

Compare:
  A: current direct CTree GPU-latent compact denominator
  B: current CompactTorchSearchServiceV1
  C: FixedShapeBatchedSearchOwnerV0 eager/preallocated
  D: C plus compile/CUDA graph attempt
  E: mock/no-search ceiling

Measure:
  roots/sec, simulations/sec, env steps/sec in compact denominator
  kernel launches per step, CPU sync count, allocations per step
  H2D bytes, D2H action bytes, D2H replay bytes
  active-root checksum, selected-action legality, visit-policy parity
  max action/value/visit delta versus LightZero CTree oracle

Kill or demote if:
  C is not at least 1.5x direct CTree search-only at R>=512,S>=8;
  D does not improve C or cannot capture without dynamic-shape fallbacks;
  mock/no-search is close to direct CTree, proving search is not the wall;
  parity/identity checks fail.
```

## Design Pattern 2: Batched Leaf Inference, Not Per-Root Inference

Pattern:

- Keep many roots alive at once.
- Traverse one path per root, collect leaf latent/action pairs, run one batched
  recurrent inference, then backpropagate all rows.
- This can be CPU/C++ tree plus GPU recurrent inference, or device-owned search.
  The important part is that the model sees real batches.
- MiniZero, OpenSpiel C++, KataGo-style analysis, and EfficientZero all use some
  form of many positions/games/trees in flight before neural inference.

Why it can be 5x-relevant:

- If the GPU is idle because each recurrent call is tiny, batching leaves is the
  first-order fix.
- It is less radical than full GPU tree traversal and may be a migration path if
  the dense Torch owner loses on irregular tree work.

Known traps:

- Root batching is not enough if each simulation still serializes CPU traversal,
  Python callback, tensor allocation, recurrent inference, D2H, and CTree
  backprop.
- More envs can lower latency per batch but increase stale replay, terminal
  identity bugs, or learner queueing.
- Batching across the two CurvyTron seats must preserve independent row-seat
  roots. Accidentally switching to joint `A=9` is an algorithm change.
- Caching transpositions or leaves can help, but stale hidden states and
  perspective bugs can silently corrupt targets.

Toy falsifier:

```text
Synthetic model, fixed root tensors, no env:
  R in {64, 128, 256, 512, 1024}
  S in {4, 8, 16}

Compare:
  A: scalar root search calls
  B: CPU tree with one batched recurrent call per sim
  C: service-tax fake tree with recurrent calls only
  D: precomputed recurrent outputs

Measure:
  average recurrent batch size, GPU utilization, recurrent sec/sim,
  tree/control sec/sim, sync count, roots/sec.

Kill or demote if:
  average recurrent batch cannot stay above roughly 128;
  precomputing recurrent outputs barely moves wall time;
  tree/control remains dominant after batching.
```

## Design Pattern 3: Compact Actor/Search/Replay Pipeline

Pattern:

```text
compact env rows
-> resident or pinned observation slabs
-> batched search service
-> compact action step returned immediately
-> compact replay payload committed later
-> learner materialization at sampler edge
```

External analogues:

- Sample Factory splits rollout workers, inference workers, batcher, learner,
  and shared-memory buffers.
- PufferLib uses contiguous tensor allocations, pinned transfer buffers, CUDA
  streams, and CUDA graph tracing.
- EnvPool attacks vector environment throughput with fixed async batch outputs.
- EfficientZero uses explicit self-play, replay, batch/reanalysis workers, and
  learner roles.
- MiniZero uses server, self-play workers, optimizer, and storage; self-play
  workers run multiple games and batch neural inference.
- OpenSpiel C++ AlphaZero adds threads, shared cache, batched inference, and GPU
  support over the slower Python implementation.

Why it can be 5x-relevant:

- If stock `train_muzero` serializes collection, replay, and learner work, a
  compact pipeline can remove wait states rather than optimizing one function.
- If scalar timestep and replay row construction are hot, index-only compact
  replay rows remove a repeated object tax.
- Two-phase search output lets the env advance while replay payloads are
  materialized or flushed off the critical path.

Known traps:

- Speed currencies multiply fake numbers. Env steps/sec, roots/sec, learner
  iterations/hour, and games/hour are different units until Gate A/T rows tie
  them together.
- Shared memory or slabs do not help if every step immediately re-expands into
  `BaseEnvTimestep`, `PolicyRowRecordV0`, and GameSegment-like objects.
- Replay "not the wall" can still be misleading if the expensive part is hot
  path construction and final-observation identity, not sampling.
- Asynchrony makes row ordering bugs easier: env row, player, policy id,
  terminal final observation, and RND latest-frame sidecars must be checked.

Toy falsifier:

```text
Use the existing compact C0-C8 and trainer T0-T5 ideas.

C rows:
  observation off, mechanics no-op, scalar materialization on/off,
  replay materialization on/off, precomputed recurrent, service-tax search,
  mock/no-search.

T rows:
  stock trainer, replay no-op, learner no-op, replay real/learner no-op,
  RND isolated, sidecars isolated.

Measure:
  same-denominator env steps/sec, action/replay identity checks,
  materialized rows/sec, bytes copied, learner calls, replay sample calls.

Kill or demote if:
  compact rows improve but trainer rows do not;
  scalar/replay materialization on/off saves less than 25 percent;
  final observation, RND, or action identity checks fail.
```

## Design Pattern 4: Resident Observation Lane Only If It Reaches Search

Pattern:

- Keep latest frame/stack tensors resident through search and RND input.
- Use pinned memory and async H2D only if full residency is not yet available.
- Avoid scalar GPU render calls that immediately copy frames back to CPU.

Why it can be 5x-relevant:

- GPU search cannot win if every step still does CPU render, CPU stack, H2D
  root input, search, and D2H replay payload.
- Observation wins compound with fixed-shape search only when they remove a
  boundary from the same loop.

Known traps:

- Faster standalone rendering is not faster training if search/replay still
  rebuilds host objects.
- GPU observation can hide row-major versus view-major or terminal/autoreset
  mistakes.
- If no-observation-refresh is not much faster than real observation, search or
  replay is probably the wall.

Toy falsifier:

```text
B=512, P=2, fixed S, compact denominator.

Compare:
  A: CPU render/stack -> H2D search input
  B: GPU or resident stack -> selected action D2H only
  C: no observation refresh

Measure:
  closed-loop steps/sec, H2D bytes, D2H bytes, row/player/latest-frame checksum.

Kill or demote if:
  B is close to A;
  C is close to A;
  row/player/final-frame checks fail.
```

## Design Pattern 5: MCTX/JAX As Ceiling, Not Shortcut

Pattern:

- MCTX is fast because search is JAX-native, batch-first, JIT-compiled, and
  expressed as pure recurrent functions over arrays.
- It is a useful ceiling for what fixed-shape accelerator search could look
  like.

Why it can be 5x-relevant:

- It demonstrates the desired shape: batched roots, static loop, accelerator
  execution, small action/readback surface.
- It can quickly tell whether dense array search is viable at CurvyTron root
  counts.

Known traps:

- `torch -> numpy -> jax -> numpy -> torch` is not production GPU search.
- A PyTorch LightZero model cannot sit inside an MCTX `recurrent_fn`.
- Real promotion requires a JAX model/learner/checkpoint path, a trustworthy
  shadow model, or an accepted algorithm change.
- MCTX mask polarity, all-invalid roots, value perspective, reward sign,
  discount, root noise, and Gumbel semantics must be explicit.
- JAX recompilation on changing shapes/static args can erase the speed win.

Toy falsifier:

```text
Visual-root benchmark only:
  obs [R,4,64,64], R in {128,512,1024}
  A=3, S in {4,8,16}, hidden_dim in {64,128}
  fixed legal masks, then realistic masks

Compare:
  A: MCTX tiny JAX model
  B: fixed-shape Torch owner with equivalent tiny model
  C: direct CTree compact denominator

Measure:
  compile time, steady search time, H2D/D2H, memory, recompile count,
  action legality, action-weight normalization, root-value shape.

Kill or demote if:
  steady search is not at least 2x direct CTree on search-only rows;
  H2D/D2H dominates;
  shape changes recompile in normal compact batches;
  semantic mismatch is not an explicitly accepted algorithm change.
```

## Design Pattern 6: Algorithmic Search Reduction Is Separate

Pattern:

- Gumbel MuZero, posterior-policy search variants, and lower simulation counts
  can reduce search work by needing fewer simulations for a usable target.
- This is different from systems speed. It changes the search target or search
  policy and needs Coach approval.

Why it can be 5x-relevant:

- For `A=3`, a high-quality low-simulation search may matter more than making
  a larger simulation count fast.
- If LightZero CTree parity is impossible, a deliberate Gumbel-style algorithm
  lane may be cleaner than pretending it is the same backend.

Known traps:

- Fewer simulations can improve throughput and lose learning progress.
- Action-match parity against CTree may be the wrong metric if the algorithm is
  deliberately different, but the change must be named.
- Short profile wins do not prove sample efficiency.

Toy falsifier:

```text
Profile-only first, then Coach-owned quality gate if approved:
  direct CTree S=8/16
  Gumbel or fixed-shape owner S=2/4/8

Measure:
  roots/sec, action agreement, visit KL, root value delta,
  short-run training health only if Coach accepts the semantic change.

Kill or demote if:
  speedup comes only from worse targets;
  training health per wall-clock does not improve;
  docs cannot state the semantic change plainly.
```

## FixedShapeBatchedSearchOwnerV0 Sketch

This is a doc-level shape, not code.

Constructor:

```text
FixedShapeBatchedSearchOwnerV0(
  policy_or_model,
  root_capacity=R,
  action_count=3,
  num_simulations=S,
  observation_shape=(4,64,64),
  hidden_shape=...,
  device="cuda",
  dtype=float32,
  compile_mode="none|torch_compile|cuda_graph",
)
```

Run contract:

```text
run(root_batch: CompactRootBatchV1) -> CompactSearchResultV1
```

Preconditions:

- `root_batch.legal_mask.shape == [R, 3]`, or the owner pads to exactly `R`.
- `active_root_mask.shape == [R]`.
- No CTree calls.
- No per-root Python records.
- No per-simulation host readback.
- No shape-changing active-root compaction inside the simulation loop.
- Padded inactive roots have harmless legal masks and are ignored by metadata.

Hot loop:

```text
copy or alias observations into owner.obs[R,C,H,W]
copy masks into owner.legal_mask[R,A] and owner.active[R]
initial_inference over fixed R
initialize tree buffers

for sim in range(S):
  select leaf for all R using masked dense tensors
  recurrent_inference over fixed R parent latents/actions
  expand/back up with active/path masks

selected_action = argmax/sample visit counts for all R
copy selected_action[active] to host
optionally leave visit/value payload resident until replay flush
validate CompactSearchResultV1 at edge
```

Required telemetry:

- `fixed_owner_profile_only=true`
- `fixed_owner_calls_train_muzero=false`
- `fixed_owner_ctree_calls=0`
- `fixed_owner_python_rows_materialized=0`
- `fixed_owner_per_sim_d2h_bytes=0`
- `fixed_owner_action_d2h_bytes`
- `fixed_owner_replay_payload_d2h_bytes`
- `fixed_owner_h2d_bytes`
- `fixed_owner_root_capacity`
- `fixed_owner_active_root_count`
- `fixed_owner_num_simulations`
- `fixed_owner_compile_status`
- `fixed_owner_allocation_count_after_warmup`
- `fixed_owner_search_sec`
- `fixed_owner_total_sec`

Failure policy:

- If a shape mismatch would require a new compiled graph, fail closed or pad.
- If root noise, illegal masks, or terminal rows break fixed-shape assumptions,
  run eager fixed-shape with masks, not a silent semantic fallback.
- If parity cannot be defined, label the row as algorithmic divergence.

## Priority Recommendation

1. Freeze direct CTree as a cleanup/baseline row unless it later beats the
   matched `train_muzero` denominator by a large margin.
2. Build the V1 two-phase fixed-shape owner behind the compact search-service
   contract.
3. Return selected actions plus stable identity only on the env-critical path.
4. Flush replay/search payload later by handle, with compact identity checks
   before anything becomes sample-visible.
5. Compare the two-phase owner against direct CTree and MCTX/JAX on the compact
   denominator.
6. Only after a compact win survives identity, RND, and replay gates, attach
   trainer T0-T5 rows. Do not wire into Coach training before that.

## Sources Checked

Local:

- `docs/working/optimizer/reorientation_2026-05-23/EXTERNAL_RESEARCH_FOLLOWUP.md`
- `docs/working/optimizer/reorientation_2026-05-23/CURRENT_STATE.md`
- `docs/working/optimizer/reorientation_2026-05-23/COMPACT_OWNERSHIP_PLAN.md`
- `docs/working/optimizer/reorientation_2026-05-23/NEXT_MOVES.md`
- `docs/working/optimizer/reorientation_2026-05-23/EXPERIMENT_QUEUE.md`
- `docs/working/optimizer/reorientation_2026-05-23/BOTTLENECK_MODEL.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/README.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/large_scale_zero_architectures.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/mctx_jax_search.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/minizero_architecture.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/efficientzero_ray_architecture.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/lightzero_stock_dataflow.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/curvytron_system_design.md`
- `src/curvyzero/training/compact_search_service.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- `src/curvyzero/training/compact_torch_search_service.py`
- `src/curvyzero/training/mctx_compact_search_service.py`

External:

- MCTX README: https://github.com/google-deepmind/mctx
- OpenSpiel AlphaZero docs: https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- MiniZero README: https://github.com/rlglab/minizero
- EfficientZero README: https://github.com/YeWR/EfficientZero
- KataGo analysis engine docs: https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md
- PufferLib docs: https://puffer.ai/docs.html
- Sample Factory architecture: https://www.samplefactory.dev/06-architecture/overview/
- Sample Factory double-buffered sampling: https://www.samplefactory.dev/07-advanced-topics/double-buffered/
- EnvPool paper: https://arxiv.org/abs/2206.10558
- Batch Monte Carlo Tree Search: https://arxiv.org/abs/2104.04278
- Parallel MCTS with batched simulations: https://arxiv.org/abs/2207.06649
- PyTorch CUDA graphs blog: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- PyTorch `torch.compile` docs: https://docs.pytorch.org/docs/2.9/generated/torch.compile.html
- JAX JIT docs: https://docs.jax.dev/en/latest/jit-compilation.html
