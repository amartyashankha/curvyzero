# Fast RL / Self-Play Architecture Patterns

Date: 2026-05-22

Scope: external architecture research for the CurvyTron optimizer lane. This is
not a framework migration recommendation and did not touch live training runs.

## Short Read

Fast game RL systems do not usually get their multiplier from a single clever
kernel. They get it from ownership boundaries:

```text
many actor/env roots alive at once
-> batched model/search service
-> compact trajectory/replay ownership
-> learner consumes ready batches
-> checkpoints refresh actors/search workers
```

The recurring pattern is to keep scalar game APIs and Python objects away from
the hot path. CurvyTron's current compact sidecar/search work is directionally
right, but a sidecar that still returns to the stock LightZero collect/search
and replay shape should be expected to land in the `~1.3x-1.7x` class, not the
`5x-10x` class. The larger architecture experiment should test whether
CurvyTron can own a compact batch from env rows through search result through
replay chunk.

## Sources Read

Local notes:

- `docs/working/optimizer/architecture_reexploration_2026-05-12/large_scale_zero_architectures.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/minizero_architecture.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/efficientzero_ray_architecture.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_external_search_systems_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/current_hot_path_bottleneck_map_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/compact_search_replay_contract_plan_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/puffer_style_contiguous_buffer_attach_audit_20260522.md`

External sources:

- OpenSpiel AlphaZero architecture docs:
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- MiniZero README architecture section:
  <https://github.com/rlglab/minizero>
- KataGo parallel analysis engine:
  <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>
- KataGo distributed training site:
  <https://katagotraining.org/>
- KataGo repository:
  <https://github.com/lightvector/KataGo>
- Polygames README:
  <https://github.com/facebookarchive/Polygames>
- EfficientZero repository:
  <https://github.com/YeWR/EfficientZero>
- EfficientZero paper supplementary material:
  <https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material>
- PufferLib vectorization docs:
  <https://pufferai.github.io/dev/build/html/rst/landing.html>
- PufferLib docs:
  <https://puffer.ai/docs.html>
- SEED RL paper:
  <https://arxiv.org/abs/1910.06591>
- SEED RL repository:
  <https://github.com/google-research/seed_rl>
- MCTX repository:
  <https://github.com/google-deepmind/mctx>
- LightZero CTree docs:
  <https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>
- AlphaZero paper:
  <https://arxiv.org/abs/1712.01815>
- MuZero paper:
  <https://arxiv.org/abs/1911.08265>

## Recurring Architecture Patterns

### 1. Actor Pools Own Environment Progress

OpenSpiel, AlphaZero/MuZero-style systems, MiniZero, EfficientZero, KataGo
distributed training, and PufferLib all separate acting from learning. The
actor side advances environments and produces trajectory records; the learner
side consumes replay and publishes weights.

The exact synchronization varies:

- OpenSpiel AlphaZero launches actors, learner, and evaluators. The C++ path
  adds threads, a shared cache, batched inference, and GPU support.
- MiniZero has a server, self-play workers, an optimization worker, and shared
  storage. Its server runs synchronous iterations: collect games with the
  latest net, stop collection, optimize, publish the next net.
- EfficientZero uses Ray actors for self-play, replay, CPU batch preparation,
  GPU reanalysis, and evaluation, with the learner loop consuming prepared
  batches.
- KataGo's public distributed training has clients download current networks,
  generate self-play/rating games, and upload data.
- PufferLib's speed story is mostly about env/vector ownership: many envs per
  worker, shared buffers, async sampling, and zero-copy or low-copy batching.

CurvyTron mapping: clean. CurvyTron already has row-batched env surfaces and
can represent physical rows plus players as compact arrays. The stock LightZero
collector turns that back into scalar env ids and timestep objects too early.

CurvyTron mismatch: most Zero systems assume alternating turns or one scalar
action per env step. CurvyTron has simultaneous two-player physical ticks, so
the actor boundary must preserve `[row, player]` provenance and commit one
joint action per physical row.

### 2. Fast Search Systems Batch Many Roots Or Leaves

MiniZero is the most direct self-play reference: one self-play worker keeps
multiple MCTS instances alive, selects leaves across them, and evaluates those
leaves in a GPU batch. KataGo's parallel analysis engine makes the same point
from the serving side: analyzing many positions can be faster than one position
at a time because the neural net sees larger batches. OpenSpiel's C++ AlphaZero
also contrasts with its Python path by adding batched inference.

MCTX is the cleaner endpoint: a batch-first, JAX-native MCTS implementation
where root arrays, recurrent functions, and tree state are visible to the
compiler. It is useful as a target shape, not a small patch for PyTorch
LightZero.

CurvyTron mapping: very clean. The natural fast batch is:

```text
B physical rows * P player views = B*P roots
fixed action count A = 3
fixed num_simulations
batched initial inference
batched recurrent inference per simulation
compact visit/action/value output
```

CurvyTron mismatch: independent per-seat search must not accidentally change
game semantics. A centralized joint-action search over `3*3=9` actions is
architecturally easy but trains a different controller. Sequentially choosing
player 0 then player 1 also changes information/order unless carefully hidden.
The semantically interesting lane is two root searches from the same physical
snapshot, followed by one simultaneous joint-action commit.

### 3. Replay Is A First-Class Owner, Not A Dumping Ground

AlphaZero/MuZero-style systems store search-improved policy targets and
outcomes. OpenSpiel's learner pulls actor trajectories into FIFO replay.
MiniZero stores self-play records in shared storage and the optimizer maintains
an in-memory replay window. EfficientZero goes further: replay stores trajectory
blocks and a separate CPU/GPU reanalysis tier prepares final training batches.

For CurvyTron, this matters because a fast compact collector can still lose the
whole-loop win if it materializes stock game segments, dicts, and target rows in
the hot cadence.

CurvyTron mapping: clean if we make `CompactReplayChunkV1` real. The existing
compact contract note already sketches the right fields:

```text
observation/action/reward/done/to_play/action_mask
visit_policy/raw_visit_counts/root_value/selected_action
row/player/env ids
checkpoint/search metadata
terminal final-observation sidecars
```

CurvyTron mismatch: LightZero replay expects its own GameSegment-style shape.
The compatibility bridge is valuable for validation, but it should not remain
the optimized production hot path if the target is a large multiplier.

### 4. GPU Utilization Comes From Batch Shape And Stable Boundaries

KataGo, MiniZero, OpenSpiel C++, MCTX, SEED RL, and PufferLib all point at the
same systems principle: accelerators help when the work arrives in sufficiently
large, stable batches. They help less when Python is issuing tiny, dynamic
requests and synchronizing after each step.

Relevant variants:

- MiniZero batches leaf evaluation across active MCTS instances.
- KataGo tunes parallel positions, per-position search threads, and NN batch
  size.
- SEED RL centralizes inference on accelerators instead of pushing model copies
  and local inference into many actors.
- PufferLib uses static memory, contiguous buffers, pinned transfers, CUDA
  streams, and CUDA graph capture in its newer native backend.
- LightZero already uses C++ for CTree traversal/backprop, so CurvyTron's
  remaining issue is not simply "move search to C++"; it is the Python/list,
  CPU/GPU, and object boundary around CTree.

CurvyTron mapping: clean. Fixed `A=3`, fixed observation shape, fixed sim count,
and row/player masks are unusually friendly to static-shape search and compact
replay.

CurvyTron mismatch: current stock training still crosses the boundary many
times:

```text
GPU model tensors
-> CPU/list root prep
-> Python simulation loop
-> GPU recurrent inference
-> CPU reward/value/policy arrays for CTree backprop
-> dict actions and stock replay objects
```

That explains why direct GPU-latent CTree can be a real win without being the
endgame.

### 5. CPU Env Workers Should Feed Contiguous Buffers

PufferLib is not a MuZero system, but it is highly relevant to CurvyTron's env
and observation side. Its vectorization docs emphasize multiple envs per
worker, shared memory, shared flags, async send/recv, and zero-copy batching.
The newer docs emphasize static allocations and chunked vectorized buffers.

CurvyTron mapping: very clean. The hybrid manager already has a pre-scalar
attach point with row-major observation/action-mask arrays. The native actor
buffer profile result in local notes supports the same lesson: removing actor
payload/merge overhead improved the compact zero-observation profile row
substantially.

CurvyTron mismatch: renderer-backed rows still need a compact render-state
contract, not just compact zero-observation buffers. Terminal/final observation
and autoreset sidecars are required for correct replay.

## What Maps Cleanly To CurvyTron

These patterns are good candidates to copy directly:

- Actor/search/learner separation with explicit checkpoint ids.
- Many live physical rows per worker.
- `[B,P]` env/player arrays rather than per-env objects.
- Batched initial inference over `B*P` roots.
- Batched recurrent inference over active roots/leaves.
- Fixed-shape search inputs: `value[B*P]`, `policy[B*P,3]`,
  `legal_mask[B*P,3]`, `last_action[B*P]`, hidden-state indices.
- Compact search output: selected action, visit policy/counts, root value,
  implementation/search metadata.
- Replay chunks that store compact arrays and terminal/final-observation
  sidecars.
- Replay age/freshness metadata: checkpoint id, model hash, actor refresh id,
  search config, software version.
- CPU env workers writing to owned contiguous buffers before any LightZero
  adapter runs.
- Compatibility adapters used for parity tests, logging, and eval rather than
  the main optimized path.

## What Does Not Map Cleanly

These are dangerous or expensive to copy without adaptation:

- Turn-based actor APIs. CurvyTron needs simultaneous joint action commit.
- Single scalar action trajectory blocks. CurvyTron needs row/player identity
  and either per-seat targets or carefully specified centralized targets.
- Pure board-game assumptions about legal actions, symmetries, and terminal
  outcome timing.
- KataGo-style transposition/NN caches as a main speed thesis. CurvyTron's
  visual continuous-ish state space likely has less repeated-state reuse than
  Go.
- EfficientZero's Atari-shaped single-agent replay contract. Its actor/replay/
  reanalysis topology is useful, but its trajectory semantics are not a direct
  fit.
- MCTX/JAX as a small local patch. It is a useful batch-first design reference,
  but a PyTorch/LightZero bridge would recreate host boundaries unless the
  whole model/search path moves.
- SEED-style central inference as an immediate MuZero answer. It is excellent
  evidence for accelerator-side inference ownership, but MuZero search needs a
  search service, not just an action-inference server.

## Ranked Next Architecture Experiments

### 1. Compact Replay Chunk Writer And Sampler

Build a profile-only `CompactReplayChunkV1` writer that consumes compact
root/search/env arrays without allocating stock LightZero `GameSegment` objects.
Then build a sampler/target builder over those arrays and compare it to the
existing target-row path on deterministic fixtures.

Why first: local notes already show compact target-row proofs starting to land.
Without replay ownership, any faster search service can just move the bottleneck
to object materialization.

Pass signal:

```text
compact chunks preserve actions, visits, rewards, dones, final observations,
row/player ids, masks, to_play, and root values across 2-3 record chunks
```

Kill signal:

```text
compact replay construction is not materially faster than stock object replay,
or cannot preserve terminal/autoreset semantics cleanly
```

### 2. Mock Search-Service Full-Loop Ceiling

Extend the mock search-service profile so it consumes real compact observations
and legal masks, emits legal compact visit/action/value arrays, writes compact
replay chunks, and optionally feeds a fixed learner-batch stub.

Why second: existing mock rows suggest compact search-service shape is
meaningfully faster than direct CTree but not obviously `5x-10x` by search
alone. Adding replay ownership prices the broader architecture before writing a
real search service.

Pass signal:

```text
env -> mock compact search -> compact replay remains far above current direct
stock-compatible train throughput
```

Kill signal:

```text
after replay and batch construction are included, the ceiling collapses near
the current direct path
```

### 3. MiniZero-Style Batched Search Service Prototype

Prototype a service that owns many active CurvyTron row/player roots and calls
the MuZero model in batches:

```text
submit CompactRootBatchV1
service owns tree + latent state
for sim in num_simulations:
  traverse many roots/leaves
  recurrent_inference batched
  backprop compact arrays
return CompactSearchResultV1
```

Start with CPU flat-array CTree or a thin array-native wrapper before attempting
GPU-resident tree state. The key proof is removing Python list/root-object
traffic and per-simulation D2H policy/value/reward conversion.

Why third: this is the first real architecture candidate for a durable
multiplier.

Pass signal:

```text
search throughput scales with root batch size and average recurrent batch size,
while replay result construction stays compact
```

Kill signal:

```text
root batching does not improve GPU utilization, or CTree/list/CPU ownership
still dominates after the service boundary is introduced
```

### 4. Native Vector Env Buffer Ownership

Move the hybrid pre-scalar attach point from a probe into an owned buffer ABI:

```text
obs[B,P,4,64,64]
action_mask[B,P,3]
reward[B,P]
done[B]
joint_action[B,P]
terminal/final_observation/autoreset sidecars
row/player/env ids
```

The actor manager should write this directly, and the compact search/replay
consumer should read it directly. Stock scalar timesteps become an adapter.

Why fourth: if search-service ceiling is only about `2x`, the remaining
multiplier has to come from Puffer-style buffer ownership and less Python
scalarization across env/search/replay.

### 5. Array-Native Fixed-`A=3` CTree

If the service prototype shows CTree boundary overhead is still the limiter,
specialize the tree ABI for CurvyTron:

```text
prepare(value[N], policy[N,3], legal[N,3])
traverse() -> hidden_index[N], last_action[N]
backprop(reward[N], value[N], policy[N,3])
output() -> action[N], visits[N,3], root_value[N]
```

Why fifth: useful and conservative, but likely a bridge-class improvement
rather than a full architecture jump if replay/env scalarization remains.

### 6. EfficientZero-Style Reanalysis Tier

Only after compact replay exists, consider a middle tier that samples compact
replay and prepares learner-ready batches, possibly with target-network
inference or policy reanalysis.

Why later: EfficientZero's topology is useful, but CurvyTron's immediate wall
is current self-play/search/replay ownership. Reanalysis adds complexity before
we have the compact data contract settled.

### 7. MCTX/JAX Scratch Lane

Keep MCTX as a design/falsification lane for static batch-first search. Use it
only if the team is willing to own a JAX model/search path or use it as a
profile oracle.

Why last: architecturally beautiful, operationally not a small LightZero patch.

## Stale Policy And Self-Play Batching Caveats

Policy staleness is not one thing. The risk depends on the collection mode.

In synchronous collect-then-train:

```text
freeze checkpoint K
collect a bounded compact batch with K
train K+1 from replay window including K data
publish K+1
```

The data is intentionally from `K`. This is not accidental fleet staleness. The
real caveat is batch size: collecting too much from `K` delays feedback from
`K+1` and can overrepresent old behavior.

In continuous actor/search service mode:

```text
actors/search workers collect while learner trains
checkpoints refresh periodically
replay contains several checkpoint ids
```

Staleness is real and normal. It needs explicit controls:

- checkpoint id and model hash on every compact chunk;
- actor/search worker refresh cadence;
- replay age windows or checkpoint windows;
- maximum allowed actor lag;
- train/data ratio controls;
- optional rejection or downweighting of too-old chunks;
- eval gates that use fresh checkpoints.

In a centralized inference/search service:

```text
actors may step envs while service uses recent weights
one trajectory can include nearby weight versions
```

This can reduce actor-side stale weights but introduces mixed-version
trajectories. That is not automatically wrong, but the replay records must say
which checkpoint produced each search target.

Careful CurvyTron read:

- Larger batches improve GPU utilization but may reduce policy freshness.
- More actors can make individual games finish later if hardware is saturated,
  which increases effective staleness.
- Replaying older self-play is standard in off-policy Zero/MuZero-style
  training, but search targets from old policies are not identical to targets
  from a current policy.
- Reanalysis can refresh targets later, but it is a second-stage architecture,
  not a reason to ignore metadata now.
- For first compact experiments, prefer bounded synchronous batches so
  staleness is controlled by construction and easier to reason about.

## Recommended Near-Term Decision

Do not spend the next major effort only polishing the current
`direct_ctree_gpu_latent` path unless the goal is a modest LightZero-compatible
speedup. It is still worth validating, but the broader architecture evidence
points to a different bet:

```text
compact vector env ownership
-> compact batched search service
-> compact replay chunks
-> stock LightZero adapter only for parity/eval
```

The ranked next move is therefore:

1. Make compact replay chunks real and parity-tested.
2. Price the full compact mock service including replay/batch construction.
3. If the ceiling is high enough, build the MiniZero/KataGo-style batched
   search service.
4. If the ceiling is not high enough, move earlier in the pipeline and make
   native vector env buffers the primary ownership boundary.

Bottom line: the 10x-class thesis is not "search in C" or "search on GPU." It
is "own the compact batch across env, search, replay, and learner input." The
external systems are remarkably consistent on that point.
