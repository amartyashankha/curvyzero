# External GPU MCTS Dataflow Research, 2026-05-22

Status: research-only optimizer note. No source code, live Coach training runs,
Modal volumes, checkpoints, evals, GIFs, or tournaments were touched.

Target question: how do fast AlphaZero/MuZero/LightZero-style and high-throughput
RL systems organize self-play, search, replay, and learner dataflow; what stays
on GPU versus CPU; where do they accept synchronization; what is realistic for
GPU MCTS; and what maps cleanly to CurvyTron's current profile-only path.

## Short Read

Fast systems do not get their main multiplier from "use GPU" as a label. They
get it from ownership boundaries:

```text
many env/game roots alive at once
-> batched model/search service
-> compact action/search-policy/value arrays
-> replay owner that avoids per-step Python objects
-> learner consumes ready tensor batches
-> checkpoints or weights refresh actors/search workers at coarse cadence
```

The strongest external match for our desired search shape is MCTX: root
logits/value/embedding, tree state, recurrent expansion, backup, invalid-action
masks, actions, and policy targets are JAX arrays over a batch. It is not a
drop-in LightZero CTree replacement because the recurrent model has to be
JAX-traceable to stay inside the compiled search loop.

LightZero is the strongest match for our current trusted training shape. It
already separates policy collect, learning, evaluation, and MCTS, and its CTree
uses C++ for core traverse/backprop. The remaining wall is not "tree code is
Python." It is the loop boundary around CTree: GPU model outputs copied to CPU,
lists/root objects, Python per-simulation control, recurrent output readback,
per-env action dicts, scalar timesteps, stock replay objects, and RND side
paths.

The realistic near-term path is not full GPU CurvyTron. The best profile-only
next architecture is:

```text
HybridCompactBatch
-> CompactRootBatchV1 without scalar LightZero timesteps
-> search service backend, initially direct CTree or MCTX/Torch sidecar
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1 / compact replay ring
-> learner materialization only at sampler/validation edge
```

Full GPU MCTS is realistic only if fixed-shape batches, masks, tree arrays, and
the recurrent model are compiled or service-owned together. It is not realistic
as "call PyTorch from JAX" or "keep CPU CTree and expect GPU residency."

## Local Cross-Check

Current local docs and code already identify the same boundary.

- `README.md` in this working directory says actual Coach training still enters
  stock LightZero `train_muzero`, while the batched/resident GPU work remains
  profile-only. It records matched H100 stock-loop speedups around `1.28x-1.31x`
  from `direct_ctree_gpu_latent`, not a production migration.
- `current_code_dataflow_map_20260521.md` separates production Coach training,
  stock full-loop profiles, profile-only hybrid batches, public LightZero
  consumer probes, direct CTree probes, and RND.
- `current_hot_path_bottleneck_map_20260522.md` names the active LightZero wall
  as GPU model tensors crossing into CPU NumPy/list CTree contracts and back
  every search simulation.
- `subagent_full_iteration_dataflow_20260522.md` names the compact-loop wall as
  next-search-input preparation: env step plus observation/stack/root handoff,
  not raw drawing alone.
- `gpu_mcts_current_flow_explainer_20260522.md` usefully labels the lanes:
  stock LightZero training, `direct_ctree_gpu_latent`, flat-A3 CTree, dense
  Torch MCTS, and MCTX/JAX compact visual roots.
- The current compact row contract is real code, not just prose:
  `src/curvyzero/training/compact_policy_row_bridge.py` defines
  `CompactRootBatchV1`, `CompactSearchResultV1`, `CompactReplayChunkV1`, and
  `CompactReplayIndexRowsV1`.
- The pre-scalar attach point is also real:
  `src/curvyzero/training/source_state_hybrid_observation_profile.py` builds
  `HybridCompactBatch`, runs `batched_stack_probe`, and only later calls
  `materialize_lightzero_scalar_timestep`.

The local profile-only path therefore already has the right first boundary:
row-major `[B,P,4,64,64]` observations, `[B,P,3]` masks, row/player ids,
rewards, done/final/autoreset sidecars, and compact search/replay outputs. The
gap is that the trusted training lane still re-enters stock LightZero's scalar
collector, CTree, replay, learner, and RND object shape.

## External Systems

### AlphaZero and OpenSpiel

AlphaZero's algorithmic contract is self-play plus search-improved policy
targets. The AlphaZero paper is the canonical source for "MCTS guided by a
neural network generates better action distributions, then the network is
trained on those self-play outcomes" ([arXiv:1712.01815](https://arxiv.org/abs/1712.01815)).

OpenSpiel's AlphaZero docs expose the systems lesson more directly. Their
implementation is actors generating self-play with MCTS, a learner pulling
trajectories into FIFO replay, evaluators, and checkpoints. The docs explicitly
contrast the Python implementation, which lacks inference batching and uses CPU,
with the C++/LibTorch implementation, which uses threads, a shared cache,
batched inference, and GPU inference/training support
([OpenSpiel AlphaZero](https://openspiel.readthedocs.io/en/stable/alpha_zero.html)).

CurvyTron read: copy the dataflow separation, not the turn-based assumptions.
Actors/search/learner/eval separation maps well. A scalar per-env game API does
not. CurvyTron must preserve simultaneous `[row, player]` provenance until one
joint action is committed per physical row.

### MiniZero

MiniZero is the cleanest external AlphaZero/MuZero-style systems reference for
batching. Its README describes a server, self-play workers, an optimization
worker, and storage. Each self-play worker keeps multiple MCTS instances alive,
selects leaf nodes across them, and evaluates the leaves with batched GPU
inference ([MiniZero README](https://github.com/rlglab/minizero)).

What stays CPU: game state, tree selection/control, record/storage plumbing in
the worker/server architecture.

What goes GPU: neural inference over collected leaf batches.

Where it accepts sync: iteration boundaries and batched leaf inference. The
worker pauses tree expansion until a leaf batch is evaluated, then continues.

CurvyTron read: this supports a search service over many row-seat roots. It
also explains why optimizing one scalar MCTS root at a time will not saturate a
GPU. Our natural leaf/root batch is `B * P`, with fixed `A=3`.

### LightZero and EfficientZero

LightZero's README says Policy covers learning, collecting, and evaluation, and
MCTS covers tree structure plus policy interaction. It implements MCTS in Python
`ptree` and C++ `ctree`
([LightZero README](https://github.com/opendilab/LightZero)). The LightZero tree
docs say MuZero CTree's `batch_traverse` and `batch_backpropagate` are in C++,
and that batch search enables parallel model inference
([LightZero tree docs](https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html)).
The current LightZero source shows the practical boundary: per simulation it
calls `tree_muzero.batch_traverse`, builds Torch tensors for latent states and
actions, runs `model.recurrent_inference`, detaches outputs to CPU NumPy, turns
reward/value/policy into Python lists, then calls `batch_backpropagate`
([LightZero `mcts_ctree.py`](https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/mcts/tree_search/mcts_ctree.py)).

EfficientZero's supplementary implementation notes are especially relevant. The
pipeline uses Ray, data/self-play workers, CPU rollout/context workers, GPU
batch workers for reanalysis/search, replay, and learner queues. It splits CPU
and GPU parts of reanalysis to keep both busy, uses C++ MCTS for atomic tree
work, uses Python for neural inference, and Cython between them
([EfficientZero supplement](https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material);
[EfficientZero repo](https://github.com/YeWR/EfficientZero)).

CurvyTron read: LightZero/EfficientZero validate the conservative path:
CPU/C++ tree plus batched GPU model calls can be good enough to build a real
trainer. They also show the ceiling of this approach. If model outputs have to
return to CPU tree structures every simulation, the system has accepted a
per-simulation synchronization point. That can be acceptable for compatibility,
but it is not the MCTX-like endpoint.

### MuZero and Gumbel MuZero

MuZero defines the learned-model search dataflow: representation encodes the
real observation, dynamics rolls hidden states forward by action, and prediction
produces policy/value. The arXiv page includes the paper and pseudocode
ancillaries ([arXiv:1911.08265](https://arxiv.org/abs/1911.08265)).

Gumbel MuZero matters for CurvyTron because it was designed to improve policy
quality when planning with few simulations. The OpenReview abstract says Gumbel
AlphaZero/MuZero improve prior performance in low-simulation settings
([Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO)).

CurvyTron read: because `A=3` and our expensive part is often system overhead,
low-simulation search quality is a first-class speed lever. Gumbel-style search
is worth using in MCTX/JAX sidecars and as a LightZero config comparison, but it
does not remove the object/sync boundary by itself.

### MCTX

MCTX is the clean GPU-resident reference shape. Its README says it is a
JAX-native MCTS library for AlphaZero, MuZero, and Gumbel MuZero, supports JIT
compilation, and operates on input batches in parallel
([MCTX README](https://github.com/google-deepmind/mctx)).

The source types are exactly the shape we want:

```text
RootFnOutput:
  prior_logits [B,A]
  value        [B]
  embedding    [B,...]

recurrent_fn(params, rng_key, action [B], embedding [B,...]):
  -> reward [B], discount [B], prior_logits [B,A], value [B]
  -> next_embedding [B,...]

PolicyOutput:
  action [B]
  action_weights [B,A]
  search_tree batched Tree
```

Primary source details:

- `RootFnOutput`, `RecurrentFnOutput`, and `PolicyOutput` shapes are in
  [`mctx/_src/base.py`](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/base.py).
- Tree state is dense arrays over `[B,N]` and `[B,N,A]` for visits, values,
  parents, children, rewards, discounts, priors, invalid-action masks, and
  embeddings in
  [`mctx/_src/tree.py`](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py).
- Search uses a JAX loop, vmapped simulation, expansion through the recurrent
  function, and JAX backup in
  [`mctx/_src/search.py`](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/search.py).

What stays GPU/device: root logits/value/embedding, tree arrays, invalid-action
masks, recurrent outputs, embeddings for every expanded node, visit counts, and
action weights.

What can sync to CPU: final selected actions if a CPU env must step; final
policy/value summaries if replay is CPU; metrics/checkpoints.

What must not sync: recurrent reward/value/policy at every simulation. If those
leave the compiled loop, MCTX's main benefit is gone.

CurvyTron read: MCTX is the right scratch sidecar or future rewrite target, not
a small LightZero patch. A PyTorch LightZero model bridged into MCTX would
recreate the host boundary. A tiny JAX model over real CurvyTron visual roots is
the correct falsifier.

### PufferLib

PufferLib is not MuZero, but its data ownership pattern is highly relevant. Its
docs emphasize static memory, single contiguous allocations, CUDA graph tracing,
chunked env buffers, CUDA streams, async transfers, and pinned memory
([PufferLib docs](https://puffer.ai/docs.html)).

What stays GPU: rollout forward pass, train minibatch/loss/update when using
the native backend, static allocations, graph-replayed work.

What may stay CPU: environment execution within chunks, with pinned async
transfers to GPU.

Where it accepts sync: epoch/rollout buffer boundaries, graph warmup/tracing,
and explicit transfer queues.

CurvyTron read: our `HybridBatchedObservationProfileManager` plus
`HybridCompactBatch` is already a Puffer-like attach point. The translation is
"own contiguous row/player buffers and feed compact search/replay directly,"
not "make scalar LightZero timesteps faster."

### SEED RL

SEED RL centralizes both inference and training on the learner. Its repo says
training and inference are performed on the learner, and the paper abstract
names centralized inference plus an optimized communication layer as the
architecture ([SEED repo](https://github.com/google-research/seed_rl);
[SEED paper](https://arxiv.org/abs/1910.06591)).

What stays accelerator-side: model inference and training.

What stays actor-side: env stepping and trajectory production.

Where it accepts sync: actors send observations or unrolls to the central
learner/inference service and receive actions back. The accepted cost is
communication, paid to avoid stale local model copies and to batch accelerator
inference.

CurvyTron read: a central batched search/model service is plausible even while
CurvyTron env state remains CPU. Selected action readback is tiny. The key is
not to read back full observations or recurrent outputs unnecessarily.

### EnvPool

EnvPool is a C++ batched environment pool with sync and async APIs. Its README
reports high raw FPS, batched APIs by default, and async `send`/`recv`
([EnvPool README](https://github.com/sail-sg/envpool)).

What stays CPU: environment simulation.

What it optimizes: C++ threads, batched env ids, async action send and result
receive, less Python subprocess overhead.

CurvyTron read: EnvPool validates the value of a native/vector env boundary and
async action/result flow. It is not a GPU-residency answer by itself.

### Isaac Gym, Brax, and PixelBrax

Isaac Gym is the clearest "all data stays on GPU" RL example. Its abstract says
physics simulation and policy training reside on GPU and communicate through
physics buffers/PyTorch tensors without CPU bottlenecks
([Isaac Gym paper](https://arxiv.org/abs/2108.10470)).

Brax is the JAX analog: environments and learning algorithms compile together
and run on the same accelerator
([Brax paper](https://arxiv.org/abs/2106.13281)). PixelBrax extends that idea
to pixel observations with a pure JAX renderer, enabling end-to-end GPU pixel
RL over thousands of parallel envs
([PixelBrax paper](https://arxiv.org/abs/2502.00021)).

CurvyTron read: these are proof that full env/render/policy residency is a real
architecture, but they are not near-term drop-ins. CurvyTron's current env is
CPU NumPy with source-state rendering experiments. The clean migration would be
a JAX/CUDA env plus JAX/CUDA search/learner. That is a rewrite lane, not a
profile patch.

## Residency And Sync Table

| System | Env / sim | Search tree | Model | Replay / learner | Accepted sync |
| --- | --- | --- | --- | --- | --- |
| OpenSpiel AlphaZero C++ | CPU game state, threaded actors | CPU MCTS with evaluator cache | GPU batched inference/training possible | FIFO replay, learner checkpoints | actor trajectory queue, checkpoint refresh, batched evaluator calls |
| MiniZero | CPU games in self-play workers | multiple CPU MCTS instances | batched GPU leaf inference | storage + optimization worker | leaf batch inference and iteration boundaries |
| LightZero | env manager objects, often CPU envs | CPU CTree for ctree mode | PyTorch GPU model if configured | stock GameSegment/replay/learner | root prep CPU, per-sim recurrent output CPU for CTree, replay/learner object edges |
| EfficientZero | CPU actors/context workers | C++ MCTS, batch MCTS | Python/PyTorch GPU inference and GPU reanalysis workers | Ray replay, queues, learner | CPU/GPU split for reanalysis, queues, batched GPU workers |
| MCTX | caller-owned env | JAX tree arrays on device | JAX recurrent/representation functions | caller-owned | final action/results out; no per-sim recurrent CPU sync |
| PufferLib | chunked env buffers, CPU or native | not MCTS | native/PyTorch policy on GPU | static rollout/train buffers | pinned async transfers, rollout/train graph boundaries |
| SEED RL | actor-side envs | not MCTS | central learner/inference accelerator | central learner | actor-service observation/action communication |
| EnvPool | C++ CPU env pool | not MCTS | external | external | async send/recv, batched env ids |
| Isaac Gym/Brax/PixelBrax | GPU/JAX sim/render | not MCTS | same-device learner | same-device training loop | metrics/checkpoint edge, not env-policy hot path |

## What Is Realistic For GPU MCTS

Realistic:

- Batched roots: `R = B * P` with fixed `A=3`, fixed simulation count, and
  masks for live roots and illegal actions.
- Dense tree arrays over `[R, num_simulations + 1]` and `[R, num_simulations +
  1, A]`.
- Device-resident recurrent model outputs inside the search loop.
- Tiny final readback: selected actions `[R]`, visit policy `[R,3]`, root value
  `[R]`.
- JAX/MCTX if the model is JAX-native.
- Dense Torch/Triton/CUDA if buffers are fixed, allocations are hoisted, and
  loops are compiled or captured.
- CPU env plus GPU search, as long as only selected actions cross back each
  decision and observation/stack handoff is coarse or resident.

Not realistic:

- Calling a PyTorch LightZero model from inside JAX/MCTX and expecting
  GPU-resident search.
- Keeping LightZero CPU CTree/list APIs and expecting MCTS to become "on GPU"
  because latents stay on CUDA.
- Eager Torch tensor MCTS with many small ops and dynamic allocation, then
  expecting automatic GPU saturation.
- Rebuilding full `[B,P,4,64,64]` root observations on CPU every decision while
  calling the search path resident.
- Treating search-only roots/sec as training speed. Replay, learner, RND,
  terminal/final observation, and checkpoint/eval sidecars must be measured in
  their own denominator.

CurvyTron-specific caveats:

- Independent per-seat roots over `A=3` preserve the current learner-view
  shape but do not explicitly search joint two-player actions. That is close to
  the current fixed-opponent/ego-seat semantics.
- Centralized joint-action search over `A=9` is technically easy and may be a
  useful control, but it changes the policy target and controller semantics.
- CPU env action readback is acceptable because selected actions are tiny. The
  problem is full observation, root prep, and recurrent payload readback.
- Low-action MCTS can be launch/control-bound. Compiler-visible loops matter
  more than raw GPU FLOPs.

## Options That Map To Our Current Profile-Only Path

### Option A: Keep LightZero, make CTree boundary array-native

Use the existing direct CTree / flat-A3 lane to remove Python list conversions
and improve stock-loop compatibility. This maps cleanly to
`direct_ctree_gpu_latent`, vendored flat-A3 CTree, and the existing promotion
gates.

Upside: lowest semantic risk and closest to `train_muzero`.

Ceiling: likely modest unless root prep, recurrent readback, replay, and scalar
collector object fanout are also attacked. The current measured full-loop gain
is already in the `~1.3x` class.

### Option B: Compact search/replay service with current CTree backend

Use `HybridCompactBatch -> CompactRootBatchV1 -> CompactSearchResultV1 ->
CompactReplayIndexRowsV1` and initially let the search service call the current
direct CTree backend. The point is to remove public LightZero scalar collect
and replay object fanout from the hot loop before replacing search.

Upside: directly tests the architecture boundary external systems recommend.

Risk: if the service still copies recurrent outputs to CPU per simulation, it
will expose the next CTree wall quickly. That is useful information.

### Option C: MCTX visual-root sidecar

Use real CurvyTron compact visual roots and masks, but a tiny JAX model. Keep
it profile-only. Measure compile separately from warmed runs, resident stack
versus fresh H2D, and final D2H for actions/policies.

Upside: clean proof of all-device search feasibility.

Risk: does not prove current LightZero learning. It proves the desired search
shape and memory behavior.

### Option D: Fixed-shape dense Torch search

Represent a fixed `A=3` tree as Torch tensors, hoist allocations, use masks
instead of shape changes, and test `torch.compile(mode="reduce-overhead")`,
CUDA graphs, or Triton kernels only after the eager loop is shape-stable.

Upside: can reuse PyTorch model ownership and avoid JAX migration.

Risk: eager small-op GPU MCTS can lose badly. This is only worth judging after
static buffers and no dynamic `nonzero`/`.item()` gates are enforced.

### Option E: Puffer-style contiguous actor/render/replay buffers

Push `native_actor_buffer`, borrowed render state, resident stack, and compact
sidecars toward one owner that writes row/player buffers in place. Keep scalar
LightZero materialization as an adapter edge.

Upside: maps exactly to the compact-loop bottleneck: state ownership and
next-search-input handoff.

Risk: renderer-backed terminal/autoreset/final-observation correctness is
non-negotiable and must fail closed.

## Five Architecture Experiments

### 1. Compact service loop with scalar LightZero disabled

Build or run the existing profile-only loop as:

```text
HybridCompactBatch
-> CompactRootBatchV1(copy_observation=False where valid)
-> direct_ctree_gpu_latent service wrapper
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> next joint_action[B,P]
```

Measure against current matched direct profile rows with the same `B`, `P`,
sim count, death mode, RND mode, and renderer mode.

Keep if it beats current direct profile rows by a clear `3x` class margin in
the compact denominator or exposes one dominant next wall with clean timing.
Kill as the 10x lane if it only reproduces current direct speed after removing
scalar materialization.

### 2. Real visual-root MCTX sidecar

Run a fixed-shape sidecar:

```text
real [B,2,4,64,64] CurvyTron compact roots
-> invalid_actions [B*2,3]
-> tiny JAX encoder H=64
-> mctx.gumbel_muzero_policy sim8/16/32
-> action/action_weights/root_value
```

Matrix: `B=64,256,512`, `sim=8,16,32`, resident stack on/off. Report compile
time, warmed p50/p95, memory, H2D, search, D2H actions only, D2H full policy,
illegal action count, and recompile count.

Keep if warmed search plus required transfers is comfortably faster than the
LightZero/direct CTree bucket and does not recompile. Kill if visual setup/H2D
dominates or memory/recompile behavior is unstable.

### 3. Precomputed recurrent output versus real recurrent output in the compact service

Use the current direct CTree hook shape, but compare:

```text
real recurrent_inference per sim
precomputed resident reward/value/policy tensors per sim
flat-A3 backprop when available
stock list backprop
```

This isolates recurrent model cost from CPU CTree/list/sync/control cost in the
same compact service denominator.

Keep array-native CTree work only if precomputed recurrent still leaves a large
CTree/list wall. If real recurrent dominates, move effort to compiled recurrent
batching and latent ownership instead.

### 4. Puffer-style native actor/render/search buffer owner

Extend the profile-only native actor buffer path so the owner exposes one
contiguous batch to the compact search service:

```text
action_in[B,P]
reward/done/final/autoreset sidecars
legal_mask[B,P,3]
resident or pinned obs_stack[B,P,4,64,64]
selected_action/visit_policy/root_value out
```

Compare copied parent render state versus borrowed/native state, host stack
versus resident stack, and explicit sync on/off. Include terminal/autoreset
fail-closed rows after the no-death canary.

Keep if state ownership reduces total closed-loop wall, not just timer labels.
Kill if actor/render ownership moves less than the measured observation and
root-handoff slices predict.

### 5. Compact replay/RND tensor ring sampler

Prototype a profile-only compact replay ring that stores index rows and search
arrays in collection, then materializes learner-shaped tensors at sample time.
Add an RND latest-frame path that slices the resident latest channel before any
CPU materialization.

Measure:

- replay-index write cost;
- full materialization cost at sample edge;
- learner-stub H2D and train batch construction;
- RND latest-frame extraction and metric sync cost;
- final-observation correctness for terminal rows.

Keep if replay/RND stays a small fraction when search/env are accelerated.
Kill any design that brings full observation target rows or CPU RND hashing
back into the collect hot cadence.

## Recommendation

Do not spend the next architecture slot on another renderer-only pass or on
claiming the current direct CTree path is GPU MCTS. The stronger research-backed
move is a compact service loop that keeps row/player/search/replay arrays
together and treats stock LightZero objects as validation adapters.

The order I would run is:

1. compact service loop with scalar materialization disabled;
2. precomputed recurrent split inside that loop;
3. real visual-root MCTX sidecar;
4. native actor/render/search buffer owner;
5. compact replay/RND tensor ring.

That sequence is conservative about semantics but honest about the 5-10x
target: the multiplier comes from keeping the full dataflow compact and
resident longer, not from making one existing LightZero function slightly
faster.

## Sources

Primary external sources:

- AlphaZero paper: <https://arxiv.org/abs/1712.01815>
- MuZero paper: <https://arxiv.org/abs/1911.08265>
- Gumbel MuZero: <https://openreview.net/forum?id=bERaNdoegnO>
- MCTX README: <https://github.com/google-deepmind/mctx>
- MCTX core types: <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/base.py>
- MCTX tree: <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py>
- MCTX search: <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/search.py>
- LightZero README: <https://github.com/opendilab/LightZero>
- LightZero tree docs: <https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>
- LightZero MCTS CTree source: <https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/mcts/tree_search/mcts_ctree.py>
- MiniZero README: <https://github.com/rlglab/minizero>
- EfficientZero repo: <https://github.com/YeWR/EfficientZero>
- EfficientZero supplementary implementation notes: <https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material>
- OpenSpiel AlphaZero docs: <https://openspiel.readthedocs.io/en/stable/alpha_zero.html>
- PufferLib docs: <https://puffer.ai/docs.html>
- SEED RL repo: <https://github.com/google-research/seed_rl>
- SEED RL paper: <https://arxiv.org/abs/1910.06591>
- EnvPool README: <https://github.com/sail-sg/envpool>
- Isaac Gym paper: <https://arxiv.org/abs/2108.10470>
- Brax paper: <https://arxiv.org/abs/2106.13281>
- PixelBrax paper: <https://arxiv.org/abs/2502.00021>

Local docs cross-checked:

- `README.md`
- `current_code_dataflow_map_20260521.md`
- `current_hot_path_bottleneck_map_20260522.md`
- `subagent_full_iteration_dataflow_20260522.md`
- `gpu_mcts_current_flow_explainer_20260522.md`
- `subagent_mctx_gpu_search_research_20260522.md`
- `puffer_style_contiguous_buffer_attach_audit_20260522.md`
- `compact_search_replay_service_contract_20260522.md`
- `direct_ctree_promotion_gates_20260522.md`
