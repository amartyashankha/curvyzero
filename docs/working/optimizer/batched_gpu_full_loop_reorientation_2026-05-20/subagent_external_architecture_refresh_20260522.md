# External Architecture Refresh

Date: 2026-05-22

Scope: critique/research note for the CurvyTron training speed lane. I reviewed
the local optimizer notes first:

- `subagent_external_search_systems_20260522.md`
- `gpu_search_fix_ladder_20260521.md`
- `dense_torch_compile_spike_feasibility_20260522.md`

No code, trainer defaults, tournament defaults, or live runs were changed.

## Executive Answer

The external systems evidence supports the current CurvyTron conclusion:
small wrappers can plausibly produce useful `1x-2x` wins, but durable `5x-10x`
training-speed wins usually come from changing the shape of the self-play/search
system:

```text
many active games/roots
-> batched accelerator inference
-> compiled or array-native search state
-> actor/search/evaluator/learner separation
-> compact replay rows at the compatibility boundary
```

LightZero already has C++ CTree internals, so the problem is not simply "rewrite
MCTS in C++". The remaining wall is the collect/search boundary: Python control,
list/object CTree APIs, recurrent model handoff, per-simulation CPU/GPU
movement, and scalar LightZero packaging.

CurvyTron should test a fixed-shape compiled/fused dense search lane first as
the smallest real-search falsifier. In parallel or immediately before that, run
a one-row mock search-service ceiling to determine whether perfect search
boundary removal is even capable of the target multiplier in the full loop.

## Local Baseline Read

Current trusted loop:

```text
stock LightZero train_muzero
source_state_fixed_opponent
```

Local notes show the easy wrapper stack is already partially harvested:

- `direct_ctree_arrays` removes a large public wrapper/list/materialization
  chunk.
- `direct_ctree_gpu_latent` keeps root latents on GPU longer while preserving
  LightZero CTree semantics.
- Recent profile-only rows show strong boundary wins over stock facade, but the
  best matched full-loop profile wins are still only around the user-reported
  `1.28x-1.31x`.
- Dense eager Torch MCTS is alive at sim8 but has already shown bad sim16
  scaling unless the loop becomes fixed-shape and compiler-visible.

Plain implication:

```text
The bottleneck has moved from renderer/output fast paths into the
LightZero-shaped collect/search shell.
```

## External Patterns

### AlphaZero And MuZero Used Massive Actor/Search Scale

The AlphaZero paper says training used 5,000 first-generation TPUs for self-play
and 64 second-generation TPUs for network training, with self-play games
generated from the latest parameters. It also notes AlphaZero searched far fewer
positions per second than Stockfish/Elmo, relying on neural search quality
rather than brute-force node count:
<https://arxiv.org/pdf/1712.01815>.

The MuZero paper reports, for each board game, 16 TPUs for training and 1,000
TPUs for self-play; Atari used fewer self-play TPUs because it used fewer
simulations per move and a smaller dynamics function:
<https://arxiv.org/pdf/1911.08265>.

CurvyTron read:

- The original large results were not single-process Python loops made fast.
- Acting/search and learning were separate throughput machines.
- Neural search can be valuable while still being latency-bound; system design
  must make the accelerator see enough batch.

### OpenSpiel AlphaZero Makes The Boundary Explicit

OpenSpiel describes the standard AlphaZero components as actors, a neural
network evaluator, a learner, and evaluators. Its docs contrast the Python
implementation, which lacks inference batching and runs on CPU, with the C++
LibTorch implementation, which uses threads, a shared cache, batched inference,
and GPU support:
<https://openspiel.readthedocs.io/en/latest/alpha_zero.html>.

CurvyTron read:

- The speed-relevant distinction is not Python versus C++ alone; it is whether
  the actors feed a shared, batched evaluator and avoid scalar inference.
- This supports a CurvyTron boundary shaped around batched row/seat roots and
  compact arrays, not per-root LightZero objects.

### MiniZero Is The Cleanest MuZero-Like Systems Match

MiniZero's architecture has a server, self-play workers, an optimization worker,
and storage. Each self-play worker keeps multiple MCTS instances alive, collects
leaf nodes across them, and evaluates those leaves with batched GPU inference:
<https://github.com/rlglab/minizero>.

CurvyTron read:

- This is almost exactly the missing architecture: multiple roots alive, one
  batched evaluator call, compact game records after search.
- For CurvyTron, the natural root batch is `rows * seats`, then many actors or
  vector-env rows if one batch is not enough to fill the GPU.
- This is more likely to scale than polishing one scalar LightZero root path.

### KataGo Optimizes Cross-Position Batching And Caching

KataGo's parallel analysis engine analyzes many positions concurrently and can
be faster than single-position GTP because it uses cross-position batching:
<https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>.

The same docs describe an asynchronous protocol and an NN result cache. KataGo's
analysis config also exposes the tuning axes that matter in practice:
parallel positions, per-position search threads, and NN batch size:
<https://github.com/lightvector/KataGo/blob/master/cpp/configs/analysis_example.cfg>.

CurvyTron read:

- The external pattern is "many positions feed one NN/search engine", not "make
  one search call a little nicer".
- Cache/transposition is less directly valuable in CurvyTron than Go, but
  batched service scheduling is directly relevant.

### LightZero Already Has C++ CTree

LightZero's README says MCTS has Python `ptree` and C++ `ctree`
implementations:
<https://github.com/opendilab/LightZero>.

Its tree-search docs say MuZero CTree's core `batch_traverse` and
`batch_backpropagate` functions are implemented in C++, and that batched roots
allow model inference to be parallelized:
<https://www.aidoczh.com/lightzero/api_doc/mcts/tree_search/index.html>.

CurvyTron read:

- This is the strongest argument against a vague "move it to C" plan.
- CTree is already C++ in the middle. The costly part left for CurvyTron is the
  shape of the API and the Python/model/list shell around it.
- Array-native CTree is a good conservative lane, but still likely boundary
  cleanup rather than a full `5x-10x` training architecture.

### MCTX Shows The Clean Device-Resident Endpoint

DeepMind's MCTX is JAX-native, supports JIT compilation, and defines search
over batches of inputs in parallel:
<https://github.com/google-deepmind/mctx>.

CurvyTron read:

- MCTX is the right conceptual target: batch-first arrays, recurrent function,
  and search state visible to the compiler.
- It is not a small patch for a PyTorch/LightZero loop. Calling a PyTorch model
  from JAX would recreate the host boundary.
- Use MCTX as a scratch lane only if CurvyTron is willing to own a JAX model and
  replay/training path.

### Central Inference Is A General RL Scaling Pattern

SEED RL is not MuZero-specific, but it is highly relevant systems evidence:
centralized accelerator inference plus optimized communication enabled millions
of frames per second, three-times-faster Atari wall time, and 40%-80% lower
experiment cost in the reported settings:
<https://research.google/pubs/seed-rl-scalable-and-efficient-deep-rl-with-accelerated-central-inference/>.

CurvyTron read:

- Shared inference service is not just a Go-engine trick.
- When actors are numerous and model inference is expensive, central batching
  beats sending tiny inference work through each actor.

### CUDA Graphs And `torch.compile` Help Only After Shape Discipline

PyTorch's CUDA Graphs writeup says graphs reduce launch overhead by replaying a
fixed set of kernels, with fixed arguments and memory addresses; static shapes
and static control flow are the usual preconditions:
<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>.

NVIDIA's CUDA Graph guidance makes the same point: graphs reduce cumulative
kernel launch overhead for many small operations, but the graph has constraints
around static memory, shape, synchronization, and control flow:
<https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html>.

CurvyTron read:

- This directly explains why eager dense Torch can win sim8 and lose sim16:
  many tiny kernels, dynamic indexing, and Python control can dominate.
- The viable test shape is fixed `N=B*2`, fixed `A=3`, fixed sim count,
  preallocated buffers, and masks instead of changing shapes.

### Even New MCTS Parallelism Is Architectural

The 2024 Speculative MCTS paper frames sequential MCTS latency as a main
AlphaZero training bottleneck. Its reported gains come from speculative
execution and NN-cache interaction: `5.81x` latency reduction in 9x9 Go and
`1.91x` end-to-end 19x19 training speedup versus KataGo:
<https://proceedings.neurips.cc/paper_files/paper/2024/hash/a19940b01b77b6acd41ff8b32b334e7c-Abstract-Conference.html>.

CurvyTron read:

- Mature search systems still need architectural parallelism for additional
  speed.
- A `1.9x` end-to-end speedup over KataGo is impressive precisely because the
  baseline already batches/caches heavily. This makes a `5x-10x` gain from
  wrapper tweaks around LightZero implausible.

## Answers To The Four Questions

### 1. How Do These Systems Get Speed?

They get speed by changing the unit of work:

- many actors or many active positions generate enough work;
- leaves/roots are batched before neural inference;
- the search tree is compact and compiled, or at least array-native;
- inference is shared through an evaluator/service rather than per actor;
- replay/materialization happens at a coarse boundary;
- caches avoid duplicate NN calls where the domain has transpositions or
  overlapping searches;
- static shapes and fixed buffers allow compilation, CUDA graphs, or hand-fused
  kernels to reduce launch overhead.

The consistent pattern is throughput architecture, not local wrapper polish.

### 2. Does This Support The Current CurvyTron Conclusion?

Yes.

The strongest evidence is LightZero itself: the current trusted loop is already
using a framework whose CTree internals are C++ and batched. If the best recent
full-loop wins are still around `1.28x-1.31x`, then the wall is not simply
"CTree should be C++"; it is the Python/list/object/model handoff and scalar
collector compatibility boundary.

The external systems that get large speedups either:

- move to many actors and batched inference/search services;
- make search state compiler-visible;
- or change algorithmic/sample-efficiency details, as KataGo did in its
  self-play acceleration work:
  <https://arxiv.org/abs/1902.10565>.

That supports this working thesis:

```text
1x-2x: wrapper, tensor lifetime, output fast-path, CTree boundary cleanup
2x-4x: fixed-shape dense/compiled search or array-native CTree
5x-10x: service-shaped batching, many actors/roots, compact replay boundary,
        and/or learner/collector architecture changes
```

The exact multipliers are hypotheses, but the ordering is well supported.

### 3. Which Architecture Should CurvyTron Test First?

Test first:

```text
fixed-shape compiled/fused dense Torch/Triton/CUDA-graph search
```

Reason:

- It is the shortest real-search experiment from the current repo state.
- CurvyTron has tiny fixed action count `A=3`, which is favorable for dense
  arrays.
- The existing dense Torch prototype already beat the practical CTree row at
  sim8 in profile-only mode, but failed the sim16 scaling gate.
- A compile/fusion spike directly tests whether the sim16 failure is launch and
  Python-control overhead.
- It does not require a LightZero fork, JAX model port, or actor service rewrite
  before the bottleneck is proven.

Recommended order:

1. Run a fixed-shape dense compile/fusion spike at sim16.
2. If it wins materially, continue toward a compiled dense search engine.
3. If it fails, stop polishing eager dense Torch and choose between:
   - array-native CTree if LightZero semantic compatibility is the priority;
   - MiniZero/KataGo-style search service if `5x-10x` training speed is the
     priority.
4. Keep MCTX/JAX as a scratch reference, not the first production lane.

Architecture judgment:

| Lane | Expected role | Critique |
| --- | --- | --- |
| Fixed-shape compiled dense Torch/Triton | First real-search falsifier | Best fit for `A=3`; validates/falsifies launch-overhead theory quickly. |
| Array-native CTree | Conservative compatibility bridge | Good for reducing list/object churn; unlikely alone to deliver `5x-10x`. |
| MiniZero/KataGo-style search service | First serious production-scale architecture | Best match to external systems; bigger rewrite, but plausible full-loop multiplier. |
| MCTX/JAX scratch | Clean endpoint/reference | Strong design target; costly unless model/search/replay move to JAX together. |
| Full C++/CUDA trainer rewrite | Later, if needed | Too broad before proving the right boundary and batching shape. |

### 4. Smallest Falsifier Experiment

Run a two-row ceiling plus one real-search row on the same denominator:

```text
H100
B512 / A16
sim16
60 measured / 15 warmup
root-noise0
uint8 stack
scalar materialization off where the profile lane allows it
```

Rows:

1. `direct_ctree_gpu_latent` repeat: practical LightZero-shaped baseline.
2. `search_service_mock_ceiling`: profile-only shim that consumes the same
   batched observations/legal masks and returns compact action/visit/value
   arrays without LightZero CTree, without per-root object packaging, and with
   deterministic legal actions or prerecorded visit/value tensors.
3. `dense_torch_mcts_compile_spike`: fixed `N=B*2`, fixed `A=3`, fixed sim16,
   preallocated buffers, root-noise0, no dynamic root filtering, no `.item()`
   gates, model recurrent eager at first, compiled/fused selection and backup.

Decision rule:

- If `search_service_mock_ceiling` is not close to the desired `5x-10x`
  full-loop target versus stock, then search/collection architecture alone
  cannot deliver the target; look at env stepping, learner, replay, multi-actor
  scale, or algorithmic changes.
- If the mock ceiling is high but `direct_ctree_gpu_latent` remains near current
  wins, the radical boundary thesis is validated: the missing multiplier lives
  in the LightZero-shaped search/collection boundary.
- If `dense_torch_mcts_compile_spike sim16` beats `direct_ctree_gpu_latent sim16`
  by a material margin, continue dense compiled search.
- If compiled dense sim16 cannot beat `direct_ctree_gpu_latent`, stop treating
  eager/dense Torch as the production path and move to array-native CTree or a
  MiniZero/KataGo-style service.

Practical threshold from the local dense compile note:

```text
compiled dense sim16 must beat the direct_ctree_gpu_latent sim16 comparator,
roughly 6.1k roots/sec on the fresh profile denominator.
```

A strong result would be `7k-8.5k` sim16 roots/sec. A weak sim8-only win should
not promote the lane.

## Recommendation

Do not start with a broad rewrite. Start with a falsifier that prices the
architecture:

1. Add/run the profile-only mock search-service ceiling row.
2. Add/run the fixed-shape compiled dense sim16 row.
3. Use the result to choose:
   - compiled dense search if sim16 wins;
   - array-native CTree if compatibility wins and expected upside can be modest;
   - MiniZero/KataGo-style batched search service if the ceiling says the
     full-loop multiplier exists but LightZero-shaped paths cannot reach it.

My critique of the current thesis:

```text
Correct direction, but sharpen the proof.
```

The external literature strongly supports "large speedups require architecture",
but the smallest next step should be a ceiling experiment that proves the
target multiplier is actually inside the search/collection boundary before
CurvyTron commits to a service rewrite.

