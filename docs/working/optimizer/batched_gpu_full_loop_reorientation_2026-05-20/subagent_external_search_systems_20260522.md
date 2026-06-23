# External Search Systems Read

Date: 2026-05-22

Scope: external-pattern research for the CurvyTron optimizer lane. No
production code, trainer defaults, tournament defaults, or live runs changed.

## Short Answer

Do not frame the next fix as "move the bottleneck to C".

The real pattern in fast AlphaZero/MuZero-style systems is:

```text
many roots / games / positions alive at once
-> batched neural inference
-> compact tree/search state
-> minimal Python/object/list churn in the inner loop
-> replay rows only at a coarse compatibility edge
```

LightZero already uses C++ for the CTree traversal/backprop pieces. CurvyTron's
current speed wall is the boundary around search: public LightZero wrappers,
Python per-simulation control, CPU/list CTree APIs, recurrent model output
copies, dynamic indexing, small GPU kernels, and scalar/timestep packaging.
GPU search is not automatically much faster because a tiny-action MCTS loop can
be launch/sync/control-bound even when the model itself is fast.

## Source Patterns

### OpenSpiel AlphaZero: C++ Threads, Cache, Batched Inference

OpenSpiel's AlphaZero docs describe the same conceptual components in Python
and C++: actors generate self-play with MCTS plus a neural evaluator, a learner
updates the network, and evaluators measure progress. The important systems
distinction is that Python is CPU-only and does not batch inference, while the
C++ implementation uses threads, a shared cache, batched inference, and GPU
inference/training support:
<https://openspiel.readthedocs.io/en/latest/alpha_zero.html>.

CurvyTron read:

- This supports replacing scalar LightZero-facing collection with a batched
  actor/search/evaluator boundary.
- The shared-cache/batched-evaluator idea is more relevant than a blind C port.
  C helps when it removes the per-root Python boundary and feeds bigger NN
  batches.
- Near-term analogy: compact arrays in/out around search, not per-env
  `BaseEnvTimestep` objects in the hot path.

### MiniZero: Multiple MCTS Instances Per Worker, Batched GPU Leaf Eval

MiniZero's README says each self-play worker maintains multiple MCTS instances,
selects leaf nodes across them, and evaluates the leaves with batched GPU
inference:
<https://github.com/rlglab/minizero>.

CurvyTron read:

- This is the cleanest external match for "why one GPU search call is not much
  faster": GPU benefit comes from collecting enough leaf/root work to batch.
- For CurvyTron, the natural batch is physical rows `[B]`, seats `[2]`, and
  active roots `[B*2]`. Do not let that collapse back to scalar dicts before
  search.
- The production-scale version is a MiniZero-like search service: many active
  roots, one batched recurrent inference per simulation step, compact visit
  arrays back.

### KataGo: Cross-Position Batching, NN Cache, Thread Tuning

KataGo's analysis engine is built to analyze many positions in parallel and can
be much faster than single-position GTP because it exploits cross-position
batching:
<https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>.
Its docs also describe an asynchronous JSON protocol and an NN result cache
used to skip duplicate neural queries.

KataGo's analysis config makes the batching point explicit: users tune the
number of parallel positions, per-position search threads, and `nnMaxBatchSize`;
the comments note that GPU batching is usually the dominant consideration:
<https://github.com/lightvector/KataGo/blob/master/cpp/configs/analysis_example.cfg>.
The repo README also advertises a match engine where bots share GPU batches and
CPU resources:
<https://github.com/lightvector/KataGo>.

CurvyTron read:

- A single fast GPU is not enough if the search topology issues tiny requests.
- Parallel positions are better than too many threads fighting inside one tree.
  For CurvyTron, that argues for many row/seat roots in one search batch rather
  than optimizing one scalar root at a time.
- NN cache/transposition is less directly useful for CurvyTron than for Go, but
  compact memoized root/model outputs may still help profile-only repeated-state
  controls.

### LightZero: CTree Is Already C++ And Batched

LightZero's docs say MuZero CTree uses C++ for `batch_traverse` and
`batch_backpropagate`, and the search method handles batches of roots with
parallel model inference:
<https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>.
The LightZero README describes MCTS implementations in Python `ptree` and C++
`ctree`:
<https://github.com/opendilab/LightZero>.

CurvyTron read:

- "Use C++" is already partly true. The remaining target is the API boundary
  and loop shell around CTree.
- Array-native CTree APIs for fixed `A=3` can be a useful conservative lane:
  fewer lists, fewer `.tolist()` conversions, fewer CPU/GPU round trips.
- But a CPU-owned tree still needs recurrent model synchronization each
  simulation. This is a plausible `>2x` cleanup path, not obviously a `5x-10x`
  endpoint.

### MCTX: JAX-Native, JIT-Compiled, Batch-First MCTS

DeepMind's MCTX is a JAX-native MCTS library for AlphaZero, MuZero, and Gumbel
MuZero. Its README says search supports JIT compilation and operates on batches
of inputs in parallel:
<https://github.com/google-deepmind/mctx>.

CurvyTron read:

- This is the clean device-resident search reference: root arrays, recurrent
  function, and tree state are all compiler-visible.
- It is not a small LightZero patch. A PyTorch model called from JAX would
  recreate the host boundary.
- Use MCTX as the design target for a scratch/profile lane or a future rewrite:
  fixed shapes, masks for invalid roots/actions, batch-first search outputs.

### CUDA Graphs / Static Shapes: Useful Only After The Loop Is Shape-Stable

NVIDIA's CUDA Graph guidance says graphs reduce cumulative kernel launch
overhead, especially for many small operations, but require constraints such as
static memory addresses, static shapes, and limited dynamic control flow:
<https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/cuda-graph.html>.
The PyTorch CUDA Graphs writeup shows the common trick: keep tensor sizes
static and use masks to mark valid entries instead of changing shapes:
<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>.

CurvyTron read:

- Dense eager Torch MCTS can lose at higher sim counts because many small tensor
  ops and dynamic indexing become launch/control overhead.
- The CurvyTron-friendly graph shape is fixed `N=B*2`, fixed `A=3`, fixed
  `num_simulations`, fixed buffers, and masks for live roots/legal actions.
- CUDA graphs or `torch.compile(mode="reduce-overhead")` are worth testing only
  after the dense search loop stops allocating and changing shapes in the hot
  path.

## Why GPU Search Is Not Much Faster Yet

1. The model root pass is already fast. Local docs show the wall after root
   inference, inside collect/search/output.
2. LightZero CTree keeps core tree operations in C++, but the surrounding loop
   still crosses Python, CPU, NumPy/list, and GPU/CPU boundaries.
3. Dense Torch search at small `A=3` does not automatically saturate a GPU.
   Without compilation/fusion, it can become many small launches and sync-prone
   dynamic indexing.
4. Static-shape tricks need a static loop. Dynamic live-root sets, `nonzero`,
   `.item()` gates, variable legal lists, and per-depth allocation all fight
   CUDA graph capture.
5. Cross-position batching is the real multiplier. MiniZero and KataGo keep
   many MCTS instances or positions active so the NN evaluator sees real
   batches.

## What Each Pattern Means For CurvyTron

| Pattern | CurvyTron implication |
| --- | --- |
| OpenSpiel C++ AlphaZero | Separate actor/search/evaluator/learner roles; use threads/cache/batched inference if we leave stock scalar collection. |
| MiniZero self-play workers | Keep many roots alive and batch leaf recurrent inference; a search service beats scalar root polishing. |
| KataGo analysis/match engines | Parallelize across positions/row-seat roots; tune batch fill and thread/search balance around GPU batch size. |
| LightZero CTree | Do not rewrite everything to C; first remove list/API/copy churn around the existing C++ tree. |
| MCTX/JAX | The clean endpoint is batch-first, JIT-visible search; not drop-in for current PyTorch/LightZero. |
| CUDA graphs/static masks | Fixed `B`, `A`, sim count, buffers, and masks are required before launch-overhead tools help. |

## Recommendation Order

1. **Keep current direct/GPU-latent CTree validation moving.** It is the closest
   LightZero-compatible path and directly attacks the proven public wrapper and
   latent CPU round-trip wall. Finish forced-case parity, root-noise/eval
   gates, and matched stock-vs-direct comparisons before any trainer claim.

2. **Make dense GPU MCTS fixed-shape before judging it.** The external pattern
   is not "eager Torch on GPU"; it is static, batched, compiler-visible search.
   For the profile lane, freeze `N=B*2`, `A=3`, `sim`, buffers, legal masks,
   live-root masks, and path arrays, then test `torch.compile`/CUDA graphs or
   Triton-style fusion. This is the best way to answer whether GPU-resident
   search can beat CTree at sim16+.

3. **If dense compiled search stalls, build array-native CTree for `A=3`.**
   This is the conservative C lane: keep LightZero semantics and CTree stats,
   but replace Python lists/root output fanout with dense arrays. Expected
   upside is boundary cleanup, not a full architecture jump.

4. **Prototype a MiniZero/KataGo-style batched search service.** Many active
   CurvyTron row-seat roots feed one evaluator/search worker, which returns
   compact action/visit/value arrays. This is the first architecture that can
   plausibly turn GPU inference/search batching into a durable full-loop win.

5. **Treat MCTX/JAX as a scratch architecture lane, not a patch.** It is the
   right all-device reference if CurvyTron is willing to own model/search/replay
   semantics in JAX. It is not the shortest route from stock LightZero.

6. **Defer full C++/CUDA env/search integration until the search boundary is
   settled.** A C rewrite can still be slow if it issues tiny GPU requests or
   re-enters Python every simulation. The next proof should remove boundaries,
   not merely change the implementation language.

## Bottom Line

For CurvyTron, the bottleneck should move to a batched, fixed-shape search
boundary, not simply to C. C is useful when it makes the boundary array-native.
GPU is useful when it sees enough roots/leaves and a stable enough loop to
avoid launch/sync churn. The next serious speed path is therefore:

```text
compact batched CurvyTron rows
-> uint8/device observation stack
-> batched root model
-> GPU-latent CTree or fixed-shape dense search
-> compact visit/action/value arrays
-> replay/timestep materialization only at the compatibility edge
```

That matches the real external systems better than another renderer-only pass
or a broad "rewrite in C" push.
