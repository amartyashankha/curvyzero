# GPU MCTS Current Flow Explainer, 2026-05-22

Status: active optimizer working memory.

Purpose: plain-language map of the current GPU/MCTS work. This is meant to
avoid mixing together three different lanes:

1. stock LightZero training;
2. the train-facing Torch `direct_ctree_gpu_latent` bridge;
3. newer profile-only dense Torch and MCTX/JAX search experiments.

## Bottom Line

The whole MCTS trainer is not yet on GPU.

The latest code has several GPU-search experiments, but they are at different
levels of reality:

```text
Stock training:
  Torch model can run on CUDA.
  Tree search is still LightZero CPU CTree.
  This is the trusted training path.

direct_ctree_gpu_latent:
  Torch model tensors and latent states stay on CUDA longer.
  CTree traversal/backprop is still CPU C++/Cython with Python/list edges.
  This is profile-gated, not normal Coach advice.

dense_torch_mcts:
  A fixed-A=3 tree is represented as Torch tensors on device.
  This is a profile-only ceiling/probe, not a production trainer.

MCTX/JAX:
  MCTS search runs inside JAX/XLA/MCTX on GPU.
  It can consume real compact CurvyTron visual roots.
  Older rows used a toy JAX model.
  Current rows also include a real immutable-checkpoint JAX shadow model.
  Neither path calls train_muzero.
  The real-shadow same-root comparator has semantic deltas.
```

The newest high-upside result is the MCTX/JAX compact-root lane. It shows
10x-class search-boundary headroom, but the repeated closed compact loop then
shows the next wall: env step, observation/stack update, replay-index
construction, and host/device synchronization around the fast search.

## Glossary

MCTS: Monte Carlo Tree Search. The policy asks, "if I try actions and imagine
what follows, which action looks best?"

Root: one current position we are deciding from. In CurvyTron profiles one env
row has two player views, so `[B, 2, 4, 64, 64]` becomes `B * 2` roots.

Latent: the neural network's hidden representation of an observation. It is
smaller than the raw image stack and is what MuZero's recurrent model expands.

CTree: LightZero's C++/Cython CPU tree implementation. It is faster than pure
Python, but its public API still passes Python lists/NumPy arrays around.

MCTX: DeepMind's JAX MCTS library. It is batch-first and JIT-compiled, so the
search loop and tree-shaped tensors can live inside XLA on the GPU.

Compact batch: our flattened row/player contract that avoids one Python object
per env. The important structs are `HybridCompactBatch`, `CompactRootBatchV1`,
`CompactSearchResultV1`, and `CompactReplayIndexRowsV1`.

Flat A=3: CurvyTron has exactly three actions. The flat-A3 CTree spike passes
`float32[N]` rewards, `float32[N]` values, and `float32[N,3]` policies instead
of nested Python policy lists.

## Lane 1: Stock LightZero Training

This is still the trusted training path.

```text
CurvyTron env/scalar LightZero observations
-> policy.collect_mode.forward / MuZeroPolicy._forward_collect
-> Torch initial_inference on CUDA if cuda=True
-> LightZero CTree roots/search on CPU
-> Torch recurrent_inference calls during search
-> CPU CTree backprop
-> per-env action dicts
-> scalar env manager, replay, learner, RND
```

The config builder sets CUDA for the policy/model when requested, but that does
not mean the CTree itself is on GPU.

Current measured profile-level gain from the direct bridge is real but modest:

```text
no-RND matched full loop:
  stock:              433.17 steps/sec
  direct output-fast: 566.19 steps/sec
  gain:               about 1.31x

RND hash-fixed matched full loop:
  stock:              351.02 steps/sec
  direct:             448.52 steps/sec
  gain:               about 1.28x
```

That is expected because the old LightZero object shape is still in the
denominator: env ids, dict observations, BaseEnvTimestep objects, CPU CTree
roots, Python list-shaped CTree APIs, replay segments, learner batches, and RND
CPU extraction/hashing.

## Lane 2: `direct_ctree_gpu_latent`

This bridge patches `MuZeroPolicy._forward_collect` in profile mode.

What it improves:

```text
Initial inference produces root latent tensors on CUDA.
A CUDA latent_pool stores latent states for every simulation depth.
CTree returns path indices and actions.
Torch gathers latent_pool[path, batch] on CUDA.
Torch recurrent_inference runs on CUDA.
The next latent state is copied into latent_pool.
```

What is still CPU:

```text
CTree root objects.
Legal action lists.
CTree traversal.
Min/max stats.
Backprop through the tree.
Root visit distributions and values.
Reward/value/policy payloads copied back to CPU every simulation.
```

So the bridge is better described as:

```text
GPU model and latent pool, CPU tree.
```

It is not:

```text
GPU-native MCTS.
```

## Lane 3: Flat-A3 Vendored CTree

Flat-A3 is a CPU CTree API cleanup, not a GPU tree.

Normal LightZero CTree backprop receives nested Python lists:

```text
reward list
value list
policy list of lists
```

Flat-A3 receives contiguous arrays:

```text
reward[N] float32
value[N] float32
policy[N,3] float32
```

That removes one list-shaped payload boundary for CurvyTron's fixed action
space. It was a good falsifier and showed no-model CTree speedups, but matched
full-loop rows did not improve enough:

```text
direct LightZero CTree: 516.55 steps/sec
flat-A3 CTree:          509.69 steps/sec
```

Plain read: flat-A3 proved list ABI cost exists, but this alone is not where
the next big full-loop win lives.

## Lane 4: Dense Torch MCTS

This is the closest Torch-side "tree as tensors" implementation.

It allocates fixed-size device tensors like:

```text
edge_child[root, node, action]
edge_visit[root, node, action]
edge_value_sum[root, node, action]
edge_reward[root, node, action]
edge_prior[root, node, action]
latent_pool[node, root, ...]
node_latent_slot[root, node]
next_node_index[root]
```

Selection and backup are vectorized Torch functions over roots and actions.
The tree shape is flattened into arrays instead of pointer-heavy CTree nodes.

This is profile-only. It is useful because it answers, "what if the fixed-A=3
tree lived in tensors instead of CPU CTree objects?" It is not yet a validated
trainer replacement.

## Lane 5: MCTX/JAX Compact Visual Roots

This is the newest direction with the clearest search-boundary headroom.

The important profile flow is:

```text
HybridCompactBatch real [B,2,4,64,64] visual observations
-> CompactRootBatchV1
-> jax.device_put observations and invalid-action mask
-> jitted representation/prediction/recurrent functions
-> mctx.gumbel_muzero_policy
-> actions, action weights, root values
-> CompactSearchResultV1 validation
-> selected actions step compact env once
-> CompactReplayIndexRowsV1
```

What runs on GPU:

```text
JAX visual encoder or synthetic representation.
JAX recurrent function.
MCTX search loop and tree arrays inside XLA.
Invalid-action masking inside the JAX/MCTX search.
```

What is still not real production:

```text
The model is a toy JAX model, not the current LightZero PyTorch model.
It does not call train_muzero.
It does not prove learning.
Replay, RND, and learner ownership are not integrated as a production path.
```

Current useful result:

```text
H100 B512/P2 real compact visual roots:
  sim16/h64/v8 fresh-boundary: 124,090 roots/sec
  sim16/h64/v8 resident:       167,516 roots/sec
  sim32/h64/v8 fresh-boundary: 51,454 roots/sec
  sim32/h64/v8 resident:       65,228 roots/sec
```

Plain read: the search can be very fast if the tree/search/model shape is
owned by JAX/XLA. But search-only roots/sec is not full training speed.

## The Current Wall After MCTX

The repeated closed compact loop added the surrounding edge back into the
denominator:

```text
build root batch
-> move observations/masks or reuse resident stack
-> MCTX search
-> copy actions/weights back
-> step compact env
-> build replay index rows
-> repeat
```

Those rows were much slower than the search-only numbers. Recent closed-loop
active-root rates were roughly:

```text
B256/P2/sim16/loop4:   3.25k roots/sec
B512/P2/sim16/loop8:   5.06k roots/sec
B512/P2/sim32/loop8:   4.87k roots/sec
B1024/P2/sim16/loop8:  6.41k roots/sec
B1024/P2/sim32/loop8:  5.03k roots/sec
```

The follow-up native actor-buffer rows helped:

```text
B512/P2/sim16/loop8:   5.79k -> 6.82k roots/sec
B1024/P2/sim16/loop8:  6.25k -> 8.92k roots/sec
```

But the env/observation edge still dominates those closed-loop rows. That means
the next work should not be "make MCTX search alone even faster." It should be
the compact loop around it: env state, observation/stack ownership, replay
index rows, RND latest-frame input, target materialization, and learner samples.

## Code Map

Train-facing bridge:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
  _install_lightzero_collect_search_backend_hook
  _direct_ctree_gpu_latent_search_for_collect
```

Profile boundary and dense Torch path:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
  _LightZeroCollectForwardStackProbe
  _run_direct_mcts_arrays
  _run_direct_ctree_gpu_latent_search
  _run_dense_torch_mcts
```

Vendored flat-A3 CPU CTree:

```text
src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/mz_tree_a3.pyx
src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/lib/cnode.cpp
src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/lib/cnode.h
scripts/build_lightzero_ctree_a3.py
scripts/benchmark_lightzero_ctree_no_model.py
```

MCTX/JAX lane:

```text
src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
```

Compact contracts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
  HybridCompactBatch

src/curvyzero/training/compact_policy_row_bridge.py
  CompactRootBatchV1
  CompactSearchResultV1
  CompactReplayIndexRowsV1
```

## What To Say Carefully

Correct:

```text
We have a profile-only MCTX/JAX lane that moves search/tree arrays onto GPU and
shows strong search-boundary headroom on real compact visual roots.
```

Correct:

```text
We have a train-facing profile hook that keeps Torch latents on CUDA while
still using CPU CTree.
```

Incorrect:

```text
The production trainer's MCTS tree is now on GPU.
```

Incorrect:

```text
The MCTX results prove the next Coach run will train 10x faster.
```

Current next proof:

```text
Make the repeated compact loop fast enough that MCTX's search win survives the
surrounding env/observation/replay/RND/learner edge, then test current-model
realism.
```
