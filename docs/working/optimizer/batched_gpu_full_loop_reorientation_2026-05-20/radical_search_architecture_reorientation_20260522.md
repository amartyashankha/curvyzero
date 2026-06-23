# Radical Search Architecture Reorientation

Date: 2026-05-22

Status: active optimizer working memory. No live Coach runs touched.

## Plain Read

The current optimization is real but too small to be the endgame.

Matched full-loop profile rows now say:

```text
no-RND:       stock 433 steps/sec -> direct output-fast 566 steps/sec, about 1.31x
rnd_meter:   stock 351 steps/sec -> direct output-fast 449 steps/sec, about 1.28x
```

That proves the collect/search boundary matters. It also proves the current
patch is not the 5-10x move.

The reason is simple:

```text
direct_ctree_gpu_latent keeps latents on GPU,
but CTree/search still crosses Python, CPU, NumPy/list, and object APIs every
simulation.
```

So the next question is no longer "can we shave another small wrapper cost?"
The next question is:

```text
Can we keep many CurvyTron roots alive in one compact batch through search,
model calls, and replay-shaped output?
```

## Why The Past Changes Were Small

Most past patches kept the stock LightZero topology:

```text
scalar env rows
-> public LightZero collect/search wrapper
-> CPU/list-shaped CTree API
-> per-env output dicts
-> stock replay/learner
```

Those patches can remove obvious waste, but they do not change the main shape
of the work. Once rendering got cheaper, the wall moved to collect/search.
Once `direct_ctree_gpu_latent` removed the latent CPU round trip, the wall
moved to CTree/list/model-output/output boundaries.

This is Amdahl's Law in the current currency:

```text
Optimizing a component that is no longer most of wall time gives only a small
whole-loop speedup.
```

## External Pattern Refresh

The external systems point in the same direction:

- OpenSpiel's AlphaZero docs distinguish a slow Python path from a faster C++
  path that uses threads, a shared cache, batched inference, and GPU
  inference/training:
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>.
- MiniZero's self-play workers keep multiple MCTS instances alive and evaluate
  collected leaves with batched GPU inference:
  <https://github.com/rlglab/minizero>.
- KataGo's parallel analysis engine is faster when it can batch many positions
  across a modern GPU, and it also uses an NN result cache:
  <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>.
- DeepMind's MCTX is JAX-native, batch-first, and JIT-compatible:
  <https://github.com/google-deepmind/mctx>.
- PufferLib is another relevant fast-RL architecture. Its public docs describe
  static memory allocation, CUDA graph replay, pinned-memory async transfer,
  chunked vectorized environment buffers, CUDA streams, and C environments that
  write observations directly into contiguous training buffers:
  <https://puffer.ai/docs.html>. Its blog describes native environments that
  write observations directly into shared-memory batches with no redundant
  copies, plus asynchronous sampling where policy inference and environment
  stepping overlap:
  <https://puffer.ai/blog.html>.
- AlphaZero and MuZero used large separate self-play and training compute pools,
  not a tiny scalar synchronous Python loop:
  <https://arxiv.org/abs/1712.01815>,
  <https://arxiv.org/abs/1911.08265>.

Plain translation for CurvyTron:

```text
The multiplier comes from batch shape and boundary ownership, not from merely
saying "GPU" or "C".
```

LightZero already uses C++ CTree for core traverse/backprop. The remaining wall
is the boundary around that C++: Python control, list conversion, device
transfers, per-simulation model calls, and public per-env output assembly.

PufferLib changes the framing slightly:

```text
If CurvyTron stays inside a scalar Python env API, we should expect small wins.
The big win needs a native/vector environment contract where observations,
actions, rewards, and terminals live in contiguous buffers and the trainer
consumes those buffers directly.
```

That is compatible with the search-service thesis. The bigger architecture is
not only "better MCTS"; it is:

```text
contiguous CurvyTron actor batch
-> contiguous observation/action-mask buffers
-> batched policy/search service
-> compact replay/target rows
-> scalar objects only for compatibility, logging, or eval
```

## Architecture Options

| option | expected role | upside | risk |
| --- | --- | ---: | --- |
| `direct_ctree_gpu_latent` promotion | tactical LightZero-compatible bridge | `~1.3x` full-loop profile so far | must pass semantic gates before Coach use |
| fixed-shape compiled/fused dense search | fastest high-upside falsifier | if sim16 beats GPU-latent, maybe `1.5x+` over current direct and a path toward more | semantics and torch compile can break |
| array-native fixed-A=3 CTree | conservative CTree boundary cleanup | likely `1.2x-2x` over current direct if list/API churn is large | Cython/C++ vendoring, still CPU tree |
| MiniZero/KataGo-style search service | real 5-10x-class architecture candidate | many roots, batched leaf inference, compact outputs | bigger rewrite, new collector/replay contracts |
| MCTX/JAX scratch lane | clean all-device reference | can test batch/JIT search shape | not a drop-in PyTorch/LightZero patch |

## Current Falsifier Ladder

Do not jump straight to a huge rewrite. Run small, sharp tests that can kill
bad ideas quickly.

1. **Train-facing validation gate.**
   Prove the profile hook returns stock-shaped output and never silently falls
   back. Exact gates are for forced masks, clear preferences, values/logits,
   schema, and target rows. Ordinary tie-heavy visits stay statistical.

2. **Fixed-shape dense compile spike.**
   This falsifier has now run. It was good at sim8 and failed at sim16:
   `dense_torch_mcts_compile_spike` reached `4872.70` roots/sec at sim16,
   while `direct_ctree_gpu_latent` reached `6153.95` on the same H100 shape.

   Decision: stop polishing this exact helper. The failure mode is dynamic
   search/update shape, mutated tree buffers, and recompiles around the
   simulation-depth loop.

3. **Array-native CTree design after dense compile failed.**
   Keep real LightZero CTree semantics but replace nested Python lists with
   dense `[N,3]` arrays for root prep, recurrent reward/value/policy inputs,
   and visit/value output.

4. **Mock search-service ceiling now.**
   Before committing to a large rewrite, build a profile-only ceiling row that
   consumes real batched observations and legal masks, skips CTree/search, and
   returns compact legal action/visit/value arrays. If this ceiling is not far
   above the current direct hook in the relevant denominator, a search-service
   rewrite cannot deliver the desired multiplier by itself.

   First durable H100 result:

   ```text
   B512/A16, 60 measured, 15 warmup, scalar materialization off:
     mock_search_service sim16:       11648.29 roots/sec
     direct_ctree_gpu_latent sim16:    5303.97 roots/sec
     recurrent_toy sim16:              8512.57 roots/sec
   ```

   Plain read:

   ```text
   compact search-service shape is about 2.20x above current direct in this
   profile. That is real, but it is not enough to claim 5-10x by search alone.
   ```

5. **Batched search service sketch.**
   If the mock ceiling is high and the array-native lane looks capped, design
   the larger MiniZero-shaped worker: many row/seat roots alive, batched
   recurrent inference, compact action/visit/value chunks, and replay
   materialization at the edge.

6. **Native/vector environment ownership.**
   PufferLib makes this unavoidable as a serious option. If the search-service
   ceiling is only about `2x`, the remaining multiplier must come from broader
   topology: contiguous env/observation buffers, fewer Python objects, async
   collection/model overlap, and compact replay. This is a larger rewrite than
   array-native CTree.

## What We Should Stop Doing

- Do not call renderer-only rows the whole-loop answer.
- Do not call roots/sec profile-only rows Coach speed.
- Do not expect more CPU cores to fix a Python/list/sync topology.
- Do not promote dense eager Torch just because sim8 looked good; sim16 already
  exposed the scaling issue.
- Do not require exact neutral/tie-heavy visit parity; stock LightZero itself
  can differ there.

## Immediate Decision

The best current order after the compile falsifier is:

```text
finish P0 validation -> use durable mock/direct/recurrent results -> decide
whether the next implementation is array-native CTree, compact search service,
or native/vector env-buffer prototype
```

The thesis to test is:

```text
Small patches give 1-2x.
Real 5-10x requires compact batched ownership across environment observation,
search, and replay boundaries.
```

That is now the optimizer lane.
