# Search Boundary Escape Plan - 2026-05-21

## Plain Summary

The `1.8x` result is not a hardware limit. It is a boundary limit.

`direct_ctree_arrays` removed a lot of public LightZero wrapper work, but it
still uses LightZero's MuZero CTree in the normal shape:

```text
Torch model on GPU
-> copy latent/logits/value to CPU
-> Python/NumPy/list root setup
-> Python loop over num_simulations
-> C++ batch_traverse
-> Python gathers latent states
-> Torch recurrent_inference on GPU
-> copy recurrent outputs to CPU
-> Python list conversion
-> C++ batch_backpropagate
-> Python/list root output
```

LightZero already has C++ tree kernels. The remaining wall is that the kernels
are surrounded by Python and list conversion every simulation. So the useful
question is not "should we use C++?" We already do. The useful question is:

```text
How much of the CTree/search boundary can become array-native or device-native?
```

## Current Evidence

Late P2 H100 profile-only boundary rows:

| probe | throughput | read |
| --- | ---: | --- |
| stock public facade | `2670.68 roots/s` | baseline collect/search wrapper |
| `direct_ctree_arrays`, host uint8 | `4764.06 roots/s` | about `1.78x` faster |
| `direct_ctree_arrays`, pinned uint8 | `3689.15 roots/s` | input copy is not the wall |
| `direct_ctree_arrays`, stale resident input | `3069.08 roots/s` | no useful win here |

Best direct row breakdown:

| bucket | time | read |
| --- | ---: | --- |
| measured wall | `12.90s` | profile window |
| direct boundary | `8.20s` | policy/search boundary |
| MCTS search call | `6.06s` | main direct-lane wall |
| model calls | `0.85s` | small |
| observation | `2.08s` | meaningful but not first wall |
| H2D input | `1.05s` | visible but not dominant |
| root prep | `0.48s` | visible but not enough alone |
| output assembly | about `0.05s` | old trap mostly fixed |

Important caveat: these are profile-only boundary rows, not Coach launch advice.
The full-loop one-process batched GPU manager is a different denominator and is
not currently the trusted training path.

## Code Pointers

Local direct boundary:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  around `_run_direct_mcts_arrays`.
- It calls `model.initial_inference`, then converts `pred_values`,
  `latent_state_roots`, and `policy_logits` to CPU NumPy arrays.
- It builds `policy_logits_np.tolist()`, `legal_actions`, `noises`, `Roots`,
  then calls `mcts.search`.

Installed LightZero:

- `.venv/lib/python3.11/site-packages/lzero/policy/muzero.py`
  `_forward_collect` does the public root setup and one output dict per env.
- `.venv/lib/python3.11/site-packages/lzero/mcts/tree_search/mcts_ctree.py`
  `MuZeroMCTSCtree.search` loops in Python over `num_simulations`.
- `.venv/lib/python3.11/site-packages/lzero/mcts/ctree/ctree_muzero/mz_tree.*.so`
  is the compiled Cython extension.

Upstream LightZero source cloned at `/private/tmp/LightZero`:

- `lzero/mcts/ctree/ctree_muzero/mz_tree.pyx`
- `lzero/mcts/ctree/ctree_muzero/lib/cnode.cpp`
- `lzero/mcts/ctree/ctree_muzero/lib/cnode.h`

The Cython wrapper currently accepts Python lists for root prepare,
backpropagate inputs, and output extraction. The C++ root object stores
`std::vector<std::vector<int>>` legal actions and returns
`std::vector<std::vector<int>>` visit distributions.

## Why A Small Patch Will Not Give 10x

Root prep and final output are no longer the whole wall. A wrapper that only
replaces:

```text
policy_logits_np.tolist()
legal_actions = [...]
roots.get_distributions()
roots.get_values()
```

can help, but it cannot remove the `6s` search bucket by itself.

Inside `MuZeroMCTSCtree.search`, every simulation still does:

1. allocate a `ResultsWrapper`;
2. call C++ `batch_traverse`;
3. return Python lists of leaf indices/actions;
4. gather latent states in Python;
5. build a Torch tensor from NumPy;
6. run `model.recurrent_inference`;
7. copy latent/logits/value/reward back to CPU;
8. convert reward/value/policy arrays to Python lists;
9. call C++ `batch_backpropagate`.

That loop is the real CTree boundary wall.

## Fix Lane A: Array-Native Cython Boundary

Goal: keep LightZero's C++ tree semantics, but stop making Python lists for
fixed small action space roots.

Prototype shape:

```text
prepare_roots_arrays(
    policy_logits: float32[N,3],
    rewards: float32[N],
    noises: float32[N,3],
    action_mask: uint8[N,3],
    to_play: int32[N],
) -> Roots

roots_get_arrays(Roots) -> visits:int32[N,3], values:float32[N]
```

Expected win:

- If it only attacks root prep/output: probably `5-15%` over current direct.
- If it also removes per-simulation `tolist()` in backpropagate: maybe
  `20-40%` over current direct.
- It is not enough for 10x, but it is a good quick falsifier.

Implementation options:

- Quickest: vendor a small Cython extension in our repo based on upstream
  LightZero `mz_tree.pyx` and `cnode.cpp/h`, with added array functions for
  action_count=3.
- Cleaner but slower organizationally: patch LightZero upstream or install a
  forked LightZero build in the Modal image.
- Avoid pybind for the first pass. LightZero already uses Cython and the source
  is Cython/C++; extending the existing style is lower risk.

Validation:

- no-noise fixed seed exact parity against current direct;
- root-noise statistical parity;
- illegal action visit mass zero;
- all-actions-legal fast path;
- masked action path;
- support-scale/value parity.

## Fix Lane B: Cythonize The Search Loop

Goal: remove most per-simulation Python/list churn while still letting PyTorch
do model inference.

Prototype shape:

```text
for sim in range(num_simulations):
    parent_idx, batch_idx, actions, virtual_to_play = ctree_traverse_arrays(...)
    latent_batch = latent_store[parent_idx, batch_idx]  # vectorized
    network_output = recurrent_inference(latent_batch, actions)
    ctree_backpropagate_arrays(reward, value, logits, ...)
```

This does not fully move search to the GPU, but it attacks the direct search
bucket itself.

Expected win:

- Plausible `1.3x-1.8x` over current direct if Python/search residual is as
  large as the profile suggests.
- Combined with current direct over stock, this could plausibly move the
  profile-only boundary toward `2.5x-3x` over stock facade.
- Still probably not 10x because PyTorch recurrent inference and CPU tree
  synchronization remain once per simulation.

This is the first "real" C++/Cython fix, not just a cleanup.

## Fix Lane C: MiniZero/KataGo-Style Actor And Inference Service

Goal: stop asking one Python LightZero collector to do every search boundary.
Keep many MCTS instances in flight and batch their leaf evaluations.

External pattern:

- MiniZero keeps multiple MCTS instances per self-play worker and evaluates a
  batch of selected leaf nodes through one GPU inference call.
- KataGo is a C++ engine with search threads and neural-network batching.
- OpenSpiel explicitly says its Python AlphaZero path does not batch inference
  and runs on CPU, while its C++ path uses threads, shared cache, batched
  inference, and GPU training/inference.

CurvyTron version:

```text
many env rows / games
-> many active roots
-> C++ or compact actor scheduler selects leaves
-> one batched recurrent inference call
-> backpropagate results
-> compact replay chunks
```

Expected win:

- Larger than Lane A/B if implemented cleanly.
- This is a custom collector/replay bridge, so it must pass stronger semantic
  gates before touching Coach training.
- It can still preserve MuZero, but it is no longer "stock LightZero with a
  small wrapper."

## Fix Lane D: MCTX/JAX Device-Resident Search

Goal: use a search implementation that is designed to live as batched arrays on
accelerators.

MCTX is the cleanest known option: its search is JAX-native, JIT-compatible, and
defined over batched inputs. It avoids the LightZero CTree/Python/list boundary
by construction.

Expected win:

- Highest search headroom, plausibly `5x-10x+` for the search boundary in the
  right batch regime.
- Highest migration cost. It means a JAX model/search lane or a bridge from
  PyTorch weights into JAX, plus replay/trainer compatibility work.

This should be a spike, not a blind rewrite:

1. synthetic CurvyTron-shaped root/dynamics benchmark;
2. compare roots/sec and action/visit shapes against LightZero on tiny fixed
   cases;
3. only then decide whether it is worth a new training lane.

## Decision

The next useful implementation plan is not one patch. It is two parallel
prototypes:

1. **Near-term LightZero-compatible path:** Cython array-native CTree boundary,
   first root/output arrays, then per-simulation backprop/traverse arrays.
2. **Ambitious architecture path:** MCTX or MiniZero-style batched leaf
   evaluator spike, kept separate from Coach training until it proves real
   throughput and semantic sanity.

If we need a concrete answer to "should we move it all to C?":

```text
Move the LightZero CTree boundary deeper into Cython/C++ first.
Do not expect that alone to give 10x.
For 10x, either run many MCTS instances through a batched inference service or
move search to an accelerator-native array system such as MCTX.
```

## Active Follow-Ups

- C++/Cython sidecar: inspect LightZero CTree and propose staged wrapper.
- Accelerator-native sidecar: compare MCTX/JAX/Torch/Triton options.
- Batched actor sidecar: inspect MiniZero/KataGo/OpenSpiel patterns and propose
  CurvyTron-compatible actor/replay architecture.

