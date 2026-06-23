# Array-Native CTree Feasibility Critique

Date: 2026-05-22

Scope: read-only critique of the current repo and installed LightZero surface.
No production code changes. No live Coach runs touched.

## Short Verdict

A fixed-`A=3` array-native CTree boundary is feasible enough to build a
micro-canary. It is the cleanest conservative path if we want to keep LightZero
CTree semantics while removing some Python/list/object overhead.

It is probably not a 10x move by itself.

The current CTree core is already C++. The hot problem is the wrapper boundary:
Torch tensors become CPU NumPy arrays, then Python lists, then Cython converts
those lists into C++ vectors. Search results come back as Python lists again.

Plain estimate:

```text
fixed-A=3 array-native wrapper only: likely 1.1x-1.4x full-loop, maybe 1.2x-1.8x on the search boundary
deeper CTree fork with dense child storage/output arrays: maybe closer to 2x on search-boundary rows
10x: unlikely unless we also change the bigger topology: batched search service, compiled/fused search, compact replay/collector edges
```

So this is worth testing, but it should be treated as a falsifier, not as the
main radical architecture by default.

## What I Inspected

- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/array_native_ctree_next_design_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/array_native_ctree_opportunity_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_hot_boundary_recritique_20260522.md`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- installed LightZero:
  `.venv/lib/python3.11/site-packages/lzero/policy/muzero.py`
- installed LightZero:
  `.venv/lib/python3.11/site-packages/lzero/mcts/tree_search/mcts_ctree.py`
- local LightZero source mirror:
  `/private/tmp/lightzero-src-optimizer-20260521/lzero/mcts/ctree/ctree_muzero/mz_tree.pyx`
- local LightZero source mirror:
  `/private/tmp/lightzero-src-optimizer-20260521/lzero/mcts/ctree/ctree_muzero/lib/cnode.cpp`

I also checked the installed Cython extension directly. `Roots.prepare(...)`
accepts Python lists but rejects NumPy arrays:

```text
prepare_numpy TypeError: Argument 'noises' has incorrect type (expected list, got numpy.ndarray)
prepare_list ok
```

## Exact Current List-Shaped Calls

### Stock LightZero collect path

In installed `lzero/policy/muzero.py`, `_forward_collect` does this:

```text
line 741: pred_values = inverse(...).detach().cpu().numpy()
line 742: latent_state_roots = latent_state_roots.detach().cpu().numpy()
line 743: policy_logits = policy_logits.detach().cpu().numpy().tolist()
line 745: legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] ...]
lines 748-750: noises = [np.random.dirichlet(...).astype(np.float32).tolist() ...]
line 754: roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
line 759: roots.prepare(root_noise_weight, noises, reward_roots, policy_logits, to_play)
line 760: self._mcts_collect.search(...)
lines 763-764: roots.get_distributions(); roots.get_values()
lines 766-793: one Python output dict per env id
```

### Stock LightZero CTree search

In installed `lzero/mcts/tree_search/mcts_ctree.py`, `MuZeroMCTSCtree.search`
does this each simulation:

```text
line 264: latent_states = []
line 267: results = tree_muzero.ResultsWrapper(num=batch_size)
lines 277-286: tree_muzero.batch_traverse(...) returns Python lists
lines 289-290: Python loop appends selected latent states
lines 292-295: NumPy arrays become Torch tensors on device
lines 306-309: recurrent outputs detach to CPU NumPy
lines 314-316: reward/value/policy become Python lists
lines 324-327: tree_muzero.batch_backpropagate(... lists ...)
```

### Our train-facing `direct_ctree_gpu_latent` hook

In `lightzero_curvyzero_stacked_debug_visual_survival_train.py`:

```text
line 1880: to_play_batch = list(to_play)
lines 1896-1904: tree_muzero.batch_traverse(...)
lines 1910-1925: Python list indices/actions become CUDA tensors
lines 1941-1949: reward/value/policy packed, then copied to CPU NumPy
lines 1959-1962: reward/value/policy become Python lists
lines 1964-1973: tree_muzero.batch_backpropagate(... lists ...)
```

This hook removed the repeated root-latent CPU copy. It did not remove the
per-simulation CTree list boundary.

### Our profile sidecar direct CTree path

In `source_state_batched_observation_boundary_profile.py`:

```text
line 5129: policy_logits_list = policy_logits_np.tolist()
lines 5130-5137: legal_actions and noises are Python nested lists
line 5138: roots = type(mcts).roots(active_root_count, legal_actions)
line 5139: roots.prepare(..., noises, reward_roots, policy_logits_list, to_play)
lines 5187-5188: roots.get_distributions(); roots.get_values()
lines 5195-5275: Python/NumPy output assembly
line 5627: to_play_batch = list(to_play)
lines 5639-5647: tree_muzero.batch_traverse(...)
lines 5668-5672: last_actions Python list becomes CUDA tensor
lines 5692-5700: recurrent outputs copied to CPU NumPy
lines 5709-5711: reward/value/policy become Python lists
lines 5713-5722: tree_muzero.batch_backpropagate(... lists ...)
```

## What The Cython/C++ Boundary Looks Like

The installed binary is:

```text
.venv/lib/python3.11/site-packages/lzero/mcts/ctree/ctree_muzero/mz_tree.cpython-311-darwin.so
```

The source mirror shows:

```text
mz_tree.pyx line 30: Roots.__cinit__(int root_num, vector[vector[int]] legal_actions_list)
mz_tree.pyx lines 34-36: Roots.prepare(..., list noises, list value_prefix_pool, list policy_logits_pool, vector[int]& to_play_batch)
mz_tree.pyx lines 74-82: batch_backpropagate(..., list value_prefixs, list values, list policies, ...)
mz_tree.pyx lines 95-100: batch_traverse(...) returns Python lists
```

The C++ side is already typed:

```text
cnode.cpp lines 321-339: CRoots::prepare(...) loops roots and expands children
cnode.cpp lines 480-499: cbatch_backpropagate(...) consumes vector<float> and vector<vector<float>>
cnode.cpp lines 755-823: cbatch_traverse(...) fills CSearchResults vectors
```

But it is not dense-array-native internally:

```text
cnode.h lines 25-28: each CNode stores std::map<int, CNode> children and vector<int> legal_actions
cnode.cpp lines 83-120: CNode::expand(...) iterates legal_actions
cnode.cpp lines 387-402: get_distributions() returns vector<vector<int>>
```

That means there are two possible scopes:

1. **Thin array-native wrapper.** Accept NumPy typed memoryviews at Cython,
   convert or loop in Cython/C++ without Python lists, keep current CNode map
   internals.
2. **Real fixed-`A=3` CTree fork.** Store children and visits in dense arrays,
   avoid `std::map`, fill caller-provided `[N,3]` output arrays.

The first is small and feasible. The second is a real CTree fork.

## Why Fixed `A=3` Helps

CurvyTron has three actions. That lets us avoid LightZero's generic
variable-action wrapper:

```text
legal mask:     uint8[N,3]
root reward:    float32[N]
policy logits:  float32[N,3]
to_play:        int32[N]
visits out:     int32[N,3]
values out:     float32[N]
```

For all-actions-legal rows, this removes sparse legal-action lists entirely.
For mixed masks, the API can still use `legal_mask[N,3]` and zero illegal visit
mass explicitly.

That is cleaner than repeatedly building:

```text
[[0, 1, 2], [0, 2], ...]
[[noise0, noise1, noise2], ...]
[[logit0, logit1, logit2], ...]
```

## Expected Headroom

The latest useful matched full-loop numbers are:

```text
no-RND:        stock 433.17 steps/sec -> direct output-fast 566.19 steps/sec, about 1.31x
rnd_meter_v0: stock 351.02 steps/sec -> direct output-fast 448.52 steps/sec, about 1.28x
```

The no-RND direct row still showed roughly:

```text
wall:           28.94s
policy collect: 10.31s
MCTS/search:     8.06s
recurrent:       4.28s
D2H bucket:      2.47s
output assembly: 0.077s
```

The thin array-native CTree wrapper can attack:

- Python list creation;
- Cython list-to-vector conversion;
- root/output sparse list wrappers;
- dense `[N,3]` output assembly;
- some per-simulation Python object churn.

It does not remove:

- GPU recurrent model work;
- GPU-to-CPU reward/value/policy transfer if CTree remains CPU;
- CPU CTree traversal/backprop itself;
- current CNode `std::map` children;
- stock collector/replay/learner/RND shell around search;
- one Python loop over simulations.

So the honest expectation is:

```text
thin fixed-A=3 Cython wrapper:
  likely: 1.1x-1.4x full-loop over current direct hook
  optimistic: 1.5x full-loop if list/vector conversion is worse than timers show
  unlikely: 2x full-loop

deeper dense fixed-A=3 CTree fork:
  possible: ~2x on the search boundary
  unknown full-loop effect until measured

10x:
  no, not from this alone
```

A 10x path needs a larger topology change: compiled/fused batched search, a
batched search service, or a compact actor/search/replay path that avoids
scalar Python objects across the whole collect loop.

## Main Risks

1. **The measured hot part may not be the lists.**
   The `.tolist()` timer itself was small in prior notes. The bigger
   D2H-labelled bucket includes GPU wait, support transforms, and transfer.
   A CPU CTree still needs reward/value/policy on CPU.

2. **CNode internals are still object-heavy.**
   Even with array inputs, current CTree stores children in `std::map<int,
   CNode>`. That is not a dense GPU-style or cache-optimal fixed-action tree.

3. **Legal-mask semantics can drift.**
   Stock stores visit counts only over legal actions. A dense `[N,3]` output
   must guarantee illegal actions get zero visits and are never sampled.

4. **Root noise must match legal actions only.**
   Dense `[N,3]` Dirichlet noise must be generated over legal slots only, then
   placed back into action ids. Generating three-way noise and masking after
   would change semantics for mixed masks.

5. **Cython memoryviews can copy silently.**
   The micro-canary must assert dtype and contiguity: `float32`, `int32`,
   C-contiguous, no object arrays.

6. **Build/repro risk.**
   Changing LightZero's Cython extension is not a normal repo-only Python patch.
   It needs an isolated source checkout or a vendored extension path. Do not
   mutate `.venv` as the durable solution.

7. **Full-loop Amdahl can erase the win.**
   Even if the search boundary speeds up, stock collector, replay, RND, env
   manager, and learner overhead may become the next wall immediately.

## Smallest Useful Micro-Canary

Do not start by wiring training. Do not start by replacing all of CTree.

Smallest real canary:

1. In an isolated LightZero source checkout, add one Cython/C++ function:

```text
batch_backpropagate_flat_a3(
  current_latent_state_index: int,
  discount_factor: float,
  rewards: float32[N],
  values: float32[N],
  policy_logits: float32[N,3],
  min_max_stats_lst,
  results,
  to_play: int32[N],
)
```

2. Keep existing `Roots`, existing `batch_traverse`, and existing CNode internals.
   This isolates the per-simulation list boundary first.

3. Build a local synthetic benchmark:

```text
N = 1024 roots
A = 3 all legal
S = 16 simulations
root_noise_weight = 0 for deterministic parity first
fake reward/value/policy arrays are fixed seeded float32
old path: reward.tolist(), value.tolist(), policy.tolist(), batch_backpropagate
new path: flat arrays, batch_backpropagate_flat_a3
compare roots.get_distributions() and roots.get_values()
time old versus new
```

4. Then add mixed-mask cases:

```text
[1,1,1], [1,0,1], [0,1,0], [1,0,0]
```

5. Only if that moves, add:

```text
Roots.prepare_arrays(...)
batch_traverse_arrays(...) that fills int arrays
get_arrays(visits[N,3], values[N])
```

The first canary should answer one question:

```text
Is Python-list-to-C++-vector conversion in per-sim backprop actually expensive enough to matter?
```

If no, stop this lane or promote it to cleanup only.

## What Would Invalidate This Lane

Stop or deprioritize array-native CTree if any of these happen:

1. The flat backprop micro-canary is less than `1.15x` faster than old
   list-shaped backprop at `N=1024`, `S=16`.

2. A profile row with array-native backprop does not improve
   `direct_ctree_gpu_latent` sim16 boundary throughput by at least `1.15x`.

3. Full-loop matched rows improve by less than `5-10%` after the boundary row
   improves. That means Amdahl has moved elsewhere.

4. Mixed-mask parity fails or illegal actions receive nonzero visits.

5. Root-noise parity requires enough special cases that the fixed-`A=3` wrapper
   stops being small.

6. Cython memoryview handling still copies into nested vectors in the hot path
   and the copy cost is not lower than Python lists.

7. Recurrent model or collector/replay overhead dominates the fresh profile
   split after direct CTree, leaving little CTree-boundary time to recover.

## Recommendation

Build the micro-canary, but keep expectations sober.

Recommended next sequence:

1. Add a local-only Cython spike for `batch_backpropagate_flat_a3`.
2. Benchmark old list backprop versus flat backprop with fixed seeded roots.
3. Require exact distributions/values for deterministic no-noise cases.
4. Require mixed-mask legal-action safety.
5. If it clears the `1.15x` boundary gate, add `prepare_arrays` and
   `get_arrays`.
6. Only then run a profile sidecar row against `direct_ctree_gpu_latent`.

Do not sell this as the 10x plan. Sell it as the conservative CTree-boundary
falsifier. The 10x plan still needs a larger array-owned search topology.
