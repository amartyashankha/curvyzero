# Array-Native CTree Next Design

Date: 2026-05-22

Status: optimizer working plan. Profile-only until parity gates pass. Do not
touch live Coach training runs.

## Plain Goal

The current best LightZero/CTree train-facing probe is:

```text
collect_search_backend=direct_ctree_gpu_latent
```

Do not read this as the current best optimizer row overall. Real-checkpoint
MCTX/JAX shadow rows now beat direct CTree in profile-only throughput, with
unresolved semantic deltas.

It keeps latent states on GPU, but it still calls LightZero CTree through a
Python/list-shaped API every simulation. The next serious optimization is to
remove that boundary without changing MuZero search semantics.

## Current CTree Shape

Installed LightZero exposes CTree as compiled modules:

```text
lzero.mcts.ctree.ctree_muzero.mz_tree.cpython-311-darwin.so
```

The upstream source shows the important wrappers:

```text
lzero/mcts/ctree/ctree_muzero/mz_tree.pyx
lzero/mcts/ctree/ctree_muzero/mz_tree.pxd
lzero/mcts/ctree/ctree_muzero/lib/cnode.cpp
lzero/mcts/ctree/ctree_muzero/lib/cnode.h
```

The current Python/Cython boundary is list-based:

```text
Roots.prepare(..., list value_prefix_pool, list policy_logits_pool, list to_play)
batch_traverse(..., list virtual_to_play_batch) -> Python vectors/lists
batch_backpropagate(..., list rewards, list values, list policies, list to_play)
Roots.get_distributions() -> vector<vector<int>> as Python object
Roots.get_values() -> vector<float> as Python object
```

Runtime probes against the installed `.so` confirmed the wrapper rejects NumPy
arrays at the hot API:

```text
Roots.prepare(...): expected list, got numpy.ndarray
batch_traverse(...): virtual_to_play_batch expected list
batch_backpropagate(...): value_prefixs expected list
```

2026-05-22 source/build refresh:

```text
The local installed LightZero wheel contains only compiled `.so` CTree modules.
It does not ship `mz_tree.pyx`, `mz_tree.pxd`, `cnode.cpp`, or `cnode.h`.
Those files exist in the upstream source clone under `/private/tmp/LightZero`,
but they are not vendored into CurvyZero.
```

Plain consequence:

```text
batch_backpropagate_flat_a3 cannot be monkeypatched into the installed binary.
The smallest real proof is a vendored/profile-only Cython/C++ extension or a
LightZero source fork. A separate extension cannot easily touch the cdef
internals of installed Roots/ResultsWrapper; the flat API likely needs to live
inside the same vendored CTree module.
```

The C++ function already takes typed vectors:

```text
cbatch_backpropagate(
  int current_latent_state_index,
  float discount_factor,
  const vector<float>& rewards,
  const vector<float>& values,
  const vector<vector<float>>& policies,
  ...
)
```

So the problem is not that CTree is missing. The problem is that our hot loop
builds Python lists and nested vectors each simulation.

## Small Patch Already Done

We packed reward, value, and policy logits into one contiguous tensor before
one CPU transfer in both:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
```

This is safe hygiene. It did not unlock a large speedup because `.tolist()` is
only about `0.08s`, and the D2H-labelled timer includes GPU wait and transform
work.

## Next Real API

For CurvyTron fixed action count `A=3`, add a Cython wrapper that accepts flat,
typed arrays instead of Python nested lists:

```text
batch_backpropagate_flat_a3(
  current_latent_state_index: int,
  discount_factor: float,
  rewards: float32[B],
  values: float32[B],
  policy_logits: float32[B, 3],
  min_max_stats_lst,
  results,
  to_play_batch: int32[B]
)
```

Inside Cython/C++ it can build or directly consume contiguous vectors without
Python per-row list objects. If the existing C++ `CNode.expand(...)` still
requires `vector<float>`, the first implementation can fill a reusable
`vector<vector<float>>` in Cython. The stronger implementation adds a C++
fixed-`A=3` overload so each row reads `policy_logits[i * 3 + a]`.

Smallest local vendored spike:

```text
src/curvyzero/vendor/lightzero_ctree_a3/
  mz_tree.pyx
  mz_tree.pxd
  lib/cnode.cpp
  lib/cnode.h
```

Add profile-only `batch_backpropagate_flat_a3(...)`, then add
`ctree-flat-a3` to:

```text
scripts/benchmark_lightzero_ctree_no_model.py
src/curvyzero/infra/modal/lightzero_ctree_no_model_benchmark.py
```

Cold gate:

```text
roots=1024, simulations=16, iterations=100, warmup=10,
root_noise_weight=0.0, legal_profiles=all3,mixed_2of3.

Compare vendored ctree-list vs ctree-flat-a3 under deterministic tie-breaking
for exact visit/value parity, then require at least 1.15x speedup on the
no-model denominator. If it misses that, stop the array-native CTree lane and
focus on the larger search-service architecture.
```

2026-05-22 implementation refresh:

```text
Implemented an opt-in vendored/profile-only module:
  src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/mz_tree_a3.pyx
  scripts/build_lightzero_ctree_a3.py

Added benchmark backend:
  ctree-flat-a3
```

Important parity lesson:

```text
Stock LightZero CTree calls get_time_and_set_rand_seed() inside each
batch_traverse() and randomly breaks near-ties. Two independent tree runs can
therefore disagree on exact action visits even with identical synthetic arrays.
That was the first apparent flat-A3 parity failure.
```

Fix:

```text
The vendored module now has set_deterministic_tie_breaking(True), used only by
the benchmark parity checker. Default runtime behavior remains stock-like. The
flat path first passed with existing C++ expand(...), then switched back to the
fixed-A=3 expand_a3(...) overload and still passed deterministic parity.
```

Local gate passed:

```text
uv run python scripts/benchmark_lightzero_ctree_no_model.py \
  --roots 64 --simulations 1,2,4,8 --iterations 2 --warmup 0 \
  --backends ctree-flat-a3 --legal-profiles all3,mixed_2of3 \
  --root-noise zero --root-noise-weight 0.0 --flat-a3-parity-check

Result: exact deterministic vendored-list vs flat-A3 visit/value parity for
all tested rows.
```

Local 1024/sim16 no-model speed gate:

```text
all3:
  ctree-list    1.01M nodes/sec
  ctree-flat-a3 2.03M nodes/sec  (~2.02x)

mixed_2of3:
  ctree-list    1.10M nodes/sec
  ctree-flat-a3 1.92M nodes/sec  (~1.75x)
```

Plain read:

```text
This is now a valid small boundary win. It is not the 10x architecture. It
removes Python nested-list backprop payload construction, but still keeps
stock roots, stock traverse, stock output extraction, and Python sim control.
```

H100 no-model gates:

```text
ap-ZaRkAcT7smnhIr410LweJ4, conservative vector-per-row flat path:
  all3:       600.6k -> 953.3k nodes/sec (~1.59x)
  mixed_2of3: 576.1k -> 888.3k nodes/sec (~1.54x)

ap-rQtLiZTWYGQi16v2rrf4Wm, final expand_a3 path:
  all3:       546.7k -> 922.1k nodes/sec (~1.69x)
  mixed_2of3: 517.3k -> 858.1k nodes/sec (~1.66x)

Both flat-A3 rows passed deterministic vendored-list parity:
  exact_visit_match=true
  max_visit_abs_diff=0
  max_value_abs_diff=0
```

Next use:

```text
Wire ctree-flat-a3 into the profile-only direct_ctree_gpu_latent train hook as
an explicit opt-in backend, then run matched full-loop rows against current
direct output-fast. Do not present it as Coach launch advice until the full-loop
gate shows a real denominator win.
```

Implementation refresh:

```text
The explicit train/profile flag now exists:
  collect_search_ctree_backend=flat_a3

It is accepted only together with:
  collect_search_backend=direct_ctree_gpu_latent

The profile grid builder emits both flags. Compact output includes an observed
runtime proof, and the summarizer rejects flat-A3 rows unless the profiler saw
flat-A3 and the flat payload timer exists.
```

Image-safety note:

```text
The vendored CTree extension is not built in the ordinary stock trainer image.
It is built in isolated CPU40 optimizer images for the flat-A3 profile route.
This keeps normal stock/live starts from depending on the experimental Cython
extension build.
```

Matched full-loop outcome:

```text
opt-flat-a3-ab-20260522a, H100 C64/sim16/3 learner:

direct LightZero CTree: 516.55 steps/sec
flat-A3 CTree:          509.69 steps/sec
```

Decision:

```text
Do not recommend flat-A3 to Coach as a speed setting. Keep it as a validated
boundary probe. The next serious speed lane must remove a larger boundary:
search service / native compact batch ownership / replay handoff, not only the
CTree backprop payload shape.
```

## Expected Headroom

This is not guaranteed 10x. The skeptical current estimate is:

```text
array-native fixed-A=3 CTree: plausible 1.3x-2x over current direct hook
compiled/fused or real batched search service: plausible 5x-10x lane
```

Current measured train-facing direct rows still spend time in:

- stock collector/env/replay shell;
- GPU recurrent inference;
- CPU CTree traversal/backprop;
- per-simulation model/tree handoff;
- learner and RND hooks.

The array-native patch attacks the part we still believe is most wrong:
per-simulation Python/list/object fanout. It should be tested against the
current full-loop profile denominator, not only profile-only roots/sec.

Fresh H100 no-model refresh:

```text
modal run ap-9hEH4WJk4kprHGTpcEiPte
roots 512,1024; simulations 16,32; legal all3,mixed_2of3.

ctree-list:       about 0.51M-0.94M nodes/sec
ctree-torch-d2h:  about 0.58M-0.82M nodes/sec
fake-flat:        about 16M-22.6M nodes/sec
```

Read: flat/vector search arithmetic is much faster than the current CTree list
ABI, but this benchmark still excludes real recurrent inference, replay, RND,
and stock collect fanout. It supports a CTree-boundary spike; it does not prove
a trainer-facing 10x by itself.

Plain version:

```text
Array-native CTree is the conservative next implementation because it preserves
LightZero CTree semantics. It is probably not the whole 10x answer. If we want
10x, the parallel research lane is a compiled/fused batched search service that
keeps search state and tensors resident instead of bouncing through Python each
simulation.
```

## Required Gates

1. Local deterministic-tie parity against vendored list CTree:
   rewards, values, policy logits, visit distributions, searched values, and
   selected actions must match within a documented tolerance. Exact neutral
   parity against installed stock CTree is not a valid gate because installed
   CTree reseeds and randomizes near-tie selection inside traversal.
2. Tiny Modal CUDA canary:
   same seed, same roots, sim8/sim16, actual CUDA model outputs.
3. Full-loop profile A/B:
   H100 C64/sim16/3-learner no-RND and RND-meter rows, matched against stock
   and current direct hook.
4. Fallback proof:
   if any mask is not all-actions-legal or any shape is not fixed `A=3`, the
   code must fall back to the existing safe path.
5. Coach-facing rule:
   do not recommend this for actual training until it is wired through the
   canonical launcher and passes the same stock `train_muzero` contract.

Use the current direct-hook promotion checklist as the baseline:
[direct_ctree_promotion_gates_20260522.md](direct_ctree_promotion_gates_20260522.md).

## Current Recommendation

Do not spend more time on CPU count. Do not polish packed D2H unless a fresh
timer proves it matters. The next optimizer lane is:

```text
array-native CTree fixed-A=3 boundary, profile-only first.
```

If that becomes too large or brittle, the fallback lane is a bounded
compiled/fused batched search spike with the same parity gates.
