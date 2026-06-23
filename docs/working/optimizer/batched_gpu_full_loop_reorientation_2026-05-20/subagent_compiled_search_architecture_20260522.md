# Compiled Search Architecture Read

Date: 2026-05-22

Scope: read-only architecture sidecar. No live training, trainer defaults,
tournament defaults, or Modal run state changed.

## Short Answer

Direct CTree arrays is the right near-term LightZero-compatible lane, but the
current `~1.8x` result is mostly a boundary win, not the end architecture.
Getting robustly beyond `2x` requires removing the remaining search-loop
boundary: per-simulation Python, NumPy/list conversion, CPU tree traversal
handoffs, latent gathers, recurrent-output CPU copies, and root/output plumbing.

The credible paths are:

1. **Small tactical path:** keep LightZero CTree semantics, but make the Cython
   boundary array-native and keep GPU latent tensors alive longer. This can
   probably cross `2x` if it attacks per-simulation churn, but it is unlikely
   to become a `5x-10x` architecture.
2. **Medium big-swing path:** profile-only dense Torch MCTS for CurvyTron's tiny
   action space (`A=3`, maybe joint `A=9` as a control), with tree arrays and
   latent pool on GPU. This is the simplest proof that a device-resident search
   shape can beat the CPU CTree boundary while preserving MuZero targets.
3. **Large architecture path:** MiniZero/KataGo-style batched self-play/search
   service or MCTX/JAX. These are realistic systems directions, but they are
   not small patches to current PyTorch/LightZero training.

## Current Evidence

Local denominator docs say the model is not the wall:

- H100 public stock-facade MCTS: about `2473 roots/sec`.
- H100 direct CTree arrays, fresh host input: about `4564 roots/sec`, or
  roughly `1.85x`.
- Direct CTree resident/stale input ceiling: only modestly higher in fresh rows,
  so input movement alone is not the big remaining win.
- Direct CTree fresh row split: search about `6.26s`, model about `0.83s`,
  root prep about `0.49s`, model-output D2H about `0.10s`, observation about
  `2.23s`, H2D about `1.14s`.
- Stock public split previously showed model calls only about `1.81s` inside
  `35.36s` collect-forward, with a large wrapper/output/root-prep residual.

Plain implication:

```text
To be robustly >2x over a 2473 roots/sec facade, the direct lane needs to stay
above about 4946 roots/sec in stable matched rows.

To approach 3x, it needs about 7420 roots/sec, which cannot come from H2D or
root/output cleanup alone. The search loop itself must shrink.
```

## Why Current CTree Hits A Ceiling

LightZero's CTree is already C++ at the `batch_traverse` and
`batch_backpropagate` calls, but the MuZero search loop is still Python-shaped.
The local LightZero clone shows the collect path:

```text
initial_inference(data)
-> latent_state_roots.detach().cpu().numpy()
-> policy_logits.detach().cpu().numpy().tolist()
-> legal_actions = np.nonzero(...).tolist()
-> roots.prepare(...)
-> _mcts_collect.search(...)
-> roots.get_distributions()
-> per-env output dict assembly
```

Inside `MuZeroMCTSCtree.search`, each simulation:

```text
batch_traverse C++ call
-> Python list of selected leaf indexes/actions
-> Python gather of latent states
-> torch.from_numpy(...).to(device)
-> recurrent_inference
-> detach recurrent latent/logits/value/reward to CPU NumPy
-> reward/value/logits .tolist()
-> batch_backpropagate C++ call
```

So "use C++" is not the question. We already do. The question is how much of
the search boundary becomes array-native or device-native.

## Is GPU-Resident MCTS/MCTX Realistic?

Yes, but only as a separate architecture lane.

MCTX is the cleanest reference shape: JAX-native MuZero/Gumbel MuZero search,
JIT-compatible, batch-first, with `RootFnOutput` and `recurrent_fn` arrays.
That maps well to:

```text
obs[B,2,4,64,64]
-> roots[B*2]
-> representation/prediction/recurrent_fn on device
-> MCTX policy output
-> actions[B,2]
```

It is not realistic as a small patch to current LightZero because a PyTorch
model inside a jitted JAX recurrent function would recreate the host boundary.
A serious MCTX path means JAX model/search ownership or a maintained
Torch-to-JAX shadow-model lane, plus replay/checkpoint semantics.

Dense Torch MCTS is the more practical immediate GPU-resident proof. CurvyTron
has tiny action count and low sim counts, so fixed GPU buffers are plausible:

```text
tree stats:  [R, num_simulations + 1, A]
latent pool: [R, num_simulations + 1, latent_shape...]
mask:        [R, A]
```

This would not use LightZero CTree, so it must remain profile-only until parity
gates pass. But it directly tests the thing CTree cannot: no CPU tree, no
Python list leaf batches, no per-simulation latent CPU round trip.

## Is C++/Pybind Array-Native CTree Enough?

Enough for `>2x`: probably, if it attacks the per-simulation boundary.

Enough for the bigger architecture: probably not.

A root/output-only CTree API such as:

```text
prepare_roots_arrays(policy_logits[N,A], rewards[N], noises[N,A],
                     action_mask[N,A], to_play[N])
roots_get_arrays() -> visits[N,A], values[N]
```

would likely be a useful cleanup, but the docs' current split says root prep
and output assembly are not the dominant remaining wall. The important patch is
array-native traverse/backprop inputs and vectorized latent selection across
the simulation loop.

Implementation note: MiniZero uses C++ plus pybind for core components and
PyTorch for NN code, but local LightZero CTree is Cython/C++. For the fastest
CurvyTron proof, extending/vendoring the existing Cython-style `mz_tree.pyx`
is lower-risk than introducing pybind first.

Expected read:

- root/output arrays only: likely small, maybe `5-15%`.
- per-simulation array boundary: plausible `20-40%` over current direct.
- full CPU CTree loop cythonized around PyTorch recurrent calls: plausible
  `1.3x-1.8x` over current direct if the search residual is as measured.
- still not `5x-10x`, because the tree remains CPU-owned and recurrent model
  synchronization still happens every simulation.

## What Actually Gets Beyond 2x

The durable path is not another renderer pass and not pinned input alone. It is
preserving batch shape through search:

```text
compact CurvyTron batch
-> uint8/device observation stack
-> batched root model call
-> batched/tree-array search
-> compact visit/action/value arrays
-> replay rows/chunks only at the compatibility edge
```

MiniZero is the relevant external pattern. Its self-play worker keeps multiple
MCTS instances active, collects a batch of selected leaf nodes, and evaluates
them with batch GPU inference. For CurvyTron, that suggests:

```text
many env rows / seats
-> many active roots
-> select leaves across roots
-> one batched recurrent inference call
-> backpropagate tree stats
-> compact replay write
```

This can preserve MuZero while changing the collector architecture. It is
bigger than `collect_mode.forward` bypassing, but smaller than a full JAX
training rewrite if implemented as a profile-only search service first.

## Simplest Next Proof

The simplest next proof should be profile-only and should not touch live
training:

1. Finish/prioritize the existing `direct_ctree_gpu_latent` matched row only as
   a quick falsifier. If keeping latent tensors on GPU barely moves search,
   stop polishing partial CTree glue.
2. Build a tiny dense Torch MCTS profile proof for `A=3`, `sim=8`, fixed batch
   `R=B*2`, using the real LightZero model's `initial_inference` and
   `recurrent_inference`, but GPU-resident tree arrays.
3. Compare against stock facade and direct CTree arrays on forced cases before
   speed claims:
   legal masks, single legal action, masked preference, clear preference,
   no-noise eval, root-noise collect, support transforms, visit distributions,
   searched values, and replay target rows.
4. Only if the dense search proof is materially faster should we decide between
   a Torch production search lane and a JAX/MCTX scratch lane.

This proof is small enough to answer the architecture question and honest
enough not to become accidental Coach advice.

## MuZero Semantics That Must Stay Identical

Any replacement must preserve:

- PUCT selection constants: `pb_c_base`, `pb_c_init`, priors, visit counts,
  Q/value normalization, `value_delta_max`, discount, and backup order.
- Root noise: collect mode Dirichlet noise with the same alpha/weight; eval
  mode no noise.
- Temperature and epsilon behavior: executed action sampling may use epsilon,
  but stored visit-count policy targets must not be rewritten by epsilon.
- Legal masks: binary `0/1`, `1` means legal, illegal visit mass zero, selected
  actions always legal, and legal-list indexes mapped back to full action ids.
- Value/reward support inverse transforms before backup.
- Batched recurrent inference semantics and reward/value/policy outputs used
  for node expansion.
- `to_play=-1` fixed-opponent/non-board-game semantics for the current lane.
- Row/player perspective, learner-seat selection, simultaneous action scatter,
  and opponent metadata.
- Replay-facing fields: observation stack, action, reward, done,
  `final_observation`, `to_play`, root value, child visits, policy source.
- Randomness controls for root noise, action sampling, resets, opponent
  selection, learner-seat selection, and action-repeat stochasticity.
- RND/death/autoreset behavior as separate axes, not folded into search-speed
  claims.

Exact neutral/tie-heavy visit parity is not a reliable approval gate because
stock CTree itself can drift in tie rows. Forced cases should be exact;
ordinary collect rows should be statistical/distributional with illegal-action
and replay-field checks.

## Source Notes

Local CurvyTron docs read:

- `README.md`
- `whole_loop_denominator_ledger_20260521.md`
- `mcts_arrays_boundary_contract_20260521.md`
- `direct_ctree_promotion_contract_20260521.md`
- `gpu_search_fix_ladder_20260521.md`
- `search_boundary_escape_plan_20260521.md`
- `resident_chunk_canary_plan_20260521.md`
- `subagent_jax_mctx_spike_critique_20260521.md`

External local clones read:

- `/private/tmp/mctx-src-optimizer-20260521/README.md`
- `/private/tmp/mctx-src-optimizer-20260521/mctx/_src/search.py`
- `/private/tmp/mctx-src-optimizer-20260521/mctx/_src/policies.py`
- `/private/tmp/minizero-src-optimizer-20260521/README.md`
- `/private/tmp/minizero-src-optimizer-20260521/docs/Development.md`
- `/private/tmp/minizero-src-optimizer-20260521/minizero/actor/zero_actor.cpp`
- `/private/tmp/lightzero-src-optimizer-20260521/lzero/policy/muzero.py`
- `/private/tmp/lightzero-src-optimizer-20260521/lzero/mcts/tree_search/mcts_ctree.py`
- `/private/tmp/lightzero-src-optimizer-20260521/lzero/mcts/ctree/ctree_muzero/mz_tree.pyx`
- `/private/tmp/lightzero-src-optimizer-20260521/lzero/mcts/ctree/ctree_muzero/lib/cnode.h`

No web source was needed beyond the local primary clones/docs for this pass.
