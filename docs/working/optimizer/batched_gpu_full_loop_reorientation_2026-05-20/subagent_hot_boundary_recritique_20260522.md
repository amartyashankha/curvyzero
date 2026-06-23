# Subagent Hot Boundary Recritique

Date: 2026-05-22

Scope: read-only audit of the current optimizer docs and the current
`direct_ctree_gpu_latent` code path. I did not edit code and did not touch live
runs.

## Verdict

The current `direct_ctree_gpu_latent + output fast path` result is real, but it
is a boundary cleanup, not a topology change. It moves matched full-loop profile
rows from:

```text
no-RND:        433.17 -> 566.19 steps/sec, about 1.31x
rnd_meter_v0: 351.02 -> 448.52 steps/sec, about 1.28x
```

That is the right size for the patch we made. It is not the right size for a
5-10x move because the loop still uses stock LightZero collection topology:
scalar env-manager rows, public collector/replay/target ownership, a CPU CTree
API, per-simulation Python/list payloads, and per-env collect dictionaries at
the output edge.

The suspected wall is indeed still collect/search topology, not rendering.

## 1. Exact Hot Boundary After `direct_ctree_gpu_latent`

The hot boundary is:

```text
Torch CUDA MuZero model tensors
-> inverse support transform / policy logits
-> CPU NumPy payload
-> Python lists per simulation
-> LightZero CTree Cython API
-> Python/root output extraction
-> stock per-env collect output dicts
```

The two code manifestations are:

- Train-facing profile hook:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  `_install_lightzero_collect_search_backend_hook(...)` and
  `_direct_ctree_gpu_latent_search_for_collect(...)`.
- Sidecar/profile boundary:
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  `_LightZeroCollectForwardStackProbe._run_direct_mcts_arrays(...)` and
  `_run_direct_ctree_gpu_latent_search(...)`.

What `direct_ctree_gpu_latent` fixes:

- it avoids repeatedly copying latent states to CPU for CTree search;
- it keeps a CUDA latent pool and indexes it during recurrent inference;
- it bypasses much of public `collect_mode.forward` fanout in the profile
  sidecar;
- in the train hook, it preserves stock `train_muzero` outside
  `_forward_collect`.

What remains hot:

- `batch_traverse(...)` still returns Python list-shaped latent path indices,
  batch indices, and actions;
- each recurrent output still becomes CPU reward/value/policy payloads for
  `batch_backpropagate(...)`;
- the hot API still wants nested Python lists, not flat arrays;
- root preparation still listifies policy logits and Dirichlet noise;
- `roots.get_distributions()` and `roots.get_values()` still extract Python
  objects at the output edge;
- the train hook still returns one LightZero-style output dict per env id;
- stock collector, replay, target builder, learner, RND, and profile sidecars
  still sit around the faster search hook.

The latest no-RND direct output-fast profile makes this visible:

```text
direct output-fast C64/sim16/3-learner:
  wall:           28.94s
  policy collect: 10.31s
  MCTS/search:     8.06s
  recurrent:       4.28s
  D2H bucket:      2.47s
  output assembly: 0.077s
```

The old output assembly issue is mostly gone. The remaining collect/search
time is not a renderer problem and not just a `.tolist()` problem. The
D2H-labelled bucket includes transform, GPU wait, and the unavoidable fact
that the CTree API still consumes CPU/list-shaped model output every
simulation.

## 2. What Would Have To Change For 5-10x

A 5-10x move cannot come from the current stock loop plus another small
renderer or output patch. It needs one of these concrete topology changes.

### A. Array-Native CTree Boundary

Keep LightZero CTree semantics, but stop feeding it Python lists each
simulation.

Concrete shape:

```text
CUDA recurrent output
-> one packed reward/value/policy tensor
-> flat typed CPU array or C++/Cython view
-> fixed-A=3 batch_backpropagate/prepare/output APIs
-> compact arrays out
```

The minimum useful API is a fixed-action CurvyTron path such as:

```text
batch_backpropagate_flat_a3(
  rewards: float32[B],
  values: float32[B],
  policy_logits: float32[B, 3],
  to_play: int32[B],
  ...
)
```

This is the conservative next implementation because it preserves the CTree
algorithm. It is probably a 1.3-2x boundary improvement if the current diagnosis
is right, not a guaranteed 10x full-loop result.

### B. Compiled/Fused Batched Search

Move tree state and search updates into fixed-shape tensor arrays, then compile
or capture the loop.

Concrete options:

- `torch.compile`/CUDA graphs/Triton around the fixed-shape dense search;
- JAX/MCTX-style batched MCTS;
- custom CUDA/C++ search state for fixed `A=3`.

This is the plausible 5-10x search-boundary lane because it removes the
per-simulation Python/control/API tax rather than polishing it.

### C. Batch-Resident Collect/Replayer Topology

Even if search gets faster, the stock loop can still erase the gain. A real
5-10x Coach-facing architecture needs:

```text
compact actor/env batch
-> device/uint8 observation stack
-> batched model/search arrays
-> compact action/value/visit arrays
-> compact replay/target writes
-> scalar objects only at compatibility/debug edges
```

This means the batch cannot die at scalar `BaseEnvTimestep` rows, and it cannot
immediately reappear as one Python dict per env after search. RND and terminal
bookkeeping must consume the same compact/latest-frame contract rather than
forcing a second scalar path.

## 3. P0 Validation Gaps Before Any Coach-Facing Use

The current local validation is useful but not enough for Coach-facing
promotion. The P0 gaps are:

1. **Output fast path parity.** The all-actions-legal fast path needs exact
   deterministic selector parity and stochastic distribution checks against
   stock. This must cover the train hook, not only the profile arrays helper.

2. **Mixed-mask forced cases.** Single legal action, one illegal action, masked
   clear preference, illegal visit mass zero, fractional masks, and empty masks
   must pass or fail closed. Current docs call these mandatory; they are not a
   Coach gate until exercised against the actual hook output.

3. **Support/noise/eval modes.** Missing forced cases remain for support
   transform behavior, no-noise eval mode, and root-noise collect mode. The
   sim16 neutral exact failure is acceptable as a critique of the gate, but it
   means we need explicit statistical thresholds for ordinary noisy collect
   rows.

4. **Replay/target canary.** Stock and direct collect output must produce the
   same replay/target meaning: action, visit counts, searched value, predicted
   value/logits, reward, done, action mask, and metadata compatibility.

5. **Terminal/live canary.** No-death rows are insufficient. Normal death,
   autoreset, terminal rows, final observation, zero masks, and live rows must
   be covered.

6. **RND canary.** For `rnd_meter_v0` and any positive RND mode, prove latest
   frame source, predictor update, frozen target, and unchanged environment
   reward semantics. The hash fix removed an overhead wall; it did not promote
   the search hook.

7. **Matched full-loop repeats.** Require at least two same-shape stock/direct
   profile repeats with `called_train_muzero=true`,
   `collect_search_backend_fallback_calls == 0`, evaluator/checkpoint/GIF
   sidecars disabled unless intentionally profiled, and stable no-RND plus RND
   reads.

## 4. Fastest Falsifier Experiments, In Order

1. **Local forced-case hook parity.**
   Extend or reuse `scripts/compare_curvytron_direct_ctree_stock.py` to hit the
   actual train-hook output contract for all-legal, single-legal,
   mixed-legal-cycle, masked preference, root-noise off/on, sim8 and sim16.
   Fail fast on illegal actions, schema drift, nonzero illegal visit mass, or
   value/logit mismatch.

2. **Same-denominator no-RND full-loop repeat.**
   Repeat the H100 C64/sim16/3-learner stock/direct output-fast pair twice,
   with sparse telemetry, eval/checkpoint/GIF disabled, `fallback_calls == 0`,
   and the same step source. If the direct row does not stay at least about
   1.2x, the current hook is a useful probe but not a launch candidate.

3. **Direct-path timer split at sim16.**
   On the sidecar boundary, rerun `direct_ctree_gpu_latent` with the current
   timer split and compare all-legal versus mixed-mask, root-noise 0 versus
   0.25. The falsifier is whether model-output D2H/list/API/search shell
   buckets are not the dominant residual. If they are not, array-native CTree
   is the wrong next patch.

4. **Array-native CTree micro-canary.**
   Build the smallest fixed-`A=3` flat-array backprop/output wrapper, profile
   it only against `direct_ctree_gpu_latent` sim16, and require both parity and
   at least about 1.15-1.2x boundary improvement. If it only saves root/output
   cleanup, stop the lane.

5. **Compiled/fused dense search sim16 gate.**
   Try only a bounded `torch.compile`/CUDA-graph/fused fixed-shape dense search
   spike. It must beat `direct_ctree_gpu_latent` sim16 by about 1.2x after
   warmup and pass forced-case parity. If not, stop eager dense polishing and
   move back to array-native CTree or a real batched search service.

6. **RND and normal-death promotion pair.**
   Only after the no-RND repeat and P0 semantic gates pass, run matched
   `rnd_meter_v0` and normal-death/autoreset stock/direct pairs. If either
   breaks semantics or collapses the win, keep the hook profile-only.

## Boundary Critique In One Sentence

`direct_ctree_gpu_latent` proves that the public LightZero collect/search
wrapper was wasteful, but the remaining hot boundary is still the
model/search/collector object boundary: CUDA tensors are repeatedly converted
into CPU/list CTree payloads and then stock per-env collect dictionaries. A
5-10x result requires keeping the batch and search state array-shaped across
that boundary, not making the current scalar/object boundary a little cleaner.
