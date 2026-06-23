# Subagent Current Bottleneck Recritique

Date: 2026-05-22

Scope: read-only audit of the current optimizer code paths and docs. I did not
touch production code and did not touch live Coach runs.

Files inspected:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/exploration_bonus.py`
- `scripts/summarize_curvytron_optimizer_profile_results.py`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md`
- related current docs: `current_hot_path_bottleneck_map_20260522.md`,
  `mock_search_service_ceiling_plan_20260522.md`, and `task_board.md`

## Short Answer

The current Amdahl wall is not rendering in the trusted full-loop denominator.
It is the stock LightZero collect/search compatibility boundary:

```text
host observation/timestep objects
-> GPU initial inference
-> CPU root prep and Python legal-action/noise lists
-> Python loop over MCTS simulations
-> CTree CPU/list traverse/backprop API
-> recurrent inference on GPU
-> reward/value/policy copied back to CPU every simulation
-> Python dict action output
-> stock replay/learner/RND object lanes
```

`direct_ctree_gpu_latent + output-fast` cleaned up one real slice of that path.
That is why it gets about `1.28x-1.31x`, not `5x-10x`.

The exact code boundary that probably blocks the larger speedup is the
per-simulation LightZero CTree boundary: CTree is already C++, but our current
hook still sends reward/value/policy through CPU NumPy and Python lists every
simulation and still receives traverse/output through Python lists.

Update after the durable H100 wave:

```text
mock_search_service sim16:       11648.29 roots/sec
direct_ctree_gpu_latent sim16:    5303.97 roots/sec
recurrent_toy sim16:              8512.57 roots/sec
```

That answers the immediate measurement request. Deleting CTree/search ownership
in this ceiling is about `2.20x` over current direct. This is enough headroom
to keep compact search-service work alive, but not enough to claim a standalone
5-10x search rewrite.

Update after the local native-vector boundary probe:

```text
B512/A16/steps100/zero-observation/uint8/no-pickle
no scalar + native probe: 23515 timesteps/sec
scalar-only:              18604 timesteps/sec
scalar + native probe:    17380 timesteps/sec

scalar materialization: about 2.07s over 102400 timesteps
native compact probe:   about 0.62s over 102400 timesteps
actor_step_wall:        about 3.42s in the no-scalar row
```

That says the object edge is real, but it is not the only wall. In the
no-scalar native-probe row, actor/env scheduling is already about `3.42s` of a
`4.35s` measured local run. A Puffer-style compact boundary helps, but a 5-10x
design also has to own actor/env scheduling and replay/RND materialization.

## Current Amdahl Wall

Trusted matched full-loop rows in the current docs:

| row | stock | direct output-fast | gain |
| --- | ---: | ---: | ---: |
| no-RND C64/sim16/3-learner | `433.17 steps/sec` | `566.19 steps/sec` | `1.31x` |
| `rnd_meter_v0` hash-fixed | `351.02 steps/sec` | `448.52 steps/sec` | `1.28x` |

Important no-RND direct row split:

```text
wall:             28.94s
policy collect:   10.31s
MCTS/search:       8.06s
recurrent:         4.28s
model-output D2H:  2.47s
output assembly:   0.077s
```

Plain read:

- Output assembly used to be bad; the fast path mostly fixed it.
- RND hashing used to be bad; the hash fix mostly removed that specific wall.
- Rendering can still dominate long no-death observation-only profiles, but it
  is not the main wall in the current matched stock `train_muzero` full-loop
  rows.
- The active wall is collect/search topology and stock object boundaries. The
  latest local native-probe row also exposes actor/env scheduling as a visible
  wall once scalar materialization is skipped.

## Exact Code Boundary Blocking 5-10x

### Train-facing direct hook

In
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`,
the direct hook still pays these boundaries.

Root prep:

```text
2159-2171:
  pred_values -> detach().cpu().numpy()
  policy_logits -> detach().cpu().numpy().tolist()
  legal_actions as Python lists
  Dirichlet noises as Python lists
  roots.prepare(... lists ...)
```

Search loop:

```text
1885-1974:
  for simulation_index in range(num_simulations):
    ResultsWrapper(...)
    tree_muzero.batch_traverse(...) -> Python lists
    Python list indices/actions -> CUDA tensors
    model.recurrent_inference(...)
    reward/value/policy_logits -> detach().cpu().numpy()
    reward/value/policy_logits -> .tolist()
    tree_muzero.batch_backpropagate(... lists ...)
```

Output:

```text
2198-2247:
  roots.get_distributions()
  roots.get_values()
  dict output per env id
```

Output is no longer the next obvious target because the all-legal fast path
dropped it to about `0.077s` in the matched no-RND row.

### Profile sidecar direct CTree path

In
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`,
the same shape appears:

```text
5129-5141:
  policy_logits_np.tolist()
  legal_actions lists
  noises lists
  roots.prepare(...)

5631-5724:
  per-sim batch_traverse -> Python lists
  last_actions list -> CUDA tensor
  recurrent_inference
  model_output -> CPU NumPy
  reward/value/policy -> .tolist()
  batch_backpropagate(...)
```

The sidecar also has dense GPU and compile experiments:

```text
6570-6849: dense_torch_mcts
6851-7005: compile helper gate
```

Those are useful falsifiers, but they are profile-only and not LightZero CTree.
The compile spike already failed sim16 against `direct_ctree_gpu_latent`.

### Hybrid observation/profile manager

`src/curvyzero/training/source_state_hybrid_observation_profile.py` makes the
batch/scalar tension explicit:

```text
405-412: in-process actor loop
419-427: payload merge then observation update
459-515: optional batched stack/search probe
519-530: materialize LightZero scalar timestep
588-625: shift stack, render latest frame, write host stack
```

This manager is profile-only (`calls_train_muzero=false` in its result payload).
It is good for measuring pre-scalarization costs, but its high roots/sec does
not automatically translate to stock `train_muzero`.

### RND path

`src/curvyzero/training/exploration_bonus.py` is not the current largest wall
after the hash fix, but it is still a separate CPU/object lane:

```text
669-680: state hash copies tensors to CPU
762-787: collect_data extracts latest frame and stores cloned tensors in a list
789-819: train_with_data samples Python list, trains, hashes before/after
821-871: estimate copies MSE to CPU, min/max normalizes per batch, deep-copies reward
```

RND is not the reason direct search is only `1.3x`, but a future resident
pipeline will have to redesign it or it will become the next wall.

### Summary tooling

`scripts/summarize_curvytron_optimizer_profile_results.py` has useful
attestation now:

```text
117-220: required semantic identity fields
173-217: direct_ctree_gpu_latent requires backend counters and zero fallbacks
223-270: row summary includes manager, steps source, timers, root batch, GPU
```

This is good. The caveat is that attestation proves the row says what it
measured; it does not prove the denominator is the one we meant to optimize.

## Evidence That Is Weak Or Lost

1. **Profile-only roots/sec is not train speed.**
   `source_state_batched_observation_boundary_profile.py` rows can show large
   sidecar throughput, but many do not call `train_muzero`, do not touch replay,
   do not run learner updates, and do not use stock collector semantics.

2. **The mock search-service ceiling is measured now, but still profile-only.**
   It reached `11648.29` roots/sec at sim16 versus `5303.97` for
   `direct_ctree_gpu_latent`. That is meaningful, but it deliberately skips
   real MCTS, replay, learner, and RND semantics.

3. **The dense compile spike is not a production answer.**
   It won sim8 but lost sim16 to `direct_ctree_gpu_latent`. It also hit
   compile/recompile/CUDA-graph warnings. It should stay as evidence, not as the
   next polish target.

4. **Nested timers are not additive.**
   `policy_forward_collect` includes MCTS/model work. `mcts_search` includes
   recurrent model work. CUDA syncs in profile timers are useful for
   attribution but can distort throughput if treated as production cost.

5. **RND rows have a step-source caveat.**
   The docs say RND rows use an MCTS-root profile fallback for step count, so
   compare matched RND stock/direct rows only. Do not mix them with cleaner
   no-RND counters.

6. **Direct hook semantic validation is not complete.**
   We have local compare evidence and fallback counters, but Coach-facing
   promotion still needs stronger gates around masks, root noise, terminal
   rows, normal death/autoreset, RND latest-frame semantics, replay targets,
   and deterministic tolerance.

7. **The denominator keeps changing.**
   C64/sim16 full-loop, B512/A16 sidecar, C256/C512 manager rows, no-death,
   normal-death, RND, and zero-observation rows all answer different questions.
   The docs mostly label them, but conclusions sometimes leak across them.

8. **Learner/replay effects are under-separated.**
   The direct no-RND row reports learner time much lower than the stock row.
   That may be real, noise, or indirect timing interaction. A pure search
   conclusion should not depend on that learner difference.

9. **The native-vector probe is topology evidence, not training evidence.**
   It is local, zero-observation, no-pickle, profile-only, and it skips real
   search. Its value is that it prices scalar materialization and compact array
   consumption. It does not prove the real trainer will get `23515`
   timesteps/sec.

## What Probably Blocks 5-10x

The blocker is not one single `.tolist()` call. It is the shape of the loop:

```text
stock LightZero collector owns one ready-env batch
-> CTree owns CPU root/tree objects
-> Python owns simulation loop control
-> GPU recurrent inference runs one simulation batch at a time
-> CTree requires CPU reward/value/policy lists every simulation
-> stock collector receives Python dict output
-> actor/env scheduling keeps producing scalar-ish batches
-> replay/RND/learner continue as stock objects
```

This topology prevents a large jump because the batch does not remain a compact
array/tensor object across collection, search, replay, and RND.

Expected headroom by lane:

| lane | likely full-loop headroom | why |
| --- | ---: | --- |
| More output assembly cleanup | tiny | already `~0.077s` in direct no-RND row |
| More packed transfer/listify cleanup | small | listify was `~0.08s`; D2H bucket is mostly sync/transform/boundary |
| Array-native fixed-`A=3` CTree wrapper | likely `1.1x-1.4x` over current direct | removes list/vector boundary, keeps CPU CTree and stock topology |
| Better renderer only | denominator-dependent | useful for long observation-heavy profiles, not current matched train-loop wall |
| Mock/real compact search service | about `2.2x` profile-only over current direct ceiling so far | useful, but not standalone 10x |
| Native/vector boundary without scalar materialization | about `1.26x` over scalar-only in the local zero-observation probe | proves object edge matters, while actor/env scheduling remains large |
| Compact collect/search/replay/RND topology | only credible `5x-10x` lane | removes scalar/object compatibility as the hot path |

## Measurement Just Run

The first durable H100 wave used the same B512/A16 denominator for:

```text
mock_search_service sim8/sim16
direct_ctree_gpu_latent sim8/sim16
recurrent_toy sim8/sim16
```

Result:

```text
mock_search_service was >1.5x direct, but <3x direct.
```

Plain decision:

```text
Search-service is not dead.
Search-service alone is probably not the 10x lane.
The next architecture critique must include the collector/env/replay object
boundary, not only MCTS.
```

The local native-vector boundary probe then added this B512/A16/steps100
zero-observation result:

| row | timesteps/sec | measured sec over 102400 timesteps | read |
| --- | ---: | ---: | --- |
| no scalar + native probe | `23515` | `4.35s` | fastest local compact-boundary row |
| scalar-only | `18604` | `5.50s` | scalar LightZero object edge costs real time |
| scalar + native probe | `17380` | `5.89s` | paying both edges is worse |

Measured components:

```text
scalar materialization: about 2.07s, about 20.2 us/timestep
native compact probe:   about 0.62s, about  6.1 us/timestep
actor_step_wall:        about 3.42s, about 33.4 us/timestep
```

Plain decision:

```text
The object edge matters.
The compact native probe itself is cheap.
Actor/env scheduling is now a first-class wall in this local shape.
```

Next measurement:

```text
Run the PufferLib-shaped native/vector buffer falsifier with realer boundaries:
contiguous obs/mask/reward/done/action buffers in,
mock_search_service or recurrent_toy out,
compact replay chunk materialized only at the edge.
```

It should report actor scheduling, env step, observation/update, compact
consumer, scalar materialization, and compact replay materialization separately.
The key falsifier is whether actor/env scheduling still dominates after the
consumer stays compact.

## Tiny Instrumentation Recommendation

No production code change is required before the next measurement.

If instrumentation is added, keep it doc-safe/profile-only:

- add a summary table that places `mock_search_service`, `direct_ctree_gpu_latent`,
  and `recurrent_toy` on the same B512/A16/sim16 row;
- include explicit fields for `calls_train_muzero`, `profile_only`,
  `real_ctree_calls`, `recurrent_inference_calls`, and `public_output_edge`;
- keep direct rows requiring `collect_search_backend_fallback_calls == 0`.

## Bottom Line

The current `1.3x` result is not mysterious. It is what a partial boundary
cleanup looks like when the larger stock LightZero collect/search/replay/RND
contract is still intact.

The next useful question is not "can we shave another small timer bucket?" It is:

```text
If search ownership became compact arrays instead of CTree/Python objects,
does the matched sidecar denominator jump enough to justify a real rewrite?
```

The mock search-service ceiling wave answered that partially. The answer is:

```text
yes, search boundary ownership matters;
no, the measured ceiling is not enough by itself.
```

So the next real question is broader:

```text
Can we keep the entire collect/search/replay-shaped path as compact arrays long
enough to avoid scalar Python object churn?
```

The fresh native-probe result sharpens that question:

```text
Can we also make actor/env scheduling feed those compact arrays cheaply enough,
or does actor/env scheduling become the next Amdahl wall?
```
