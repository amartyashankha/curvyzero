# GPU MCTS Architecture Review, 2026-05-23g

Scope: sidecar optimizer review only. I did not edit runtime code, trainer
defaults, live Coach launches, checkpoints, evals, tournaments, or Modal state.

## Short Answer

After model parity, the next big move should not be "optimize MCTX harder."
It should be:

```text
real-model compact search service
-> closed compact env/action/replay loop
-> matched stock-vs-candidate full-loop profile
```

The current MCTX/JAX rows are good architecture evidence. On strict H100
80/20 rows they are about `1.8x-3.2x` faster than direct CTree in the compact
profile denominator. But they still use a toy JAX model, not the LightZero
model, and the faster search exposes the next wall: observation/env/handoff.

Plain recommendation:

```text
Finish model parity first.
Then build one trainer-shaped compact denominator where search, action commit,
replay rows, RND/final-observation sidecars, and learner sample materialization
are all measured together.
Only optimize the hottest wall in that denominator.
```

## Local Facts I Trust

The trusted training lane is still stock LightZero `train_muzero` with
`source_state_fixed_opponent` and CPU policy observations. The compact/MCTX path
is profile-only and must not be treated as Coach launch advice.

Current optimizer shape:

```text
HybridCompactBatch
-> compact env/observation state
-> CompactRootBatchV1
-> compact/fixed-shape search or MCTX
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

Latest strict MCTX read:

```text
H100 B512/sim16:  direct 5,864 steps/sec, MCTX 11,826 steps/sec, 2.02x
H100 B512/sim32:  direct 4,781 steps/sec, MCTX  8,667 steps/sec, 1.81x
H100 B1024/sim16: direct 4,947 steps/sec, MCTX 11,700 steps/sec, 2.36x
H100 B1024/sim32: direct 4,400 steps/sec, MCTX 13,964 steps/sec, 3.17x
```

The search sub-bucket win is much larger than the full-profile win. That is the
important Amdahl warning. In the earlier H100 B1024/sim32 MCTX row, measured
total was `5.238s`, compact slab was `1.084s`, search was `0.628s`, H2D was
`0.296s`, observation was `2.283s`, and actor wall was `1.828s`. If search were
free, that row would still not become 10x faster.

The current `MctxCompactSearchServiceV1` is correctly fenced as profile-only:
it labels itself as JAX/MCTX/Gumbel MuZero, not LightZero CTree, not
`train_muzero`, and it hard-fails inactive roots by default for fixed-shape
profiling. That is the right safety posture while model parity is unfinished.

## External Pattern Read

Most AlphaZero/MuZero-style systems do not get speed by putting an arbitrary
Python object tree on GPU. They either:

- keep tree ownership/control in CPU/C++ and batch neural inference across many
  games/searches; or
- use an array-native stack where model, recurrent function, tree arrays, and
  search loop all live in the accelerator program.

Examples:

- OpenSpiel's AlphaZero docs describe separate actors, learner, evaluator, and
  replay; their docs also distinguish the Python path from the C++ path with
  batched inference and GPU support.
- EfficientZero uses Ray-style distributed self-play, replay, and learner
  pieces, with C++/Cython tree work and PyTorch inference. This is a conservative
  "many actors plus batched GPU inference" pattern.
- MiniZero and KataGo-style engines keep many positions/games in flight so
  neural evaluation batches are large enough to keep the accelerator busy.
- MCTX is the clean accelerator-native reference: dense tree arrays, batched
  roots, invalid action masks, and recurrent model calls stay inside JAX.

CurvyTron should borrow the pattern, not the scale:

```text
many row-seat roots in flight
-> fixed-shape root tensors and masks
-> one search/inference service call
-> compact action/result/replay tensors
-> scalar LightZero objects only at validation or learner-materialization edges
```

The trap is a half-port:

```text
Torch model -> NumPy -> JAX MCTX -> NumPy -> Torch learner
```

That can look like GPU search while preserving the host-sync problem. If MCTX
is promoted past profile-only, the recurrent model must be device-native in the
same compiled/search-owned world, or the bridge will eat the win.

## Next Big Move After Model Parity

Build the smallest real-model compact service that can be compared against
stock in a trainer-shaped denominator.

Target shape:

```text
CompactRootBatchV1
  observation [R,4,64,64] uint8 or resident device tensor
  legal_mask [R,3]
  root/env/player/policy ids

real-model search service
  model parity with current LightZero heads/supports/perspective
  fixed action count A=3
  fixed or padded active-root shape
  no per-simulation D2H payload if possible

CompactSearchResultV1
  selected_action [active_R]
  visit_policy [active_R,3]
  root_value [active_R]
  raw visit/logit fields if needed for parity

CompactRolloutSlab
  staged actions applied to next env step
  previous search rows committed as compact replay rows
  terminal rows use final_observation
  RND/latest-frame sidecars stay identity-aligned
```

Then run one matched profile matrix:

```text
stock LightZero train_muzero profile
direct_ctree_gpu_latent compact profile
real-model compact service profile
MCTX/JAX profile-only ceiling
mock/service-tax ceiling
```

Use the same batch, actors, simulations, warmup, measured steps, RND setting,
death/autoreset setting, scalar materialization setting, and model checkpoint.
Anything else keeps the result in "interesting but not a decision" territory.

## Amdahl Risks

1. **Search gets faster and reveals observation/env/handoff.** This is already
   visible. MCTX search is cheap enough that observation, actor wall, H2D, root
   packaging, and action/replay glue become the denominator.

2. **Model parity increases search cost.** The toy JAX model is deliberately
   small. A real LightZero-equivalent model may move time back into recurrent
   inference or H2D/D2H unless the model/search loop stays resident.

3. **Active-root masking can cause recompiles or bad padding economics.** Real
   training has deaths, resets, inactive roots, and non-prefix active ids. The
   first trainer-shaped version should prefer fixed padded shapes with masks
   over shape-changing compaction unless the compile cache is proven stable.

4. **Replay materialization can erase search wins.** The compact replay proof is
   strong, but trainer-facing sample batches, stock target hooks, RND, terminal
   rows, and learner input construction must be in the measured path before a
   speedup becomes launch-relevant.

5. **Actor fanout can move the bottleneck to merge/learner.** Large actor pools
   help only if searched data generation is the wall. Once collection is fast,
   replay write/merge, sample building, learner updates, checkpoint IO, or policy
   freshness can become first-order.

6. **JAX/PyTorch ownership can become the real tax.** MCTX is attractive because
   it is compiled and array-native. If the real model remains PyTorch and every
   simulation crosses frameworks or host memory, the architecture loses the
   reason MCTX helped.

## What Not To Optimize Yet

Do not spend primary effort on these until a fresh matched denominator promotes
them:

- renderer-only work that does not also reduce stack/root/search-input handoff;
- CPU count or actor count around the current scalar LightZero collector;
- dense eager Torch tree-loop polish;
- flat-A3 CTree as the main architecture;
- more direct CTree wrapper cleanup unless a profile says output/list glue is
  hot again;
- scalar timestep materialization ergonomics in the profile-only lane;
- custom full-GPU tree kernels before model parity and active-root masking are
  validated;
- JAX/Torch bridge experiments that call PyTorch inside or between JAX search
  simulations.

The repo already has enough evidence that small patches can produce useful
`1.3x`-ish wins. The missing move is not another small patch. It is deleting a
whole object/sync boundary while preserving training semantics.

## Concrete Gates

P0 before any promotion:

- Real model parity against the current LightZero model outputs and target
  semantics.
- `CompactSearchResultV1` parity on fixed known roots: selected legal actions,
  visit policy shape/mass, root value, player perspective, and no all-one-action
  collapse.
- Non-prefix active-root ids and terminal/final-observation rows.
- RND latest-frame identity stays tied to the same compact record.
- No measured-loop JAX recompiles or hidden framework crossings.

P1 profile:

- Same manifest compares direct CTree, candidate service, MCTX ceiling, and
  mock/service-tax ceiling.
- Report env step, observation/update, root packaging, H2D, initial model,
  recurrent/search loop, D2H action payload, D2H replay payload, replay row
  commit, sampler/materialization, and learner update time separately.
- Claim full-profile speedup only from aggregate measured time, not last-call
  search telemetry.

P2 trainer-facing:

- Matched stock-vs-candidate `train_muzero` smoke.
- Learner-facing sample batches match trusted immediate replay rows.
- Policy version, search version, checkpoint id, learner seat, and observation
  backend are carried through replay.
- Coach explicitly decides the learning-quality risk before any default changes.

## Practical Recommendation

Keep MCTX/JAX as the architecture ceiling and model-parity target pressure. Do
not treat it as the product yet.

After model parity lands, spend the next serious optimizer slice on one
real-model compact service plus full compact-loop profile. If that profile says
search is still the wall, move toward compiled/fixed-shape search. If it says
observation/env/handoff is now the wall, attack resident observation/root
ownership before touching search again.

The winning shape is probably:

```text
fixed compact buffers
many row-seat roots in flight
batched real-model search/inference
compact replay rows
coarse LightZero materialization only at sampler/validation edges
```

That is the practical middle between "stock LightZero scalar objects forever"
and "rewrite the whole trainer in JAX tomorrow."

## Files Read

- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/README.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/current_state_audit_20260523.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/mctx_scaling_grid_summary_20260523e.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_mctx_result_critique_20260523e.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_next_search_backend_plan_20260523d.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_external_fast_rl_patterns_20260523b.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/large_scale_zero_architectures.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_external_gpu_rl_patterns_20260521.md`
- `src/curvyzero/training/mctx_compact_search_service.py`
- `src/curvyzero/training/compact_search_service.py`
- `src/curvyzero/training/compact_rollout_slab.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `tests/test_mctx_compact_search_service.py`
- `tests/test_compact_search_replay_contract.py`

## External Sources Checked

- MCTX: <https://github.com/google-deepmind/mctx>
- OpenSpiel AlphaZero docs: <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- EfficientZero supplement and repo:
  <https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material>,
  <https://github.com/YeWR/EfficientZero>
- MiniZero: <https://github.com/rlglab/minizero>
- KataGo analysis engine:
  <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>
- AlphaZero paper: <https://arxiv.org/abs/1712.01815>
- MuZero paper: <https://arxiv.org/abs/1911.08265>
- SEED RL: <https://arxiv.org/abs/1910.06591>
