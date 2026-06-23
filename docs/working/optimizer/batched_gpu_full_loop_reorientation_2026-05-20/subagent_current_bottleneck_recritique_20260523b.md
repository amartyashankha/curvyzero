# Subagent Current Bottleneck Recritique, 2026-05-23b

Scope: read-only optimizer critique plus this report. I did not touch live
Coach runs, Modal volumes, checkpoints, evals, GIFs, tournaments, or source
code.

## Short Answer

We are ready for an optimizer-lane architecture reorientation. We are not ready
for a Coach/training architecture cutover.

The latest same-denominator H100 evidence says:

- Direct LightZero CTree still beats the current eager compact Torch service.
- Service-tax and mock rows still show headroom.
- Actual Coach training speedup is still unproven.

So the next move should not be "promote compact Torch" or "polish the eager
Torch tree." The next serious move is a gated compact ownership/search-service
architecture: compact root/result/replay identity stays alive through closed
loop, and a real fixed-shape or array-native search backend must beat direct
CTree on the same sim16 and sim32 denominator before it earns more investment.

## 1. Current Hot Path Being Measured

The latest hot path is the profile-only compact boundary in:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/run_curvytron_hybrid_observation_profile_manifest.py
```

The headline artifacts are:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_direct_20260523a
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_compacttorch_20260523a
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_servicetax_20260523a
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_mock_20260523a
```

Common shape:

```text
H100 and L4, B512, actor_count=16, 80 measured steps, 20 warmup,
sim16/sim32, host_uint8_pinned input, scalar timestep materialization off,
compact replay proof on, root_noise_weight=0.0,
profile_only=true, calls_train_muzero=false, touches_live_runs=false.
```

Plain path:

```text
previous selected actions
-> CPU CurvyTron profile env step
-> compact row/player sidecars
-> JAX persistent renderer builds a uint8 [512,2,4,64,64] stack
-> LightZero-style compact search probe
-> compact selected_action / visit_policy / root_value arrays
-> compact replay-index proof
-> next selected actions
```

This is not stock `train_muzero`. In the row JSON, `profile_only=true`,
`calls_train_muzero=false`, `stock_lightzero_integrated=false`, and
`materialized_timestep_count=0`.

Code evidence:

- `source_state_hybrid_observation_profile.py` requires a batched stack probe
  when scalar timestep materialization is off and marks results
  `profile_only=True`, `calls_train_muzero=False`.
- `source_state_batched_observation_boundary_profile.py` describes
  `_run_direct_mcts_arrays` as "profile-only direct compact arrays boundary over
  LightZero's real CTree MCTS".
- The direct CTree path still does CPU root prep and list conversion:
  `policy_logits_np.tolist()`, legal-action lists, Dirichlet-noise lists, and
  `roots.prepare(...)`.
- `_run_direct_ctree_gpu_latent_search` explicitly says CTree still needs CPU
  policy/reward/value arrays. Its sim loop calls `batch_traverse`, runs Torch
  recurrent inference, copies reward/value/policy back to CPU NumPy, listifies,
  then calls `batch_backpropagate`.
- `compact_torch_search_service.py` marks the Torch service
  `profile_only`, `not_lightzero_ctree`, and `trainer_ready=false`.

## 2. Training Numbers Versus Profile-Only Numbers

Actual Coach training-loop speed is only represented by launch/run throughput
docs, not by the H100 compact boundary grid.

Real Coach run speed from
`actual_training_speed_read_20260521.md`:

| denominator | metric |
| --- | ---: |
| `rnd-blank-sweep-fastckpt-20260519a`, L4/T4-class, stock path | mean `~18.4k` checkpoint iters/hour |
| same batch | median `~19.7k` checkpoint iters/hour |
| older `r18fresh` H100 batch | `~31.5k` learner iters/hour, not directly comparable |

Stock `train_muzero` profile rows, closer to the Coach loop but still profile
mode:

| row | stock | direct output-fast | gain |
| --- | ---: | ---: | ---: |
| no-RND C64/sim16/3-learner | `433.17 steps/sec` | `566.19 steps/sec` | `1.31x` |
| `rnd_meter_v0` hash-fixed | `351.02 steps/sec` | `448.52 steps/sec` | `1.28x` |

Latest profile-only H100 compact boundary rows from `dataflow_wave2_*_20260523a`:

| sims | direct CTree | compact Torch | service-tax | mock |
| ---: | ---: | ---: | ---: | ---: |
| 16 | `5,467 steps/sec` | `4,047 steps/sec` | `7,812 steps/sec` | `7,462 steps/sec` |
| 32 | `3,137 steps/sec` | `2,674 steps/sec` | `5,192 steps/sec` | `9,171 steps/sec` |

Those H100 numbers are optimizer profile-only evidence. They do not prove Coach
learning speed, replay/learner throughput, RND cadence, checkpoint cadence, or
run quality.

## 3. Actual Current Amdahl Bottleneck

In the trusted full-loop denominator, the bottleneck is still the LightZero
collect/search/object topology, not raw rendering:

```text
GPU model tensors
-> CPU NumPy/list root and CTree contracts
-> Python simulation loop
-> GPU recurrent inference
-> CPU NumPy/list reward/value/policy backprop
-> compact/public action output
-> replay/learner/RND object lanes
```

The stock full-loop rows support that: direct output-fast improves no-RND
`433.17 -> 566.19 steps/sec` and RND `351.02 -> 448.52 steps/sec`, but that is
only `~1.3x`, not a 5x move. The no-RND direct row still had `19.41s` collect,
`10.31s` policy collect, `8.06s` MCTS, and `2.47s` direct D2H. Output assembly
was already down to `0.077s`, so more output-wrapper polish is not the wall.

In the current profile-only compact denominator, the bottleneck is the real
search/control ownership path:

| H100 row | measured sec | probe sec | model sec | search sec |
| --- | ---: | ---: | ---: | ---: |
| direct CTree sim16 | `14.98` | `6.17` | `1.66` | `4.82` |
| direct CTree sim32 | `26.11` | `13.90` | `3.77` | `11.66` |
| compact Torch sim16 | `20.24` | `9.55` | `0.36` | `8.18` |
| compact Torch sim32 | `30.64` | `22.26` | `0.36` | `21.10` |
| service-tax sim16 | `10.49` | `2.01` | `1.64` | `0.21` |
| service-tax sim32 | `15.78` | `4.39` | `3.60` | `0.58` |
| mock sim32 | `8.93` | `0.48` | `0.35` | `0.00` |

Read:

- Direct CTree is paying real model calls plus the CPU/list/CTree boundary.
- Eager compact Torch removes CTree semantics but is slower on H100 at both sim
  counts. Its own tree/recurrent loop is the new wall.
- Service-tax keeps real model calls but removes real CTree/search updates, and
  wins `1.43x` at sim16 and `1.65x` at sim32 versus direct CTree.
- Mock gives the no-search ceiling and reaches `2.92x` over direct at sim32.

Amdahl implication: there is real headroom, but not from one leaf. At sim16,
even perfect search removal cannot give 5x because the non-search compact
profile wall remains large. At sim32, search is large enough to justify a real
search-service rewrite, but observation/ownership/replay materialization still
has to shrink for any 5x+ claim.

The separate closed-loop docs say the same thing from the other side:
refresh-off is only about `1.6-1.7x`, raw GPU draw is tiny, and the expensive
leaves are production-to-compact conversion, delta pack, renderer H2D/update,
public packaging, and search.

## 4. Wrong Next Move

Wrong next moves:

- Promote `compact_torch_search_service` or dense/eager Torch MCTS to Coach.
  It loses the latest H100 same-denominator sim16 and sim32 rows and is not
  LightZero CTree semantics.
- Spend the next main wave polishing the eager Torch tree loop without a
  fixed-shape compiled or array-native replacement plan.
- Treat service-tax, mock, MCTX/JAX, or compact replay proof rows as training
  speed. They are profile-only.
- Optimize `compact_service_replay_proof_sec` as if it were a claimed training
  hot path. In these rows it is validation cost.
- Do renderer-kernel-only work as the main plan. Raw GPU drawing is not the wall.
- Keep pushing parent-side full-copy compact render buffers. The docs show they
  move work into actor/render-state writes instead of deleting it.
- Start broad Coach/trainer rewrites before the compact path proves identity,
  terminal/final observation, RND latest-frame, replay targets, and sampler
  parity.

## 5. Best Next Big Move

Best next move: a gated compact ownership plus real search-service move, still
inside the optimizer lane.

The shape should be:

```text
compact env/row/player state
-> CompactRootBatchV1
-> real CompactSearchServiceV1 backend
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> learner/sample materialization only at coarse validated edges
```

The backend should not be the current eager compact Torch loop unless a new
same-denominator run overturns the latest result. Better candidates are:

- fixed-shape compiled Torch/Triton search with stable `[R,3]` arrays;
- array-native CTree/flat fixed-action API;
- MCTX/JAX comparator using real compact roots as architecture evidence;
- later, a many-producer batched search/inference service if compact identity
  and replay/RND gates hold.

Concrete gate before calling it a big win:

- same H100 B512/A16 sim16 and sim32 denominator;
- root noise off first, then seeded noise gate;
- `profile_only=true` until promoted;
- no scalar timestep materialization in the compact hot path;
- selected action from search step `k` drives env transition `k+1`;
- compact replay index rows rebuild the trusted target rows on demand;
- RND latest frame and terminal final observation attach to the same record;
- candidate beats direct CTree on both sim16 and sim32 with measured wall, not
  only probe roots/sec.

If that fails while service-tax/mock still show headroom, reevaluate the search
strategy instead of polishing wrappers. If it passes, then run a small matched
stock-vs-candidate full-loop smoke. Only after that should anyone talk about
Coach-facing speed.
