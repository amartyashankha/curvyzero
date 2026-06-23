# H100/L4 Optimizer Handoff

Date: 2026-05-16

Status: historical hardware/cost handoff. Keep the production-surface warnings,
but do not treat this as the current optimizer search-service handoff.

Current 2026-05-23 correction:

```text
Coach training stays on stock LightZero train_muzero with
source_state_fixed_opponent and cpu_oracle observations.

The active optimizer lane is profile-only compact search/dataflow. The remote
compact_torch_search_service smoke now passes replay proof, but it is not a
Coach backend. The latest H100 B512/A16/sim16 profile says compact Torch is only
around direct CTree speed to about 1.4x faster, with most time inside the eager
Torch tree/recurrent loop.
```

## Plain Read

The r18fresh 18-run batch used the correct current production observation
semantics:

```text
browser_lines + simple_symbols -> cpu_oracle -> [4,64,64] stack
```

It did **not** use the new batched GPU observation renderer. That is expected:
the batched GPU path is still an optimizer sidecar/prototype. The existing
trainer flag `policy_observation_backend=jax_gpu` is the old scalar hook, and it
has already profiled slower than CPU. Do not flip that flag for production.

The optimizer's latest decision is:

- `float32` is now the aggressive default for the **batched GPU observation
  boundary sidecar**.
- `float64` is the exact-parity/debug reference.
- The known float32 mismatch was one pixel by one luma value (`100` vs `101`) at
  a trail edge. That is not meaningful enough to block a learned policy
  observation candidate by itself.
- The reason not to use GPU observations in training today is integration, not
  that float32 is unacceptable.

## What r18fresh Actually Ran

Manifest:
`artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json`

Shared settings across all 18 rows:

| Field | Value |
| --- | --- |
| compute | `gpu-h100-cpu40` |
| trainer app | `curvyzero-lightzero-curvytron-visual-survival-train-v2` |
| train entry | `lightzero_curvytron_visual_survival_h100_cpu40` |
| env variant | `source_state_fixed_opponent` |
| policy trail render | `browser_lines` |
| policy bonus render | `simple_symbols` |
| policy observation backend | default `cpu_oracle` |
| learner batch size | `32` |
| collector env count | `256` |
| MCTS simulations | `8` |
| checkpoint cadence | every `10000` learner iterations |
| source max steps | `1048576` |
| learner seat mode | `random_per_episode` |

Important meaning: `gpu-h100-cpu40` put the LightZero model/search/learner work
on an H100. It did not move CurvyTron observation rendering to the GPU.

The code path is:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
  -> lightzero_curvytron_visual_survival_h100_cpu40
  -> _run_visual_survival_train(compute="gpu-h100-cpu40")
  -> stock LightZero train_muzero
  -> source_state_fixed_opponent env
  -> cpu_oracle browser_lines + simple_symbols observations
```

The matching L4 function exists:

```text
lightzero_curvytron_visual_survival_gpu_cpu40
  -> _run_visual_survival_train(compute="gpu-l4-t4-cpu40")
```

That gives us a clean place to run matched H100/L4 profiles without changing the
learning contract.

## Actual H100 Throughput From r18fresh

The r18fresh rows reached roughly `270k..311k` learner iterations. Using the
checkpoint-window runtime from the local status artifact
`/tmp/r18fresh_eval_status_clean.json`, the 18-row H100 batch averaged about
`31.5k` learner iterations/hour.

Observed range:

| Slice | Approx rate |
| --- | ---: |
| all rows mean | `31.5k` learner iters/hour |
| fastest reward family, sparse outcome | `37.9k` learner iters/hour |
| no-outcome family | `29.9k` learner iters/hour |
| plus-outcome family | `26.6k` learner iters/hour |
| checkpoint cadence | about `14..25` minutes per `10000` learner iterations |

This is useful for wall-clock planning, but it is not an L4 comparison because
all 18 rows used H100.

## What We Know About H100 vs L4

2026-05-16 update: current-surface matched L4-vs-H100 profile data now exists.
These rows use the current production policy surface:

```text
browser_lines + simple_symbols -> cpu_oracle -> [4,64,64] stack
```

They are profile rows, not live learning runs: no LightZero eval, no background
eval, no GIF, checkpoint cadence set beyond the run, and no-death mode enabled
to force long trajectories.

Fresh grid:

| Profile row | L4/T4 CPU40 | H100 CPU40 | Read |
| --- | ---: | ---: | --- |
| C128, batch32, sim8, no-death512 | `591.39` env steps/s | `755.28` env steps/s | H100 faster |
| C128, batch64, sim8, no-death512 | `583.06` env steps/s | `523.83` env steps/s | L4 faster in this bad H100/batch64 shape |
| C256, batch32, sim8, no-death512 | `601.92` env steps/s | `1001.94` env steps/s | best H100 row |
| C256, batch64, sim8, no-death512 | `713.83` env steps/s | `793.63` env steps/s | best L4 row |

Plain read: H100 is faster in absolute throughput at its best row, but L4 is not
catastrophically slower. Best L4 is `713.83`; best H100 is `1001.94`. L4
throughput is about `28.8%` lower, and the same amount of work takes about
`1.40x` as long. That is acceptable for broad cheaper experiments.

Concrete Coach recommendation for the next L4 run:

```text
gpu-l4-t4-cpu40
collector_env_num=256
n_episode=256
batch_size=64
num_simulations=8
browser_lines + simple_symbols + cpu_oracle
sparse checkpoints/evals
```

Important caveat: batch64 helped on L4/C256 but hurt on H100. This is a
hardware/profile-shape recommendation, not a universal "batch64 is better" rule.

Historical speed-only profiles are still useful as a sanity check, but they used
the old CPU `body_circles_fast + simple_symbols` surface. They should not be
treated as current production observation evidence.

Historical profile hints:

| Profile row | L4/T4 CPU40 | H100 CPU40 | Read |
| --- | ---: | ---: | --- |
| C64 sim8 fast path | `591.6` env steps/s | `498.5` env steps/s | H100 worse at this narrow shape |
| C256 sim8 fast path | `805.6` env steps/s | `1081.9` env steps/s | H100 about `1.34x` faster |
| best sim8 fast path | `946.1` env steps/s at C384 | `1204.0` env steps/s at C768 | H100 about `1.27x` faster |
| C64 sim16 fast path | `376.1` env steps/s | `549.1` env steps/s | H100 about `1.46x` faster |
| C64 sim32 fast path | `354.7` env steps/s | `528.2` env steps/s | H100 about `1.49x` faster |

Interpretation: old rows made H100 look mixed at sim8; the fresh current-surface
rows say the next broad lane should prefer L4 for cost/parallelism, while H100
stays available for expensive sentinels or heavier-search profiles. H100 still
is not GPU observation rendering; it accelerates the LightZero
model/search/learner side.

Do not use current Modal price claims here without rechecking them. The decision
rule is simple: L4 is more cost-effective if it is less than the current
H100/L4 hourly-price ratio slower. Recheck live prices before a large spend.

## Batched GPU Observation Candidate

Optimizer sidecar:
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

Current sidecar default:

```text
DEFAULT_BOUNDARY_GEOMETRY_DTYPE = float32
```

Latest H100 sidecar numbers:

| Boundary row | Candidate observation | Reference/debug |
| --- | ---: | ---: |
| B64/S1024 | float32 `255ms` | float64 `379ms` |
| B256/S1024 | float32 `1.14s` | float64 `1.38s` |
| B64/S1024 float64 vs CPU reference | `379ms` | CPU render+stack `1.09s` |
| B64/S1024 timeout/autoreset float64 | median `376ms`, p95 `920ms` | exact terminal stack parity |

This is the promising next architecture. It still is not the trainer default:
the sidecar reads frames back to host and has not passed a full LightZero loop
canary.

## Concrete Optimizer Ask

Use the small manifest tooling instead of hand-launching rows:

```text
docs/working/optimizer/modal_profile_tooling_2026-05-16.md
scripts/build_curvytron_profile_grid.py
scripts/run_curvytron_optimizer_profile_manifest.py
scripts/summarize_curvytron_optimizer_profile_results.py
```

The immediate matched profile has been run. For follow-up profiles, keep using
the current production surface, not the old body-circles surface:

```text
--mode profile
--env-variant source_state_fixed_opponent
--source-state-trail-render-mode browser_lines
--source-state-bonus-render-mode simple_symbols
--policy-observation-backend cpu_oracle
--batch-size 64
--collector-env-num 256
--num-simulations 8
--source-max-steps 1048576
```

Current broad training default:

```text
gpu-l4-t4-cpu40
```

Use H100 as an explicit sentinel/ablation, not the default. Run enough learner
calls to get stable buckets, but keep checkpoint/eval/GIF I/O off for pure speed
comparisons. Use direct blocking rows for small grids. Use `--detach` with
`--profile-spawn` for background grids. Do not use `--profile-spawn` without
`--detach`.

Profile buckets to report:

- env step;
- observation render/stack;
- frozen-opponent inference;
- policy collect;
- MCTS/search;
- replay/sample;
- learner/update;
- checkpoint/commit;
- background eval/GIF/artifact I/O if enabled.

Also run the same comparison for the future batched GPU boundary once wired:

```text
batched_gpu float32
batched_gpu float64 debug/reference
cpu_oracle
```

## Do Not Do

- Do not infer GPU observation rendering just because `compute=gpu-h100-cpu40`.
- Do not use `body_circles_fast` as a production policy surface.
- Do not switch production to scalar `policy_observation_backend=jax_gpu`.
- Do not copy old fast-stock launch commands unless they are explicitly labeled
  historical/speed-only controls.
- Do not claim the batched GPU observation path is trainer-ready until a full
  LightZero loop canary passes.

## Open Questions

- Does C512 beat C256 enough to justify the extra process pressure?
- Is `policy_observation_backend=cpu_oracle` written explicitly enough in future
  manifests, or should we force it into every launch artifact even when it is the
  default?
- How much of r18fresh wall time was checkpoint/eval/GIF overhead versus train
  loop work?
- After the batched GPU backend is wired, does host readback erase too much of
  the render win?
- Does L4 remain cost-effective for sim8 but not for higher-search sentinel rows?
