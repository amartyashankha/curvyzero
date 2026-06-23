# Subagent Amdahl Next Priorities

Date: 2026-05-21

Scope: prioritization critique from existing profile artifacts only. No live
runs or trainer defaults touched.

## Bottom Line

The current bottleneck is no longer cleanly "GPU render is slow." The cleaner
C256/C512 real-vs-zero pairs still bound render/observation cleanup at roughly
`1.25x-1.58x`, but the new C768 pair flips the expected result: real observation
is faster than zero observation (`1420.45` vs `1191.71` steps/s). That makes the
current wall look like one-process stock-loop saturation: policy/search,
manager barriers, scalar timestep construction, stack/update/package work, and
batch-collapse under normal death.

Render work is still visible and worth shaving where it is cheap. It is not the
path to `5-10x` under the current stock LightZero-shaped collection loop.

## Existing Throughput Anchors

Use these as fixed-workload throughput ratios where the workload is matched.
Do not sum the inclusive timers as disjoint wall buckets.

| row family | shape | real obs | zero obs | read |
|---|---:|---:|---:|---|
| C256 prior | no death, no RND, sim2 | `1193.48` | `1748.18` | zero/real `1.46x`; render-observation tax about `32%` |
| C256 post-patch | no death, no RND, sim2 | `1096.18` | `1735.15` | zero/real `1.58x`; patch was noisy at C256 |
| C512 prior/post | no death, no RND, sim2 | `1352.47 -> 1439.84` | `1805.22` | best zero/real `1.25x`; render-only headroom about `20%` |
| C512 repeat | no death, no RND, sim2 | `1379.36` | use prior `1805.22` | same regime; zero/real `1.31x` if paired to prior zero |
| C768 new | no death, no RND, sim2 | `1420.45` | `1191.71` | zero is slower; treat as saturation/pathology, not a ceiling |

The best current full-loop no-RND profile remains roughly `1.4k steps/s`
(`1439.84` at C512, `1420.45` at C768). Width no longer buys clean throughput:
C768 real is basically flat versus C512, and C768 zero gets worse.

## Amdahl Read

For matched no-death real/zero rows:

```text
max render-only speedup ~= zero_observation_steps_per_sec / real_observation_steps_per_sec
observation tax share ~= 1 - real_steps_per_sec / zero_steps_per_sec
```

That gives:

| anchor | max speedup from perfect observation | current wall removed by perfect observation |
|---|---:|---:|
| C256 prior | `1.46x` | `31.7%` |
| C256 post-patch | `1.58x` | `36.8%` |
| C512 best | `1.25x` | `20.2%` |
| C512 repeat vs prior zero | `1.31x` | `23.6%` |
| C768 measured | `0.84x` | not meaningful; zero row is slower |

The conservative conclusion: render-only work can plausibly move a C512-like
row from about `1.4k` to about `1.8k steps/s`, not to `7k+`. At C768, the
diagnostic says the rest of the loop can dominate so hard that removing render
does not improve end-to-end throughput.

## Timer Clues

The C768 pair is the most useful warning:

| C768 row | wall | collector | manager step | renderer render | stack update | policy forward | MCTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| real | `553.65s` | `544.36s` | `273.34s` | `111.55s` | `136.23s` | `185.79s` | `30.23s` |
| zero | `659.92s` | `649.53s` | `243.67s` | `0.26s` | `43.00s` | `262.48s` | `53.25s` |

Zero observation removes renderer work, but policy forward and MCTS get much
worse and wall time increases. Whether that is GPU scheduling, memory behavior,
batching shape, or run noise, it means C768 is not a clean renderer experiment.
It is evidence that stock-loop topology is now the bigger uncertainty.

## RND Meter Rows

The C512 RND meter rows are useful overhead rows, not positive-RND learning
proof. They did prove the important meter invariants: `predictor_changed=true`,
`target_changed=false`, `last_target_reward_changed=false`,
`last_target_reward_delta_abs_max=0.0`, and no small-buffer skips.

| row | RND cadence | denominator | steps/s | vs same-grid no-RND | RND collect/train/estimate |
|---|---:|---|---:|---:|---|
| C512 no-RND repeat | off | collector envstep delta | `1379.36` | baseline | n/a |
| C512 RND meter | update10 | MCTS-root fallback | `1218.94` | `0.884x` throughput, about `11.6%` slower | `9.07s / 0.95s / 0.19s` |
| C512 RND meter | update100 | MCTS-root fallback | `1234.55` | `0.895x` throughput, about `10.5%` slower | `8.66s / 3.72s / 0.13s` |

RND is not the dominant wall in these rows, but it is large enough that it must
stay a separate cadence axis. Do not mix RND overhead into renderer claims.

## Normal Death

The latest normal-death C256 gate finally completed:

| row | shape | env steps | manager steps | root batch mean | steps/s | GPU util |
|---|---|---:|---:|---:|---:|---:|
| normal-death C256 | real obs, no RND, sim2 | `36014` | `333` | `108.15` | `485.01` | `32%` |

This is a correctness/gate milestone, not a throughput regression to compare
directly against no-death C256. Death/autoreset changes the workload: active
root batch collapsed from `256` to about `108`, env steps dropped to `36014`,
and GPU utilization fell. The current bottleneck in normal death appears to be
live-row/batch-collapse plus scalar-manager semantics, not the renderer alone.

No normal-death C768 artifact exists in the current local profile directory.

## What 5-10x Requires

A `5x` result over the best current no-RND row means roughly `7.2k steps/s`.
A `10x` result means roughly `14.4k steps/s`. Against the C256 CPU-oracle
anchor (`722.43 steps/s`), `10x` still means `7.2k steps/s`.

The measured render-only ceilings top out near `1.8k steps/s`. Therefore:

- render cleanup alone is at most a useful `20-60%` local win in the current
  full-loop shape;
- `5x` needs roughly another `4x` after perfect observation;
- `10x` needs roughly another `8x` after perfect observation from the current
  best GPU lane.

The plausible `5-10x` changes are architectural:

- preserve subprocess-style actor parallelism while batching render/observation
  centrally;
- keep env state, reward, reset, stack update, and latest-frame extraction in a
  more vectorized accelerator-side form;
- reduce or bypass scalar Python timestep payload churn at the LightZero
  boundary;
- make policy/search batching scale with live roots instead of degrading at
  C768 or under normal death;
- treat RND as a separate scheduled model workload, not incidental renderer
  overhead.

## Best Next Grid

The next grid should reduce the biggest uncertainty: whether C768 is a noisy
outlier, a zero-observation path artifact, or a real stock-loop saturation
boundary.

Launched core grid:

| axis | values | reason |
|---|---|---|
| width | C512, C768 | bracket the current plateau/anomaly |
| observation | real, zero | keep Amdahl visible |
| simulations | sim2, sim4 | separate renderer cost from policy/search scaling |
| death | disabled | keep workload denominator fixed |
| RND | off | keep cadence out of the main bottleneck read |
| repeats | launched grid first; repeat only if the C768 inversion remains decision-critical | test whether the inversion is stable |

That is `8` core rows. If there is room for sidecars later, add:

| sidecar | rows | reason |
|---|---|---|
| RND meter | C512 no-RND repeat, C512 update100 | confirm the `~10-12%` overhead without rerunning update10 unless cadence is the focus |
| normal death | C256 and C512 real obs, no RND, sim2 | measure live-root collapse and autoreset overhead separately from no-death Amdahl |

Expected-outcomes matrix:

| observation | expected conclusion |
|---|---|
| C768 zero stays slower than C768 real | zero-observation is no longer a valid monotonic ceiling at high width; prioritize topology/search/manager scheduling over renderer-only work |
| sim4 widens the real-vs-zero gap | policy/search scaling is interacting with observation path; inspect batching/scheduling and GPU contention before more render kernels |
| sim4 narrows or preserves the gap while both flatten | manager/topology barrier is likely the wall; width alone is exhausted |
| C512 real gets within `10-15%` of C512 zero | renderer is mostly paid down for the current loop; move primary effort to actor parallelism and scalar-boundary reduction |
| C512 real remains `20%+` below C512 zero | targeted render/stack/package cleanup is still worth doing, but it cannot explain `5-10x` |

Decision rules:

- If C768 zero repeats remain slower than real, stop using zero-observation as
  a monotonic ceiling at high width and prioritize topology/search/manager work.
- If C512 real remains within about `20-25%` of C512 zero, stop spending primary
  effort on renderer-only kernels.
- If sim4 widens the real/zero gap, search/policy scheduling is part of the
  wall; if both real and zero flatten, manager/topology is the wall.
- If normal-death C512 collapses root batch like C256, optimize live-row
  compaction and terminal/autoreset batching before judging renderer speed.

## Priority Order

1. Treat current C512/C768 as a plateau near `1.4k steps/s`, with render-only
   upside bounded near `1.8k steps/s`.
2. Run the C512/C768 x real/zero x sim2/sim4 no-death grid before deeper render
   work.
3. Keep RND meter and normal-death rows as sidecar gates, not renderer
   benchmarks.
4. Start designing the actor-parallel plus batched-observation topology, because
   that is the first path with enough Amdahl room for `5-10x`.
