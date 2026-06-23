# Subagent Amdahl Sanity

Date: 2026-05-21

Scope: math and optimization-priority critique only. No code changes and no live Coach
training runs touched.

## Bottom Line

The current batched GPU manager is now a real full-loop profile win at wider
collector counts, but it is not a 10x story. The useful way to read the latest
rows is:

- Subprocess CPU-oracle is a strong baseline because it hides scalar env and
  observation work across workers.
- Batched GPU real observation now beats fresh subprocess CPU-oracle at C128
  and C256 in the no-death, no-RND, sim2 profile shape.
- Zero-observation rows show the ceiling for "same stock loop, but observation
  is free-ish." That ceiling is only about `1.25x` above the latest C512 real
  row, and about `1.6x` above the C128/C256 real rows.
- Therefore, pure render/stack cleanup has bounded value. It still matters, but
  a 10x win requires changing the rest of the loop: manager architecture,
  actor parallelism, policy/search batching, and avoiding scalar Python/NumPy
  timestep churn.

## Trusted Throughput Anchors

Use these as fixed-workload throughput ratios, not as summed timer buckets.

| width | CPU-oracle subprocess | batched GPU real obs | batched GPU zero obs | read |
|---|---:|---:|---:|---|
| C64 early | `883.03 steps/s` | `416.89 steps/s` | n/a | one-process batched GPU lost badly to subprocess parallelism |
| C128 | `857.48 steps/s` | `978.96 steps/s` | `1557.42 steps/s` | GPU real is `1.14x` CPU; zero is `1.59x` real |
| C256 prior/matched | `722.43 steps/s` | `1193.48 steps/s` | `1748.18 steps/s` | GPU real is `1.65x` CPU; zero is `1.46x` real |
| C256 post-patch | n/a | `1096.18 steps/s` | `1735.15 steps/s` | noisy/neutral versus prior; zero is `1.58x` real |
| C512 prior/post | no matched sim2 CPU anchor in docs | `1352.47 -> 1439.84 steps/s` | `1805.22 steps/s` | post-patch real improved about `6.5%`; zero is now only `1.25x` real |

C512 should not be claimed as a direct CPU-oracle speedup without a matched C512
CPU-oracle sim2 row. The C512 duplicate-seed CPU-oracle numbers in the docs are
useful variance context, but they are sim4/background diagnostics rather than a
clean C512 sim2 A/B.

## Amdahl Read

For fixed env-step workloads:

```text
wall_time ~= env_steps / steps_per_sec
max speedup from perfecting observation ~= zero_observation_steps_per_sec / real_observation_steps_per_sec
observation tax share of current real wall ~= 1 - real_steps_per_sec / zero_steps_per_sec
```

This gives:

| width | zero/real ceiling | implied real-wall observation tax |
|---|---:|---:|
| C128 | `1.59x` | about `37%` |
| C256 prior | `1.46x` | about `32%` |
| C256 post-patch | `1.58x` | about `37%` |
| C512 post-patch | `1.25x` | about `20%` |

The C512 row is the clearest warning against over-investing in renderer-only
work. If the current real C512 row is `1439.84 steps/s`, a perfect observation
path under the current stock-loop shape is bounded near `1805.22 steps/s`.
That is useful, but it is not remotely 10x.

The C128/C256 rows still show meaningful observation headroom, but even there
perfect observation would only move the current real GPU rows by about
`1.5-1.6x`. After that, the remaining wall is policy forward, MCTS/search,
manager/env stepping, scalar timestep construction, replay/learner overhead,
and host synchronization.

## Inclusive Timer Caution

Do not add `collector_collect`, `policy_forward_collect`, `mcts_search`,
manager step, renderer aggregate, and device render as if they are disjoint.
The docs strongly suggest several of these are nested or overlapping views of
the same wall time:

- `collector_collect` is the broad collection envelope.
- policy forward and MCTS happen during collection.
- manager step is part of collection.
- renderer aggregate and device render are sub-buckets of manager/observation
  work, not extra time to add on top.

The safe use of these timers is directional attribution:

- Early C64 timed batched GPU still looked observation/render-stack dominated:
  manager step about `109.44s`, renderer aggregate about `92.02s`, device render
  about `82.28s`, with policy forward about `27.67s` and MCTS about `8.23s`.
- The newer zero-observation rows are the cleaner Amdahl instrument because
  they remove most observation work while preserving stock collection/search/
  replay/learner structure.

## CPU-Oracle Versus Batched GPU

The story changed with width:

- At C64, subprocess CPU-oracle was still much faster: `883.03` versus
  `416.89 steps/s`. That was the "one-process batched GPU manager lost the
  subprocess CPU parallelism" warning.
- At C128, batched GPU real observation crossed over modestly:
  `978.96` versus `857.48 steps/s`.
- At C256, batched GPU real observation won more clearly in the prior matched
  row: `1193.48` versus `722.43 steps/s`. The post-patch C256 real row
  regressed/noised to `1096.18`, so do not over-read that patch as a general
  win.
- At C512, the latest real GPU row is the best listed profile probe
  (`1439.84 steps/s`), but the docs do not include a clean matched CPU-oracle
  C512 sim2 anchor.

Plain explanation: subprocess CPU-oracle is good because it runs many scalar
workers in parallel. Batched GPU is good because it amortizes observation over
a big surface. The current batched GPU manager is still only one process and
still returns scalar LightZero timesteps, so it wins only once the batch is wide
enough and still leaves a lot of non-observation work in the loop.

## What A 10x Win Would Require

A 10x claim needs an explicit baseline:

- 10x over C256 CPU-oracle `722.43 steps/s` means roughly `7224 steps/s`.
- 10x over C128 CPU-oracle `857.48 steps/s` means roughly `8575 steps/s`.
- 10x over the current C512 real GPU row `1439.84 steps/s` means roughly
  `14398 steps/s`.

The current zero-observation ceiling is only about `1.6-1.8k steps/s`. So:

- Relative to CPU-oracle baselines, even "free observation" still leaves about
  `4-5x` additional speedup needed for a 10x result.
- Relative to the best current real GPU row, "free observation" leaves about
  `8x` additional speedup needed.

That means a real 10x probably needs one of these architecture moves:

- preserve actor parallelism while batching GPU render requests, instead of one
  blocking batched manager;
- keep more env/observation/reward/reset state on accelerator;
- batch policy and search more aggressively across roots;
- reduce scalar Python timestep construction and per-step payload churn;
- make RND and normal-death paths pass without adding a cadence wall.

Renderer cleanup can contribute, but it cannot carry the 10x alone under the
current stock-loop shape.

## Recommended Priority

1. Keep Coach/default training on stock CPU-oracle until the batched manager has
   the missing semantic gates: normal-death autoreset, final observation, RND
   latest-frame extraction, registry/backend identity, and no hidden fallback.
2. Use C512 real-vs-zero repeats as the main near-term Amdahl probe. If C512
   real stays near `80%` of zero, move on from renderer-only work.
3. Run C768 real-vs-zero only to see whether width still buys throughput or the
   one-process manager saturates.
4. Start search/topology sweeps next: C256/C512/C768 across sim2/sim4/sim8 with
   matched CPU-oracle anchors where comparison matters.
5. Keep RND separate. Measure no-RND versus RND update10/update100 after the
   observation/backend comparison is stable.
6. Treat hybrid actor-parallel plus batched-GPU-render service as the next
   serious architecture candidate if C512/C768 zero rows remain far below the
   desired 10x target.

## One-Sentence Read

The batched GPU path is now a promising profile lane and a real C128/C256
full-loop win over fresh subprocess CPU-oracle controls, but zero-observation
ceilings show that the next order-of-magnitude win must come from loop
architecture, not just making the current renderer faster.
