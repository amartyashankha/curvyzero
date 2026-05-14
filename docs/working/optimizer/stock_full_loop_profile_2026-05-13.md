# Stock Full-Loop Profile

Date: 2026-05-13

Purpose: keep the current optimizer speed picture in one place. This page is
about the trusted stock LightZero CurvyTron path, not the old custom
two-seat path and not render-only microbenchmarks.

## Current Trusted Path

Use:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode profile/train
--env-variant source_state_fixed_opponent
--source-state-trail-render-mode browser_lines
```

This calls stock `lzero.entry.train_muzero`. LightZero owns collection,
MCTS/search, replay, target construction, and learner updates. CurvyZero owns
the environment wrapper, visual tensor, metadata, Modal scaffolding, and
profile hooks.

The current profile rows use one policy action per one source physics frame:

```text
decision_source_frames=1
decision_ms=16.666666666666668
policy_action_repeat_min=1
policy_action_repeat_max=1
policy_action_repeat_extra_probability=0.0
```

For optimizer profiles, background eval/GIF and final volume commits are off.
Checkpoint cadence is set high so checkpoint/eval artifacts do not dominate the
measurement.

## What Has Actually Been Optimized

1. We moved current profiling to the trusted stock LightZero path. The old
   custom `--mode two-seat-selfplay` path is historical evidence only.
2. We split learner GPU use from the frozen/fixed opponent path. The live
   MuZero policy can use CUDA while the opponent stays CPU-safe inside env
   workers.
3. We widened collection with subprocess envs. This is the biggest measured
   full-loop speedup so far.
4. We improved the CPU reference `browser_lines` renderer with dirty/incremental
   cache work. This matters in env-only and long-survival regimes, but it is
   no longer the only full-loop bottleneck.
5. We reset action cadence to one source frame per LightZero env step. This is
   mostly a correctness/fidelity fix, not a free speedup.

Important distinction: the trusted production renderer has **not** moved to GPU.
The GPU is currently used by the LightZero MuZero model/search/learner path. The
GPU-renderer work is a separate prototype lane and is not trusted training
plumbing yet.

## Fresh Speed Numbers

All rows below are source-state fixed-opponent, no-death profile rows with
`browser_lines`, background eval/GIF off, sparse checkpointing, and final volume
commit off.

| Row | Compute | Collectors | Sims | Steps | Wall sec | Steps/sec | Collector sec | Policy collect sec | MCTS sec | Learner sec | GPU max util |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | L4/T4 | 1 | 8 | 128 | 11.80 | 10.8 | 6.06 | 4.32 | 2.58 | 1.56 | 16% |
| C32 | L4/T4 + 40 CPU | 32 | 8 | 4096 | 26.66 | 153.6 | 16.60 | 7.89 | 4.50 | 2.01 | sampled 0% |
| C64 | L4/T4 + 40 CPU | 64 | 8 | 16384 | 40.11 | 408.4 | 30.39 | 13.59 | 7.38 | 1.33 | 25% |
| C96 | L4/T4 + 40 CPU | 96 | 8 | 24576 | 50.43 | 487.3 | 38.77 | 15.86 | 7.64 | 1.36 | sampled 0% |
| C64 | H100 + 40 CPU | 64 | 8 | 16384 | 51.04 | 321.0 | 37.39 | 16.49 | 8.56 | 1.81 | 5% |
| C64 | L4/T4 + 40 CPU | 64 | 16 | 16384 | 44.68 | 366.7 | 34.88 | 18.99 | 12.87 | 1.30 | sampled 0% |
| C64 | H100 + 40 CPU | 64 | 16 | 16384 | 33.55 | 488.4 | 25.76 | 14.23 | 10.05 | 1.02 | sampled n/a |
| C64 | L4/T4 + 40 CPU | 64 | 32 | 16384 | 64.11 | 255.6 | 52.11 | 34.29 | 27.54 | 1.74 | sampled n/a |
| C64 | H100 + 40 CPU | 64 | 32 | 16384 | 44.25 | 370.2 | 36.25 | 23.67 | 19.09 | 1.01 | sampled n/a |

Plain read:

- C32 is about `14.2x` faster than C1 in collected env steps/sec.
- C64 is about `37.7x` faster than C1 and about `2.7x` faster than C32.
- C96 is only about `1.19x` faster than C64, so widening collection is already
  hitting diminishing returns around this profile shape.
- H100 is slower than L4/T4 at C64/sim8. That says the row is not GPU-bound.
  It is CPU/process/search-scheduling bound enough that the bigger GPU does not
  help.
- Doubling search from sim8 to sim16 on C64/L4 only reduced throughput by about
  `10%` (`408.4 -> 366.7` steps/sec). MCTS/search got more expensive
  (`7.38s -> 12.87s`), but collection wall grew much less
  (`30.39s -> 34.88s`). Search matters, but it is not yet the whole wall.
- H100 changes from bad at sim8 to good at sim16. C64/H100/sim16 measured
  `488.4` steps/sec versus C64/L4/sim16 at `366.7`. Plain read: bigger GPU is
  not a blanket default, but it may be useful when search pressure is high.
- H100/sim32 measured `370.2` steps/sec. It is slower than H100/sim16, and MCTS
  grew to `19.1s` inside `36.2s` collector wall.
- L4/sim32 measured `255.6` steps/sec, with MCTS `27.5s` inside `52.1s`
  collector wall. H100 is about `1.45x` faster than L4 at sim32, but H100/sim16
  is still faster overall than H100/sim32.
- GPU utilization samples are weak evidence because kernels are bursty and
  `nvidia-smi` samples coarsely. But the H100 row still argues against using
  H100 for the current sim8 profile shape.

Note: `body_circles_fast` rows were dropped as a decision proxy. They are not
the target and can mislead. The only render acceleration question that matters
now is CPU `browser_lines` reference versus a byte-equivalent GPU/compiled
`browser_lines` path.

## Render-Only Context

Local env-only one-frame `browser_lines` profile:

| Steps | Wall sec | Env steps/sec | Render/observation share |
| ---: | ---: | ---: | ---: |
| 100 | 0.372 | 268.7 | about 72% render |
| 500 | 1.560 | 320.5 | about 76% render |

Older dirty-cache long no-death env-only comparison:

| Steps | Before cache | After cache | Speedup |
| ---: | ---: | ---: | ---: |
| 500 | 39.1s | 10.5s | about 3.7x |
| 2000 | 175.9s | 46.9s | about 3.8x |

Plain read: render optimization is real, especially for long trajectories, but
the current full stock loop is not simply "render only." The stock loop also
pays for LightZero collection, policy action selection, MCTS/search, model
inference, replay sampling, learner calls, and subprocess overhead.

## Full Optimizer Map

The current C64/C96 rows say the biggest wall bucket is still collection. That
bucket includes env-manager waiting/stepping, policy action selection, MCTS
calls, model calls, and episode assembly. The named sub-buckets show search and
policy collection are significant, but learner is small in these short profile
rows.

Do not add the nested timers together. `policy_forward_collect` sits inside
`collector_collect`, and `mcts_search` sits inside policy collection.

The optimizer surface is broader than render/search/game engine. Track these
lanes:

| Lane | What it includes | Current read | Risk/ease |
| --- | --- | --- | --- |
| Browser-lines observation | RGB source-state render, bonus sprites, luma/downsample, stack packing, possible GPU/compiled replacement | real cost in env-only and long trajectories; full-loop fraction still needs cleaner measurement | high upside, medium/high fidelity risk |
| Game engine | physics ticks, collision, bonuses, reward, reset, no-death long horizon | likely smaller than render in env-only rows, but action cadence changed the denominator | medium ease if profiling finds a hot subpart |
| LightZero collection | env manager, subprocess IPC, collector batching, episode assembly | biggest measured full-loop wall bucket | medium ease; width helps but plateaus |
| MCTS/search/model | tree bookkeeping, root/recurrent inference batching, sim count, CPU/GPU split | sim16 makes H100 useful; sim8 does not | medium/high, algorithm contract must stay unchanged |
| Replay/learner/targets | replay push/sample, target construction, learner update, batch size | small in short profile rows; long rows that never reach learner are not full-loop proof | lower priority until collection/search is cleaner |
| Hardware layout | CPU count, collector width, L4 vs H100, multi-GPU later | C64/C96 good; H100 only promising at higher sims so far | easy to test, can waste money if guessed |
| Artifacts/observability | checkpoint, eval, GIF, volume writes, telemetry | should be sparse/off for profiles; important for real runs but amortized | easy to control |
| Training shape | self-play batch per learner update, sims, horizon, checkpoint cadence | affects both speed and learning quality; Coach owns learning claims | high impact but must be coordinated |
| Distributed/fanout | separate collect workers, chunked GameSegments, merge/import into replay, learner freshness | potentially large future win; not trusted for overnight | high difficulty and high learning risk |

Current best optimizer conclusion:

- Use L4/T4 + 40 CPU for current sim8 profiles and likely near-term training
  probes.
- Use around C64 to C96 collectors for stock fixed-opponent throughput tests.
- Do not recommend H100 for sim8. Do consider H100 for sim16/sim32 if Coach
  chooses higher-search training, because the first sim16 H100 row beat the L4
  row materially.
- Next useful rows should increase search pressure (`num_simulations=32`) and
  compare L4/T4 vs H100 again.
- If future policies survive much longer, render and observation become more
  important again. Keep no-death/long profiles as a standing guardrail.

## 2026-05-13 Direct Fast Hot-Path Rerun

After wiring `body_circles_fast` to the direct gray64 player-perspective helper
inside the stock fixed-opponent wrapper, I reran matched stock LightZero
profiles. These are real `lzero.entry.train_muzero` profiles, not env-only
microbenchmarks.

Common scope:

- `compute=gpu-l4-t4-cpu40`
- `env_variant=source_state_fixed_opponent`
- `opponent_policy_kind=fixed_straight`
- `opponent_use_cuda=false`
- `reward_variant=survival_plus_bonus_no_outcome`
- `num_simulations=8`
- `batch_size=32`
- `stop_after_learner_train_calls=10`
- background eval/GIF off
- profile final volume commit off

| row | render | C | death | env steps | wall | steps/sec | collector | MCTS | policy forward | learner |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `opt0513c-brow-n-c32` | `browser_lines` | 32 | normal | 5,072 | 37.25s | 136.2 | 26.19s | 9.38s | 14.39s | 2.63s |
| `opt0513c-fast-n-c32` | `body_circles_fast` direct gray64 | 32 | normal | 5,072 | 24.42s | 207.7 | 14.36s | 6.86s | 10.75s | 2.32s |
| `opt0513c-brow-n-c64` | `browser_lines` | 64 | normal | 10,131 | 39.85s | 254.2 | 27.10s | 7.83s | 13.60s | 2.30s |
| `opt0513c-fast-n-c64` | `body_circles_fast` direct gray64 | 64 | normal | 10,131 | 31.20s | 324.8 | 19.92s | 8.53s | 14.26s | 1.97s |
| `opt0513c-brow-nd-c32` | `browser_lines` | 32 | no-death | 16,384 | 52.57s | 311.6 | 43.60s | 15.35s | 23.53s | 1.97s |
| `opt0513c-fast-nd-c32` | `body_circles_fast` direct gray64 | 32 | no-death | 16,384 | 39.10s | 419.1 | 29.65s | 13.57s | 21.06s | 2.18s |

Plain read:

- The direct fast path is a real full-loop speedup: about `1.53x` at C32
  normal, `1.28x` at C64 normal, and `1.34x` at C32 no-death.
- This does not mean render is the only bottleneck. Even after the fast render
  path, collector/search/policy-forward still dominate wall time.
- The fast path is an approximation, not CPU-reference fidelity. It is now a
  reasonable candidate for paired training rows because the 212-run matched
  learning read did not show a browser-vs-fast gap.
- Browser-lines remains the reference surface. Fast direct gray64 is a speed
  candidate, not a replacement by decree.

212-run knob reconciliation:

- Render: same-checkpoint `browser_lines` and `body_circles_fast` rows were
  basically tied in learning readout. This weakens the old "full fidelity at
  all costs" stance. Approximate render paths are allowed candidates if they
  keep heads, trails, walls, and bonus identities distinguishable and are tested
  as matched rows.
- Learner `batch_size`: matched 212-run contrasts were negative for `batch64`
  versus `batch32`: median latest eval about `-17` steps and median latest
  iteration about `-10k`. Learner time is also small in the short full-loop
  profiles, so there is no current speed reason to pay that learning risk.
  Keep `batch32`.
- Collector width: matched 212-run contrasts for `collector64` were small and
  mixed on eval (`+1.9` median latest steps) but positive on latest iteration
  (`+10k`). Optimizer profiles also show much better throughput from wider
  collection. This remains the cleanest speed knob to probe.
- Search sims: `sim16` was negative in matched 212-run contrasts and slower in
  profiles. Keep `sim8` as default; use `sim16` only as a small search-sensitivity
  sentinel or if Coach wants to test higher-search quality.

The browser-lines GPU-render prototype should stay active but separate. It has shown
`20x-46x` renderer-side wins versus scalar CPU production render in small
no-bonus parity-ish probes, but it has not yet beaten the current CPU dirty
cache inside the stock LightZero loop, and bonus sprite parity is still open.
Use Amdahl directly:

```text
overall speedup = 1 / ((1 - render_fraction) + render_fraction / render_speedup)
```

If render is only `20%` of full wall, even a `10x` render win gives only about
`1.22x` overall. If render is `60%`, the same render win gives about `2.17x`.
That is why long `browser_lines` rows with real worker telemetry matter. They
estimate the render fraction in the long-survival regime. Approximate render
rows are not enough.

## Next Work

- Active profile calls:
  - C64/H100/sim32/browser-lines:
    `opt-amdahl-c64-h100-sim32-browser256-20260513a`,
    `fc-01KRHYZPAB1XDF6JW7CBAGV14X`.
  - C64/L4/sim32/browser-lines:
    `opt-amdahl-c64-l4-sim32-browser256-20260513a`,
    `fc-01KRHZ09QB8CPXV0SH13D1Q32K`.
  - C64/L4/sim32/browser-lines completed:
    `255.6` env steps/sec, MCTS `27.5s`, collector `52.1s`.
  - C64/H100/sim32/browser-lines completed:
    `370.2` env steps/sec, MCTS `19.1s`, collector `36.2s`.
  - C32/L4/browser-lines long `source_max_steps=1024` rerun for at least one
    learner call:
    `opt-bl-l4-c32-long1024-loop1-20260514a`,
    `fc-01KRJ7KQTVQZ55ZHFRVFR08675`.
  - Long C64/L4/sim8/browser-lines at `source_max_steps=1024` collected
    `65,536` env steps but failed before learner with
    `ValueError: 'a' and 'p' must have same size`. Treat its collection timing
    as partial evidence only, not a full-loop training profile.
- Run paired C64/L4 and C64/H100 rows at sim16/sim32 to check whether H100
  becomes useful when search pressure is higher.
- Diagnose why the C64/L4 1024-step `browser_lines` row collected episodes but
  did not reach learner. Rerun a browser-lines-only long row that reaches at
  least one learner call.
- Continue GPU browser-lines work only against exact CPU-reference parity,
  including bonus sprites and real-state fixtures. Do not use `body_circles_fast`
  as a proxy.
- Add a finer split for collection residuals:
  `collector_collect - policy_forward_collect`,
  `policy_forward_collect - mcts_search`, and
  `mcts_search - model_recurrent_inference`.
- Keep profiling artifacts out of live Coach runs and website/GIF surfaces.
