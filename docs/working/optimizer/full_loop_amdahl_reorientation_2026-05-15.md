# Full-Loop Amdahl Reorientation

Date: 2026-05-15

Purpose: answer the simple question plainly: what have we actually profiled,
what is still estimated, and what does Amdahl say in the current stock
LightZero CurvyTron path?

## Plain Answer

Yes, we have run the real stock LightZero full loop. The fresh 2026-05-15 rows
below call `lzero.entry.train_muzero` through:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode profile
--env-variant source_state_fixed_opponent
--source-state-trail-render-mode browser_lines
--source-state-bonus-render-mode simple_symbols
```

They are not live training jobs. Eval/GIF and final volume commit were off.
The profile used no-death blank/noop opponent to force long enough trajectories
inside each profile row.

We have now run a full loop with the experimental scalar GPU observation backend.
It is wired, but it is slower than the CPU oracle. The isolated batched GPU
renderer remains promising; the scalar env-wrapper hook is the wrong shape.
See [GPU observation full-loop canary](gpu_observation_full_loop_canary_2026-05-15.md).

## Fresh Full-Loop Rows

Common settings:

- compute: `gpu-l4-t4-cpu40`
- stock LightZero `train_muzero`
- env manager: `subprocess`
- search: `num_simulations=8`
- learner batch: `256`
- stop after `12` learner train calls
- no eval/GIF/checkpoint artifact noise
- current policy observation surface: `browser_lines + simple_symbols`

| row | collectors | env steps | wall | steps/s | collector | policy collect | MCTS | learner | replay sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c8-sim8-current` | 8 | 4,096 | `36.60s` | `111.9` | `25.45s` | `16.59s` | `12.49s` | `3.54s` | `2.53s` |
| `c32-sim8-current` | 32 | 16,384 | `50.34s` | `325.5` | `37.78s` | `22.43s` | `14.55s` | `3.41s` | `2.45s` |

Plain read:

- C32 processes 4x as many env steps as C8, but wall only grows
  `36.60s -> 50.34s`, so widening collection still helps a lot.
- C32 is about `2.91x` higher throughput than C8 in this fresh row.
- MCTS/search is a major named bucket: `12.49s` at C8 and `14.55s` at C32.
- Learner and replay sample are not the current wall-clock bottleneck in these
  short profile rows.

## Env-Side Timing Inside Full Loop

These are per-env telemetry means. Do not sum them directly into wall time,
because subprocess env work happens in parallel.

| row | mean observation/step | mean vector step/step | mean env step before info |
| --- | ---: | ---: | ---: |
| C8 | `5.90ms` | `1.68ms` | `7.64ms` |
| C32 | `6.46ms` | `1.86ms` | `8.39ms` |

Plain read:

- Inside each env worker, observation/render is still the largest env-side
  piece.
- But full-loop wall is not one env worker. With 8 or 32 subprocess workers,
  observation work is parallelized across workers, while policy/MCTS/collector
  orchestration is still a central wall-clock bucket.

## Env-Only Amdahl By Trajectory Length

Fresh local no-death env-only profile, current `browser_lines + simple_symbols`
surface:

```text
artifacts/local/curvytron_render_profiles/current_browser_lines_symbols_amdahl_20260515.json
```

| steps | wall | render | render fraction | max speedup if render free | speedup if render 10x faster |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | `0.253s` | `0.195s` | `77.1%` | `4.36x` | `3.27x` |
| 200 | `0.521s` | `0.400s` | `76.7%` | `4.29x` | `3.23x` |
| 500 | `1.282s` | `0.993s` | `77.5%` | `4.44x` | `3.31x` |
| 1000 | `2.597s` | `2.018s` | `77.7%` | `4.49x` | `3.33x` |
| 2000 | `5.554s` | `4.269s` | `76.9%` | `4.32x` | `3.24x` |

Plain read:

- For a single long env rollout, render is absolutely worth optimizing.
- Render-only optimization in this narrow env-only regime tops out around
  `4.3x-4.5x` even if render became free.
- A 10x render win would look like about `3.2x-3.3x` in env-only rollouts.

## Full-Loop Amdahl Read

The full loop behaves differently from env-only rollouts.

At C32, each env runs about 512 steps. Observation per worker is roughly:

```text
512 * 6.46ms ~= 3.31s per worker
```

But the C32 full-loop wall is `50.34s`, and collector wall is `37.78s`.

That means current CPU observation/render is still expensive inside each worker,
but it is not the whole full-loop wall. It is parallelized across env workers.
MCTS/search/policy collection and collector orchestration now dominate more of
the wall-clock picture.

Approximate full-loop implication:

- Making render 10x faster probably does **not** make the current C32 stock
  loop 3x faster.
- The expected current-loop win from only making render faster through a
  batched GPU path is likely closer to a modest full-loop gain unless the
  integration also removes broader collector/host overhead.
- This matches the earlier full-loop fast-observation row: a large env
  observation improvement only produced about `1.17x` end-to-end at C8.

## Concrete Recommendations

1. Keep current training observation surface at `browser_lines + simple_symbols`.
   Do not fall back to `body_circles_fast` as the main recommendation.

2. Do not use scalar `policy_observation_backend=jax_gpu` as the production
   training backend. It now runs through `train_muzero`, but the matched 512-step
   canary was `8.03` steps/s versus `32.94` steps/s for `cpu_oracle`. Observation
   time was `80.31ms` versus `4.42ms`.

3. If pursuing GPU observation rendering, wire it as a **batched backend**.
   The promising shape is B64/H100: `10.88ms` end-to-end for 64 real-env
   frames at S64, or `31.92ms` at S256. A scalar hook is the wrong shape.

4. For near-term stock LightZero training/profile runs, prioritize collector
   width and search settings before renderer rewrites:
   - C32 is a safe lower bound profile shape.
   - Existing older rows showed C64/C96 can improve throughput further, with
     diminishing returns.
   - Use L4/T4+40CPU for sim8 unless a fresh H100 row proves otherwise.
   - Consider H100 when raising search pressure to sim16/sim32.

5. Run the next full-loop matrix before making a Coach-facing final config:
   - C32/C64/C96 at sim8 on L4/T4+40CPU.
   - C64 sim16 on L4/T4 and H100.
   - One C64/C96 row with telemetry stride reduced to confirm telemetry itself
     is not distorting wall time.
   - Later, after a batched GPU backend exists, rerun the same matrix with that
     batched backend flag/name, not the current scalar `jax_gpu` canary.

6. Treat larger observations as a separate experiment. Renderer-only H100 rows
   make `96x96` plausible, but full-loop model/search/replay cost is unmeasured.
   Keep `[4,64,64]` for the next training recommendation.

7. Clean `/runs` inode pressure before serious long runs. The profile container
   warned `/runs` was about `97.7%` of inode capacity. That is not a speed
   conclusion, but it is an operational risk.

## Next Prototype Gate

The next optimizer prototype should test a boundary, not just a kernel.

Profile-only mock collector shape:

1. Create many source-state env rows in one parent process.
2. Step CPU physics normally.
3. Gather compact render state after each step/reset.
4. Render both controlled-player views in one batched backend call.
5. Update `[4,64,64]` stacks and terminal/final observations.
6. Optionally run a policy-forward stub and replay-copy stub.
7. Compare wall time against CPU `cpu_oracle`.

This does not call `train_muzero`, does not touch live runs, and does not change
stock defaults. It answers the practical Amdahl question: does batched GPU
observation help after packing, readback, stack update, policy/search, and reset
costs are counted?
