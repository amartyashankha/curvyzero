# Stock Fast-Path Scaling Grid

Date: 2026-05-14

Purpose: measure the fastest practical current stock LightZero profile shape
after the V8 fast observation wiring. This is speed evidence only.

## Scope

All rows use the trusted stock path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode profile
--env-variant source_state_fixed_opponent
```

Shared profile controls:

- reward: `survival_plus_bonus_no_outcome`;
- opponent: `fixed_straight` with `blank_canvas_noop` runtime for no-death
  profiling;
- render: `body_circles_fast`;
- bonus render: `simple_symbols`;
- learner batch: `batch_size=32`, because this is the current learning-safe
  batch from the Coach/leaderboard docs;
- no-death: enabled, to force long trajectories;
- source horizon: `source_max_steps=512`;
- profile stop: `12` learner train calls;
- env manager: `subprocess`;
- env telemetry stride: `128`;
- stock LightZero eval, background eval, background GIF, checkpoint commits,
  and final profile volume commit: off.

Do not compare this directly to older custom two-seat rows or render-only
microbenchmarks. The denominator here is collected env steps per
`train_muzero` wall time, with learner calls included.

## Launch Ledger

| attempt | compute | collectors | sims | status |
| --- | --- | ---: | ---: | --- |
| `c32-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 32 | 8 | completed |
| `c64-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 64 | 8 | completed |
| `c96-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 96 | 8 | completed after rerun |
| `c64-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 64 | 8 | completed |
| `c64-l4-sim16-b32-fastv8` | `gpu-l4-t4-cpu40` | 64 | 16 | completed |
| `c64-h100-sim16-b32-fastv8` | `gpu-h100-cpu40` | 64 | 16 | completed |
| `c64-l4-sim32-b32-fastv8` | `gpu-l4-t4-cpu40` | 64 | 32 | completed after rerun |
| `c64-h100-sim32-b32-fastv8` | `gpu-h100-cpu40` | 64 | 32 | completed |
| `c64-l4-sim8-b32-browser-ref` | `gpu-l4-t4-cpu40` | 64 | 8 | completed |
| `c128-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 128 | 8 | completed |
| `c192-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 192 | 8 | completed |
| `c256-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 256 | 8 | completed |
| `c384-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 384 | 8 | completed |
| `c512-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 512 | 8 | completed |
| `c128-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 128 | 8 | completed |
| `c128-l4-sim8-b32-browser-ref` | `gpu-l4-t4-cpu40` | 128 | 8 | completed |
| `c128-l4-sim16-b32-fastv8` | `gpu-l4-t4-cpu40` | 128 | 16 | completed |
| `c128-l4-sim8-b32-fastv8-source1024` | `gpu-l4-t4-cpu40` | 128 | 8 | failed before learner |
| `c192-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 192 | 8 | completed |
| `c256-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 256 | 8 | completed |
| `c192-l4-sim8-b32-browser-ref` | `gpu-l4-t4-cpu40` | 192 | 8 | completed |
| `c768-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 768 | 8 | completed |
| `c384-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 384 | 8 | completed |
| `c512-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 512 | 8 | completed |
| `c256-h100x2-sim8-b32-fastv8-multigpu` | `gpu-h100x2-cpu40` | 256 | 8 | failed before collection |
| `c256-l4-sim8-b32-browser-ref` | `gpu-l4-t4-cpu40` | 256 | 8 | completed |
| `c128-h100-sim16-b32-fastv8` | `gpu-h100-cpu40` | 128 | 16 | completed |
| `c128-h100-sim32-b32-fastv8` | `gpu-h100-cpu40` | 128 | 32 | completed |
| `c128-l4-sim8-b64-fastv8` | `gpu-l4-t4-cpu40` | 128 | 8 | completed |
| `c768-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 768 | 8 | completed |
| `c1024-h100-sim8-b32-fastv8` | `gpu-h100-cpu40` | 1024 | 8 | cancelled |
| `c1024-l4-sim8-b32-fastv8` | `gpu-l4-t4-cpu40` | 1024 | 8 | cancelled |
| `c256-h100-sim8-b32-browser-ref` | `gpu-h100-cpu40` | 256 | 8 | completed |
| `c384-h100-sim8-b32-browser-ref` | `gpu-h100-cpu40` | 384 | 8 | completed |
| `c256-h100-sim16-b32-fastv8` | `gpu-h100-cpu40` | 256 | 16 | completed |
| `c384-h100-sim16-b32-fastv8` | `gpu-h100-cpu40` | 384 | 16 | completed |
| `c384-l4-sim8-b64-fastv8` | `gpu-l4-t4-cpu40` | 384 | 8 | completed |
| `c256-h100-sim8-b32-fastv8-repeat` | `gpu-h100-cpu40` | 256 | 8 | cancelled |
| `c384-h100-sim8-b32-fastv8-repeat` | `gpu-h100-cpu40` | 384 | 8 | cancelled |
| `c384-l4-sim8-b32-fastv8-repeat` | `gpu-l4-t4-cpu40` | 384 | 8 | cancelled |

The first C96 and L4/sim32 launches reused the same `run_id` as earlier cells
and failed before training because profile auto-resume saw another cell's
checkpoint. The reruns use unique run ids. Future profile grids should use one
run id per cell unless resumed-state profiling is intentional.

## Results So Far

| attempt | render | compute | collectors | sims | env steps | wall s | env steps/s | collector s | MCTS s | policy collect s | learner s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c32-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 32 | 8 | 16,384 | 40.57 | 403.8 | 30.51 | 13.98 | 21.80 | 2.42 |
| `c64-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 64 | 8 | 32,768 | 55.39 | 591.6 | 43.25 | 15.76 | 27.92 | 2.25 |
| `c96-l4-sim8-b32-fastv8-rerun` | fast V8 | L4/T4 CPU40 | 96 | 8 | 49,152 | 72.60 | 677.1 | 57.27 | 16.83 | 34.21 | 2.23 |
| `c64-h100-sim8-b32-fastv8` | fast V8 | H100 CPU40 | 64 | 8 | 32,768 | 65.73 | 498.5 | 50.73 | 18.06 | 32.92 | 2.69 |
| `c64-l4-sim16-b32-fastv8` | fast V8 | L4/T4 CPU40 | 64 | 16 | 32,768 | 87.12 | 376.1 | 73.04 | 39.36 | 56.03 | 3.16 |
| `c64-h100-sim16-b32-fastv8` | fast V8 | H100 CPU40 | 64 | 16 | 32,768 | 59.67 | 549.1 | 49.09 | 26.44 | 35.70 | 2.05 |
| `c64-l4-sim32-b32-fastv8-rerun` | fast V8 | L4/T4 CPU40 | 64 | 32 | 32,768 | 92.38 | 354.7 | 80.46 | 52.60 | 65.23 | 2.16 |
| `c64-h100-sim32-b32-fastv8` | fast V8 | H100 CPU40 | 64 | 32 | 32,768 | 62.04 | 528.2 | 53.70 | 35.77 | 43.45 | 1.33 |
| `c64-l4-sim8-b32-browser-ref` | browser ref | L4/T4 CPU40 | 64 | 8 | 32,768 | 66.69 | 491.4 | 54.10 | 15.75 | 27.67 | 2.35 |
| `c128-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 128 | 8 | 65,536 | 87.53 | 748.8 | 69.54 | 19.32 | 39.92 | 2.42 |
| `c192-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 192 | 8 | 98,304 | 116.74 | 842.0 | 95.31 | 21.29 | 51.36 | 2.27 |
| `c256-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 256 | 8 | 131,072 | 162.70 | 805.6 | 137.55 | 26.13 | 65.91 | 2.26 |
| `c128-h100-sim8-b32-fastv8` | fast V8 | H100 CPU40 | 128 | 8 | 65,536 | 65.36 | 1002.7 | 52.65 | 15.53 | 30.49 | 1.62 |
| `c128-l4-sim8-b32-browser-ref` | browser ref | L4/T4 CPU40 | 128 | 8 | 65,536 | 104.02 | 630.0 | 87.25 | 20.27 | 41.20 | 2.36 |
| `c128-l4-sim16-b32-fastv8` | fast V8 | L4/T4 CPU40 | 128 | 16 | 65,536 | 106.47 | 615.5 | 90.11 | 37.24 | 58.34 | 2.32 |
| `c128-l4-sim8-b32-fastv8-source1024` | fast V8 | L4/T4 CPU40 | 128 | 8 | 131,072 | 148.67 | 881.6 | 133.91 | 35.40 | 74.61 | 0.00 |
| `c384-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 384 | 8 | 196,608 | 207.81 | 946.1 | 169.25 | 30.71 | 87.02 | 2.60 |
| `c512-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 512 | 8 | 262,144 | 305.84 | 857.1 | 255.76 | 42.14 | 122.45 | 2.35 |
| `c192-h100-sim8-b32-fastv8` | fast V8 | H100 CPU40 | 192 | 8 | 98,304 | 112.44 | 874.3 | 94.82 | 20.54 | 46.40 | 2.00 |
| `c256-h100-sim8-b32-fastv8` | fast V8 | H100 CPU40 | 256 | 8 | 131,072 | 121.15 | 1081.9 | 101.83 | 20.72 | 47.61 | 1.59 |
| `c192-l4-sim8-b32-browser-ref` | browser ref | L4/T4 CPU40 | 192 | 8 | 98,304 | 138.09 | 711.9 | 116.11 | 21.49 | 51.43 | 2.39 |
| `c768-l4-sim8-b32-fastv8` | fast V8 | L4/T4 CPU40 | 768 | 8 | 393,216 | 564.57 | 696.5 | 489.89 | 73.26 | 235.68 | 2.82 |
| `c384-h100-sim8-b32-fastv8` | fast V8 | H100 CPU40 | 384 | 8 | 196,608 | 268.33 | 732.7 | 233.11 | 33.75 | 85.98 | 2.23 |
| `c512-h100-sim8-b32-fastv8` | fast V8 | H100 CPU40 | 512 | 8 | 262,144 | 347.87 | 753.6 | 286.77 | 47.92 | 140.80 | 2.68 |
| `c256-h100x2-sim8-b32-fastv8-multigpu` | fast V8 | H100x2 CPU40 | 256 | 8 | 0 | 1.16 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| `c256-l4-sim8-b32-browser-ref` | browser ref | L4/T4 CPU40 | 256 | 8 | 131,072 | 194.95 | 672.4 | 163.65 | 26.91 | 65.90 | 2.43 |
| `c128-h100-sim16-b32-fastv8` | fast V8 | H100 CPU40 | 128 | 16 | 65,536 | 99.89 | 656.1 | 83.90 | 37.46 | 54.08 | 2.27 |
| `c128-h100-sim32-b32-fastv8` | fast V8 | H100 CPU40 | 128 | 32 | 65,536 | 175.94 | 372.5 | 153.61 | 90.23 | 116.86 | 2.97 |
| `c128-l4-sim8-b64-fastv8` | fast V8 | L4/T4 CPU40 | 128 | 8 | 65,536 | 87.72 | 747.1 | 69.74 | 19.57 | 39.98 | 2.31 |
| `c256-h100-sim16-b32-fastv8` | fast V8 | H100 CPU40 | 256 | 16 | 131,072 | 175.79 | 745.6 | 150.63 | 55.65 | 85.87 | 2.21 |
| `c384-h100-sim16-b32-fastv8` | fast V8 | H100 CPU40 | 384 | 16 | 196,608 | 206.76 | 950.9 | 175.90 | 58.96 | 97.39 | 1.78 |
| `c768-h100-sim8-b32-fastv8` | fast V8 | H100 CPU40 | 768 | 8 | 393,216 | 326.59 | 1204.0 | 279.52 | 46.33 | 123.39 | 1.53 |
| `c256-h100-sim8-b32-browser-ref` | browser ref | H100 CPU40 | 256 | 8 | 131,072 | 228.18 | 574.4 | 195.09 | 29.82 | 76.48 | 2.49 |
| `c384-h100-sim8-b32-browser-ref` | browser ref | H100 CPU40 | 384 | 8 | 196,608 | 317.06 | 620.1 | 264.39 | 36.10 | 104.89 | 3.04 |
| `c384-l4-sim8-b64-fastv8` | fast V8 | L4/T4 CPU40 | 384 | 8 | 196,608 | 269.63 | 729.2 | 231.68 | 40.17 | 121.82 | 2.88 |

Failure details:

- `c256-h100x2-sim8-b32-fastv8-multigpu` failed before collection with
  `ValueError: Default process group has not been initialized`. Plain read:
  the current LightZero `policy.multi_gpu=True` path is not a drop-in Modal
  multi-GPU speedup.

## Plain Read

- Current fast V8 observation is a real whole-loop speedup at C64/L4/sim8:
  `591.6` versus `491.4` env steps/sec, about `1.20x`.
- The model-facing observation itself is much cheaper than browser, but the
  whole loop still spends a lot of wall time in collection, policy forward,
  MCTS, subprocess/env waiting, replay, and learner work.
- C64/L4/sim8 is about `1.47x` faster than C32/L4/sim8 for the same current
  fast observation path.
- C384/L4/sim8 is the fastest completed L4 row so far, at `946.1` env
  steps/sec. C512/L4/sim8 dropped to `857.1`, and C768/L4/sim8 dropped to
  `696.5`, so the L4 width knee is probably around C384 for this exact
  no-death/source512 profile.
- C768/H100/sim8 is the fastest completed overall sim8 row so far, at
  `1204.0` env steps/sec. This is speed-only evidence: the learning docs still
  treat very wide collector counts as probes, not proven quality defaults.
- C256/H100/sim8 remains the cleanest aggressive speed/quality compromise from
  the completed grid, at `1081.9` env steps/sec. H100 was worse at C64/sim8,
  but wins once root batches are wide enough.
- H100 is much better than L4/T4 at sim16/sim32, but those higher-sim rows are
  not supported as quality defaults by the 212-run learning evidence. Treat
  them as sentinels unless Coach explicitly wants a higher-search experiment.
- C128 fast V8 versus C128 browser reference: `1002.7` on H100 fast is not a
  fair render comparison because hardware changed; the L4 render comparison is
  `748.8` fast versus `630.0` browser, about `1.19x`.
- C192 L4 render comparison is `842.0` fast versus `711.9` browser, about
  `1.18x`. The fast render keeps a real but not giant full-loop advantage at
  wide collector counts.
- C256 L4 render comparison is `805.6` fast versus `672.4` browser, about
  `1.20x`. The full-loop fast-render gain is stable around `1.18x-1.20x` on
  L4, not a 10x whole-training-loop win.
- C256 H100 render comparison is `1081.9` fast versus `574.4` browser, about
  `1.88x`. C384 H100 comparison is `732.7` fast versus `620.1` browser, about
  `1.18x`. The H100 browser rows are noisy, but they reinforce the same
  decision: use fast V8 for speed probes and keep browser as a matched fidelity
  sentinel, not as the default for every high-throughput run.
- The source1024 row collected `131,072` env steps at `881.6` env steps/sec but
  failed before learner with `ValueError: 'a' and 'p' must have same size`.
  Treat it as collection-only speed evidence and a bug/footgun for long
  no-death profiles, not a full-loop training proof.
- Batch64 did not help speed. At C128/L4 it was `747.1` versus `748.8` env
  steps/sec for batch32; at C384/L4 it was `729.2` versus `946.1`. Since
  learning evidence is negative for batch64 too, do not recommend it.
- Sim16 at C384/H100 measured `950.9` env steps/sec, but the learning evidence
  still says sim16 is a sentinel, not a default. Do not turn a speed-only row
  into a learning recommendation.
- Multi-GPU is currently not available by flipping `--lightzero-multi-gpu`; the
  H100x2 row failed before collection because no distributed process group was
  initialized.
- GPU utilization samples are still coarse and often show 0%, so use wall time
  and named timers rather than `nvidia-smi` samples for this decision.

## Decision Questions

1. Does C64 still beat C32 clearly under the current fast observation path?
2. Does C96 still give enough extra throughput to justify more subprocess
   overhead, or is C64 the practical knee?
3. Is H100 useful at sim8 now, or still wasted?
4. Does H100 become useful only at sim16/sim32, and if so is the learning
   evidence strong enough to justify those higher-search rows?
5. Does wider collection keep improving after C128, or do C192/C256/C384/C512
   hit subprocess overhead and collapse?
6. Does the fast-vs-browser speed gap persist at C128, or shrink once collection
   dominates?
7. Does a longer no-death horizon (`source_max_steps=1024`) change the full-loop
   bottleneck read?
8. Does H100 continue winning at C192/C256/C384/C512, or was C128 the sweet
   spot?
9. Does `gpu-h100x2-cpu40 --lightzero-multi-gpu` actually help, or is the
   second GPU idle/overhead?
10. Does `batch_size=64` matter for speed at all, even though learning evidence
   says not to use it as a default?

## Current Best Prior

Before this grid, the most recent exact V8 full-loop profile was only C8. It
showed the model observation got about `10.5x` faster, but the whole loop only
improved about `1.17x`. That row is too narrow to choose the next training
hardware.
