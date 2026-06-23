# Profile Validation Results

> [!IMPORTANT]
> Superseded/archive note (2026-05-15): production policy observation is CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU `browser_lines + simple_symbols` remains lab-only until trainer contract parity. `body_circles_fast` is historical/control only.

Date: 2026-05-12

Grid:

```text
opt-stock-frozen-profile-first-wave-20260512c
```

Scope: six validation rows from the trusted stock LightZero profile path:
`source_state_fixed_opponent`, frozen checkpoint opponent, opponent on CPU,
eval/GIF off, checkpoint saving effectively off, and
`stop_after_learner_train_calls=10`.

## Result Table

| row | shape | death | render | steps | wall sec | steps/s | collect sec | MCTS sec | learner sec | obs sec | opponent sec | vector sec | GPU max |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | base C1 | normal | browser | 68 | 17.15 | 3.96 | 9.36 | 3.52 | 2.76 | 1.69 | 2.50 | 1.05 | 17% |
| 02 | base C1 | no death | browser | 256 | 74.64 | 3.43 | 66.76 | 13.00 | 2.85 | 41.47 | 8.50 | 6.21 | 16% |
| 03 | base C1 | no death | fast | 256 | 34.12 | 7.50 | 28.30 | 10.02 | 2.06 | 7.84 | 8.26 | 4.25 | 24% |
| 04 | subprocess C8 | normal | browser | 74 | 13.14 | 5.63 | 4.26 | 0.40 | 2.66 | 1.04 | 4.64 | 0.25 | 17% |
| 06 | subprocess C32 | normal | browser | 316 | 16.84 | 18.76 | 5.89 | 0.45 | 2.61 | 3.06 | 35.85 | 1.02 | 0% |
| 12 | subprocess C16 | no death | browser | 4096 | 112.44 | 36.43 | 103.10 | 8.43 | 2.42 | 19.24 | 13.63 | 1.80 | 0% |

Notes:

- `obs`, `opponent`, and `vector` are sampled env telemetry buckets. In base
  rows they cover all steps. In subprocess rows they are sampled worker-side
  CPU seconds, so use them for direction, not exact wall percentages.
- `wall sec` is `train_muzero_wall`, not local Modal CLI time. Final artifact
  volume commits are intentionally excluded from the speed table.
- Row 02 had a final commit error in stdout, but `summary.json` later appeared
  in the volume. Treat final commit time as an artifact durability concern, not
  training-loop timing.

## Plain Read

Rendering/observation matters again on long trajectories.

The cleanest comparison is row 02 versus row 03. Both are base manager, one env,
no death, 256 steps, 8 MCTS sims, same checkpoint. The only intended difference
is render mode:

```text
browser_lines:      3.43 steps/s, obs 41.47 sec
body_circles_fast:  7.50 steps/s, obs  7.84 sec
```

That is about a 2.2x total speedup and about a 5.1x observation-bucket speedup
for the fast render lens. This does not prove the fast render is the final
training surface. It proves browser-style observation is a real long-trajectory
cost.

Subprocess collection helps.

Row 06, subprocess C32 normal-death browser, reaches 18.76 steps/s versus row
01 base C1 at 3.96 steps/s. The comparison is not perfect because the short
episodes differ slightly, but it says the stock path can use CPU parallelism.

MCTS is not the main bottleneck at this setting.

In the long row 12, collection took 103.10 sec out of 112.44 sec. MCTS took
8.43 sec and learner took 2.42 sec. At 8 simulations, optimizing learner or
buying a bigger GPU is not the first lever.

Frozen-opponent CPU action time is visible.

In subprocess C32 normal-death row 06, sampled opponent time is much larger
than sampled observation time. That does not mean opponent time is the whole
wall clock, but it does say the fixed-opponent path is spending meaningful CPU
time in env workers.

## Amdahl Read

Current highest-leverage areas:

1. Reduce observation/render cost for browser-like visual input, especially for
   no-death or future good-policy long trajectories.
2. Keep increasing collector parallelism until subprocess scaling bends.
3. Investigate frozen-opponent CPU inference overhead separately; it is a
   control-lane cost, not necessarily final current-policy self-play cost.
4. Run the MCTS simulation ladder before spending time on H100 or MCTX. Search
   may become important at higher sims, but it is not dominant at sim8 here.

Current lower-leverage areas:

- Learner optimization: learner is only a few seconds in these profiles.
- Checkpoint/eval/GIF: disabled for profiles and sparse in training.
- Multi-GPU: not useful from these rows; GPU use is low.

## Tooling Problem Found

Profile result persistence needed a cleaner strategy before wide detached grids.

Facts:

- Without a final profile commit, some rows printed valid summaries but did not
  reliably publish `summary.json` to the Modal volume.
- With a final profile commit, summaries appeared, but commits were slow and one
  row reported a commit error even though the summary later became visible.
- The training timing table excludes final commit time, but wide detached grids
  still need reliable readback.

## Readback Fix

Use Modal function-result readback for optimizer profiles.

The working shape is:

```text
parent launch: `modal run --detach`
child launch: `--profile-spawn`
profile result: `modal.FunctionCall.from_id(function_call_id).get()`
volume commit: off
```

Validation:

```text
manifest: opt-stock-frozen-profile-first-wave-20260512e
row: 01
result path: artifacts/local/curvytron_optimizer_profile_results/opt-stock-frozen-profile-first-wave-20260512e/row_01_result.json
status: complete
steps/wall: 69 steps in 12.93 sec, 5.34 steps/s
final volume commit: not attempted
```

The failed `20260512d` attempt is also useful: `--profile-spawn` without parent
`modal run --detach` can return a call id, but the parent app can stop before
the child profile finishes. Do not use that shape for wide grids.

## First Wave Function-Readback Results

Grid:

```text
opt-stock-frozen-profile-first-wave-20260512e
```

Result files:

```text
artifacts/local/curvytron_optimizer_profile_results/opt-stock-frozen-profile-first-wave-20260512e/
```

| row | C | sims | death | render | reward | steps | wall | steps/s | collect | MCTS | learner | obs | opp | GPU max |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 1 | 8 | normal | browser_lines | sparse_outcome | 69 | 12.93 | 5.34 | 7.05 | 2.62 | 1.96 | 1.29 | 2.12 | 4.0 |
| 05 | 16 | 8 | normal | browser_lines | sparse_outcome | 147 | 13.35 | 11.01 | 4.14 | 0.37 | 2.30 | 1.02 | 12.68 | 22.0 |
| 06 | 32 | 8 | normal | browser_lines | sparse_outcome | 316 | 15.15 | 20.86 | 4.86 | 0.39 | 2.37 | 2.26 | 41.51 | 0.0 |
| 07 | 64 | 8 | normal | browser_lines | sparse_outcome | 605 | 18.85 | 32.09 | 6.37 | 0.45 | 2.27 | 5.85 | 132.63 | 0.0 |
| 08 | 96 | 8 | normal | browser_lines | sparse_outcome | 925 | 22.55 | 41.01 | 7.95 | 0.44 | 2.00 | 6.23 | 293.53 | 0.0 |
| 09 | 32 | 4 | normal | browser_lines | sparse_outcome | 299 | 14.51 | 20.61 | 4.18 | 0.20 | 2.06 | 1.97 | 37.73 | 0.0 |
| 10 | 32 | 16 | normal | browser_lines | sparse_outcome | 305 | 18.15 | 16.80 | 7.47 | 0.84 | 2.57 | 3.07 | 51.73 | 0.0 |
| 11 | 32 | 32 | normal | browser_lines | sparse_outcome | 306 | 19.50 | 15.69 | 8.10 | 1.55 | 2.79 | 2.87 | 46.19 | 0.0 |
| 12 | 16 | 8 | nodeath | browser_lines | sparse_outcome | 4096 | 112.35 | 36.46 | 103.36 | 8.45 | 2.31 | 19.18 | 14.13 | 0.0 |
| 13 | 16 | 8 | nodeath | body_circles_fast | sparse_outcome | 4096 | 61.98 | 66.09 | 52.15 | 8.37 | 2.58 | 4.99 | 14.70 | 0.0 |
| 14 | 64 | 16 | normal | browser_lines | sparse_outcome | 599 | 19.13 | 31.32 | 6.83 | 0.71 | 1.86 | 4.66 | 140.23 | 0.0 |
| 17 | 32 | 8 | normal | browser_lines | dense_survival_plus_outcome | 316 | 16.52 | 19.13 | 5.88 | 0.46 | 2.73 | 2.82 | 34.64 | 0.0 |

Plain read:

- Collector width helps. C1 -> C32 -> C96 moves about 5.3 -> 20.9 -> 41.0
  steps/s on normal-death browser rows. Scaling is positive but already
  sublinear.
- Search is not the first bottleneck here. At C32, sim4/sim8/sim16/sim32 are
  about 20.6/20.9/16.8/15.7 steps/s. MCTS time rises, but it is still only a
  few seconds in these short profiles.
- Long trajectories make rendering matter. At C16 no-death, `body_circles_fast`
  is about 66.1 steps/s versus 36.5 steps/s for `browser_lines`, a 1.8x total
  speedup. Observation sampled time drops from 19.18 sec to 4.99 sec.
- Dense survival reward bookkeeping is not a major cost in this profile:
  row 17 is close to row 06.

Current recommendation:

Use `gpu-l4-t4-cpu40`, `subprocess`, `collector_env_num` around 64-96, sparse
checkpoint/eval/GIF cadence, frozen opponent on CPU, and the browser renderer
for trusted fidelity runs. Use `body_circles_fast` as an optimizer speed lens
and possibly as a training-speed option only if Coach/Environment accept the
fidelity tradeoff.

## Second Wave Read

Detailed second-wave rows are in `second_wave_profile_tensor.md`.

The recommendation tightened after the second wave:

- Use C96 as the default wide profile/training starting point on
  `gpu-l4-t4-cpu40`. C128 and C160 did not buy enough over C96 to be the default
  first choice.
- Do not chase H100 or multi-GPU yet. Sim16 at C96 was about as fast as sim8 at
  C96, and MCTS/search time is still not the main wall-clock cost.
- Keep browser render for trusted runs unless the training owner explicitly
  accepts the `body_circles_fast` approximation. Fast render is a real speed
  lever for long trajectories, about 1.75x at C32 no-death, but it is a visual
  fidelity decision.
- Frozen checkpoint opponent inference is a real cost in the stock
  fixed-opponent lane. Fixed-straight rows are much faster. Treat that as an
  optimizer warning about the control setup, not as a learning recommendation.
- Dense reward bookkeeping is not a large bottleneck compared with collection,
  render, and frozen-opponent work.
