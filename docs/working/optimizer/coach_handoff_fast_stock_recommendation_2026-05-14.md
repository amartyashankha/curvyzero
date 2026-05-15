# Coach Handoff: Fast Stock CurvyTron Recommendation

Date: 2026-05-14

## Plain Verdict

Use the trusted stock LightZero path. Do not use the old custom two-seat path
for learning claims.

2026-05-15 correction: this handoff is superseded as launch guidance. It
recommended the best *wired and profiled* path at the time,
`body_circles_fast + simple_symbols`, but current production policy
observations are CPU `cpu_oracle` `browser_lines + simple_symbols`. The GPU
renderer is lab/profiling only, even after first H100 real-env smoke rows match
the CPU oracle. Keep the speed numbers as historical evidence only. Do not copy
the body-circles commands into new runs unless the row is explicitly an
ablation.

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
```

Superseded recommendation, kept only to explain old speed rows:

```text
compute: gpu-h100-cpu40
collector_env_num: 256
num_simulations: 8
batch_size: 32
trail render: body_circles_fast
bonus render: simple_symbols
```

This was the best wired compromise when the handoff was written. It is no
longer the optimizer recommendation because it is CPU body-circles rendering,
not `cpu_oracle` `browser_lines + simple_symbols`.

## Superseded Primary Training Recommendation

Do not copy this block into new launches unless the row is explicitly a
historical ablation:

```text
reward_variant: survival_plus_bonus_no_outcome
num_simulations: 8
batch_size: 32
collector_env_num: 256
compute: gpu-h100-cpu40
source_state_trail_render_mode: body_circles_fast
source_state_bonus_render_mode: simple_symbols
stochasticity: medium and light core; heavy bounded; steady/control if budget
mix repeats: repM anchor, repH upside, rep0 control
```

Reason: the 212-run read says survival learning is real, latest-212 tournament
evidence is real, `sim8/batch32` is supported, `batch64` is bad, and `sim16`
has not earned the quality cost. The speed grid says H100/C256 with the fast
V8 observation path is the cleanest aggressive throughput point.

Important plain-language note: "H100" and "GPU renderer" are different things.

- `compute=gpu-h100-cpu40` means the Modal container has an H100 and LightZero
  runs the model/search/learner on that GPU.
- It does **not** mean the CurvyTron observation renderer is the separate GPU
  renderer prototype.
- This old handoff's recommended observation path was CPU-side
  `body_circles_fast + simple_symbols`; that recommendation is superseded.
- The current production observation path is CPU `cpu_oracle`
  `browser_lines + simple_symbols`.
- The faithful GPU browser-lines renderer is still lab/profiling only: promising
  and exact on first H100 smoke rows after same-owner/cursor/owner-priority
  fixes, but not wired into the stock trainer as the training observation path.

When a faithful batched GPU renderer is wired into the trainer and passes parity
plus full-loop profiling, it can replace `cpu_oracle`. That is not the current
state.

Learning evidence docs:

- `docs/working/training/curvytron_architecture_research_2026-05-12/fair_comparison_212_run_investigation_2026-05-13.md`
- `docs/working/training/leaderboard_to_training_2026-05-13/overnight_run_decision.md`
- `docs/working/training/leaderboard_to_training_2026-05-13/top100_tournament_stress_2026-05-14.md`

## Speed Probe Block

These are speed recommendations only. They should not override Coach's learning
judgment.

Strong speed rows from the stock profile grid:

| row | env steps/sec | read |
| --- | ---: | --- |
| C64/L4/sim8 fast | 591.6 | safe-ish speed probe above C32 |
| C384/L4/sim8 fast | 946.1 | best L4 row, speed-only wide probe |
| C256/H100/sim8 fast | 1081.9 | cleanest aggressive H100 candidate |
| C768/H100/sim8 fast | 1204.0 | fastest completed row, very wide speed-only probe |

Fast render gives a real whole-loop gain but not a magic 10x by itself:

- L4 matched fast-vs-browser is about `1.18x-1.20x` at C64-C256.
- H100 C256 fast-vs-browser was `1.88x`, but H100 browser rows were noisier.
- The main bottleneck is still collection/search/env-manager wall time, not
  learner time alone.

Do not use `body_circles_fast + simple_symbols` as the current main
recommendation. The current production backend is CPU `cpu_oracle`
`browser_lines + simple_symbols`; CPU body-circles rows are historical/control
evidence.

## What Not To Use

- Do not use `batch64`. It is worse in learning evidence and did not help speed.
- Do not make `sim16` a default. It is a sentinel only.
- Do not use multi-GPU for this run. The current `--lightzero-multi-gpu` profile
  failed before collection because no distributed process group was initialized.
- Do not let the trainer poll live Elo or Modal Dict state inside the learning
  loop. Use immutable assignment files at launch/resume/clean refresh only.

## Checkpoint Cadence

Use a cadence that gives visible curves without making artifacts dominate.

Recommendation for the next serious launch:

```text
save_ckpt_after_iter: 5000 to 10000
```

Use the lower end for first canaries or new settings. Use the higher end once
the run is healthy. Keep background eval/GIF on for real training visibility,
but do not include that overhead in optimizer profile rows.

## Concrete Coach Ask

Launch this main row:

```text
gpu-h100-cpu40
collector_env_num=256
num_simulations=8
batch_size=32
reward_variant=survival_plus_bonus_no_outcome
source_state_trail_render_mode=body_circles_fast
source_state_bonus_render_mode=simple_symbols
save_ckpt_after_iter=5000 to 10000
```

Optional extras only if Coach wants a matrix:

1. C64/L4 fast as a conservative comparison.
2. C768/H100 fast as a pure speed/aggression probe.
3. One browser-lines sentinel, not a full browser matrix.

Do not use batch64, multi-GPU, or sim16 as defaults.

Optimizer docs with the raw speed grid:

- `docs/working/optimizer/stock_fast_path_scaling_grid_2026-05-14.md`
- `docs/working/optimizer/active_working_memory_2026-05-14.md`
