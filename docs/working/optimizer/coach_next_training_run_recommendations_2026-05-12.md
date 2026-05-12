# Coach Next Training Run Recommendations

Date: 2026-05-12

Copy-paste handoff: use the canonical CurvyTron two-seat current-policy
self-play launcher, run a broad but readable overnight matrix, and keep the
fast visual lane honest with browser-lines controls. The default recommendation
is mostly L4/T4, `fast_gray64_direct`, normal death, accumulated replay,
sparse checkpoints, CurvyZero checkpoint eval/GIF on, and stock LightZero
in-loop eval off. Do not use the old two-seat Modal smoke wrapper.

## Current Truth

- Canonical path:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
- This is current-policy two-seat self-play. One shared LightZero MuZero policy
  chooses both players from the same pre-step state, then CurvyTron advances
  once with the joint action.
- The custom part should stay small: CurvyTron environment plus simultaneous
  action bridge. Keep the learner/search/replay path close to LightZero.
- `browser_lines` is the closer visual reference. It renders source-state RGB
  and downsamples to gray64.
- `fast_gray64_direct` is the speed lane. It is a strong semantic visual
  approximation, not browser pixel fidelity. It keeps trail/head positions,
  self/other contrast, bonus presence, and bonus type luma. It drops exact
  connected-line rasterization, sprite texture, antialiasing, and exact
  downsample coverage.
- `body_circles_fast` is historical/comparison only. Do not use it for new
  Coach runs.
- `profile_no_death` is only for optimizer timing. Real training should use
  normal death.

## What Changed For Coach

- `--two-seat-learning-rate` is now wired through the canonical Modal launcher,
  two-seat runner, policy config patch, progress metadata, and checkpoints.
  Omit it to use the LightZero config default.
- Profiling GIF/eval clutter can be disabled with
  `--no-background-eval-enabled --no-background-gif-enabled`. Default training
  should leave CurvyZero checkpoint eval/GIF on.
- Default stochasticity is not what we casually assumed earlier:
  observation noise is on at `0.10`, but random action no-op and policy repeat
  skip are off by default.

## Evidence

- Before fast direct rendering, long no-death profiles were render-bound. A
  B64/L4/browser-lines no-death profile took about `768s`; visual stack was
  around `40s/iteration`.
- With `fast_gray64_direct`, the comparable B64/L4 profile took about `203s`;
  visual stack dropped to about `2s/iteration`. The bottleneck moved toward
  policy/search, environment stepping, and observation noise.
- B128/L4 fast was slower per replay row than B64/L4. B128/H100 was much
  better than B128/L4, but not enough to make H100 the default.
- Plain Amdahl read: use fast direct for most overnight learning probes, keep
  browser-lines controls, and only pay H100 where batch/search is big enough to
  expose GPU benefit.

## Base Command Shape

Use this shape for most runs and vary only the table columns:

```text
modal run src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  --mode two-seat-selfplay \
  --compute gpu-l4-t4 \
  --batch-size 64 \
  --max-train-iter <ITERATIONS> \
  --num-simulations 8 \
  --two-seat-collect-steps-per-iteration 64 \
  --two-seat-updates-per-iteration 4 \
  --two-seat-replay-scope accumulated \
  --two-seat-learner-sample-size 256 \
  --two-seat-max-replay-rows 65536 \
  --two-seat-death-mode normal \
  --two-seat-trail-render-mode fast_gray64_direct \
  --save-ckpt-after-iter 50
```

For profiling-only runs, add:

```text
--two-seat-death-mode profile_no_death \
--no-background-eval-enabled \
--no-background-gif-enabled \
--two-seat-save-initial-checkpoint false
```

For real overnight training, do not add those profiling flags. Let CurvyZero
checkpoint eval/GIF run at the checkpoint cadence.

Brief checkpoint recommendation:

- Choose cadence from the first 20-50 warm-up iterations, not from a fixed
  guess. Target wall-clock spacing is the real rule.
- First canary wave: aim for one checkpoint every 5-10 minutes. Start with
  `--save-ckpt-after-iter 50` if no better rate estimate exists.
- Overnight matrix after the launch path is healthy: aim for one checkpoint
  every 10-20 minutes. Use `50`, `100`, or `250` iterations depending on the
  measured iteration speed.
- Very long follow-up runs: checkpoint every 30-60 minutes if eval/GIF artifacts
  are already working and intermediate visibility is less important. Relax
  further only after the first curves look sane.
- Profiling-only runs: avoid initial checkpoints and set checkpoint cadence
  high enough that artifact work is not part of the timing result.

## Recommended 30-Run Matrix

Use distinct run IDs. For paired ablations, keep the same seed family so the
comparison is not pure noise. If this is too many, run the first 12 plus the two
browser controls.

```text
id  lane                 gpu       render              B    sim  collect  upd  sample  lr       reward             stochastic
01  main                 L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
02  main_seed            L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
03  main_seed            L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
04  search16             L4/T4     fast_gray64_direct  64   16   64       4    256     unset    default            default
05  search32             L4/T4     fast_gray64_direct  64   32   64       4    256     unset    default            default
06  small_batch          L4/T4     fast_gray64_direct  32   8    64       4    128     unset    default            default
07  large_batch_l4       L4/T4     fast_gray64_direct  128  8    64       4    512     unset    default            default
08  large_batch_h100     H100      fast_gray64_direct  128  8    64       4    512     unset    default            default
09  large_search_h100    H100      fast_gray64_direct  128  16   64       4    512     unset    default            default
10  collect128           L4/T4     fast_gray64_direct  64   8    128      4    256     unset    default            default
11  updates8             L4/T4     fast_gray64_direct  64   8    64       8    256     unset    default            default
12  learner512           L4/T4     fast_gray64_direct  64   8    64       4    512     unset    default            default
13  lr_1e-4              L4/T4     fast_gray64_direct  64   8    64       4    256     1e-4     default            default
14  lr_3e-4              L4/T4     fast_gray64_direct  64   8    64       4    256     3e-4     default            default
15  lr_1e-3              L4/T4     fast_gray64_direct  64   8    64       4    256     1e-3     default            default
16  lr_h100_3e-4         H100      fast_gray64_direct  128  8    64       4    512     3e-4     default            default
17  no_bonus             L4/T4     fast_gray64_direct  64   8    64       4    256     unset    no_bonus           default
18  terminal_only        L4/T4     fast_gray64_direct  64   8    64       4    256     unset    terminal_only      default
19  stronger_terminal    L4/T4     fast_gray64_direct  64   8    64       4    256     unset    terminal_x2        default
20  survival_only_ctrl   L4/T4     fast_gray64_direct  64   8    64       4    256     unset    survival_only      default
21  no_obs_noise         L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            obs_noise_0
22  action_repeat        L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            repeat_20pct
23  action_noop_05       L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            action_noop_5pct
24  no_stochasticity     L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            none
25  browser_control      L4/T4     browser_lines       32   8    64       4    256     unset    default            default
26  browser_search       L4/T4     browser_lines       32   16   64       4    256     unset    default            default
27  browser_lr_3e-4      L4/T4     browser_lines       32   8    64       4    256     3e-4     default            default
28  browser_no_bonus     L4/T4     browser_lines       32   8    64       4    256     unset    no_bonus           default
29  h100_search32        H100      fast_gray64_direct  128  32   64       4    512     unset    default            default
30  h100_collect128      H100      fast_gray64_direct  128  16   128      4    512     unset    default            default
```

## Reward Flags

Default reward means no reward flags:

```text
alive +0.01 while alive
bonus pickup +0.05 per same-step catch
terminal outcome = sparse_outcome * 0.01 * episode_step_count
return discount = 1.0
```

Use these named variants:

```text
no_bonus:
  --two-seat-bonus-pickup-reward-per-catch 0

terminal_only:
  --two-seat-alive-reward 0
  --two-seat-bonus-pickup-reward-per-catch 0

terminal_x2:
  --two-seat-terminal-outcome-reward-per-step 0.02

survival_only:
  --two-seat-terminal-outcome-reward-per-step 0
  --two-seat-bonus-pickup-reward-per-catch 0
```

Plain warning: `terminal_only` is not pure AlphaZero sparse +1/-1. The current
implementation still scales terminal outcome by episode length. That may be a
reasonable delayed-reward control, but label it correctly.

## Stochasticity Flags

Default stochasticity means observation noise only:

```text
--two-seat-observation-noise-std 0.10
--two-seat-action-noop-probability 0
--two-seat-policy-action-repeat-min 1
--two-seat-policy-action-repeat-max 1
--two-seat-policy-action-repeat-extra-probability 0
```

Use these named variants:

```text
obs_noise_0:
  --two-seat-observation-noise-std 0

repeat_20pct:
  --two-seat-policy-action-repeat-max 3
  --two-seat-policy-action-repeat-extra-probability 0.20

action_noop_5pct:
  --two-seat-action-noop-probability 0.05

none:
  --two-seat-observation-noise-std 0
  --two-seat-action-noop-probability 0
  --two-seat-policy-action-repeat-min 1
  --two-seat-policy-action-repeat-max 1
  --two-seat-policy-action-repeat-extra-probability 0
```

Optimizer recommendation: do not make random action no-op the default tonight.
It changes the agent-action contract and was recently added. Include one small
no-op probe if Coach wants to test robustness. Prefer default obs noise and a
separate no-stochasticity control.

## Hardware Read

- Use L4/T4 for most runs. It is cheaper and enough for B64 fast-direct runs.
- Use H100 only for the five listed scale probes. H100 helps B128 more than L4,
  but it is not proven to improve learning per dollar or per wall-clock yet.
- Skip multi-GPU tonight. The canonical two-seat path has not shown a clean
  model-throughput bottleneck where multi-GPU is the obvious next move.

## What To Watch

- Primary learning read belongs to Coach: survival length, sparse outcome,
  terminal causes, collapse checks, and action histograms.
- Optimizer read: wall time per checkpoint, replay rows/sec, self-play rows/sec,
  policy/search time, env step time, visual stack time, learner time, checkpoint
  artifact time.
- If fast-direct improves but browser-lines fails, the approximation may be too
  lossy. If both fail the same way, look at reward/search/training setup before
  blaming rendering.
- Larger self-play batches are useful only if the learner actually turns the
  extra collected rows into better checkpoint behavior. They are not magic; they
  trade fresher, more frequent updates for bigger batches and more compute per
  update.

## Minimal Set If Time Is Short

Run these 12 first:

```text
01 main
02 main_seed
03 main_seed
04 search16
07 large_batch_l4
08 large_batch_h100
13 lr_1e-4
14 lr_3e-4
17 no_bonus
18 terminal_only
21 no_obs_noise
25 browser_control
```

Then add the remaining rows if the launch surface is healthy.
