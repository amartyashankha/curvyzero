# Coach Next Training Run Recommendations

Date: 2026-05-12

## Superseded By Postmortem

This file is historical launch guidance for the custom two-seat adapter. Do not
use it for new Coach learning runs.

The May 12 no-learning audit found that the recommended `--mode
two-seat-selfplay` path did not call stock `train_muzero` and did not use
LightZero `GameSegment` / `MuZeroGameBuffer` targets. Current research lives in
`docs/working/training/curvytron_architecture_research_2026-05-12/`.
Current Optimizer speed evidence for the trusted stock frozen-opponent path is
in `docs/working/optimizer/stock_frozen_optimizer_pivot_2026-05-12.md`.
Everything below this paragraph is historical unless it is explicitly copied
into a newer stock-path handoff.

Historical copy-paste handoff: use the canonical CurvyTron two-seat current-policy
self-play launcher and run an aggressive approximation-heavy overnight matrix.
The main training surface is `fast_gray64_direct`, not `browser_lines`.
Browser-lines runs are mandatory small sentinels, but they should not gate or
slow the main approximation launch. Pair each browser sentinel with a matched
small fast-direct row so the comparison is not nonsense. The default
recommendation is mostly L4/T4, `fast_gray64_direct`, normal death, accumulated
replay, CurvyZero checkpoint eval/GIF on, and stock LightZero in-loop eval off.
Do not use the old two-seat Modal smoke wrapper.

## Historical Two-Seat Truth

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
- GIFs are spawned from checkpoint artifacts. For a wide matrix, cap GIF length
  explicitly; the default GIF max-steps value means no GIF-specific cap.
- Per-update model-hash verification is now opt-in. Leave it off for training
  speed; use `--two-seat-verify-model-update-hash` only for a short debug smoke.
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
- A read-only check of live 2026-05-12 fast-direct runs sharpened that picture:
  H100/B128/sim8 at iteration 120 spent `54.9s` in `policy_search_sec` and
  `2.25s` in `visual_stack_update_sec`; H100/B256/sim8 at iteration 20 spent
  `323.5s` in `policy_search_sec` and `5.95s` in `visual_stack_update_sec`.
  That does not prove learning quality, but it does say the next optimizer
  target is search/MCTS scaling, not another render rewrite.
- B128/L4 fast was slower per replay row than B64/L4. B128/H100 was much
  better than B128/L4, but not enough to make H100 the default.
- Later read-only live rows sharpened the batch-size read: compare replay rows
  per second, not iteration count. B64/L4 and B32/L4 were close
  (`~272` versus `~266` replay rows/s in representative rows). B128/L4 was much
  worse (`~146` rows/s). Keep B64 as the main default, B32 as a fast-feedback
  canary shape, and B128 as an H100/scale probe rather than an L4 default.
- Plain Amdahl read: use fast direct for the main overnight learning probes.
  Browser-lines is a slow fidelity sentinel, not a control lane that gets equal
  budget. Pay H100 where batch/search is big enough to expose GPU benefit.
- Fresh isolated no-death render A/B, same seed and same B8/sim2 workload:
  `browser_lines` took `53.6s` wall with `31.2s` visual stack time;
  `fast_gray64_direct` took `25.5s` wall with `2.4s` visual stack time. That is
  a real long-survival render bottleneck signal, and it is also why the main
  matrix should stay fast-direct while keeping matched browser sentinels.
- Full-loop fast-direct learner-on profile before the model-hash patch:
  B64/L4/sim8, 24 iterations, 4 updates/iteration, normal death, took `401.5s`.
  Per iteration, collect spent about `4.8s` in policy/search, `2.9s` in
  observation/replay noise, `1.4s` in visual stack, `1.2s` in autoreset, and
  `0.4s` in env step. The missing wall time was learner/sample/checkpoint/
  progress overhead; the old per-update model hash was a likely CPU sync tax and
  is now off by default.

Plain split: for `fast_gray64_direct` normal training, optimize search, learner
timing visibility, noise, and reset. For `browser_lines` long-survival
profiling, optimize render. Do not mix those two stories.

- Follow-up no-hash full-loop profile, B64/L4/sim8, 12 iterations, same broad
  shape, took `289.4s`. It did not prove a hash-off speedup. It did prove that
  actual learner forward was small (`~0.36s/iteration`) while replay sampling
  plus learner batch/target construction cost about `2.9s/iteration`. Leave
  model-hash verification off. This was the evidence that motivated the
  replay-cache patch below.
- Replay-cache follow-up, same broad B64/L4/sim8 fast-direct shape, 12
  iterations, took `234.9s`. Learner-side replay context/sample/batch work fell
  to about `0.5s/iteration`, and iteration wall before progress fell from about
  `22.8s/iteration` to about `18.5s/iteration` in the comparable attribution
  profile. This keeps the same target semantics; it just avoids rebuilding the
  full replay return context on every learner update. Coach should use the
  patched launcher for future runs. Replay/sample is no longer the main named
  fast-direct bottleneck after this patch.

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
  --save-ckpt-after-iter 25 \
  --background-gif-max-steps 2048 \
  --background-gif-frame-stride 4
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
  `--save-ckpt-after-iter 25` for B64 if no better rate estimate exists.
- Overnight matrix after the launch path is healthy: aim for one checkpoint
  every 10-20 minutes. Use `10`, `25`, `50`, or `100` iterations depending on
  measured iteration speed. If B64 is around 30 seconds/iteration, `50` is too
  sparse; use `25`.
- Very long follow-up runs: checkpoint every 30-60 minutes if eval/GIF artifacts
  are already working and intermediate visibility is less important. Relax
  further only after the first curves look sane.
- Profiling-only runs: avoid initial checkpoints and set checkpoint cadence
  high enough that artifact work is not part of the timing result.
- Each checkpoint can spawn eval plus GIF. Keep background eval/GIF on for real
  training visibility, but cap GIF work with `--background-gif-max-steps 2048`
  and `--background-gif-frame-stride 4` until artifact cost is proven harmless.
- Background checkpoint eval is fixed-opponent/survival observability. It is not
  a two-seat self-play strength proof. Coach should prioritize survival curves,
  terminal causes, sparse outcome, action histograms, and collapse checks.

## Recommended 40-Run Matrix

Capacity is fine, so bias hard toward the approximation lane. Use distinct run
IDs. For paired ablations, keep the same seed family so the comparison is not
pure noise. Browser-lines rows are mandatory sentinels at the end; launch them
with matched fast-direct rows, but do not wait for them before launching the
main approximation matrix.

```text
id  lane                 gpu       render              B    sim  collect  upd  sample  lr       reward             stochastic
01  main                 L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
02  main_seed            L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
03  main_seed            L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
04  main_seed            L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
05  main_seed            L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            default
06  search16             L4/T4     fast_gray64_direct  64   16   64       4    256     unset    default            default
07  search32             L4/T4     fast_gray64_direct  64   32   64       4    256     unset    default            default
08  small_batch          L4/T4     fast_gray64_direct  32   8    64       4    128     unset    default            default
09  large_batch_l4_probe L4/T4     fast_gray64_direct  128  8    64       4    512     unset    default            default
10  large_search_l4_probe L4/T4    fast_gray64_direct  128  16   64       4    512     unset    default            default
11  collect128           L4/T4     fast_gray64_direct  64   8    128      4    256     unset    default            default
12  collect256           L4/T4     fast_gray64_direct  64   8    256      4    256     unset    default            default
13  updates8             L4/T4     fast_gray64_direct  64   8    64       8    256     unset    default            default
14  updates16            L4/T4     fast_gray64_direct  64   8    64       16   256     unset    default            default
15  learner512           L4/T4     fast_gray64_direct  64   8    64       4    512     unset    default            default
16  learner1024          L4/T4     fast_gray64_direct  128  8    64       4    1024    unset    default            default
17  lr_3e-5              L4/T4     fast_gray64_direct  64   8    64       4    256     3e-5     default            default
18  lr_1e-4              L4/T4     fast_gray64_direct  64   8    64       4    256     1e-4     default            default
19  lr_3e-4              L4/T4     fast_gray64_direct  64   8    64       4    256     3e-4     default            default
20  lr_1e-3              L4/T4     fast_gray64_direct  64   8    64       4    256     1e-3     default            default
21  lr_3e-3              L4/T4     fast_gray64_direct  64   8    64       4    256     3e-3     default            default
22  no_bonus             L4/T4     fast_gray64_direct  64   8    64       4    256     unset    no_bonus           default
23  terminal_only        L4/T4     fast_gray64_direct  64   8    64       4    256     unset    terminal_only      default
24  stronger_terminal    L4/T4     fast_gray64_direct  64   8    64       4    256     unset    terminal_x2        default
25  survival_only_ctrl   L4/T4     fast_gray64_direct  64   8    64       4    256     unset    survival_only      default
26  bonus_heavy          L4/T4     fast_gray64_direct  64   8    64       4    256     unset    bonus_x2           default
27  no_obs_noise         L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            obs_noise_0
28  obs_noise_05         L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            obs_noise_05
29  obs_noise_20         L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            obs_noise_20
30  action_repeat_diag   L4/T4     fast_gray64_direct  64   8    64       0    256     unset    default            repeat_20pct
31  action_noop_05       L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            action_noop_5pct
32  no_obs_noise_seed    L4/T4     fast_gray64_direct  64   8    64       4    256     unset    default            obs_noise_0
33  large_batch_h100     H100      fast_gray64_direct  128  8    64       4    512     unset    default            default
34  large_search_h100    H100      fast_gray64_direct  128  16   64       4    512     unset    default            default
35  h100_search32        H100      fast_gray64_direct  128  32   64       4    512     unset    default            default
36  h100_collect128      H100      fast_gray64_direct  128  16   128      4    512     unset    default            default
37  h100_b256            H100      fast_gray64_direct  256  8    64       4    1024    unset    default            default
38  h100_lr_3e-4         H100      fast_gray64_direct  128  8    64       4    512     3e-4     default            default
39  fast_sentinel_b16    L4/T4     fast_gray64_direct  16   8    64       4    128     unset    default            default
40  browser_sentinel_b16 L4/T4     browser_lines       16   8    64       4    128     unset    default            default
```

Rows 09-10 are L4 scale probes, not default recommendations. Rows 39-40 are a
matched visual-surface sentinel pair; read them as approximation smoke, not as a
full browser-lines control lane. A second browser-lines learning-rate sentinel
can be added later if the first pair shows a concerning split.

Rows 09-10 and 16 are stress/attribution probes, not Coach-priority rows. Row
38 confounds H100 with learning rate, so treat it as a learning probe, not clean
optimizer evidence. Row 30 is diagnostic only because repeat/no-op skips can
drop terminal targets; do not use it as a real training recommendation.

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

bonus_x2:
  --two-seat-bonus-pickup-reward-per-catch 0.10
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

obs_noise_05:
  --two-seat-observation-noise-std 0.05

obs_noise_20:
  --two-seat-observation-noise-std 0.20

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

`none` is currently identical to `obs_noise_0` because random action no-op and
policy repeat skip are already off by default. Use it only as a no-noise seed
replicate or rename it to avoid fake coverage.

Optimizer recommendation: do not make random action no-op the default tonight.
It changes the agent-action contract and was recently added. Include one small
no-op probe if Coach wants to test robustness. Prefer default obs noise and a
separate no-stochasticity control.

Coach death-signal audit update, 2026-05-12: `repeat_20pct` is diagnostic only
until skipped physical ticks have reward/return accounting. The trainer now
refuses real optimizer training when `policy_action_repeat_*` can create skipped
policy ticks, because a death or bonus on a skipped tick would not be a replay
target. Keep learning runs at repeat min/max `1` and extra probability `0`.

## Hardware Read

- Use L4/T4 for most runs. It is cheaper and enough for B64 fast-direct runs.
- Use H100 only for the six listed scale probes. H100 helps B128 more than L4,
  but it is not proven to improve learning per dollar or per wall-clock yet.
- Skip multi-GPU tonight. The canonical two-seat path has not shown a clean
  model-throughput bottleneck where multi-GPU is the obvious next move.

## What To Watch

- Primary learning read belongs to Coach: survival length, sparse outcome,
  terminal causes, collapse checks, and action histograms.
- Optimizer read: wall time per checkpoint, replay rows/sec, self-play rows/sec,
  policy/search time, env step time, visual stack time, learner time, checkpoint
  artifact time.
- For runs started after the 2026-05-12 optimizer instrumentation patch, split
  `policy_search_sec` with `policy_tensor_prepare_sec`,
  `policy_collect_forward_sec`, `policy_output_decode_sec`, and
  `policy_batch_fallback_sec`. The old total still matters; the new fields tell
  us whether the next speed target is tensor movement, LightZero/MCTS, or Python
  decoding.
- Also read `learner_timing_summary` and `iteration_timing_summary`. These are
  the fields that tell us whether replay sampling, learner update, checkpoint
  save, progress write, or progress Volume commit is eating the missing wall
  time.
- Ignore warm-up when reading speed. A tiny isolated GPU smoke showed first
  policy forward at about `1.06s` and the next at about `0.041s`; use later
  iterations for steady-state timing.
- If fast-direct improves but browser-lines fails, the approximation may be too
  lossy. If both fail the same way, look at reward/search/training setup before
  blaming rendering.
- Larger self-play batches are useful only if the learner actually turns the
  extra collected rows into better checkpoint behavior. They are not magic; they
  trade fresher, more frequent updates for bigger batches and more compute per
  update.

## Minimal Set If Time Is Short

Run these first. This is still approximation-heavy, but it includes one matched
visual sentinel pair:

```text
01 main
02 main_seed
03 main_seed
04 main_seed
05 main_seed
06 search16
08 small_batch
13 updates8
18 lr_1e-4
19 lr_3e-4
22 no_bonus
23 terminal_only
27 no_obs_noise
33 large_batch_h100
34 large_search_h100
39 fast_sentinel_b16
40 browser_sentinel_b16
```

Then add the remaining rows if the launch surface is healthy.
