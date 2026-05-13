# CurvyTron Custom Two-Seat Adapter Handoff - 2026-05-12

## Superseded Naming

This file used to call the path "canonical." That is no longer correct. Treat
this as historical/operational notes for the custom two-seat adapter only.

Current research and gates live in:
`docs/working/training/curvytron_architecture_research_2026-05-12/`.

## Copyable Agent Handoff

2026-05-12 correction after the no-learning audit: this file describes the
current operational two-seat launcher, but it is not a trusted stock-LightZero
learning lane. Do not scale it again as the main proof until it either calls
stock `train_muzero`, feeds native `GameSegment` / `MuZeroGameBuffer`, or has a
parity-tested repo-owned learner-target contract. See
[train-muzero reconciliation](curvytron_train_muzero_reconciliation_2026-05-12.md).

Current operational two-seat launcher is
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
The old `lightzero_curvytron_two_seat_train_smoke.py` Modal wrapper is deleted;
do not use old commands except as historical notes translated to the canonical
launcher. Stock LightZero eval is off by default; use CurvyZero checkpoint eval,
inspection, and GIF jobs for observability. Default checkpoint cadence is sparse,
about every `100` iterations. The default two-seat episode cap is now `65,536`
ticks, not the old `16,384`/`2,000` caps, so overnight training should not be
cut short by a small env horizon. Keep the path close to LightZero: only the
environment and the small simultaneous-action/self-play bridge should be custom.
Default starts are varied by generated reset seeds. The run may keep one fixed
top-level seed for reproducibility, but training reset/restart seeds must be
fresh deterministic derivatives, never the same explicit reset seed replayed
for every episode. Replay rows and step records keep `reset_seed` so this can be
audited. The default ruleset uses source-default natural bonus
spawning; no-bonus is now only an explicit ablation. Default robustness noise is
mild: the legacy `policy_action_repeat_*` flags now mean policy no-op skips, not
held actions. They are now off by default: `min=1`, `max=1`,
`extra_probability=0.0`. This is intentional for the first serious runs because
skipped no-op ticks currently do not create replay rows or reward targets, so a
death or bonus during a skipped tick can be miscredited. The trainer now fails
fast if real optimizer training enables this skip path. Visual input gets
Gaussian noise `0.10`, and random no-op/drop is off.
The default trainer reward is now shaped
but labeled: each per-seat replay row gets a tiny alive helper `+0.01`, an
immediate same-step bonus pickup helper `+0.05` per bonus caught by that player,
plus the sparse terminal outcome scaled by `0.01 * episode_step_count`.
Components are logged separately as training reward, dense helper, bonus pickup
helper, sparse outcome, and terminal outcome.
Next real work is not to launch more long runs from this path. It is to restore
stock-loop controls or prove a native replay/target bridge before scaling true
two-seat learning again.

Current optimizer setup recommendation supersedes the older B32/browser-lines
canary: run an aggressive approximation-heavy matrix. The main surface is
`fast_gray64_direct`, not `browser_lines`. Use mostly `gpu-l4-t4`, B64, sim8,
collect64, updates4, accumulated replay, learner sample 256, normal death, and
background CurvyZero eval/GIF on. Browser-lines is only a tiny sentinel, not a
control lane and not a gate. Choose checkpoint cadence from warm-up wall time:
target one checkpoint every 5-10 minutes for canaries and 10-20 minutes for the
overnight matrix. Full optimizer matrix:
[optimizer recommendations](../optimizer/coach_next_training_run_recommendations_2026-05-12.md).

## Short Version

Use one launcher:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Run it with:

```text
--mode two-seat-selfplay
```

The older Modal wrapper
`src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py` has
been deleted. Historical commands must not be treated as learning guidance
without the May 12 postmortem checks.

## What Works Now

- This path performs current-policy two-seat action collection, but it is not
  the default trusted learning path.
- One live LightZero MuZero policy chooses actions for both CurvyTron seats from
  the same pre-step state.
- The env advances once with the joint action.
- The learner updates that same policy before later collection.
- Modal GPU L4/T4 works; smoke saw model parameters on `cuda:0`.
- Current measured reference path uses full source-state visual input:
  `trail_render_mode=browser_lines`, browser-sprite bonus rendering, and
  LightZero policy/search on `cuda:0`. Env stepping, visual render/downsample,
  replay packaging, and observation noise are still CPU-side. Warmed profiles
  say browser-lines rendering is the largest bucket. The active optimizer
  training recommendation therefore uses `fast_gray64_direct`, a strong
  semantic approximation that is much faster but not browser pixel fidelity.
- Reset starts are varied: the trainer seeds the vector env, then calls
  `reset(seed=None)` and `autoreset_done_rows(seed=None)`, so each row/reset
  gets a generated reset seed. Replay rows and step records carry `reset_seed`.
  The native LightZero env wrapper also preserves config-level `dynamic_seed`
  when an env manager calls `env.seed(...)`, so the trainer cannot accidentally
  force fixed repeated starts.
- Source-default natural bonus spawning is on by default. Do not use no-bonus as
  the main Coach ruleset; disable natural bonuses only for controlled ablations.
- Default two-seat env horizon is `65,536` ticks. Background checkpoint survival
  eval also defaults to `65,536` max steps so the eval does not hide long
  survival. GIFs stay shorter by default so artifacts do not become huge.
- Default stochasticity is mild and simple: legacy `policy_action_repeat_*`
  means policy no-op skipping, not action holding, and is disabled by default
  with `policy_action_repeat_min=1`, `policy_action_repeat_max=1`, and
  `policy_action_repeat_extra_probability=0.0`. Keep it off for the baseline
  overnight run until reward/replay accounting for skipped physical ticks is
  proven. `observation_noise_std=0.10`, `action_noop_probability=0.0`, and no
  warmup schedule.
- Default trainer reward is the single reward float LightZero consumes from
  replay rows: dense alive helper `+0.01`/step while alive, plus immediate
  same-step bonus pickup helper `+0.05` per bonus caught by that player, plus
  env sparse terminal outcome `(+1/-1/0) * 0.01 * episode_step_count`. This
  keeps terminal outcome on the same scale as the accumulated survival helper,
  while bonus pickup stays local to the step where it happened. Eval survival
  length remains separate telemetry. Reward contract:
  [curvytron_two_seat_reward_contract_2026-05-12.md](curvytron_two_seat_reward_contract_2026-05-12.md).
- Death signal audit: for normal fresh policy decisions, wall/trail/body deaths
  propagate into replay and learner targets through `alive_after`,
  `sparse_outcome_reward`, `terminal_outcome_reward`, `reward_batch`, and
  discounted `target_value`. Death cause labels are now preserved on replay rows
  for debugging, but cause type does not change the reward value.
- Checkpoints write to:
  `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero`.
- Progress writes to:
  `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train`.

## Launch Shape

Preferred first overnight baseline shape:

```text
--mode two-seat-selfplay
--compute gpu-l4-t4
--batch-size 64
--num-simulations 8
--two-seat-collect-steps-per-iteration 64
--two-seat-updates-per-iteration 4
--two-seat-replay-scope accumulated
--two-seat-learner-sample-size 256
--two-seat-death-mode normal
--two-seat-trail-render-mode fast_gray64_direct
--save-ckpt-after-iter 50
```

Run the broader approximation-heavy matrix from the optimizer recommendation
doc when capacity is available. Keep at most one or two `browser_lines`
sentinels, and do not wait for them before launching fast-direct runs.

Baseline stochasticity proof check before launch:

```text
control_stochasticity.counts.policy_noop_skip_rows == 0
fresh_policy_action_summary.decision_count ==
  control_stochasticity.physical_action_summary.executed_action_count
```

If this fails, do not treat the run as the clean baseline. Stochastic variants
are still allowed, but they must be named as variants and read with the skip
accounting caveat. As of the death-signal audit on 2026-05-12, policy no-op
skip variants are blocked for real optimizer training until skipped physical
ticks have explicit reward/return accounting.

## Run Naming

Every serious run and attempt must start with a clear purpose prefix, then name
the important variant knobs. Do not use bare seed names or vague labels.

Recommended prefix format:

```text
curvy2seat-selfplay-<purpose>-<variant>-<date_or_batch>
```

Examples:

```text
curvy2seat-selfplay-baseline-noskip-b32-sim8-20260512
curvy2seat-selfplay-ablate-noobsnoise-b32-sim8-20260512
curvy2seat-selfplay-variant-obsnoise10-b32-sim8-20260512
curvy2seat-selfplay-variant-noopskip20-accounting-b32-sim8-20260512
```

Baseline means no policy no-op skip. If a run enables policy no-op skip,
action drop, extra observation noise, changed reward scale, changed replay
scope, or changed render mode, put that in the run id and attempt id.

Use B128/H100/search-depth variants as explicit matrix rows, not as the single
baseline. Keep `profile_no_death` and background eval/GIF off for profiling
only; real training uses normal death and background observability.

## Pre-Overnight Timing

Historical wait-mode timing canaries on 2026-05-12 used normal death,
`browser_lines`, sim8, collect64, updates4, accumulated replay, background
eval/GIF off, and checkpoint cadence 100. These are now browser-lines
historical context, not the active fast-direct recommendation.

| shape | elapsed for 4 iters | fresh rows | rows/sec | rough checkpoint 100 |
| --- | ---: | ---: | ---: | ---: |
| B32/sample128 | `626s` | `6,734` | `10.8` | `4.3h` |
| B64/sample256 | `1,183s` | `12,590` | `10.6` | `8.2h` |

Both runs were healthy plumbing smokes: `ok=true`, model parameters changed,
CUDA model, accumulated replay, and no trainer problems. B64 did not improve
rows/sec in browser-lines mode. That does not apply to the current
`fast_gray64_direct` recommendation, where B64 is the main baseline and
browser-lines is only a slow sentinel.

## Action Collapse Gate

Do not treat a single-action greedy GIF by itself as proof of training collapse.
The GIF path is deterministic eval, and older checkpoints were also mostly one
greedy action, just often a different action. Use it as a warning.

The primary launch gate is fresh policy decisions from the trainer progress:
per-player action counts should have nonzero support beyond one action, top
action fraction should stay below the collapse threshold, and entropy should
not be near zero. Physical action counts are secondary because policy no-op
skips intentionally send NOOP and can make executed actions look skewed.

If fresh decisions collapse too, stop and debug before overnight. If fresh
decisions are healthy but greedy GIFs collapse, launch can proceed as a canary
with the collapse warning tracked by checkpoint.

## Proof Smoke

Run:

```text
run_id: curvytron-canonical-two-seat-wrapup-smoke-20260512
attempt_id: wrapup-smoke
```

Result:

- `ok=true`
- learner forward succeeded
- model parameters changed
- `lightzero_policy_model_device=cuda:0`
- wrote `iteration_0.pth.tar`
- wrote `iteration_1.pth.tar`
- no trainer problems

## Eval Caveat

Background checkpoint eval, inspection, and GIF spawning are on by default. The
loader infers the checkpoint model support-head size before loading, so two-seat
checkpoints with the LightZero Atari-style 601-wide heads can be inspected.

This is CurvyZero checkpoint observability, not stock LightZero's in-training
evaluator. Stock LightZero eval stays off by default with `lightzero_eval_freq=0`.

The current checkpoint survival read still uses a fixed-opponent eval surface,
so do not treat it as proof of two-seat self-play strength. A later
two-seat-specific eval surface should replace it.

## Observability Proof Smoke

Run:

```text
run_id: curvytron-canonical-two-seat-observability-smoke-20260512
attempt_id: obs-smoke
```

Result:

- trainer `ok=true`
- model parameters changed
- wrote `iteration_0.pth.tar` and `iteration_1.pth.tar`
- checkpoint poller completed
- scheduled and completed 2 eval/inspection jobs
- scheduled and completed 2 GIF jobs
- GIF summaries loaded the checkpoints strictly after inferring support heads:
  `model_reward_support_size=601`, `model_value_support_size=601`
