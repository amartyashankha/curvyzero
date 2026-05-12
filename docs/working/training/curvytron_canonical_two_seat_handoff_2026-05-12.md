# CurvyTron Canonical Two-Seat Coach Handoff - 2026-05-12

## Copyable Agent Handoff

Current Coach main line is CurvyTron two-seat current-policy self-play through
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
held actions. With `min=1`, `max=3`, `extra_probability=0.20`, a seat takes one
real policy action, then may skip the next one or two policy chances by sending
NOOP. Skipped no-op ticks do not create replay rows or reward targets. Visual
input gets Gaussian noise `0.10`, and random no-op/drop is off.
The default trainer reward is now shaped
but labeled: each per-seat replay row gets a tiny alive helper `+0.01`, an
immediate same-step bonus pickup helper `+0.05` per bonus caught by that player,
plus the sparse terminal outcome scaled by `0.01 * episode_step_count`.
Components are logged separately as training reward, dense helper, bonus pickup
helper, sparse outcome, and terminal outcome.
Next real work is to launch and monitor clean long CurvyTron self-play runs from
this canonical path, with survival curves and collapse checks.

Current setup recommendation for the first overnight run: use `gpu-l4-t4`,
`batch_size=32`, `num_simulations=8`, `collect_steps_per_iteration=64`,
`updates_per_iteration=4`, accumulated replay, `learner_sample_size=128`,
normal death, sparse checkpoints around every `100` iterations, and background
CurvyZero eval/GIF on. Latest wait-mode timing makes B64/sample256 only
interesting after early B32 checkpoints show healthy fresh decisions; B64 is
slower to checkpoint and is not better on rows/sec. Do not assume B128,
sim16, or multi-GPU helps until render/search stops dominating.

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
been deleted. Historical commands must be translated to the canonical launcher.

## What Works Now

- The default Coach path is current-policy two-seat self-play.
- One live LightZero MuZero policy chooses actions for both CurvyTron seats from
  the same pre-step state.
- The env advances once with the joint action.
- The learner updates that same policy before later collection.
- Modal GPU L4/T4 works; smoke saw model parameters on `cuda:0`.
- Current measured path uses full source-state visual input:
  `trail_render_mode=browser_lines`, browser-sprite bonus rendering, and
  LightZero policy/search on `cuda:0`. Env stepping, visual render/downsample,
  replay packaging, and observation noise are still CPU-side. Warmed profiles
  say render is the largest bucket, so B64 is useful for more self-play per
  checkpoint but is not a free throughput win.
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
- Default stochasticity is mild and simple: legacy
  `policy_action_repeat_max=3` / `policy_action_repeat_extra_probability=0.20`
  means policy no-op skipping, not action holding. Skipped ticks send NOOP and
  stay out of replay/reward targets. `observation_noise_std=0.10`,
  `action_noop_probability=0.0`, and no warmup schedule.
- Default trainer reward is the single reward float LightZero consumes from
  replay rows: dense alive helper `+0.01`/step while alive, plus immediate
  same-step bonus pickup helper `+0.05` per bonus caught by that player, plus
  env sparse terminal outcome `(+1/-1/0) * 0.01 * episode_step_count`. This
  keeps terminal outcome on the same scale as the accumulated survival helper,
  while bonus pickup stays local to the step where it happened. Eval survival
  length remains separate telemetry. Reward contract:
  [curvytron_two_seat_reward_contract_2026-05-12.md](curvytron_two_seat_reward_contract_2026-05-12.md).
- Checkpoints write to:
  `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero`.
- Progress writes to:
  `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train`.

## Launch Shape

Preferred first overnight shape once the user gives the exact launch command:

```text
--mode two-seat-selfplay
--compute gpu-l4-t4
--batch-size 32
--num-simulations 8
--two-seat-collect-steps-per-iteration 64
--two-seat-updates-per-iteration 4
--two-seat-replay-scope accumulated
--two-seat-learner-sample-size 128
--two-seat-death-mode normal
--two-seat-trail-render-mode browser_lines
--save-ckpt-after-iter 100
```

Use `--batch-size 64 --two-seat-learner-sample-size 256` later only for a
deliberate slower-feedback run with more self-play rows per checkpoint.
Use `num_simulations=16` only if we deliberately want stronger search and accept
slower wall time. Keep `profile_no_death` and background eval/GIF off for
profiling only; real training uses normal death and background observability.

## Pre-Overnight Timing

Wait-mode timing canaries on 2026-05-12 used normal death, `browser_lines`,
sim8, collect64, updates4, accumulated replay, background eval/GIF off, and
checkpoint cadence 100.

| shape | elapsed for 4 iters | fresh rows | rows/sec | rough checkpoint 100 |
| --- | ---: | ---: | ---: | ---: |
| B32/sample128 | `626s` | `6,734` | `10.8` | `4.3h` |
| B64/sample256 | `1,183s` | `12,590` | `10.6` | `8.2h` |

Both runs were healthy plumbing smokes: `ok=true`, model parameters changed,
CUDA model, accumulated replay, and no trainer problems. B64 did not improve
rows/sec enough to justify delayed feedback for the first overnight canary.

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
