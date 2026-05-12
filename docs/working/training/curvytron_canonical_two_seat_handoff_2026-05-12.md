# CurvyTron Canonical Two-Seat Coach Handoff - 2026-05-12

## Copyable Agent Handoff

Current Coach main line is CurvyTron two-seat current-policy self-play through
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
The old `lightzero_curvytron_two_seat_train_smoke.py` Modal wrapper is deleted;
do not use old commands except as historical notes translated to the canonical
launcher. Stock LightZero eval is off by default; use CurvyZero checkpoint eval,
inspection, and GIF jobs for observability. Default checkpoint cadence is sparse,
about every `100` iterations. Keep the path close to LightZero: only the
environment and the small simultaneous-action/self-play bridge should be custom.
Default starts are already varied by generated reset seeds. The default ruleset
uses source-default natural bonus spawning; no-bonus is now only an explicit
ablation. Default robustness noise is mild: policy actions use repeat `min=1`,
`max=3`, `extra_probability=0.20`, which is about `80%` normal, `16%` held one
extra step, and `4%` held two extra steps. Visual input gets Gaussian noise
`0.10`, and random no-op/drop is off.
Next real work is to launch and monitor clean long CurvyTron self-play runs from
this canonical path, with survival curves and collapse checks.

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
- Reset starts are varied: the trainer seeds the vector env, then calls
  `reset(seed=None)`, so each row gets a generated reset seed.
- Source-default natural bonus spawning is on by default. Do not use no-bonus as
  the main Coach ruleset; disable natural bonuses only for controlled ablations.
- Default stochasticity is mild and simple: `policy_action_repeat_max=3`,
  `policy_action_repeat_extra_probability=0.20`, `observation_noise_std=0.10`,
  `action_noop_probability=0.0`, and no warmup schedule.
- Checkpoints write to:
  `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero`.
- Progress writes to:
  `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train`.

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
