# Granular Action Cadence Plan

Date: 2026-05-13

## Plain Goal

The trusted stock LightZero lane should let the policy choose an action at each
granular CurvyTron game step.

The old pattern of one policy action covering a bundle of internal engine ticks
is not acceptable for this lane unless a run explicitly asks for action repeat.

## Current Status

Implemented for the trusted `source_state_fixed_opponent` stock train lane.

- The default wrapper cadence is one source physics frame.
- The trainer config writes `decision_source_frames=1` explicitly.
- Trusted `--mode train` and `--mode dry` reject stale multi-frame
  `decision_ms` values.
- The active survivaldiag and opponent-mixture manifest builders now emit the
  one-frame timing value.
- Regression tests cover default cadence, explicit repeat accounting,
  `source_max_steps` cap semantics, telemetry cadence fields, background
  eval/GIF config metadata, and manifest defaults.

## What Must Stay Simple

- Trusted launcher remains
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`.
- Stock LightZero still owns collector, search, replay, learner, and checkpoint
  creation.
- The env remains the env boundary. The trainer may configure cadence, but it
  should not implement game mechanics.
- Existing stochasticity knobs stay as explicit knobs. Do not redesign them in
  this cut.
- The simultaneous-action detail stays hidden inside the env wrapper: LightZero
  sends one scalar ego action; the env resolves the opponent action and advances
  the source game.

## Questions To Answer First

1. Does the source-state CurvyTron env already support one action per granular
   game step?
2. Which config fields currently stretch one LightZero action across multiple
   game ticks?
3. Are the trusted `--mode train` defaults already `policy_action_repeat_min=1`,
   `policy_action_repeat_max=1`, and
   `policy_action_repeat_extra_probability=0`?
4. Does `decision_ms` or any source env timing setting still imply a hidden
   multi-tick step?
5. What test can prove a single `env.step(action)` advances exactly one
   granular source tick in the trusted config?

## Test-First Plan

1. Audit the path from CLI args to `_build_visual_survival_configs` to
   `CurvyZeroSourceStateVisualSurvivalLightZeroEnv.step`.
2. Add a local regression test proving the trusted stock config uses no action
   repeat by default.
3. Add or tighten an env-boundary test proving one LightZero `step(action)`
   maps to one granular source game advance when repeat is disabled.
4. Patch only the trainer/config plumbing if the defaults or config handoff are
   wrong.
5. Run the focused gate:

```text
uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q
```

## Non-Goals

- No reward redesign.
- No opponent-mixture redesign.
- No launch of training batches.
- No custom learner/collector/replay loop.
- No broad environment rewrite unless the audit proves the env cannot express
  one granular action per step.
