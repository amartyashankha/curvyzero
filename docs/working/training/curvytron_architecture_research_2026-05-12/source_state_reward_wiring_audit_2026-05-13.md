# Source-State Reward Wiring Audit

Date: 2026-05-13

Scope: `source_state_fixed_opponent` through stock LightZero
`train_muzero` in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.

## Findings

- `survival_plus_bonus_no_outcome` is now implemented for the stock
  fixed-opponent source-state path.
- Trainer reward is post-transition ego survival helper plus same-step
  `bonus_catch_count_step[row, ego_player]` reward.
- Sparse terminal outcome is excluded from `trainer_reward` in this variant.
  It remains available through telemetry/eval fields.
- Bonus catch counts already exist in vector env public info as
  `bonus_catch_count_step[row, player]`, reset each step and incremented at the
  catch site.
- The custom two-seat lane already has the intended reward shape:
  dense survival helper plus immediate same-step bonus pickup plus configurable
  terminal outcome scale. That pattern should be copied narrowly, not the
  two-seat training path.
- LightZero v0.2.0 uses one shared `model.support_scale` for reward and value
  heads. After the e2e mismatch fix, both reward and value supports are
  effective `601` bins from `support_scale=300`; uncapped/capped metadata is
  still recorded for audit.
- Modal dry mode now accepts the same capped support config for the stock
  LightZero path and returns `ok=true`.

## Implemented Contract

`survival_plus_bonus_no_outcome` has:

- trainer reward: post-transition ego survival helper plus
  `bonus_catch_count_step[0, ego_player_index] * bonus_pickup_reward_per_catch`;
- sparse outcome reward excluded from `trainer_reward`;
- sparse terminal outcome kept only in telemetry/eval fields;
- explicit schema id/hash, reward policy metadata, reward space, LightZero
  support scale, CLI choice, and readiness/compile-config coverage;
- shared LightZero model support capped at 300 bins-per-side (`601` output
  bins) with uncapped/capped metadata in `lightzero_target_config`.

## Verification

Local verification passed:

- reward variant tests;
- same-step bonus credit tests;
- terminal outcome exclusion tests;
- launcher/config support-size tests;
- `uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_eval_curves.py`
  -> `74 passed, 1 skipped`;
- ruff and py_compile on the changed reward/tooling/probe files.

Modal runtime verification passed:

- dry run:
  `--mode dry --env-variant source_state_fixed_opponent --reward-variant survival_plus_bonus_no_outcome --source-max-steps 65536 --num-simulations 1`;
- result: `ok=true`, no readiness problems, stock LightZero config compiled far
  enough to validate the reward/value support surface.
- first real e2e canary exposed the previous reward/value support mismatch
  (`5` reward bins vs `601` target bins); the shared-support fix resolved it.
- tiny stock `train_muzero` Modal e2e canaries now pass for:
  `blank_canvas_noop/body_circles_fast`, `blank_canvas_noop/browser_lines`, and
  `normal/body_circles_fast`.
- control stochasticity smoke passed on the stock trainer path:
  action-repeat knobs are CLI/config/surface fields, env accounting treats
  repeat as one LightZero transition with multiple physical source ticks, and a
  tiny Modal stock `train_muzero` smoke completed with repeat
  `min=1/max=3/extra_probability=0.25`, blank canvas, and 46 telemetry rows.

The stock repeat knob is held-action repeat. It is not the older custom
two-seat no-op/dropout mechanism.

Strict-stop high-cap runtime canaries and real live survivaldiag rich-status/eval
snapshots now pass for the first-wave exact lanes. The remaining launch gate is
the final manifest/test/lint pass against the current tree. Reward support
sizing, tiny e2e plumbing, and live reward/status export are no longer the
active blockers.
