# Granular Action Cadence Trainer-Loop Side Effects

Date: 2026-05-13

Scope: trusted `source_state_fixed_opponent` lane in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`,
which calls stock `lzero.entry.train_muzero`.

## Short Verdict

The cadence change is mostly an env/config semantic change, not a custom trainer
loop change. Stock LightZero still owns collector, MCTS/search, replay, learner,
and stock checkpoint creation.

The main side effect is that LightZero loop counters now count granular source
physics frames. A run with the same `max_env_step`, `source_max_steps`,
`td_steps`, or checkpoint interval now covers much less game time than an old
bundled-cadence run.

## Facts From Code

- Default trusted trainer settings are small: `max_env_step=8192`,
  `max_train_iter=64`, `source_max_steps=256`, `num_simulations=8`,
  `batch_size=16`, and `save_ckpt_after_iter=100`.
- `DEFAULT_DECISION_MS` now comes from the source-state env default, and the
  trainer records `decision_source_frames=1`,
  `source_physics_step_ms=DEFAULT_SOURCE_PHYSICS_STEP_MS`, and
  `source_max_steps_semantics=source_physics_steps`.
- `mode="train"` and `mode="dry"` reject stale multi-frame `decision_ms` for
  `source_state_fixed_opponent`. Explicit repeat must use
  `policy_action_repeat_*`.
- `_build_visual_survival_configs` still starts from
  `zoo.atari.config.atari_muzero_config`, then patches LightZero policy/env
  config. It does not replace LightZero collector, replay, search, or learner.
- The trainer still calls `train_muzero([main_config, create_config], seed=...,
  max_train_iter=..., max_env_step=...)`.
- `source_max_steps` is written to both env `source_max_steps` and `max_ticks`.
  With one source frame per decision, this means one episode cap unit is one
  source physics frame.
- The env step still receives one scalar ego action from LightZero. The env
  resolves the opponent action, builds one joint action, advances the source
  env, sums reward across any explicit repeat, and returns one LightZero
  timestep.
- `td_steps` is patched to `source_max_steps` for the trusted source-state
  reward variants. The unit is now granular source frames.
- Reward/value support is still derived from reward variant and
  `source_max_steps`, capped at `SOURCE_STATE_FIXED_OPPONENT_MAX_MODEL_SUPPORT_SCALE`
  for fixed-opponent variants.
- Batch sizing is unchanged by the cadence patch. `batch_size` is still passed
  into LightZero policy config.
- Checkpoint cadence is unchanged in trainer-iteration units. Stock checkpoint
  save runs first, then CurvyZero writes progress, resume sidecars, mirrors, and
  optional background artifact triggers.
- `progress_latest.json` records LightZero checkpoint iteration and
  `learner_train_iter`; it does not convert progress to game-time seconds or
  old bundled-decision units.
- Resume sidecars save collector/evaluator counters, learner
  `collector_envstep`, policy extras, RNG state, and replay metadata. Raw
  LightZero `GameSegment` objects are still not restored, so resume is
  operational continuity, not exact uninterrupted training equivalence.

## Trainer-Loop Side Effects

- Collector: no code change to stock collector, but each collected env step now
  represents one source frame. At the same `max_env_step`, collector data covers
  about one old source frame per transition instead of a bundled decision
  window. This is the intended control semantics, but it changes data horizon.
- Replay: no code change to stock replay. Replay rows are now finer-grained.
  Any fixed replay capacity or sample mix inside LightZero will cover less
  physical game time for the same number of stored transitions.
- Learner: no code change to learner cadence. Same `max_train_iter` and
  `batch_size` mean the learner sees the same number of update calls and sample
  shapes, but those samples represent shorter transitions.
- Search: `num_simulations` is unchanged. Search is still per LightZero action,
  so total search work per source second rises compared with the old bundled
  cadence if `max_env_step` is scaled up to recover the old physical horizon.
- Checkpoints: `save_ckpt_after_iter` is unchanged and counts learner
  iterations, not env frames. Checkpoint labels remain `iteration_N`. A
  checkpoint at the same `N` is not directly comparable to an old bundled-cadence
  checkpoint unless cadence metadata is considered.
- `max_env_step`: now means granular source-frame steps for this lane. Keeping
  the old value makes runs physically shorter. Raising it to match old physical
  time increases collection, search, replay, and telemetry volume.
- `max_train_iter`: unchanged unit. It may need retuning only because each
  training sample covers less game time.
- `td_steps`: now spans `source_max_steps` granular frames. This is clear and
  consistent, but for dense survival rewards it changes the physical value
  horizon relative to old bundled runs.
- Value/reward support: support ranges still scale from `source_max_steps`, not
  from old bundled physical duration. If `source_max_steps` is increased to
  recover old physical duration, support caps can become active more often.
- Batch sizing: no immediate compatibility risk. Throughput risk exists if
  future runs increase `max_env_step` or `source_max_steps` to recover old
  physical horizons without lowering search or batch knobs.
- Resume state: counters are LightZero/env-step counters. Resuming across the
  cadence change should be treated as mixed-semantics unless the old run's
  command/config proves the same one-frame cadence.
- Checkpoint progress: status/progress names still look like iteration progress.
  They do not by themselves tell the reader that each transition is now a
  source physics frame.

## Main Risks

- Old and new checkpoints may be compared as if `iteration_N` means the same
  training exposure. It does not if cadence differs.
- A smoke with `max_env_step=8192` may complete cleanly but cover much less game
  time than older smokes.
- Dense survival/value-target settings may need retuning after real learning
  runs because per-transition reward is now per source frame.
- `td_steps=source_max_steps` can become expensive or poorly scaled if
  `source_max_steps` is raised to recover old physical duration.
- Resume from an old bundled-cadence run can silently mix replay/counter
  semantics unless blocked or clearly labeled.
- Background eval/GIF may look fine while training-loop progress is still being
  judged in iteration units.

## Recommended Tests And Smokes

1. Fresh tiny train smoke after the cadence patch:
   `--mode train --compute cpu --max-env-step 32 --max-train-iter 2 --source-max-steps 8 --save-ckpt-after-iter 1`.
   Check that `called_train_muzero=true`, checkpoints exist, telemetry rows
   have `decision_source_frames=1`, and `progress_latest.json` advances.
2. Same smoke with `background_eval_enabled=false` to isolate the trainer loop
   from artifact workers.
3. Same smoke with background eval/GIF enabled to prove checkpoint polling still
   sees every stock checkpoint under the new cadence.
4. Resume smoke from a same-cadence checkpoint. Check that auto-resume selects
   the checkpoint, sidecar load result is recorded, and progress continues.
5. Negative resume guard: try to auto-resume from a known old bundled-cadence
   run and require either a refusal or a loud mixed-cadence warning.
6. Config assertion test for trainer-loop knobs: compiled config should report
   `decision_source_frames=1`, `source_max_steps_semantics=source_physics_steps`,
   unchanged `batch_size`, unchanged `num_simulations`, and `td_steps ==
   source_max_steps`.
7. Learning-readiness smoke with slightly larger horizon, enough to create more
   than one checkpoint. Compare env-step count, learner iteration, checkpoint
   iteration, telemetry row count, and wall time.
8. Support-scale test for large `source_max_steps`: prove the value/reward
   support cap is visible in summary/config when it activates.

## Do Not Claim Yet

- Do not claim old and new checkpoints are comparable by iteration number.
- Do not claim exact resume equivalence.
- Do not claim the new cadence learns better until a fresh post-patch learning
  run shows it.
