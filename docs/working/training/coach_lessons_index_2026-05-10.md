# Coach Lessons Index - 2026-05-10

Purpose: one compact map of today's CurvyTron/LightZero training lessons. Keep
details in the linked notes; do not expand this into another ledger.

No pytest was run for this index.

## Current Read

- CurvyTron training evidence is mechanically cleaner after the target,
  batch-size, reset-row, and replay-scope fixes, but the survival curves are
  still flat.
- Pong now has a real survival signal in the normal LightZero lane, so the
  broad stack can carry signal; Pong is a control, not the CurvyTron gate.
- The custom two-seat adapter proved one live policy can control both seats and
  train from both seats, but it is too much private trainer machinery to scale
  as the main path.
- Native LightZero should own the boring trainer pieces. The missing CurvyTron
  piece is a joint-action/current-policy collector for true simultaneous
  self-play.

## Fixes And Mechanical Lessons

- [Target fix / scale eval](curvytron_targetfix_eval_and_scale_2026-05-10.md):
  corrected-target smoke loaded strictly on 96 checkpoint/seed jobs, but means
  fell from `192.969` at `iteration_0` to `151.406` at `iteration_2`; the later
  scale run wrote 17 checkpoints and changed the model, but did not show a
  learning win.
- [Batch-size and target audit](curvytron_two_seat_bug_audit_2026-05-10.md):
  two-seat replay rows now carry `iteration`, `env_row_id`, `player_id`, and
  `decision_index`, so learner targets use discounted survival returns instead
  of the old immediate-reward fallback. Tiny per-iteration samples remain a
  training risk.
- [Reset-row fix](curvytron_two_seat_bug_audit_2026-05-10.md): reset rows are
  refreshed into the visual stack without rolling or overwriting live rows,
  avoiding stale terminal frames and accidental extra frame shifts.
- [Accumulated replay patch](curvytron_accumulated_replay_patch_2026-05-10.md):
  added `--replay-scope current_iteration|accumulated` and optional
  `--learner-sample-size`; accumulated mode reuses all in-run replay rows while
  preserving the metadata arrays needed by the survival target path.
- [Accumulated replay audit](curvytron_accumulated_replay_audit_2026-05-10.md):
  fixed reporting so top-level `replay.sample` describes final learner-visible
  rows, while `row_count` remains total collected rows.

## Result Curves

- [Current-iteration flat curve](curvytron_targetfix_eval_and_scale_2026-05-10.md):
  corrected scale eval over 64 seeds stayed in a narrow `192.703-197.453` mean
  steps band; final `iteration_16` was only `+1.297` mean steps over
  `iteration_0`.
- [Accumulated replay flat curve](curvytron_accumulated_replay_run_2026-05-10.md):
  accumulated run completed with `4096` replay rows and 33 checkpoints; 64-seed
  eval was `191.688` at `iteration_0`, then `201.844` from `iteration_1`
  through `iteration_32`. Mechanically clean, weak learning evidence.
- [Pong survival signal](../../experiments/2026-05-10-lightzero-wave11-pong-survival-curves.md):
  Wave11 normal seeds show late survival gains: s74 reaches the `2048` cap at
  `30000`, `37000`, and `37542`; s76 reaches positive mean stock return at
  `40000+`. This is stack-health evidence, not a CurvyTron pass.

## Architecture Lessons

- [Native LightZero reuse critique](curvytron_lightzero_native_reuse_critique_2026-05-10.md):
  CurvyTron can look Pong/Atari-like to LightZero for single-ego visual
  training: `[4,64,64]` frames, three actions, survival reward, done. The exact
  blocker for true self-play is that `train_muzero` chooses one ego action, not
  a simultaneous joint action for both seats.
- [Native train_muzero probe](curvytron_native_train_muzero_probe_2026-05-10.md):
  an existing Modal native trainer calls `lzero.entry.train_muzero` for
  `CurvyZeroStackedDebugVisualSurvivalLightZero-v0`, using LightZero's
  collector, replay, MCTS, learner, checkpoints, and artifacts for the
  single-ego/fixed-or-frozen-opponent lane.
- [CurvyTron vs Pong reflection](curvytron_vs_pong_architecture_reflection_2026-05-10.md):
  the custom adapter was a useful smoke bridge, but it should shrink to the
  CurvyTron-specific pieces: simultaneous action ownership, seat metadata,
  survival contract, and multiplayer eval panels.
- [Self-play truth after target fix](curvytron_selfplay_truth_after_targetfix_2026-05-10.md):
  bounded two-seat smoke is current-policy enough for a first learning-signal
  gate: one live `MuZeroPolicy` controls both seats, both seats enter one shared
  learner update, and checkpoints eval. It is not full distributed LightZero
  self-play.
- [Current-policy reality](curvytron_current_policy_selfplay_reality_2026-05-10.md):
  main native `train_muzero` remains learner-vs-fixed or learner-vs-frozen,
  never current-policy self-play. Frozen/checkpoint-lagged opponents are useful
  bridges only if labeled exactly.

## Next Gate

- Keep CurvyTron evals as survival-time distributions over reproducible random
  seed panels, not fixed-seed score chasing.
- Scale native LightZero single-ego/frozen-opponent work for trainer plumbing.
- Keep the two-seat adapter as a narrow research harness until the needed
  joint-action collector shape is explicit.
