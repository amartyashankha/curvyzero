# Training Coach Active Board - 2026-05-10

Short current board. This is the working-memory page; detailed evidence lives
in linked docs.

## Start Here

North Star: [coach_north_star_2026-05-10.md](coach_north_star_2026-05-10.md).

Lead metric: survival time. Score/return is secondary unless survival is also
moving.

## Current State

Updated 2026-05-11 21:08 EDT.

- Pong replication has passed the basic learning-signal gate. Stock-like visual
  LightZero Pong learned to survive longer across several same-run checkpoint
  curves.
- Active Modal Pong training jobs have been stopped. Checkpoints and eval
  artifacts remain on the Modal Volume.
- Pong is now a control/reference lane, not the main training lane.
- CurvyTron is the active Coach lane. The source-state turn-commit wrapper,
  `env_variant=source_state_turn_commit`, is useful as a stock LightZero
  plumbing smoke/profile path only. It is not a trainable/default path and not
  learning-quality current-policy self-play.
- Turn-commit shape: player 0's scalar step records a private pending action,
  does not advance physics, and gets reward `0`; player 1's scalar step commits
  the full joint action, advances the real source tick, and gets the survival
  reward. Stock LightZero stores both scalar steps as normal GameSegment
  transitions, so value targets may give player 0 states credit for player 1
  survival. Treat this as a reward-credit risk.
- The old custom two-seat trainer is now diagnostic only. Do not scale it as
  the main answer unless the next explicit task is to design/prove the missing
  joint-action collection semantics.
- Two-seat repeat/dropout plumbing smoke passed on Modal:
  `curvytron-two-seat-repeat-smoke-s6101-20260511` /
  `repeat-smoke-s6101`. It ran on GPU, used one live LightZero policy for both
  seats, changed model weights, saved `iteration_0` and `iteration_1`, and
  logged per-seat repeat behavior (`16` active seat rows, `8` fresh decisions,
  `8` reused actions).
- Four detached long two-seat CurvyTron runs were launched after that smoke:
  `clean-detached-s6301`, `repeat-mild-detached-s6302`,
  `repeat-strong-detached-s6303`, and `obsnoise002-detached-s6304`. All use
  current-policy two-seat collection, accumulated replay, survival reward,
  `batch_size=32`, `collect_steps_per_iteration=128`,
  `updates_per_iteration=8`, `num_simulations=4`, `max_ticks=16384`, and
  checkpoint every `100` iterations. Active containers were visible in
  `modal container list` after launch.
- CurvyTron native LightZero trainer now exposes `source_state_turn_commit` and
  `source_state_joint_action` in addition to fixed-opponent controls. Treat
  fixed/frozen opponent, turn-commit, and centralized joint-action runs as
  controls. None of those are true current-policy competitive self-play.
- Source-state turn-commit plumbing smoke passed after cleanup:
  `curvytron-source-state-turncommit-smoke-s20260511b` /
  `profile-smoke-sim2-c2-steps64-20260511b`. It used stock LightZero
  `train_muzero`, GPU model/search, MCTS, replay sample, one learner step, and
  copied `iteration_0`.
- Target/replay audit smoke then confirmed the blocker:
  `curvytron-source-state-turncommit-audit-smoke-s20260511c` /
  `profile-audit-smoke-sim2-c2-steps64-20260511c`. LightZero GameSegments
  stored fake pending rows with alternating rewards like `0,1,0,1`, and sampled
  value targets backed those commit rewards through pending rows. `mode=train`
  is blocked for `source_state_turn_commit` until reward credit is redesigned.
- That smoke wrote env-step telemetry: `36` scalar rows, `18` pending rows,
  `18` physical-commit rows, balanced acting-player rows (`18/18`), natural
  source mechanics, and no action collapse in sampled rows.
- Env-wrapper cleanup now covers the safe plumbing issues from the audit:
  source-state env knobs, metadata, telemetry, stack schema hash, and
  non-mutating render. This does not fix reward credit.
- One CurvyTron app is still active:
  `ap-pzRnD0oXuFYb4N7yWzORA3` /
  `curvyzero-lightzero-curvytron-two-seat-train-smoke`.

## Pong Facts

Final cleanup evals confirmed that several Pong runs learned to survive longer.
The curves are noisy, so compare each run to its own `iteration_0`.

| run | checkpoints | mean survival steps | mean score |
| --- | --- | --- | --- |
| `s114` L4/T4 stock64 | `0/13159` | `759.625 -> 1014.75` | `-21 -> -19.875` |
| `s120` L4/T4 stock64 | `0/14012` | `759.625 -> 1125.88` | `-21 -> -19.75` |
| `s121` L4/T4 stock64 | `0/17013` | `871.125 -> 759.625` | `-20.875 -> -21` |
| `s142` H100 stock64 repeat | `0/15000/20000/26000/26806` | `759.625 -> 873 -> 1214.62 -> 910.5 -> 1308.88` | `-21 -> ? -> ? -> ? -> -18.5` |
| `s122` H100 stock64 | `0/26000/26672` | `759.625 -> 2395.12 -> 2970.62` | `-21 -> -14.25 -> -12.375` |
| `s123` exact control | `0/20000/30000` | `759.625 -> 1147.25 -> 2008.12` | `-21 -> -19.5 -> -10.75` |
| `s113` exact control | `0/20000/30000` | `759.625 -> 862.125 -> 1481.75` | `-21 -> -20.625 -> -18.25` |

Common eval sampler seed for this cleanup wave:
`202605111551`.

## CurvyTron Facts

Current native trainer path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Despite the stale filename, the native path is source-state visual CurvyTron.
There are now two important variants:

- `env_variant=source_state_fixed_opponent`: fixed/frozen opponent control, not
  self-play.
- `env_variant=source_state_turn_commit`: stock LightZero `train_muzero` with a
  lightweight turn-commit env wrapper. This is a native plumbing smoke/profile,
  not a learning-quality self-play claim. The trainer now blocks
  `mode=train` for this variant after the target audit.
- `env_variant=source_state_joint_action`: stock LightZero control candidate
  where one scalar action indexes the centralized pair
  `(player_0_action, player_1_action)`. This is one real source tick per
  transition and avoids turn-commit pending rows, but it is centralized control,
  not true competitive self-play.
- Both use non-ALE visual stack `[4,64,64]`.

Optimizer smoke evidence:

- `opt-native-train-smoke-c16-s1121-wait`
- `train-smoke-c16-sim16-sparse-wait`
- `called_train_muzero=true`
- `ok=true`
- checkpoints copied for `iteration_0`, `iteration_35`, and `ckpt_best`

This is setup evidence only. It is not a CurvyTron learning claim.

Reward-credit caveat for turn-commit: the first scalar step is bookkeeping and
gets zero reward because physics has not advanced yet. The second scalar step
commits the real tick and gets the physical-step survival reward. Because stock
LightZero stores both scalar steps as normal transitions, value targets can
credit the pending/player0 state for player1 survival. Use turn-commit for
stock plumbing smoke/profile only; do not call it trainable or current-policy
self-play.

Turing recommendation, candidate/control only until tested: a 9-action
centralized joint-action wrapper. One LightZero scalar action maps to
`(p0_action, p1_action)`, one real CurvyTron tick, one reward,
`to_play=-1`, and `action_space_size=9`. Loud caveat: centralized control, not
true competitive self-play.

Implementation note: the first diagnostic scalar is `+1` only while both
players are alive after the real tick, otherwise `0`. That is a single control
reward for the centralized policy. It is not per-player reward, not zero-sum,
and not a competitive self-play objective. Do not apply two-seat
winner/loser-return shaping to this wrapper unless it grows an honest per-player
target surface.

Related cleanup ref: [native reuse critique](training/curvytron_lightzero_native_reuse_critique_2026-05-10.md).

Reward-shaping note: a shared `+1 per survived step` signal is acceptable as a
short-term diagnostic, but log sparse outcome and shaped survival separately.
Long-loss-vs-short-win scale issues are recorded for later reward design, not
the current plumbing blocker.

The active two-seat path now exposes policy-action-repeat/dropout knobs. The
default is no repeat. Robustness variants may set these knobs, but learning
claims should say whether the run used them.

Current launched two-seat run refs:

| run | attempt | function call | variant |
| --- | --- | --- | --- |
| `curvytron-two-seat-selfplay-clean-detached-s6301-20260511` | `clean-detached-s6301` | `fc-01KRCAT6FCJ1CMC1G1WW3973SW` | no repeat, no visual noise |
| `curvytron-two-seat-selfplay-repeat-mild-detached-s6302-20260511` | `repeat-mild-detached-s6302` | `fc-01KRCAT6F1A4F0Q2Q1BEWY18XR` | repeat max `3`, extra probability ramps to `0.10` over `1000` iterations |
| `curvytron-two-seat-selfplay-repeat-strong-detached-s6303-20260511` | `repeat-strong-detached-s6303` | `fc-01KRCAT6E0JD54V57ZNHXFFZ3R` | repeat max `4`, extra probability ramps to `0.25` over `2000` iterations |
| `curvytron-two-seat-selfplay-obsnoise002-detached-s6304-20260511` | `obsnoise002-detached-s6304` | `fc-01KRCAT6EAPMT6AAKGTYXBQR89` | observation noise std `0.02`, no repeat |

The earlier non-detached `s6201`-`s6204` launch attempt printed refs but did
not show active containers or progress directories. Treat those refs as
superseded unless later Volume evidence appears.

Check at 2026-05-11 16:36 EDT:

- Detached `s6301`-`s6304` apps are active in Modal.
- Progress files exist for all four runs.
- `progress_latest.json` is still the `start` row for all four runs.
- `iteration_0`, `latest.pth.tar`, and `ckpt_best.pth.tar` exist for all four.
- No iteration-25 progress row or iteration-100 checkpoint yet.

## CurvyTron Auto Resume

Implemented 2026-05-11 in:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Facts:

- Resume is automatic. There is no resume flag.
- At startup, the trainer scans the current run directory for
  `iteration_*.pth.tar` checkpoints.
- It checks the current attempt, older attempts under the same run, and the
  run-level `checkpoints/lightzero` mirror.
- It picks the highest numbered iteration checkpoint and sets
  `policy.learn.learner.hook.load_ckpt_before_run`.
- LightZero/DI-engine then restores learner `last_iter` and `last_step` when
  present, plus MuZero model, target model, and optimizer state.
- If a matching CurvyZero sidecar exists, the trainer also restores safe state:
  collector progress, evaluator progress, policy helper state when visible, and
  Python/NumPy/Torch random state.
- Replay GameSegments are not restored yet. Raw LightZero GameSegment objects
  include state that did not reload cleanly through this sidecar path.
- Live environment manager internals are not restored.
- Sidecars use `iteration_N.resume_state.pkl`. The checkpoint-name parser uses
  plain string checks, not regex.

Validation:

- Dry smoke `autoresume-dry-smoke-20260511a` found
  `iteration_35.pth.tar` and wrote it into `load_ckpt_before_run`.
- Train smoke `autoresume-train-smoke-20260511a` loaded the same checkpoint,
  called `train_muzero`, saved `iteration_35/36/37`, and had no action
  collapse in telemetry (`0:14, 1:15, 2:16`).
- Two-phase sidecar smoke
  `curvytron-autoresume-sidecar-smoke-20260511c` passed:
  phase 2 found `iteration_2.pth.tar`, found
  `iteration_2.resume_state.pkl`, resumed, collected new env rows, saved four
  checkpoints, and had no action collapse (`0:6, 1:6, 2:8`).
- This validates resume mechanics only. It is not a CurvyTron learning claim.

## Reporting Rules

- Report survival first.
- Compare each run to its own `iteration_0`.
- Do not call early flat Pong reads failures before enough horizon.
- Do not mix Pong and CurvyTron claims.
- Do not report a training claim without checkpoint refs, eval settings,
  action histogram or collapse check, and a plain non-claim.
- Treat seeds as reproducibility tools, not a thing to overfit.

## Current Gates

- Keep CurvyTron as the active training lane.
- Use CurvyTron checkpoint eval/inspection artifacts for survival curves,
  action behavior, and death causes.
- Treat no survival growth in CurvyTron as a stop-and-debug signal.

## Links

- Pong monitor:
  [lightzero_pong_replication_monitor_2026-05-11.md](lightzero_pong_replication_monitor_2026-05-11.md).
- Pong stock64 comparison:
  [training/pong_stock64_signal_comparison_2026-05-11.md](training/pong_stock64_signal_comparison_2026-05-11.md).
- Training knowledge index:
  [training/coach_training_knowledge_index_2026-05-11.md](training/coach_training_knowledge_index_2026-05-11.md).
- CurvyTron eval/inspection handoff:
  [training/curvytron_checkpoint_eval_inspection_handoff_2026-05-11.md](training/curvytron_checkpoint_eval_inspection_handoff_2026-05-11.md).
- Optimizer Coach handoff:
  [optimizer/coach_handoff_2026-05-11.md](optimizer/coach_handoff_2026-05-11.md).
