# CurvyTron Train-MuZero Reconciliation - 2026-05-12

Purpose: clarify why CurvyTron drifted away from stock LightZero
`train_muzero`, what code paths exist now, and what must happen before another
large CurvyTron training run.

No code was edited for this note.

## Short Verdict

We did have CurvyTron paths that call stock LightZero `train_muzero`.

The path that was scaled on May 12 was not one of them. It was the custom
two-seat path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode two-seat-selfplay
```

That path owns collection, replay rows, learner target construction, and
checkpointing. It uses LightZero policy/search and `learn_mode.forward`, but it
does not call `train_muzero` and does not use LightZero's collector/GameBuffer.

This was the wrong thing to scale as the main learning proof.

## Existing Train-MuZero Paths

### 1. `source_state_fixed_opponent`

Status: real stock LightZero training path, but not current-policy self-play.

Shape:

```text
LightZero train_muzero -> one ego action -> env.step(action)
env owns player_1 as fixed straight or frozen checkpoint opponent
```

Evidence:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  imports `lzero.entry.train_muzero` and calls it in `mode=train` / `mode=profile`.
- `docs/working/training/curvytron_native_frozen_opponent_probe_2026-05-10.md`
  says this path mechanically works and recorded `called_train_muzero: true`.

Why it was not treated as the main line:

- It is learner-vs-fixed/frozen opponent, not both seats controlled by the live
  current policy.
- The env cannot ask the live collector policy for player 1 because LightZero
  only passes one scalar ego action into `env.step(action)`.

Plain read: this path should remain the first stock-loop control. It is the
closest Pong-like CurvyTron route.

### 2. `source_state_turn_commit`

Status: stock `train_muzero` plumbing/profile only; blocked for training.

Shape:

```text
step 1: LightZero action for player 0 -> store pending action, no physics
step 2: LightZero action for player 1 -> commit joint action, advance physics
```

Why it was attractive:

- It lets stock LightZero call `env.step(action)` with one scalar action.
- Both player actions can come from the same current policy over alternating
  scalar calls.

Why it got blocked:

- LightZero's collector/GameBuffer sees the fake pending step as a normal
  transition.
- The replay sequence becomes artificial pending/commit scalar steps, not one
  physical CurvyTron tick per transition.
- The target audit found reward-credit risk: pending player-0 rows can receive
  value credit from player-1 commit rewards.

Plain read: the fake-step idea was not obviously crazy, but in stock LightZero
it is not safe unless we change what gets stored as replay or make one replay
transition correspond to one physical tick.

### 3. `source_state_joint_action`

Status: real stock `train_muzero` route, but centralized control, not
competitive self-play.

Shape:

```text
one scalar action 0..8 -> decode to (player_0_action, player_1_action)
env advances one physical tick
LightZero sees one normal scalar transition
```

Why it matters:

- It preserves the stock LightZero training loop.
- It avoids fake pending transitions.
- It can be a clean sanity check that the source-state visual env, reward, and
  stock train loop can learn something with real joint physics.

Why it is not the final target:

- One policy controls both players at once.
- It is centralized single-agent control, not two-player competitive self-play.

Plain read: this is a better control than the custom two-seat direct learner if
the question is "can stock LightZero learn from the CurvyTron visual surface?"
It should not be labeled as competitive self-play.

### 4. Custom `two-seat-selfplay`

Status: current operational launcher, but not a trusted learning path.

Shape:

```text
custom collector flattens active [B,P] seats
LightZero MuZeroPolicy.collect_mode.forward chooses actions
env steps once with joint_action[B,P]
custom replay rows are sampled
custom target arrays are built
MuZeroPolicy.learn_mode.forward is called directly
```

Origin problem:

- It imports `_learn_mode_batches` from
  `curvyzero_stacked_debug_visual_survival_profile.py`, a file whose header says
  it is a profile/smoke artifact and "not a trainer and not a learning claim."
- The result payload itself says `called_train_muzero: false`, no LightZero
  collector, and no upstream GameBuffer target builder.

Why it existed:

- It solves the immediate simultaneous-action collection problem: both seats
  can see the same pre-tick state, actions are chosen, then the env advances
  once with the joint action.

Why scaling it was a mistake:

- It replaced the trusted part of LightZero: GameSegment/GameBuffer target
  construction and learner lifecycle.
- Shape-compatible targets can still be semantically wrong.
- The May 12 artifact curves stayed flat.

Plain read: keep this only as a collector prototype or profiling tool until it
feeds native-compatible replay, not as the main trainer.

## What Went Wrong In The Paper Trail

1. We correctly found that CurvyTron is Pong-like at the single-ego boundary:
   visual stack, discrete action, scalar reward, done.

2. We correctly found that stock `train_muzero` works for fixed/frozen opponent
   CurvyTron controls.

3. We tried to get current-policy two-player behavior through a fake-step
   turn-commit wrapper.

4. The turn-commit target audit found that stock replay would store fake
   pending transitions and risk wrong credit assignment.

5. Instead of moving to a stock-compatible physical-tick replay representation,
   the active docs drifted toward the custom two-seat adapter and called it the
   Coach main line.

6. The old warning was already written in the archive:
   "Stop scaling the private two-seat trainer as the main story." That warning
   got buried by later "canonical two-seat" docs.

## Current Blockers To The Correct MuZero Path

The core blocker is not visual input, frame stacking, action count, Modal, or
GPUs.

The blocker is replay semantics for simultaneous actions:

```text
CurvyTron wants: one physical tick -> both player actions -> per-seat rewards
Stock LightZero wants: one env row -> one scalar action -> one scalar reward
```

There are three clean ways forward:

1. **Stock single-ego controls now.**

   Use `source_state_fixed_opponent` or frozen checkpoint opponents through
   stock `train_muzero`. This proves the visual/reward/train loop while staying
   honest that it is not current-policy self-play. It may also be a practical
   near-term curriculum route if the opponent is periodically chosen from
   recent checkpoints and eval uses an opponent panel.

2. **Stock centralized joint-action control now.**

   Use `source_state_joint_action` through stock `train_muzero` as a control:
   one scalar action chooses both players. This is not competitive self-play,
   but it keeps one physical tick per replay transition and may quickly tell us
   whether the stock loop can learn the source-state visual surface.

3. **Native-buffer bridge before more two-seat scaling.**

   Keep the custom simultaneous collector, but convert each seat perspective
   into native-compatible LightZero GameSegments, push them through
   `MuZeroGameBuffer`, and compare native sampled targets against the current
   hand-built targets. This keeps true simultaneous collection while reusing
   LightZero's replay/target machinery.

## Immediate Cleanup Rule

Do not call `--mode two-seat-selfplay` the main learning lane until one of these
is true:

- it calls stock `train_muzero`; or
- it feeds native `GameSegment` / `MuZeroGameBuffer` targets; or
- it has a written repo-owned learner-target contract with passing parity tests
  against a tiny known trajectory.

Until then:

- `source_state_fixed_opponent`: stock-loop control.
- `source_state_joint_action`: stock-loop centralized joint-action control.
- `source_state_turn_commit`: profile/smoke only, not train.
- `two-seat-selfplay`: collector prototype / profiling / experimental, not a
  learning claim.

## Next Concrete Checks

1. Run or inspect a tiny waited `source_state_joint_action` stock
   `train_muzero` smoke with checkpoints and target telemetry.
2. Run or inspect a small `source_state_fixed_opponent` or frozen-opponent stock
   `train_muzero` curve with survival-first eval.
3. Fix or remove misleading docs that present `two-seat-selfplay` as the trusted
   main training path.
4. Start the GameSegment/GameBuffer bridge spike if we still need true
   two-seat current-policy self-play.
