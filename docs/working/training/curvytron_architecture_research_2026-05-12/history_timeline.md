# History Timeline

Purpose: record how the CurvyTron training lane moved from stock LightZero
controls to the scaled custom two-seat runs, and which claims were stronger than
the evidence.

## Concise Timeline

1. **May 9: single-ego LightZero framing.**

   CurvyTron was framed as Pong-like enough for a stock scalar-action LightZero
   row if the wrapper owned the opponent: one visual observation, one discrete
   ego action, one scalar reward. This was always a control shape, not full
   simultaneous self-play. See `stock_lightzero_dataflow.md` and the later
   reconciliation note.

2. **May 10: stock fixed/frozen opponent paths existed.**

   `source_state_fixed_opponent` and frozen-checkpoint opponent wiring could
   call `lzero.entry.train_muzero`; LightZero owned collector, `GameSegment`,
   `MuZeroGameBuffer`, learner, checkpoints, and in-loop eval. The frozen route
   was mechanically plausible and later got a strict CPU canary:
   `stock-frozen-canary-source-state-s304-20260512`. This proved plumbing, not
   live same-policy self-play.

3. **May 10: the target shifted to live two-seat collection.**

   Fixed/frozen was demoted because it could not let the same current policy pick
   both players' actions from the same pre-tick state. The desired physical
   contract became:

   ```text
   one CurvyTron tick -> player 0 action + player 1 action -> per-seat outcomes
   ```

4. **May 10-11: turn-commit tried to preserve stock LightZero.**

   `source_state_turn_commit` let stock `train_muzero` ask for one scalar action
   at a time: store player 0, then commit player 1 and advance physics. The
   target audit blocked it for training because stock replay would store fake
   pending rows beside physical commit rows, creating reward-credit risk. Current
   code now raises on train mode for this variant.

5. **May 11: centralized joint-action remained a stock control.**

   `source_state_joint_action` maps one scalar action `0..8` to both player
   actions and advances one real tick. It keeps stock replay semantics but is
   centralized single-agent control, not competitive self-play.

6. **May 10-12: custom two-seat became the operational path.**

   `--mode two-seat-selfplay` used one live LightZero `MuZeroPolicy` object to
   choose actions for both seats, built `joint_action[B,P]`, stepped
   `VectorMultiplayerEnv`, wrote local per-seat replay rows, sampled local
   batches, and called `MuZeroPolicy.learn_mode.forward` directly. It did not
   call stock `train_muzero`, did not use LightZero's collector, and did not use
   native `MuZeroGameBuffer` target construction.

7. **May 12: scaled custom runs were flat.**

   Overnight and mixpast rows launched through the custom adapter. Survival
   stayed near early-life/random baseline, and pure current-policy rows often had
   little or no sparse terminal signal. This shows the custom training contract
   was unsafe to scale; it does not show CurvyTron is unlearnable.

## What Changed Too Much

The intended change was simultaneous action collection. The scaled path also
changed the training contract:

- collection moved from LightZero collector to repo-owned two-seat collection;
- replay moved from native `GameSegment` / `MuZeroGameBuffer` to local rows;
- targets moved from native buffer sampling to hand-built arrays;
- learner lifecycle moved from `train_muzero` to direct `learn_mode.forward`;
- checkpoint and `ckpt_best` meaning diverged from stock LightZero assumptions;
- eval/inspection mixed fixed-opponent readers with two-seat checkpoints;
- `to_play` semantics became suspect because custom rows used player ids `0/1`
  where non-board-game CurvyTron/Pong-like rows were expected to use `-1`.

## Overclaims

| Claim | What the repo supports instead |
| --- | --- |
| `two-seat-selfplay` is canonical/main CurvyTron training. | It is an experimental custom adapter until it calls stock `train_muzero`, feeds native replay/targets, or passes target parity tests. |
| "LightZero trained it" because `MuZeroPolicy` and `learn_mode.forward` ran. | LightZero components ran, but the stock LightZero learning loop did not. |
| Fixed/frozen opponent runs are irrelevant. | They are not live self-play, but they are the cleanest stock-loop controls and possible recent-opponent curriculum route. |
| Turn-commit preserves stock LightZero and is trainable. | It preserves calls into stock plumbing, but replay contains fake pending rows, so training is blocked. |
| Centralized joint-action is self-play. | It is a stock-loop control over real joint physics, with one policy controlling both players centrally. |
| Weights changed / checkpoints exist means learning. | A broken target builder can update weights; learning needs strict-load checkpoint curves with survival, outcome, reward components, action histograms, and opponent source. |
| Flat May 12 curves prove CurvyTron cannot be learned. | They prove the scaled custom contract did not produce learning evidence. Stock-loop and native-buffer gates remain open. |
