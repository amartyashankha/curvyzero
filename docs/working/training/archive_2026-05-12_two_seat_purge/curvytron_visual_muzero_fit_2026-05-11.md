# CurvyTron Visual MuZero Fit - 2026-05-11

Purpose: answer what CurvyTron must look like for a stock-ish visual
MuZero/LightZero path to train it, without treating simultaneous action as an
automatic blocker.

No long jobs were run. This is code/docs inspection only.

## Short Answer

CurvyTron should look like Atari Pong at the LightZero boundary whenever we can:

```text
visual frame stack -> one discrete action -> scalar reward -> done
```

That path already exists for single-ego CurvyTron:

```text
LightZero policy chooses player_0 action.
CurvyTron env steps the full game.
The env fills player_1 with fixed or frozen opponent policy.
LightZero owns collector, replay, MCTS, learner, checkpoints, and evaluator.
```

True current-policy two-seat self-play needs one extra custom layer:

```text
collector asks the same current policy for every live seat
-> folds policy rows into joint_action[B,P]
-> env.step(joint_action)
-> emits native-ish game segments/replay metadata
```

Do not make the env responsible for current-policy orchestration. The env should
own CurvyTron rules. The collector should own policy calls.

## Comparison

### Stock Visual Atari Pong Path

Representative evidence:

- `docs/experiments/2026-05-09-modal-lightzero-pong-8192-sim25.md`
- `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`

Shape:

- registered env type: `atari_lightzero`
- env id: `PongNoFrameskip-v4`
- observation: visual conv input `[4,64,64]`
- action: one scalar ALE action from size `6`
- reward/done: stock Atari
- trainer: `lzero.entry.train_muzero`

LightZero owns the useful machinery: env manager, collector, MCTS, replay
buffer, learner, checkpoint cadence, and evaluator. Pong is useful here because
it proves the stock visual MuZero pipe shape, even when a particular checkpoint
curve is not good.

### Current Custom CurvyTron Two-Seat Path

Representative code/docs:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`
- `src/curvyzero/training/policy_row_mapping.py`
- `docs/working/training/custom_vs_native_lightzero_diff_2026-05-10.md`
- `docs/working/training/curvytron_no_learning_investigation_2026-05-11.md`

Shape:

- observation: player-perspective `float32[B,P,4,64,64]`
- policy rows: active live seats are flattened to `[R,4,64,64]`
- action: one shared policy chooses each active seat action
- env step: selected rows are folded into `joint_action[B,P]`
- learner: local code calls `MuZeroPolicy.learn_mode.forward`
- replay/checkpoints: local bespoke rows, targets, sampling, and checkpointing

This path was useful. It proved simultaneous-action plumbing and showed that one
live policy object can choose both seats before one CurvyTron step.

It should now shrink. It duplicates too much native LightZero trainer behavior,
and recent mechanically clean runs were flat. Keep it as a diagnostic harness
for joint-action mapping, player-perspective observation bugs, and replay
metadata probes; do not keep growing it as the main trainer.

### Native CurvyTron Single-Ego / Frozen-Opponent Path

Representative code/docs:

- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_env.py`
- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `docs/working/training/curvytron_native_train_muzero_probe_2026-05-10.md`
- `docs/working/training/curvytron_native_frozen_opponent_probe_2026-05-10.md`

Shape:

- env type: `curvyzero_stacked_debug_visual_survival_lightzero`
- observation: wrapper-owned visual stack `[4,64,64]`
- action: one scalar CurvyTron action from `{left, straight, right}`
- reward: named survival-time reward
- opponent: env-owned fixed straight or frozen LightZero checkpoint provider
- trainer: `lzero.entry.train_muzero`

This is the smallest viable training architecture today. It is stock-ish visual
MuZero because LightZero owns the training spine and CurvyTron is presented as a
single-ego visual MDP row.

It is not current-policy self-play, and the code labels that honestly:
`current_policy_self_play=false`.

### Clean Joint-Action / Current-Policy Collector

This is the missing durable architecture for true simultaneous self-play.

Minimum contract:

- collect observations as `obs[B,P,C,H,W]` with player perspective baked in;
- build active policy rows with `env_row_id`, `player_id`, `legal_action_mask`,
  `episode_id`, and `decision_index`;
- batch those rows through one current policy object;
- map selected row actions back to `joint_action[B,P]`;
- call the public multiplayer env exactly once per simultaneous decision;
- store per-seat rows or game-segment data with enough metadata to recover
  reward, done, next observation, action weights, root value, and seat identity;
- hand storage, sampling, priority, learner scheduling, and checkpoint cadence
  back to LightZero-like machinery wherever possible.

Design choice to make explicit: this can start as independent per-seat MCTS with
the opponent action treated as another current-policy sample. It does not need
full joint-action tree search on day one. For two players, full joint action is
only `3^2=9`, but using independent row searches first is a reasonable,
documented approximation if evaluation is honest about it.

## Smallest Viable Architecture

1. Main lane: run native single-ego visual survival training with fixed and
   frozen opponents through `train_muzero`.
2. Evaluation lane: use strict checkpoint loads, seed panels, seat swaps,
   survival distributions, action histograms, terminal reasons, and explicit
   opponent labels.
3. Diagnostic lane: keep the two-seat smoke for narrow tests only.
4. Self-play lane: build one small joint-action collector. It should replace
   only the stock collector action boundary, not replay, learner, checkpointing,
   or evaluator wholesale.

In simple words: make CurvyTron look like Pong for the boring parts, and add a
custom collector only where Pong is not enough.

## Keep, Shrink, Delete

Necessary custom pieces:

- registered CurvyTron visual env wrappers and schema metadata;
- player-perspective visual rendering and frame stacking;
- fixed/frozen opponent policy providers for native single-ego training;
- `policy_row_mapping.py`, because it is the clean bridge from `[B,P,...]` to
  policy rows and back to `joint_action[B,P]`;
- explicit telemetry for opponent kind, reward schema, observation schema,
  seed/episode ids, terminal reasons, and checkpoint refs;
- a future joint-action collector with current-policy row batching.

Shrink:

- `curvytron_two_seat_lightzero_train_smoke.py`: keep as a bounded diagnostic
  and contract test, not a feature-growing trainer.
- local target/replay/checkpoint logic: preserve only enough to test metadata
  and demonstrate LightZero batch compatibility.
- action-noise and shaping variants: keep as labeled experiments after the clean
  path learns, not as default architecture.

Delete or avoid extending:

- private trainer features that duplicate LightZero collector/replay/learner
  lifecycle;
- env-local hacks that try to query the live current policy from inside
  `env.step`;
- claims that fixed or frozen opponents are current-policy self-play;
- comparisons that mix different `decision_ms`, reward schemas, or opponent
  relations as if they were one curve;
- more scale on the bespoke two-seat trainer unless it is testing one named
  collector/replay hypothesis.

## Decision

CurvyTron does not need to become an exotic custom RL stack to fit visual
MuZero. The smallest viable path is:

```text
native LightZero visual single-ego first
-> frozen-opponent ladder
-> narrow joint-action current-policy collector
```

Simultaneous action is a collector design problem. It is not a reason to throw
away the stock visual MuZero path.
