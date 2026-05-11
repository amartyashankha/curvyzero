# CurvyTron LightZero Native Reuse Critique - 2026-05-10

No pytest was run.

## 2026-05-11 Correction

The earlier correction was still too strong. We do not need to own a full
trainer just because CurvyTron actions are simultaneous, but the turn-commit
wrapper is not yet proven learning-quality current-policy self-play.

The useful stock-plumbing trick is a small turn-commit env wrapper:

```text
LightZero step 1: record player 0 action; no physics advance; reward = 0
LightZero step 2: record player 1 action; commit joint_action[2]; reward = survival
real source tick advances once
```

From LightZero's point of view this is a scalar-action environment, so stock
`train_muzero`, GameSegment, GameBuffer, learner, checkpointing, and evaluator
can stay in charge. The repo wires the source-state version as:

```text
env_variant=source_state_turn_commit
```

The concern is reward credit. Stock LightZero stores both scalar steps as normal
GameSegment transitions. Because the pending/player0 step has no physics advance
but is followed by the commit/player1 survival reward, value targets can credit
player0 states for player1 survival. That is good enough for a native smoke or
control path, not for a learning-quality self-play claim until fixed or directly
ruled out.

Code-audit note: safe env-wrapper plumbing fixes now cover source-state knobs,
metadata, telemetry, stack schema hash, and making render non-mutating with
respect to the observation stack.

Smoke evidence after cleanup:

```text
run:     curvytron-source-state-turncommit-smoke-s20260511b
attempt: profile-smoke-sim2-c2-steps64-20260511b
result:  stock train_muzero called, MCTS ran, replay sampled, learner stepped,
         iteration_0 copied, telemetry rows written
rows:    36 scalar rows = 18 pending rows + 18 physical commit rows
claim:   plumbing only; reward credit still untrusted
```

The custom two-seat trainer should now be treated as diagnostic scaffolding,
not the main lane.

## Short Verdict

The custom CurvyTron two-seat adapter was necessary as a proof tool at the
time. It answered a real question that native `train_muzero` could not answer:
can one live `MuZeroPolicy` choose both CurvyTron seats before one simultaneous
CurvyTron step, then learn from both seats?

But the reason is not that CurvyTron is fundamentally unlike Pong.

For LightZero, CurvyTron can be made to look very Pong/Atari-like:

```text
visual frame stack -> discrete action -> survival/win reward -> done
```

The real problem is our integration shape. Our normal LightZero wrapper exposes
one ego action per env row. CurvyTron source stepping needs a full joint action
for the simultaneous players. Today the wrapper fills the opponent internally.
That is fine for Pong-like single-ego training. It is not live two-seat
current-policy self-play.

But it should not become the main trainer path.

The native LightZero path is now good enough for the boring parts: env manager,
collector, MCTS call path, GameBuffer, learner, checkpoints, and trainer
artifacts. We should move as much as possible back there and keep the two-seat
adapter as a narrow self-play experiment or diagnostic harness.

For `source_state_turn_commit`, "good enough" currently means stock plumbing
smoke/control, not proven current-policy self-play.

## Can CurvyTron Look Like Pong To LightZero?

Yes.

For ordinary native LightZero training, CurvyTron can be wrapped like a
Pong/Atari-style visual env:

- observation: `float32[4,64,64]` or LightZero-stacked visual frames;
- action space: discrete `{left, straight, right}`;
- reward: survival time, terminal win/loss, or a named mix;
- done: death, round end, or timeout;
- info: source metadata, final observation, seed, opponent id.

That is enough for `train_muzero` mechanically. The repo already has a native
debug visual survival path that calls `lzero.entry.train_muzero`.

So we should stop saying, or implying, that LightZero cannot train CurvyTron
because CurvyTron is too different from Pong. It can train a CurvyTron-shaped
single-ego env.

The exact blocker is narrower:

```text
normal train_muzero collector chooses one action for one env row;
our true two-seat self-play needs the current policy to choose both player
actions before one simultaneous source step.
```

That is a collector/action-interface blocker, not a game-genre blocker.

## Why The Custom Adapter Was Necessary Then

LightZero's normal MuZero trainer treats the env row like a single actor:

```text
policy chooses one action -> env.step(action)
```

CurvyTron 1v1 current-policy self-play needs:

```text
same live policy chooses player 0 and player 1 -> env.step(joint_action[B,P])
```

The registered single-ego wrappers cannot do that exact two-seat current-policy
thing. In those wrappers LightZero controls one ego seat, and the wrapper fills
the other seat with a fixed, frozen, or otherwise env-owned opponent. That is a
normal and useful Pong-like setup. It only fails when we want the opponent to be
the same live policy object that the collector is currently using for ego.

The code says this plainly: the env cannot ask the live collector policy for
the opponent action because the live policy and learner weights live outside
`env.step`.

So the custom two-seat adapter was the smallest honest way to test the core
mechanic:

- build `[B,P,4,64,64]` observations;
- map live seats into policy rows;
- call one shared `MuZeroPolicy.eval_mode.forward` for both seats;
- rebuild `joint_action[B,P]`;
- step `VectorMultiplayerEnv`;
- build replay rows for both seats;
- call `MuZeroPolicy.learn_mode.forward` on those rows.

That was useful. It proved a boundary that `train_muzero` could not prove.

## Why It Is Now Too Custom

The adapter now duplicates too much of what LightZero already owns.

It has its own collector loop, replay rows, sample logic, value-target builder,
batch-size patching, optimizer-step switch, checkpoint writer, and iteration
state. That was acceptable while we were discovering the shape of the problem.
It is dangerous as a training lane.

It is easy to get a green run while silently drifting from LightZero's real
trainer behavior. The most suspicious parts are:

- local replay instead of LightZero GameBuffer;
- local target construction instead of LightZero's normal segment targets;
- manual batch-size patching;
- manual policy checkpoint format;
- independent one-row policy calls for each seat;
- no native collector/evaluator lifecycle;
- no actor weight-refresh story;
- no native priority/update-priority path.

This is a lot of private trainer machinery. Private trainer machinery becomes
private trainer bugs.

## What Can Move Back To Native LightZero Now

Move these back or keep them in the native `train_muzero` lane:

- Single-ego debug visual survival training.
- Registered DI-engine env wrappers.
- `BaseEnvTimestep` reset/step contract.
- LightZero collector and evaluator env managers.
- LightZero MCTS/search calls through the policy.
- LightZero GameBuffer storage and sampling.
- LightZero learner train loop.
- LightZero checkpoint cadence and trainer artifact directory.
- Native config compilation from the Atari MuZero template.
- Fixed-opponent and frozen-checkpoint-opponent runs.

The repo already has this path in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
It calls `lzero.entry.train_muzero`, records artifacts, mirrors checkpoints,
and labels itself honestly as debug-fidelity single-ego training.

That is the path to keep scaling first.

## What Still Blocks Direct `train_muzero` For True Two-Seat Self-Play

Direct `train_muzero` is still blocked for true learning-quality current-policy
two-seat CurvyTron.

Main blocker:

```text
train_muzero gives env.step one ego action, not a full simultaneous joint action.
```

More exact:

```text
LightZero Collector owns policy.forward.
CurvyTron env.step owns the simultaneous source transition.
Our current wrapper puts opponent-action choice inside env.step.
Therefore env.step cannot use the same live Collector policy for the opponent.
```

If the opponent is fixed, frozen, checkpoint-lagged, scripted, or sampled by an
env-local provider, the normal path works. If the opponent must be the same
current policy at the same collection moment, the normal path lacks the hook.

More blockers:

- The env does not receive the live collector policy object.
- The env cannot ask the current learner weights for the opponent action.
- A frozen checkpoint opponent is useful, but it is not current-policy
  self-play.
- LightZero's normal replay path is not carrying our two-seat metadata:
  `iteration`, `env_row_id`, `player_id`, `decision_index`, seat-swapped
  observations, and per-seat survival returns.
- GameBuffer target building has not been proven for simultaneous two-seat
  CurvyTron rows.
- Turn-commit target credit is specifically risky: the pending/player0 scalar
  transition can receive value credit from the later commit/player1 survival
  reward even though physics did not advance on player0's step.
- Search semantics are not settled. We currently do independent per-seat
  searches with `to_play=-1`; that is not joint-action MCTS and not a clear
  multiplayer solution concept.
- The debug visual surface is not source-fidelity pixels.
- Full reset/autoreset/final-observation lifecycle is still narrower than full
  CurvyTron.
- 3P/4P makes joint search explode as `3^P`.

So: native `train_muzero` is good for Pong-like CurvyTron training. It is not
yet a drop-in answer for live two-seat current-policy self-play because the
missing piece is a collector that chooses all simultaneous player actions
before calling `env.step(joint_action)`.

## Simplest Native-ish Path

Use native `train_muzero` as the spine.

The simplest path is:

1. Keep the registered single-ego visual survival env.
2. Keep LightZero's collector, GameBuffer, learner, checkpointing, and eval.
3. Use fixed opponents only as controls.
4. Use frozen checkpoint opponents as the first ladder.
5. Rotate ego seat and seed sets in evaluation.
6. Promote checkpoints only by survival distribution, not one lucky seed.
7. Later, add a small upstream-compatible custom collector only for the part
   native LightZero cannot express: choose all live seats with the current
   policy before `env.step(joint_action)`.

That collector should feed native-ish game segments into LightZero replay if at
all possible. It should not grow into another private trainer.

If we want "current enough" before writing a collector, use a checkpoint-lagged
opponent:

```text
learner vs latest published checkpoint from N iterations ago
```

That is not full self-play, but it is closer than fixed straight and stays on
the native trainer path.

## What We Should Stop Doing

Stop treating the two-seat adapter as the future trainer.

Stop arguing from "CurvyTron is not Pong." That is too broad and mostly false
at the LightZero boundary. The better argument is: "our current wrapper is
single-ego, while live self-play needs a joint-action collector."

Stop adding trainer features to it unless they test one specific two-seat
question.

Stop scaling runs that bypass LightZero's collector/replay/learner path and
then comparing them beside native `train_muzero` runs as if they are the same
kind of evidence.

Stop using fixed-opponent improvement as self-play evidence.

Stop saying "current-policy" when the opponent is fixed or frozen.

Stop saying "proven current-policy self-play" for turn-commit until reward
credit is fixed or directly disproven.

Stop building more private replay unless the goal is to learn exactly what
metadata native replay must receive.

Stop hand-writing target logic in the main lane. If native LightZero target
logic is wrong for CurvyTron, prove the mismatch and make the smallest adapter
at the replay/segment boundary.

Stop hiding behind debug visuals. They are useful plumbing, not source-fidelity
CurvyTron vision.

Stop making the env responsible for policy orchestration. The env should own
CurvyTron rules. The trainer or collector should own policy choice.

## Recommendation

Keep two lanes, with hard labels.

Native lane:

```text
single-ego LightZero train_muzero
fixed/frozen/checkpoint-lagged opponents
native collector/replay/learner/checkpoints/eval
```

Two-seat research lane:

```text
custom current-policy joint-action collector
minimal replay bridge
no extra trainer features
used only to design the eventual native-compatible collector
```

The creative move is not to make the custom adapter more powerful. The creative
move is to make it smaller until only the missing LightZero abstraction remains.

## Sharp Findings

1. The custom adapter was necessary to prove live shared-policy two-seat action
   selection before a simultaneous CurvyTron step.
2. It is now carrying too much private trainer code and should be demoted to a
   diagnostic/research harness.
3. CurvyTron can look Pong/Atari-like to LightZero for single-ego visual
   training: frames, discrete actions, reward, done.
4. The exact normal-path blocker is that the LightZero collector chooses one
   ego action, while live two-seat self-play needs one current policy to choose
   a full simultaneous joint action before `env.step`.
5. The simplest next path is native `train_muzero` plus frozen or lagged
   checkpoint opponents, using turn-commit only as stock plumbing/control, while
   designing one small custom collector or replay boundary fix for true
   joint-action current-policy play.
