# Pong vs CurvyTron LightZero Equivalence Audit - 2026-05-10

## Verdict

CurvyTron is not fundamentally different from Pong at the LightZero boundary.
For one controlled player it can be:

```text
visual frame stack -> discrete action -> survival/win reward -> done
```

That is the same broad shape as the Pong/Atari lane that showed signal. The
main gap is two-seat current-policy self-play:

```text
stock LightZero: policy -> one action -> env.step(action)
two-seat CurvyTron: policy -> all seat actions -> env.step(joint_action)
```

So the blocker is not "CurvyTron is unlike Pong." It is that stock
`train_muzero` does not expose the joint-action collector boundary we need.

## New Result

The custom accumulated-replay two-seat run was mechanically valid but flat.

- run: `curvytron-two-seat-accum-b4-s21-32x16-u4-sim8-20260510a`
- setup: accumulated replay, learner sample `128`, `B=4`, `32x16` collection,
  `4` updates per iteration, `num_simulations=8`
- training: `ok=true`, `4096` replay rows, `iteration_0..32`, optimizer step
  allowed, model parameters changed
- eval: strict 64-seed sim8 eval, no failures
- curve: `191.688` mean steps at `iteration_0`, `201.844` at `iteration_1`,
  then exactly `201.844` through `iteration_32`

Read: accumulated replay made the custom scaffold less obviously underpowered,
but it still did not learn. The likely issue is missing LightZero machinery or
the wrong two-seat approximation, not CurvyTron being fundamentally non-Pong.

Sources: inspected the Pong exact/native lane, Pong survival wrapper, CurvyTron
visual wrappers, native-ish trainer, two-seat scaffold, and public multiplayer
env/replay files under `src/curvyzero`.

## Contract Differences

### Observation / Frame Stack

Pong exact lane uses ALE-backed `PongNoFrameskip-v4`, conv model,
`observation_shape=[4,64,64]`, `frame_stack_num=4`.

CurvyTron stacked visual survival also exposes `float32[4,64,64]`, but the stack
is wrapper-owned and pixels are debug/source-state occupancy. The two-seat
scaffold uses `float32[B,P,4,64,64]` and flattens active seats into policy rows.

Classification: not fundamental. Single-ego is already Pong-like; the player
axis is adapter/collector work.

### Action API

Pong chooses one action per env row from Atari action space size `6`.
CurvyTron single-ego chooses one action per env row from size `3`
(`left`, `straight`, `right`). CurvyTron two-seat needs `joint_action[B,P]`.

Classification: `3` vs `6` is adapter detail. Joint action is the real API gap.

### Opponent / Current Policy

Pong opponent dynamics live inside ALE. CurvyTron single-ego wrappers choose the
opponent inside `env.step`: fixed straight or frozen checkpoint. They correctly
mark `current_policy_self_play=false` because the live collector policy is
outside the env.

The two-seat scaffold proves one shared policy object can choose both seats,
but it bypasses native `train_muzero`.

Classification: fixed/frozen opponents are native-compatible. Live
current-policy self-play needs a collector, not an env-local opponent hack.

### Reward

Pong exact uses stock Atari reward. Pong survival-shaped adds an opt-in per-step
survival bonus. CurvyTron base visual uses terminal `+1/-1/0`; stacked survival
uses post-transition alive `1.0`, dead `0.0`, no terminal bonus.

Classification: not fundamental. Reward is a named wrapper contract; do not mix
stock return, survival return, and win/loss claims.

### Reset / Done

Pong inherits Atari/LightZero reset, done, collector, and evaluator behavior.
CurvyTron wrappers use `done = terminated | truncated`, explicit reset after
done, source max-step truncation, and `final_observation`/`final_reward_map`.
Public multiplayer rows that finish must reset/autoreset before another step.

Classification: adapter work, not fundamental.

### Replay / Collector

Pong exact calls `lzero.entry.train_muzero`, so LightZero owns collector,
GameBuffer, learner, checkpointing, and evaluator.

CurvyTron stacked visual survival trainer also calls `train_muzero`, so native
single-ego fixed/frozen-opponent training is available.

The two-seat scaffold owns collection, replay rows, sampling, targets, and
checkpointing. Accumulated replay ran and stayed flat. `multiplayer_replay_v0`
is metadata-only and does not claim trainer replay readiness.

Classification: this is the main issue. The custom scaffold is probably missing
native LightZero machinery or using an inadequate approximation.

## Blockers

- Not blockers: visual input, discrete actions, survival/win signal, done/reset,
  conv frame stack, native single-ego training.
- Adapter details: action count, wrapper-owned stack, debug visual fidelity,
  reward labels, final-observation wiring.
- Real architecture gap: stock collector does not ask the current policy for
  all simultaneous actions before `env.step(joint_action)`.
- False lead now tested: accumulated replay in the private two-seat trainer.
  It ran, changed weights, and stayed flat.

## Shortest Native-Like Path

Use native LightZero as the spine.

1. Scale registered single-ego CurvyTron stacked visual survival with fixed and
   frozen-checkpoint opponents.
2. Keep Pong-style eval discipline: strict load, no fallback, survival steps,
   returns, action histograms, terminal reasons, seed panels.
3. Treat fixed/frozen improvement as single-ego evidence, not current-policy
   self-play evidence.
4. Stop scaling the private two-seat trainer as the main story.
5. Add only one custom piece: a narrow joint-action collector that asks the
   shared current policy for every active seat, emits `joint_action[B,P]`, and
   hands native-ish segments/replay back to LightZero machinery.

Shortest answer: make CurvyTron look like Pong wherever possible, and customize
only the simultaneous joint-action collector.
