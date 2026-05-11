# Coach strategy red-team - 2026-05-10

No pytest was run.

## Harder verdict

The custom two-seat adapter story is mostly self-inflicted complexity.

CurvyTron is closer to Pong than our plan has treated it: visual input,
discrete turns, two players, and survival reward. Pong worked when we stopped
inventing trainer pieces and let LightZero run its own collector, replay,
learner, checkpoints, and evaluator. CurvyTron should do that first.

The accumulated custom run stayed weak too: `191.688` to `201.844`, then
basically flat through `iteration_32`.

## Recommended next decision

Do not scale the bounded custom trainer unless a specific diagnostic reason is
named before the run. Make CurvyTron use native LightZero `train_muzero` in the
same style as Pong:

- stock Atari MuZero config as template;
- patched CurvyTron env type;
- conv model;
- observation `[4,64,64]`;
- action space `3`;
- `to_play=-1`;
- survival reward;
- LightZero collector, replay, learner, checkpoints, evaluator;
- checkpoint curve versus `iteration_0`.

This is already the shape of
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
That lane should be the main CurvyTron LightZero path.

## Invented complexity

### 1. We made "two players" mean "custom self-play trainer"

Pong also has an opponent. The useful Pong proof did not require live
two-policy self-play. It used a normal single-agent LightZero view of a
two-player game.

CurvyTron can start the same way:

- LightZero controls `player_0`;
- the env controls `player_1`;
- the env logs the full joint action;
- eval reports survival curves.

Live current-policy self-play is a later feature, not the first proof.

### 2. We turned a wrapper into a trainer

The bounded adapter now owns collection, replay rows, sampling, value targets,
learner calls, checkpoint writes, and summary semantics.

Those are trainer jobs. LightZero already has them. Rebuilding them around
`MuZeroPolicy.learn_mode.forward` makes us debug our imitation of LightZero
instead of testing CurvyTron.

### 3. We promoted mechanics to learning evidence

Strict checkpoint loads, model hash changes, metadata targets, and accumulated
replay are useful. They are not survival improvement.

The flat target-fixed two-seat curve is the important read:

```text
iteration_0  mean steps 196.156
iteration_16 mean steps 197.453
```

That does not earn more adapter complexity. The accumulated replay result makes
the same point.

### 4. We over-weighted self-play purity

The first question is not "do we have full simultaneous self-play?"

The first question is:

```text
Can a visual MuZero policy learn to keep one CurvyTron player alive longer?
```

If no, live two-seat self-play will not save us. If yes, then frozen opponents
and later live self-play have a foundation.

### 5. We split into too many lanes

Current lanes include native LightZero, bounded adapter, frozen opponents,
repo-native PPO, Pong controls, scalar/ray, and visual debug. The next move
should reduce lanes, not add one.

## Mainline now

Use the native CurvyTron visual survival LightZero trainer.

- debug visual frame stack `[4,64,64]`;
- controlled player is unambiguous;
- actions are left, straight, right;
- reward is `1.0` while controlled player is alive after the step, else `0.0`;
- done when controlled player dies or max cap is reached;
- opponent is fixed, random, scripted, or frozen checkpoint;
- LightZero sees a normal single-agent non-board-game env.

Do not require custom two-seat collection, custom replay, custom targets,
custom learner calls, repo-native PPO, source-perfect rendering, or live
current-policy self-play before this gate.

## Shortest run plan

1. Pick one native LightZero CurvyTron config.
2. Keep it close to the Pong recipe.
3. Use `train_muzero`, not direct `learn_mode.forward`.
4. Use the existing stacked debug visual survival env.
5. Start with fixed or random opponent.
6. Run enough steps to create a real checkpoint curve.
7. Eval `iteration_0`, early, middle, latest, and best on one random seed panel.
8. Report survival distribution, capped count, actions, terminal reason, and baselines.
9. Promote only if survival improves over `iteration_0`.

## Stop doing

- Stop adding or scaling bounded two-seat adapter work.
- Stop using accumulated replay as the main story.
- Stop treating current-policy self-play as the blocker for first learning.
- Stop requiring repo-native PPO before the simple native LightZero attempt.
- Stop chasing source fidelity before debug visual survival learns.
- Stop reading tiny fixed-opponent runs as evidence against the whole path.

## Keep doing

- Keep Pong as the external proof that this LightZero stack can learn.
- Keep native LightZero CurvyTron visual survival as the main MuZero path.
- Keep frozen-checkpoint opponents as a later curriculum bridge.
- Keep repo-native PPO as backup and diagnostic.
- Keep the custom two-seat adapter only as a sketch or targeted diagnostic.

## Decision rule

Run the simple native LightZero CurvyTron lane first.

If survival improves, scale that lane, add opponent variety, then try frozen
checkpoint refresh.

If survival is flat, check budget, action collapse, reward/done semantics,
eval caps, and opponent choice before blaming LightZero.

If survival stays flat after a fair native run, move to repo-native PPO/IPPO to
test basic learnability. Leave the custom two-seat trainer frozen unless a
specific bug probe needs it.

## Biggest risk

The biggest risk is continuing to debug a custom trainer we did not need. Make
CurvyTron boring to LightZero: one visual ego policy, three actions, survival
reward, native `train_muzero`, checkpoint curve. Then believe the curve.
