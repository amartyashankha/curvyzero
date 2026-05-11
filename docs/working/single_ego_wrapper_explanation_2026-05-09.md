# Single-Ego Wrapper Explanation - 2026-05-09

Ownership: this note only.

## Short Answer

A simultaneous multiplayer game can be presented as a single-agent environment
from one player's point of view.

That is not cheating by itself. The learner chooses one player's action. The
wrapper chooses the other players' actions using named opponent policies. Then
the wrapper applies the resulting control snapshot while the real game advances.

So "simultaneous" is not automatically fatal. It means the wrapper decision
needs a complete per-live-player control choice before advancing the source
frames. The learner does not have to personally choose every part of that
wrapper joint action, as long as the missing choices are supplied in a clear,
reproducible, and well-recorded way.

The hard part is not the idea. The hard part is the bookkeeping. If the wrapper
hides who controlled the opponents, hides their actions, loses reset seeds,
auto-resets before saving the last state, or calls something "self-play" when it
is really "learner versus frozen opponent," then the training data becomes hard
to trust.

LightZero can host this shape for one live learner versus a scripted or frozen
opponent. It is still awkward for true simultaneous self-play because its normal
collection loop wants one action per environment row, while real simultaneous
self-play wants one action per live player before the game advances.

## The Simple Mental Model

Imagine a two-player Pong-like tick.

The real game wants:

```text
left paddle action + right paddle action -> one physics update
```

A single-ego wrapper gives LightZero only one side:

```text
LightZero chooses left paddle action
wrapper chooses right paddle action
real game steps once with both actions
wrapper returns left player's observation and reward
```

From the left player's view, this is a normal learning problem. The world reacts
to the left player's action. Part of that reaction comes from physics. Part of
it comes from the opponent. That is allowed.

If the opponent is random, the environment is random. If the opponent is a
script, the environment includes that script. If the opponent is a frozen
checkpoint, the environment includes that frozen checkpoint. All three are
valid as long as the run says exactly which one it used.

## Why Simultaneous Is Not Fatal

For CurvyZero wrappers, "simultaneous action" means the wrapper collects one
control choice per live player at the same decision boundary, then maps those
choices to held source controls over the elapsed-ms frame window. It does not
mean the CurvyTron source itself exposes discrete simultaneous actions.

The single-ego wrapper remains valid when:

1. The wrapper applies a complete live-player control snapshot before advancing.
2. The ego action is the action the learner actually chose.
3. Every non-ego action comes from a named opponent policy.
4. Randomness is seeded or recorded well enough to reproduce or audit the run.
5. Replay records the full action set, not just the ego action.
6. Evaluation says which seat, opponent, checkpoint, and seed set were used.

In other words, the simultaneous game is not being denied. It is being viewed
from one seat.

The bad version would be pretending the game is turn-based when it is not:

```text
left acts -> fake half-step -> right acts -> real step
```

That can distort rewards, observations, timing, and the model the learner is
trying to learn. The cleaner wrapper does this instead:

```text
left action is provided by learner
right action is provided by opponent policy
both actions are applied to the same real tick
```

That is why simultaneous play is compatible with a single-ego wrapper, but not
with sloppy hidden half-turns.

## What Makes The Wrapper Valid

The wrapper is valid if it behaves like a real environment from the ego player's
perspective.

The ego player should see only the observation it is supposed to see. It should
choose one legal ego action. The wrapper should fill in the other actions using
documented opponent policies. The game should advance according to the real
rules. The reward should be the ego player's reward. The terminal state should
be the true terminal state, not a state after an accidental reset.

This is the same idea as training against a built-in computer opponent. The
opponent is part of the environment for that run.

It becomes invalid or misleading when the opponent changes without being
recorded, when opponent randomness cannot be traced, when the wrapper gives the
ego player hidden information it would not have in the real game, or when replay
cannot prove what actually happened.

## Opponent Policy

The opponent policy must be explicit.

Good labels are things like:

```text
random_uniform
track_ball
lagged_track_ball_1
frozen_checkpoint:iteration_000123
current_policy:self_play_generation_7
```

The label matters because "the same ego actions" can mean very different things
against different opponents. A checkpoint that beats random actions may still
lose badly to a simple tracking bot. A checkpoint that beats one frozen parent
may fail against a different seat or seed set.

If the opponent is random, the random stream is part of the environment. If the
opponent is a checkpoint, the exact checkpoint file, model key, adapter mode,
and inference settings are part of the environment. If those are missing, the
run is not really reproducible.

## Seed And Reset

Reset needs to say exactly what episode was created.

At minimum, keep:

```text
episode id
reset seed
reset source
rules/schema version
ego seat
opponent policy id
opponent checkpoint id, if any
```

This matters more in multiplayer than it first appears. The starting state, the
opponent's random choices, spawn order, and any hidden game randomization can all
change the meaning of a result.

A reset should not silently erase the terminal evidence from the previous
episode. First save the final observation, final reward, terminal reason, and
summary metadata. Then reset.

## Hidden Joint Actions

The learner may only choose one action, but the wrapper transition used a joint
action/control snapshot: one choice per live player.

That joint action can be hidden from the learner's observation and still be
recorded in replay. This distinction is important:

```text
hidden from learner: okay
missing from replay: not okay
```

Replay should preserve the full wrapper action/control set for the transition.
Otherwise we cannot later answer simple questions such as:

```text
Did the opponent cause the collision?
Was the ego action legal?
Did both players choose the same turn?
Did the checkpoint collapse because the opponent policy changed?
```

The ego view can stay clean while the audit trail remains complete.

## Replay

Replay should usually store one row per ego perspective, but each row should
point back to the real shared transition.

For a two-player tick, that can mean two ego rows:

```text
row A: player 0's observation, action, reward, value target, metadata
row B: player 1's observation, action, reward, value target, metadata
```

Both rows came from the same real wrapper transition. Both should reference the
same episode id, step id, reset seed, rules version, and joint action/control
snapshot.

For a single-ego LightZero wrapper, there may be only one training row for the
learner-controlled seat. That is fine for learner-versus-opponent training. But
it should not be mistaken for full live self-play replay unless the other live
seat also produced a policy row and training row.

## Final Observations

Final observations are easy to lose.

Many environment stacks auto-reset after an episode ends. That is convenient for
throughput, but dangerous for evidence. If the wrapper returns the first
observation of the next episode where the final observation should have been,
then replay and evaluation are polluted.

The rule should be simple:

```text
when done, save the last real observation before reset
```

Also save whether the episode ended naturally, timed out, was truncated, or was
stopped by some wrapper condition. Those cases can have different training
meaning.

## Self-Play

"Self-play" needs careful labels.

There are several different things people call self-play:

```text
current learner rolls out against a scripted opponent
current learner rolls out against a frozen older checkpoint
same current policy controls both seats live
league of checkpoints controls different seats across games
```

Only the third and fourth are really multiplayer self-play in the strongest
sense. The first is ordinary learner-versus-scripted training. The second is a
useful bridge, but it is frozen-checkpoint opponent training.

The single-ego wrapper is excellent for the first two. It can be part of the
third and fourth only if something outside the wrapper asks the live policy for
each live player's wrapper action before the shared transition advances.

## Checkpoint Assignment

Every run needs to say which checkpoint controlled which seat.

For training:

```text
ego seat: player_0
learner checkpoint at start: ckpt_10
opponent seat: player_1
opponent policy: frozen checkpoint ckpt_06
```

For evaluation:

```text
candidate checkpoint: ckpt_10
opponent: track_ball
seat pairing: candidate as player_0 and player_1
seed set: held_out_eval_v3
```

Without this, results can look better or worse for accidental reasons. A model
may be strong from one seat and weak from the other. A child checkpoint may beat
its parent only because it got easier seeds. A run may accidentally compare a
policy-head opponent against an MCTS opponent and treat them as the same thing.

Checkpoint assignment is not clerical. It is part of the experiment.

## Evaluation

Evaluation should be separate from training.

A useful evaluation report says:

```text
candidate checkpoint
opponent policy or opponent checkpoint
seat assignment
seed set
number of games
win/loss/return/survival metrics
terminal reasons
action histograms
feature/schema versions
```

For multiplayer games, seat swapping is important. A policy that only works as
player 0 is not generally solved.

Evaluation should include more than one opponent type. Beating random is a
floor, not a proof of strength. A small ladder might include random, simple
scripted opponents, frozen older checkpoints, and the current best checkpoint.

## Why LightZero Is Still Awkward

LightZero is comfortable when an environment row needs one action:

```text
observation -> LightZero action -> env.step(action)
```

That matches CartPole, Atari, and a single-ego Pong wrapper.

True simultaneous self-play wants a different shape:

```text
player 0 observation -> policy action
player 1 observation -> policy action
combine both actions
env.step(wrapper joint action/control snapshot)
store player 0 ego row
store player 1 ego row
```

That means the collector, not just the environment, needs to understand that
one real wrapper transition produced multiple player decisions.

LightZero board-game self-play is not a direct answer because board games often
use alternating turns:

```text
player 0 acts, then player 1 acts, then player 0 acts...
```

Pong and CurvyTron-style games are different:

```text
player 0 and player 1 wrapper controls are chosen, then the world advances
```

A LightZero env can hide the opponent action inside a single-ego wrapper, which
is practical for learner-versus-scripted or learner-versus-frozen-checkpoint
training. But hiding a live current-policy opponent inside the env is awkward:
the env would need to call back into the learner policy, manage model state,
handle exploration settings, and decide checkpoint timing. That mixes roles
that should stay separate.

The cleaner future shape is a joint-action collector:

```text
build one policy row per live player
ask the policy for all needed actions
map those actions back to the real players
step the wrapper transition once
write replay rows with full metadata
```

That is more work than a single-ego wrapper, but it matches the auditable game
view needed for training.

## Bottom Line

The user's confusion is reasonable: if the game is simultaneous, it sounds like
a single-action API must be wrong.

The answer is: a single-action API is okay when it is explicitly "one player
acting inside an environment that supplies the other players." The underlying
wrapper step still uses a full joint action/control snapshot. The opponent
policies are part of the environment for that run.

What would be wrong is pretending that this automatically gives full
multiplayer self-play, or failing to record the hidden parts of the transition.

So the rule is:

```text
single-ego wrapper: valid for one perspective versus named opponents
true live all-player self-play: needs explicit multi-player action collection
LightZero: usable for the first, awkward for the second
```
