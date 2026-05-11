# CurvyTron Mechanics Notes For Eval - 2026-05-11

Purpose: explain the game mechanics in plain language so evals measure the right
thing.

## Simple Game Model

CurvyTron is like a continuous two-player snake/light-cycle game.

- Each player moves forward.
- Actions choose turning direction: left, straight, or right.
- Players leave trails.
- A player dies if it hits a wall or an active trail.
- The round ends when one or zero players are alive.

For the first training target, bonuses and advanced source-game details are not
the main point. Survival time is the point.

## Why Short Episodes Can Be Normal

One trainer step is not necessarily one tiny physics frame. In the current
trainer path, one decision can advance hundreds of milliseconds of movement.

At `decision_ms=300`, each action is held for 0.3 seconds. With the current
movement scale, that means each decision moves the player several map units. A
bad policy can hit a wall or loop into its own trail in about 10 decisions.

So "10 steps" is not automatically a bug. It means about 3 seconds of game time
at `decision_ms=300`.

The eval report should always show both:

- decisions survived
- game seconds survived

## What Death Means

The important death categories are:

- wall hit
- own trail hit
- opponent trail hit
- both players died
- timeout or truncation

Current public metadata can show terminal reason and death player in some paths,
but it does not always explain the collision geometry. That is the main
observability gap.

## What The Policy Should Learn First

The first useful skill is not winning. It is avoiding immediate death.

A weak learning signal would look like:

- longer survival over checkpoints
- fewer immediate wall or self-trail deaths
- less one-action collapse
- better performance than straight/random baselines

A stronger signal would look like:

- beating wall-avoid or ray-clearance scripted baselines
- varied action choices
- traces showing the policy steers away from danger

## Comparability Rules

Do not compare survival numbers unless these match or are shown clearly:

- `decision_ms`
- map size
- max ticks
- reward settings
- eval policy/opponent setup
- checkpoint id
- number of eval episodes
- random seed set generation method

This matters because the same behavior can produce very different decision
counts if the timestep changes.

## Inspector Priority

The inspector should start broad:

1. read existing run/eval artifacts
2. produce survival curves
3. flag action collapse
4. show death reason summaries
5. save a few short episode traces

After that, add richer geometry only where it explains failures.
