# Opponent Diagnostic Design

Purpose: track the opponent designs we may use for survival-first CurvyTron
training. This is design work, not a launch record.

## New Axis: Repeated Copies

Some rows should be repeated several times with different seeds. This is not
noise; it is a real axis.

Use repeated copies when:

- the opponent policy is randomly initialized;
- the environment reset has meaningful randomness;
- the opponent is stochastic or creates different trail layouts;
- the row is important enough that one lucky or unlucky seed would mislead us.

Default rule for important stochastic rows: run about five copies with different
seeds. That rule applies especially to random-policy opponents and stochastic
opponent behavior rows.

Use fewer copies, usually one or two, for frozen ancestor checkpoint rows if
they are mainly controls, because the policy itself is less random after
training. Increase that only if an ancestor checkpoint becomes the main claim.

## Randomness To Track Separately

Do not collapse these into one vague "seed":

| Seed | Meaning |
| --- | --- |
| training seed | learner initialization and LightZero-level randomness |
| reset seed | starting positions, headings, bonuses, and environment randomness |
| opponent policy seed | random initial opponent weights or sampled frozen policy |
| opponent behavior seed | stochastic actions/noise inside the opponent policy |
| eval seed | held-out observation/eval randomness |

Run names and manifests should expose the important ones. A copy label such as
`copy_id` should identify repeated copies without pretending all seed meanings
are interchangeable.

Recommended manifest fields:

| Field | Vary when |
| --- | --- |
| `training_seed` | checking learner/init stability |
| `reset_seed` | checking start-state, heading, bonus, or environment-reset stability |
| `opponent_policy_seed` | repeating randomly initialized opponent policies |
| `opponent_behavior_seed` | repeating stochastic/noisy opponent behavior |
| `eval_seed` | creating held-out eval comparisons |
| `copy_id` | labeling repeated copies of the same logical row |

## Opponent Families

| Opponent family | What it means | Why it matters |
| --- | --- | --- |
| Blank canvas | Opponent is effectively absent: no trail, no death pressure, no useful outcome. | Simplest wall-avoidance sanity check. |
| Immortal fixed-straight | Opponent follows the fixed-straight rule but cannot die. | Tests whether an invincible trail-maker creates useful survival pressure. |
| Immortal random policy | Opponent is a randomly initialized frozen policy and cannot die. | Creates varied trails; should get repeated copies. |
| Immortal ancestor checkpoint | Opponent is an old/mid/recent frozen checkpoint and cannot die. | Tests curriculum-like pressure without easy opponent death. |
| Scripted wall-avoidant | Opponent is a simple hand-coded policy that turns away from walls. | Stronger baseline opponent, closer to a useful Pong-style baseline. |

Naming rule: use **blank canvas** for the clean no-op/no-trail/no-opponent
anchor; use **passive immortal** for the current `opponent_death_mode=immortal`
implementation; use **wall-avoidant** only for legal-action steering policies.
Do not let "immortal" imply safe, wall-avoidant, reflecting, or no-trail.

## Current Wiring Facts

The stock source-state LightZero env currently exposes two opponent kinds:
`fixed_straight` and `frozen_lightzero_checkpoint`.

A "random learned frozen opponent" is not a first-class setting yet. It should
mean one of:

- generate and store a random/iteration-0 LightZero checkpoint, then use the
  existing `frozen_lightzero_checkpoint` path;
- add a small explicit random-policy opponent kind to this source-state env.

The repo has a `SeededRandomOpponentPolicy` helper, but it is not currently
wired into this stock source-state trainer. Do not confuse this with reset
randomness.

Training config already uses dynamic reset seeds, so each reset derives a fresh
start seed from the run seed and reset index. Repeat-copy rows should still use
different copy/run seeds so learner initialization, reset sequence, and opponent
policy randomness can vary deliberately.

## Current Immortal Behavior

`opponent_death_mode=immortal` currently means passive death immunity for
player 1. It does not stop, clamp, reflect, remove trail, or turn away from
danger.

Confirmed 2026-05-13:

- Wall hit: player 1 stays alive and can keep moving out of bounds. No terminal
  reward or death metadata is emitted for the suppressed wall death.
- Own trail hit: collision is detected, but player 1 death is suppressed.
  Player 1 phases through and keeps leaving normal trail/body points.
- Learner trail/body hit by player 1: same suppression; player 1 phases
  through and player 0 is not killed just because player 1 collided.
- Learner hits player 1 trail/body: player 0 is still mortal and can die.

Code path: the source-state env passes `death_immunity_player_ids` for player 1,
the vector env turns that into a mask, and the runtime applies it to wall/body
death. It is death suppression only.

This is not yet clean enough for the main training lane. It is useful evidence,
but a large batch needs either a deliberate passive-immortal canary, a reflecting
opponent, a no-trail blank opponent, or a scripted wall-avoidant opponent.

## Desired Behavior Candidates

There are multiple possible meanings of "immortal opponent." They should be
separated.

### Passive Immortal

The opponent ignores death and keeps moving. This is the minimal current-style
diagnostic, but it may create weird out-of-bounds state if wall collisions are
only suppressed.

### Reflecting Immortal

If the opponent would hit a wall, it reflects like a ray or takes an equivalent
turn away from the wall. This keeps it in the arena and may create useful trail
layouts without requiring learned behavior.

### Trail-Phasing Immortal

If the opponent hits any trail, it passes through. No one dies from that
opponent collision. This makes the opponent a trail generator rather than a
competitive player.

### No-Trail Blank

The opponent exists only as plumbing and does not write collision trails. This
is close to single-player wall avoidance.

Design note: [blank_canvas_noop_opponent_lane.md](blank_canvas_noop_opponent_lane.md)
recommends a wrapper-owned `opponent_runtime_mode=blank_canvas_noop`, not
passive immortal and not `remove_player`. The key contract is: keep the
two-player LightZero shape, but make player 1 inert, hidden from observation,
unable to write collision/visual trail, unable to catch bonuses, and ignored by
the no-outcome training reward.

## Scripted Wall-Avoidant Policy Sketch

Goal: make a simple opponent that survives and creates meaningful trails.

Policy idea:

- Sense distance and angle to the nearest wall danger zone.
- If safely far from walls, go straight.
- If near a wall, turn left or right according to which turn points the heading
  away from the wall faster.
- Ignore trails if the opponent is immortal/trail-phasing.

This should be tested by running many simulated episodes through the real
environment, not by assuming the geometry is right.

Acceptance check:

- it stays inside the arena for long horizons;
- it does not kill the learner through suppressed death plumbing;
- its actions are not a single constant action;
- it creates diverse enough trail layouts to be useful.

Latest probe result: proactive force field with margin `20` is the preferred
first scripted opponent. It uses legal left/right/straight actions only; it does
not bounce, teleport, clamp position, or flip heading. It stayed in bounds for
`0/384` tested 1024-step starts with normal trail writing and mostly went
straight. Contact-only and pure reflected-heading target policies failed.
Stronger reflection-like target policies can be made better, but they are
turn-heavy and less clean. See
[scripted_wall_avoidant_opponent_baseline_2026-05-13.md](scripted_wall_avoidant_opponent_baseline_2026-05-13.md).

## Matrix Implications

The next tensor should include repeat groups, not just unique rows.

Likely staged blocks:

1. Blank canvas wall-avoidance sanity rows.
2. Immortal fixed-straight rows.
3. Immortal random-policy rows, about five copies per important setting.
4. Immortal ancestor-checkpoint rows, fewer copies unless promising.
5. Scripted wall-avoidant rows once the policy survives reliably.

Render, stochasticity, and reward axes should be layered on top only after the
opponent family is clear enough to interpret. Do not build the next tensor as a
single crossed product of all axes; add blocks only when the previous block's
question has an interpretable answer.
