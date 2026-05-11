# Fidelity Measurement Plan

Status: draft

This plan says what we measure when comparing the Python environment to the
CurvyTron reference. Each run should say what passed, what failed first, and
whether the evidence is state, event, outcome, observation, image, or human review.

## Stages

| Stage | What we measure | Match rule |
| --- | --- | --- |
| Setup | Scenario id, ruleset id, config hash, seed, player count, player order, map size, spawn state, action script, time-step policy. | Exact. |
| State trace | Per-tick player state, global state, trail state, bonuses, scores, phase, terminal flag. | Exact for ids, flags, counts, phases, and scores. Tolerance for floats. |
| Events | Inputs applied, movement, trail print, collision, death, score, bonus, round end, server-visible events. | Exact event names, order, ids, causes, and discrete payloads. Tolerance for numeric payloads. |
| Outcomes | Winner, draw, terminal tick, death causes, final scores, ranks, episode length. | Exact when the same time policy is used. Use a named window only when the reference cannot be stepped exactly. |
| Observations | Per-player observation shape, dtype, masks, visible players, perspective, hidden-state leaks, reward inputs. | Exact for shape, masks, order, perspective, and visibility rules. Tolerance for numeric channels. |
| Image similarity | Browser or renderer screenshots and short frame sequences. | Exact viewport, scale, crop, and frame index. Tolerance for pixels or perceptual score. |
| Human review | Speed, turning, trail rhythm, crash timing, score rhythm, multiplayer feel. | Reviewer pass/fail with notes. This is acceptance evidence, not a rule oracle. |

## Exact Versus Tolerant

Keep exact checks for choices that should not drift:

- Scenario metadata, config hashes, seed ids, action inputs, and tick indices.
- Player ids, player order, alive/dead flags, phase names, terminal flags, winner,
  death cause, killer id, rank, and score.
- Event names, event order, event count, and discrete event fields.
- Observation shape, dtype, action mask, player perspective, visibility mask, and
  hidden-state exclusion.
- Multiplayer rules such as map-size formula, reverse update order, same-frame death
  order, survivor scoring, and per-player message perspective.

Use tolerance for values that can differ because JavaScript and Python do math
slightly differently:

- Position, heading, speed, angular velocity, elapsed time, and interpolated render
  positions.
- Trail point coordinates, bonus coordinates, distances, and collision margins.
- Numeric observation channels after scale or normalization.
- Screenshot pixels, antialiasing, canvas scaling, and video frame timing.

Every tolerance must live in the fixture or comparison config. Do not hide tolerance
only inside test code.

## Run Summary

Each comparison run should emit one compact summary:

- Status: `pass`, `fail`, or `blocked`.
- Blocked reason, if any: missing reference artifact, nondeterministic reference,
  schema mismatch, runner crash, or unsupported scenario.
- First mismatch: stage, tick, player id if any, field path, expected value, actual
  value, absolute error, relative error, tolerance, and previous tick path.
- Counts by mismatch type: exact state, numeric state, event, outcome, observation,
  image, human-review note, schema, or artifact.
- Worst numeric error: field path, tick, expected value, actual value, absolute error,
  relative error, and tolerance ratio.
- Artifact links: reference trace, Python trace, event diff, replay, observation diff,
  screenshot diff, and review notes when present.

## Multiplayer Fidelity

A run is not enough if it only checks one player or 1v1. The promoted suite should
include 2-player, 3-player, and 4-player cases.

Measure these multiplayer points directly:

- Player order, spawn order, and map size by player count.
- Same-frame deaths and death event order.
- Score rules with no survivor, one survivor, and ordered deaths across ticks.
- Head-to-head, own trail, and opponent trail collisions.
- Per-player observation perspective, including which players and events each player
  can see.
- Server or message payload differences by recipient, if browser fidelity is in scope.

## Observation Fidelity

Observation checks are separate from game-state checks. A state can be correct while
the agent sees the wrong thing.

Check:

- Shape, dtype, channel names, scaling, clipping, and missing-value policy.
- Player-centric ordering and rotation, if used.
- Visibility rules for hidden information, dead players, trails, bonuses, scores, and
  terminal state.
- Action masks and legal actions.
- Reward inputs and episode-end fields.
- Observation hash after agreed quantization for stable regression tests.

Observation numeric channels should use tolerance. Observation masks, perspective,
shape, and hidden-state rules should be exact.

## Promotion Rule

Promote a scenario into the regression set only when it has provenance, fixture
tolerances, expected outcome, trace schema version, and a clear status from the last
run. If behavior is intentionally different from CurvyTron, label it `v0-choice` or
`source-inspired` instead of calling it source-fidelity.
