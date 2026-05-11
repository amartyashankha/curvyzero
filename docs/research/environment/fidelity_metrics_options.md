# Fidelity Metrics Options

Status: research note

This note lists metric choices for environment fidelity. The design plan is in
`docs/design/environment/fidelity_measurement_plan.md`.

## Recommended Set

Use a small stack of metrics instead of one score:

1. Exact state match for discrete state.
2. Tolerated numeric match for floats.
3. Event match for semantic behavior.
4. Outcome match for winners, deaths, scores, and episode end.
5. Observation match for what each agent sees.
6. Image similarity for renderer and browser checks.
7. Human review for feel.

This keeps failures explainable. A single combined score can hide the first real bug.

## Metric Options

| Metric | Best use | What should be exact | What needs tolerance |
| --- | --- | --- | --- |
| Exact state match | Setup, flags, ids, scores, phases, terminal state. | Booleans, enums, ids, counts, scores, player order, winner, death cause. | Not for floats. |
| Tolerated numeric match | Motion, angles, time, distances, numeric observation channels. | Field presence and units. | Absolute or relative error for x/y, angle, speed, time, trail and bonus coordinates. |
| Event match | Debugging update order and rule drift. | Event names, order, ids, causes, score events, terminal events. | Numeric payloads and event timestamps. |
| Outcome match | Golden tests and replay acceptance. | Winner, draw, deaths, death causes, final scores, ranks, terminal tick when time is fixed. | Terminal tick window only when the reference runner cannot control time exactly. |
| Observation match | Agent-facing fidelity and training safety. | Shape, dtype, masks, perspective, visible entities, hidden-state rules. | Normalized floats, distances, angles, image-like channels. |
| Image similarity | Visual renderer, browser protocol, and replay artifacts. | Viewport, scale, crop, frame index, color config. | Pixel error, perceptual similarity, antialiasing, frame timing. |
| Human review | Feel and acceptance after objective checks pass. | Review checklist completion and decision. | Human judgment notes, not numeric equality. |

## Tolerance Choices

Good default tolerance types:

- Absolute error for positions, distances, angles, and scores that are derived from
  small numbers.
- Relative error for large values where scale matters.
- Wrapped angle error for headings.
- Quantized hash for observations after scaling and clipping.
- Pixel threshold plus perceptual score for screenshots.

Bad defaults:

- Percentage-only checks near zero.
- Pixel checks before state and event checks pass.
- Tolerances that live only in code and are not written into the fixture.
- Human feel as a replacement for trace evidence.

## Run Summary Metrics

Every run should produce:

- `status`: `pass`, `fail`, or `blocked`.
- `first_mismatch`: stage, tick, field path, player id if any, expected, actual,
  tolerance, and error.
- `mismatch_counts`: counts for exact state, numeric state, event, outcome,
  observation, image, schema, artifact, and human-review notes.
- `worst_numeric_error`: field path, tick, absolute error, relative error, tolerance,
  and tolerance ratio.
- `artifacts`: trace files, event logs, replay, observation diff, screenshot diff, and
  review notes.

## Multiplayer Metrics

Multiplayer needs its own counts because some bugs only appear with 3 or 4 players.

Track:

- Player-count coverage: 1P smoke, 2P, 3P, and 4P.
- Same-frame death count and event order.
- Score deltas by death order and survivor state.
- Collision type counts: wall, own trail, opponent trail, head-to-head, and multi-head.
- Per-player observation mismatches by recipient.
- Per-recipient server-message mismatches when browser fidelity is tested.

## Observation Metrics

Observation fidelity should report both schema and content:

- Schema pass/fail: shape, dtype, channel names, bounds, and version.
- Mask pass/fail: legal actions, visibility, alive/dead, and terminal masks.
- Perspective pass/fail: ego player position, player ordering, rotation, and recipient
  view.
- Numeric error: max and mean error per channel after normalization.
- Leak check: count of fields that expose hidden state.
- Hash match: exact hash after agreed quantization for stable fixtures.

## Practical Recommendation

For day-to-day work, fail fast on the first exact mismatch, then the first numeric
mismatch outside tolerance, then event mismatch, then outcome mismatch. Run observation
checks before image checks. Use human review only after traces, outcomes, and key
observations are already trusted.
