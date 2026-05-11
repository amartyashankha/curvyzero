# Trace Schema

Status: Draft

## Common Trace View V1

The first comparison view is intentionally small. It removes runner-only metadata
such as JS `loadedSources` before running the first-mismatch diff.

Fields. `steps[]` are common-trace elapsed-ms source frames; they are not
trainer wrapper decisions:

- `schema`: `curvyzero_common_trace/v1`
- `scenario_id`
- `steps[]`
- `steps[].step_index`
- `steps[].step_ms`
- `steps[].players[]`
- `steps[].players[].player_id`
- `steps[].players[].x`
- `steps[].players[].y`
- `steps[].players[].angle`
- `steps[].players[].alive`
- `steps[].players[].score`, if present
- `steps[].players[].roundScore`, if present

The current Python scenario trace includes a reset frame before step frames. The
common projection drops that reset frame when the action script length shows it
is extra.

## Still Intentionally Mismatched

- The JS runner is the CurvyTron reference path; the Python runner is still
  labeled toy-v0 and is not a source-fidelity claim.
- JS has metadata such as `loadedSources`; Python does not. That metadata is
  ignored by the common trace projection.
- JS currently reports avatar `score` and `roundScore`; Python toy-v0 traces may
  not include those fields yet.
- Player coordinates and angles are expected to differ for the first forced
  fixture because movement rules are not aligned yet.
