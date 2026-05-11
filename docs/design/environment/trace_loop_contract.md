# Trace Loop Contract

Status: Draft
Date: 2026-05-09

This is the contract for the current environment fidelity loop:

```text
scenario JSON -> raw JS trace -> raw Python trace -> common trace -> diff -> artifacts
```

The scenario file is the input contract. The common trace is the comparison
contract. Raw traces are debug output. Keep this loop boring and local first.

## Responsibilities

| Piece | Job |
| --- | --- |
| Scenario format | Defines players, initial state, source control moves, elapsed-ms frames, provenance, and comparison mode. |
| JS runner | Reads one scenario, runs CurvyTron JS, writes raw JS trace and events. |
| Python runner | Reads the same scenario, runs Python, writes raw Python trace and events. |
| Normalizer | Converts each raw trace to `curvyzero_common_trace/v1`. |
| Differ | Compares common traces and writes `pass`, `fail`, or `blocked`. |
| Local loop | Owns command order, common-trace sidecars, timeline sidecars, and the per-scenario artifact folder. |
| Batch runner | Runs the same local loop for a list of scenarios and writes one small summary. |
| Modal wrapper | Later runs the same batch shape remotely and commits artifacts. |

## Scenario Shape

Use this accepted shape for new fixtures until the schema migration is explicit.
The code still reads older aliases such as `id`, but new fixtures should prefer
`scenario_id`.

```json
{
  "schema_version": "environment-scenario-v0",
  "scenario_id": "forced_two_player_turn_step",
  "ruleset_id": "curvytron-v1-reference",
  "provenance": {
    "labels": ["source-derived"],
    "source_target": "third_party/curvytron-reference",
    "source_commit": "8fec14c"
  },
  "source_setup": {
    "player_count": 2,
    "map_size": 88,
    "game": {
      "started": true,
      "in_round": true,
      "borderless": false,
      "world_active": true
    }
  },
  "player_count": 2,
  "time_policy": {
    "kind": "fixed",
    "step_ms": 16.666666666666668
  },
  "players": [
    {
      "id": "p0",
      "avatar_id": 1,
      "initial": {
        "x": 20,
        "y": 40,
        "angle_rad": 0,
        "alive": true,
        "printing": true
      }
    }
  ],
  "steps": [
    {
      "tick": 0,
      "step_ms": 16.666666666666668,
      "moves": [
        {"player_id": "p0", "move": -1},
        {"player_id": "p1", "move": 1}
      ]
    }
  ],
  "comparison": {
    "mode": "shape-only",
    "common_trace_schema": "curvyzero_common_trace/v1",
    "python_target": "curvyzero-v0",
    "include_events": false,
    "tolerances": {
      "position_abs": 0.000001,
      "angle_rad_abs": 0.000001
    }
  }
}
```

Rules:

- Use `scenario_id`, not `id`.
- Use `players[].id` and `steps[].moves[].player_id`.
- Use `time_policy.kind: "fixed"` for fixed elapsed-ms source-kinematics
  fixtures. This is fixture timing, not trainer policy cadence.
- Moves are CurvyTron control values: `-1` left, `0` straight, `1` right.
- The scenario does not include artifact paths.

## Common Trace Shape

The normalizer writes one common trace per runner:

```json
{
  "schema": "curvyzero_common_trace/v1",
  "scenario_id": "forced_two_player_turn_step",
  "map_size": 88,
  "steps": [
    {
      "step_index": 0,
      "step_ms": 16.666666666666668,
      "players": [
        {
          "player_id": "p0",
          "x": 20.266376,
          "y": 39.98756,
          "angle": -0.046667,
          "alive": true
        }
      ]
    }
  ]
}
```

Do not put runner-only fields here, such as loaded JS sources, rules hashes,
trace fingerprints, rewards, termination flags, or trainer `joint_action`
metadata.

Common trace now includes `map_size` when the scenario or raw trace supplies it,
so border-sensitive batches can compare the same world size explicitly.

### Event Comparison

Event comparison is opt-in. A scenario only includes events in the common trace
when `comparison.include_events` is exactly `true`. Without that flag, common
trace diffs compare state only.

Narrow event contract names and fields are:

- `position`: `event`, `player_id`, `x`, `y`
- `point`: `event`, `player_id`, `x`, `y`, `important`
- `die`: `event`, `player_id`, `killer_id`, `old`
- `score:round`: `event`, `player_id`, `score`, `roundScore`
- `score`: `event`, `player_id`, `score`, `roundScore`
- `round:end`: `event`, `winner_id`

Events are ordered per step. Unknown event names are projected as
`{"event": <name>}` only. Event ordering is part of the diff because events are
compared as arrays.

## Diff Status

Every diff report includes both:

- `match`: the existing boolean compatibility field.
- `status`: one of `pass`, `fail`, or `blocked`.

Status meanings:

- `pass`: requested fields match. `match` is `true`.
- `fail`: requested fields differ. `match` is `false`.
- `blocked`: the diff could not make a valid comparison because input was
  invalid or trace normalization failed. `match` is `false`.

Local loop summaries also copy the diff classification to top-level
`diff_status`, while keeping the loop-level `status` for orchestration states
such as `match`, `mismatch`, `js_failed`, `python_failed`, or `diff_failed`.

For `fail`, the current local loop summary copies the first mismatch fields that
the diff emits: field path, left value, right value, reason, and message when
present. Step-index and previous-step context remain planned observability work;
the current code does not add those fields yet.

## Timeline Sidecars

When common-trace mode succeeds, the local loop writes compact text timelines
derived from the common traces:

- `js.timeline.txt`
- `python.timeline.txt`

These files are observability-only. They include one line per common-trace step
with `step`, `step_ms`, player alive/position/angle/score fields when present,
body counters when present, and projected events when present. Diffing still uses
the JSON traces, not these text files.

Current status: common-trace diff is the default. The default toy-v0 Python
runner still reaches the expected first game-field mismatch at
`$.steps[0].players[0].angle`: JS `-0.046667`, Python toy-v0 about `-0.08`.

That mismatch is expected. Python toy-v0 uses fixed per-tick turn and time
handling. The source updates turn and position from elapsed milliseconds,
source angular velocity, and source speed. Last recorded source-kinematics
checks covered seven movement cases, including varied-elapsed multi-step
movement. That path is movement proof only, not a full source clone.

Last recorded checks for `--python-runner source-normal-wall` matched the
current forced normal-wall death fixtures. The runner remains narrow: source
movement plus normal-wall death state and the event fields listed above, not
body collisions, trails, bonuses, or full game rules.

Last recorded checks for `--python-runner source-borderless-wrap` matched the
current forced borderless wrap, PrintManager wrap, destination-body skip, and
exact-edge/corner-axis fixtures. The `--python-runner source-border-rules`
dispatcher is for mixed border batches; it delegates to the narrow normal-wall
or borderless runner based on the scenario id.

For current whole-environment status, start from
`docs/working/environment/active_lanes.md`. This contract describes the trace
loop shape; it is not the current gap queue.

Last recorded checks for `--python-runner source-body-canary` matched the six
current body canary fixtures. The runner remains narrow: opponent strict
overlap/tangent behavior, own-body latency at the `> 3` point-number gate, and
the two direct same-frame point materialization fixtures only, not
print-manager holes, broader trail storage, bonuses, or full game rules.

Last recorded checks for `--python-runner source-print-manager-canary` matched
the eight deterministic print-manager fixtures. The runner remains narrow:
print-to-hole, hole-to-print, exact-zero toggle, active no-toggle control,
delayed start, active printing stop-on-death, active already-hole
stop-on-death, and active body-collision stop-on-death only, not broader trail
cadence, collision gaps, bonuses, or full game rules.

Verified wall/border event batch:
`uv run python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --artifact-root /private/tmp/curvy-source-border-events-batch`.
Result: `6` pass, `0` fail, `0` blocked, `diff_mode: common-trace`, and no
first mismatches. This marks only the narrow wall/border event contract done.

Verified normal-wall multiplayer batch:
`uv run python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-batch-final`.
Result: `3` pass, `0` fail, `0` blocked, `diff_mode: common-trace`, and no
first mismatches. This marks the narrow 3P/4P normal-wall death/scoring
canaries done.

Verified source-body canary batch:
`uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-same-frame`.
Result: `6` pass, `0` fail, `0` blocked, `diff_mode: common-trace`, and no
first mismatches. This marks only the narrow opponent-body, own-latency, and
same-frame point-materialization canaries done.

Verified source print-manager canary batch:
`uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-active-hole-stop`.
Result: `8` pass, `0` fail, `0` blocked, `diff_mode: common-trace`, and no
first mismatches. This marks deterministic print-manager toggle basics, delayed
start, and the active printing/already-hole/body-collision stop-on-death
canaries done.

Toy-v0 diffs remain state-only by default. Add `comparison.include_events: true`
only for fixtures where both runners emit the event fields above.

## Artifact Layout

Use the simple local layout first:

```text
<artifact_root>/<scenario_id>/
  js.json
  js.common_trace.json
  js.timeline.txt
  js.stderr.txt
  python.json
  python.common_trace.json
  python.timeline.txt
  python.stderr.txt
  diff.json
  diff.stderr.txt
  summary.json
```

When batch mode runs, it should write one extra summary:

```text
<artifact_root>/summary.json
```

This is enough for the next phase. Add deeper Modal manifests only when remote
runs make them necessary.

## Modal Rule

Modal runs batches only:

```text
load scenarios
run JS
run Python
normalize
diff
write artifacts
commit Volume
return summary
```

No Modal calls inside `env.step()`, JS ticks, normalization, or diff loops.
