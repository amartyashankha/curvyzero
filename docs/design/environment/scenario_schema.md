# Environment Scenario Schema

Status: Draft

This is the small JSON shape for JS/Python trace comparison scenarios. A
scenario says how to set up the game and what moves to apply. Runners emit
traces separately.

Scenario files live under:

```text
scenarios/environment/<scenario_id>.json
```

## Required Fields

- `schema_version`: use `environment-scenario-v0`.
- `id`: stable scenario id.
- `title`: short name.
- `ruleset_id`: rule target, such as `curvytron-v1-reference`.
- `provenance`: labels and notes that say where the scenario facts came from.
- `source_setup`: game-level setup.
- `players`: ordered forced player setup.
- `steps`: ordered elapsed-ms source input frames.
- `comparison`: basic trace comparison policy.

## Provenance

Use plain labels:

- `source-derived`: copied from original source behavior or JS oracle output.
- `toy-v0`: related to the current simplified Python environment.
- `v0-choice`: an intentional toy-v0 difference from source behavior.
- `source-inspired`: based on source behavior, but not exact.
- `unresolved`: known gap or open question.

Example:

```json
{
  "provenance": {
    "labels": ["source-derived", "toy-v0", "unresolved"],
    "source_target": "third_party/curvytron-reference",
    "source_commit": "8fec14c",
    "notes": ["No scenario runner consumes this file yet."]
  }
}
```

## Setup

`source_setup` should include the whole-game facts needed before the first step:

```json
{
  "source_setup": {
    "player_count": 2,
    "map_size": 88,
    "random": {
      "math_random": 0.5
    },
    "room": {
      "name": "oracle-probe",
      "max_score": 10,
      "bonuses": [],
      "bonus_rate": 0
    },
    "game": {
      "started": true,
      "in_round": true,
      "borderless": false,
      "world_active": true
    }
  }
}
```

Random controls are deliberately small:

- `source_setup.random.math_random`: constant `Math.random()` value for the JS
  oracle. The default is `0.5`.
- `source_setup.random.math_random_sequence`: exhausting tape of `Math.random()`
  values for call-order probes. Each value must be in `[0, 1)`. If the source
  consumes more values than the tape provides, the JS oracle errors.
- JS oracle output includes `randomCalls`, a top-level call log with
  `{index, value}` entries in consumption order. The Python
  `source-print-manager-canary` runner also reports this field for
  PrintManager distance calls. The log proves stream order; state fields still
  attribute which source object received each value.

Python random-tape support is currently scoped to PrintManager distance calls
in the `source-print-manager-canary` runner. Keep other random-tape fixtures
marked pending until their matching Python/common-trace runner support lands.

Each player has a forced start:

```json
{
  "id": "p0",
  "client_id": "p0-client",
  "avatar_id": 1,
  "color": "#ff0000",
  "initial": {
    "x": 20,
    "y": 40,
    "angle_rad": 0,
    "printing": true
  }
}
```

## Steps

Each `steps[]` entry stages source control values, then calls
`game.update(step_ms)`. These are oracle/replay fixture frames, not trainer
`step(joint_action)` decisions.

Source control move values:

- `-1`: left
- `0`: straight
- `1`: right

Example:

```json
{
  "tick": 0,
  "step_ms_expr": "1000 / 60",
  "step_ms": 16.666666666666668,
  "moves": [
    { "player_id": "p0", "move": -1 },
    { "player_id": "p1", "move": 1 }
  ]
}
```

## Comparison

Keep this simple for now:

```json
{
  "comparison": {
    "kind": "js-python-state-trace",
    "trace_schema_version": "environment-trace-v0",
    "python_target": "curvyzero-v0",
    "source_fidelity_required": false,
    "tolerances": {
      "position": 0.000001,
      "angle": 0.000001,
      "velocity": 0.000001
    }
  }
}
```

The comparison block describes intent. It does not mean raw JS and raw Python
artifacts already have the same JSON shape.

## Trace Shape Policy

Current local artifacts for `forced_two_player_turn_step` use different output
shapes:

- JS writes a source-style frame with `game`, `avatars`, and `events`.
- Python writes a toy-v0 runner artifact with `trace.frames`, compact player
  arrays, rewards, and termination flags.

Do not compare the raw artifacts directly. First project each raw artifact into
a small common trace.

Recommended common envelope fields:

- `trace_schema_version`
- `scenario_id`
- `ruleset_id`
- `source_fidelity`
- `runner`
- `source_target`
- `source_commit`
- `provenance`

Recommended common frame fields:

- `tick`
- `phase`: `initial` or `post_step`
- `step_ms`
- `player_count`
- `map_size`

Recommended common per-player fields:

- `player_id`
- `avatar_id`
- `move`
- `alive`
- `x`
- `y`
- `angle_rad`
- `printing`

For toy-v0 Python traces, record numeric movement fields but do not treat them
as source-fidelity pass/fail checks. The current Python artifact says
`source_fidelity: false`, so position, angle, velocity, trail, collision,
score, and event equality should wait.

Fields not ready for first equality checks:

- velocity and angular velocity
- trail point counts, body counts, and print-manager internals
- source event order
- rewards, `terminated`, and `truncated`
- score, winners, deaths, bonuses, and world body count
- hashes, fingerprints, `loadedSources`, raw frame count, and raw tick numbering

## Source Kinematics Micro-Scenarios

The `source_kinematics_*` fixtures are tiny forced-state checks. They should
only set the initial player state and source move inputs. Do not add collision,
trail, bonus, score, or round-outcome expectations to these files yet.

Current one-step fixtures:

- `source_kinematics_straight_step`: proves source move `0` advances position
  without changing heading.
- `source_kinematics_left_turn_step`: proves source move `-1` applies the
  source turn rate before translation.
- `source_kinematics_right_turn_step`: proves source move `1` applies the
  source turn rate before translation.

The JS reference scenario runner can iterate multiple `steps`. The Python
`source-kinematics` CLI path now accepts the current `source_kinematics_*`
fixtures plus `forced_two_player_turn_step`. Keep it scoped to forced movement
kinematics until collision, trail, bonus, and scoring behavior are added
explicitly.
