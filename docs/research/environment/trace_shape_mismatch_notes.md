# Trace Shape Mismatch Notes

Status: Working note

Artifacts inspected:

- `artifacts/local/fidelity/js_forced_two_player_turn_step.json`
- `artifacts/local/fidelity/python_forced_two_player_turn_step.json`

Both artifacts are present in the local checkout.

## Short Answer

The two files are not the same kind of trace yet.

The JS file is a source-style game snapshot after one forced update. It is
close to the CurvyTron object model: game fields, avatar fields, and emitted
events.

The Python file is a `curvyzero-v0` toy-runner artifact. It includes a scenario
copy, metadata, an initial frame, and a post-step frame. It also says
`source_fidelity: false`, so it should not be treated as a proof that Python
matches CurvyTron.

A direct JSON diff between these two files is expected to fail.

## Current JS Shape

Top level:

- `scenario`
- `playerCount`
- `trace`
- `source`
- `loadedSources`

The trace has one frame. That frame uses `tick: 0` and `stepMs: 16.666667`.
It contains:

- `game`: source-like round state, winners, deaths, arena size, and body count.
- `avatars`: rich per-avatar state such as position, angle, velocity, alive
  flag, score, trail count, body count, and `printManager`.
- `events`: source events such as `angle` and `position`.

Example values after the forced step:

- player `p0`: `x = 20.266376`, `y = 39.98756`, `angle = -0.046667`
- player `p1`: `x = 59.733624`, `y = 39.98756`, `angle = 3.188259`

## Current Python Shape

Top level:

- `message`
- `provenance`
- `rules_hash`
- `runner`
- `scenario`
- `schema`
- `source_fidelity`
- `trace`
- `trace_fingerprint`

The trace has two frames:

- `tick: 0`: initial toy-v0 state
- `tick: 1`: post-step toy-v0 state

Each frame stores compact arrays:

- `positions`
- `headings`
- `alive`
- `rewards`
- `terminated`
- `truncated`

Example toy-v0 values after the forced step:

- player `p0`: `x = 20.996801376342773`, `y = 39.92008590698242`,
  `heading = -0.07999999821186066`
- player `p1`: `x = 59.003196716308594`, `y = 39.92008590698242`,
  `heading = 3.221592664718628`

These values are not just rounded versions of the JS values. They come from a
different toy ruleset.

## Main Mismatches

- The JS trace is source-shaped. The Python trace is toy-runner-shaped.
- The JS file has one post-update frame. The Python file has an initial frame
  and a post-step frame.
- The JS frame stores players as avatar objects. The Python frame stores player
  state in parallel arrays.
- The JS file records source events. The Python file records RL-style fields
  such as rewards and termination flags.
- The movement numbers differ because Python is still `curvyzero-v0`, not a
  source-fidelity implementation.

## Recommended Common Fields

Use a small projected trace before comparing JS and Python. The projection
should be easy to build from either raw artifact.

Recommended envelope fields:

- `trace_schema_version`
- `scenario_id`
- `ruleset_id`
- `source_fidelity`
- `runner`
- `source_target`
- `source_commit`
- `provenance`

Recommended frame fields:

- `tick`
- `phase`: `initial` or `post_step`
- `step_ms`
- `player_count`
- `map_size`

Recommended per-player fields:

- `player_id`
- `avatar_id`
- `move`
- `alive`
- `x`
- `y`
- `angle_rad`
- `printing`

For the current JS artifact, `player_id` can be derived from avatar `name`
until the oracle writes it explicitly. For the current Python artifact,
per-player state can be read by index from `scenario.players`, `positions`,
`headings`, and `alive`.

## Do Not Compare Yet

Do not use these fields as pass/fail checks between the current JS and Python
artifacts:

- position, angle, velocity, or angular velocity
- trail point counts, body counts, and `printManager`
- source event order
- score, round winner, game winner, deaths, and death count
- bonuses and world body count
- RL fields such as rewards, `terminated`, and `truncated`
- hashes, fingerprints, and `loadedSources`
- raw frame count or tick numbering

These fields are still useful to record. They are not fair equality checks
until Python is running a source-fidelity ruleset and both runners emit the
same projected trace shape.

## Recommended Next Step

Treat the current comparison as a shape smoke test only:

1. Check that both artifacts name the same scenario.
2. Check that both artifacts describe two players.
3. Build the common projected fields listed above.
4. Mark numeric JS-vs-Python movement comparison as blocked by
   `source_fidelity: false`.

