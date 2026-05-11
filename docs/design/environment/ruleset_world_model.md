# Ruleset World Model

Status: narrow wall/border events verified, 2026-05-09

This page keeps the arena and border rules explicit. The important correction is
that source CurvyTron does not have only fixed walls. It has normal wall death
and a separate timed `borderless` bonus that wraps at the map edge.

## Source-Backed Facts

- Normal mode: `Game.update(step)` checks
  `world.getBoundIntersect(avatar.body, avatar.radius)`. If the avatar body
  crosses the map edge, the avatar dies.
- Borderless mode: `Game.update(step)` checks
  `world.getBoundIntersect(avatar.body, 0)`. If the avatar body crosses the map
  edge, the avatar is moved with `World.getOposite(...)` to the opposite edge
  instead of dying.
- `BonusGameBorderless` is the source trigger. It sets the game `borderless`
  effect to `true` for a timed duration.
- Source `borderless` is not a clean torus. It wraps to the exact opposite edge,
  loses any overshoot beyond the edge, resolves only the first border axis found,
  treats exact edge equality as safe, and skips body collision on the wrap frame
  because body collision lives in the non-border branch.

Source files read:

- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/model/Bonus/BonusGameBorderless.js`
- `docs/research/curvytron_reference_notes.md`
- `docs/research/environment/curvytron_js_state_oracle.md`
- `docs/experiments/environment/2026-05-08-headless-js-oracle-probe.md`

## Current Python V0

`curvyzero-v0` is a small training scaffold. It has fixed rectangular bounds:
leaving the grid kills the player. It has no `borderless` flag, no source
borderless wrap, and no bonus system. Its wall-collision test proves toy-v0
behavior only.

## Probe Status

Ready:

- Source code supports normal-wall death and borderless wrap.
- The borderless source result is known: timed bonus, margin `0`, exact opposite
  edge, lost overshoot, one-axis-first handling, and skipped body collision on
  the wrap frame.
- The headless JS probe can step source game objects directly.
- The current source-kinematics path matches the forced two-player movement
  trace.
- The shared scenario loop matches two normal-wall death fixtures, plain
  borderless wrap, borderless PrintManager wrap, destination-body skip, and
  exact-edge/corner-axis behavior through the narrow source border runners.
- Event comparison for the narrow border slice is opt-in by contract through
  `comparison.include_events: true`. The current wall/border fixtures opt in and
  passed the mixed source-border batch.
- Verified command:
  `uv run python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --artifact-root /private/tmp/curvy-source-border-events-batch`.
  Result: `6` pass, `0` fail, `0` blocked, `diff_mode: common-trace`, and no
  first mismatches.

Pending:

- Event fidelity beyond the narrow wall/border contract, including body
  collisions, trails, bonuses, and full replay/server messages.
- Broader scoring traces beyond the current wall-death state fields.
- Body collision, trail collision, and print-manager side effects.

Narrow event contract fields:

- `position`: `player_id`, `x`, `y`
- `point`: `player_id`, `x`, `y`, `important`
- `die`: `player_id`, `killer_id`, `old`
- `score:round` and `score`: `player_id`, `score`, `roundScore`
- `round:end`: `winner_id`

Events are ordered per step. The event surface is intentionally narrow; do not
add trails, bonuses, browser messages, or full replay fields to this slice.

## Documentation Rule

Do not write plain `wall collision` when the rule matters. Use one of these:

- `normal-wall death`
- `source borderless wrap`
- `fixed toy-v0 wall`
- `border rule pending probe`

Self-reflection: source and probe output beat memory, summaries, and handoffs.
If the source/probe result is not ready, mark it pending.
