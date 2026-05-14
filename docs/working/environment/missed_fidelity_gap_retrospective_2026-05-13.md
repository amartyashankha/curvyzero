# Missed Fidelity Gap Retrospective - 2026-05-13

Status: working-memory note, docs-only.
Scope: `BonusEnemyStraightAngle` movement fidelity, default bonus probability
fidelity, `BonusSelfMaster` print-manager side effects, and test guardrails.

## What Was Missed

`BonusEnemyStraightAngle` was treated too much like a stored bonus value. The
source behavior is movement/control semantics: it switches the avatar out of
continuous direction looping, uses a right-angle turn base, applies the current
signed turn once, and then clears that turn input.

The important fact was not just "straight-angle is active." The important fact
was how active straight-angle changes the update loop.

## Why It Was Missed

The existing checks proved bonus application, expiry, and stored fields. Those
were useful, but they did not prove behavior through the movement update loop.
For `BonusEnemyStraightAngle`, that meant we had proof for fields, stack, and
expiry, but not for the update-loop behavior that actually makes the bonus
source-visible.

The docs/spec already contained the right source fact. The gap was that the
fact stayed in documentation and metadata-style checks instead of being promoted
into native movement tests that exercised control state, elapsed source frames,
and resulting trajectory.

## New Guardrail

For every source property or effect, require at least one behavior proof when
the effect changes movement, collision, rendering, or lifecycle behavior.

Metadata/property checks are still allowed, but they are not enough for effects
that alter runtime behavior. The proof should drive the public/native path far
enough to show the source-visible consequence, not only the stored state.

Input semantics need the same honesty. The trainer wrapper uses
`joint_action`/held action decisions and can re-emit controls for each
decision. Source-native behavior is input-event/current-turn state advanced by
elapsed-ms frames. Tests can bridge those surfaces, but they should name which
surface they prove.

## Fixes Completed

`BonusEnemyStraightAngle` is fixed as movement behavior. Native/vector movement
now carries source-like `current_angular_velocity` and `direction_in_loop`, and
internal source-frame loops arm source moves only once per outer decision. The
proof includes a low-level snap test and a public seeded catch test across
`decision_source_frames=4`.

Default bonus probabilities were also a real bug. Scalar and vector paths now
use the source default map: inverse `0.8`, straight-angle `0.6`, borderless
`0.8`, dynamic clear probability, and `1.0` for the other source-default
bonuses. Probability and spawn tests were updated.

Dirty render stats aggregation was fixed too: numeric top-level fields are
summed, while nested dict counters are merged.

`BonusSelfMaster` print-manager side effects are fixed in vector runtime and
covered by public env proof. Catch gives `invincible=true` and `printing=-1`
for `7500` ms. While active it blocks body/trail death, but normal wall death
still kills unless project-only `profile_no_death`, `death_immunity`, or
opponent-immortal modes are enabled. Expiry restores `invincible=false` and
restarts printing/PrintManager; death before expiry clears the stack and must
not restart printing or leave invincible true.

Validation after the right-angle/default/dirty-render fixes: the focused
controls/bonus/runtime/render-surface suite reported `260 passed`; the broad
environment sweep reported `578 passed, 2 skipped`; `ruff`, the environment
doc guard, and `git diff --check` passed. Focused SelfMaster validation later
reported runtime `self_master or print_manager` `11 passed`, public
`self_master` `5 passed`, and broader focused environment suite `321 passed`;
`ruff` and diff checks passed. The full environment sweep after SelfMaster
reported `591 passed, 2 skipped`.

## Active Audit Threads

- Bonus semantics cross-audit: recheck source bonus effects for places where a
  stored field is not the whole behavior.
- Movement/control state audit: verify held input, one-shot input, turn-rate,
  and source-frame semantics where bonuses or controls modify movement.
- Source inventory/test proof audit: map each source inventory claim to at
  least one behavior proof when the claim affects runtime behavior.
- SelfMaster follow-up: broad validation is complete for the current
  environment sweep; the print-manager side-effect implementation gap is no
  longer open.

## Post-Right-Angle Audit Queue

This is a ranked queue from the new audits, not a claim that every item is
currently broken.

Completed from this queue:

- Fix `BonusEnemyStraightAngle` native/vector movement semantics and public
  seeded catch proof across `decision_source_frames=4`.
- Fix default bonus probabilities: `0.8` inverse, `0.6` straight-angle, `0.8`
  borderless, dynamic clear probability, and `1.0` for the other
  source-default bonuses.
- Fix `BonusSelfMaster` print-manager side effects and public env proof,
  including active invincibility/`printing=-1`, expiry restart, and
  death-before-expiry cleanup.
- Add focused runtime proof for velocity and inverse active-turn behavior:
  speed bonuses refresh an already held turn's rate, and inverse preserves the
  active turn sign on catch/expiry until the next source input event.
- Add focused runtime proof for radius collision/render lifecycle:
  `BonusSelfSmall` radius changes affect normal wall checks, body collision
  checks, raw browser-like RGB rendering, and downsampled gray64 observations.
- Add focused direct borderless catch-to-wrap proof: after
  `BonusGameBorderless` catch, an active edge crossing wraps and does not
  normal-wall-die.
- Add focused AllColor visual/observation proof: after `BonusAllColor` catch,
  rotated `avatar_color` reaches the browser-like RGB frame and gray64 path,
  then expiry restores the baseline frame.

Remaining queue:

- No open items from this queue.
