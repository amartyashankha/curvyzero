# Trail Gap Investigation

Status: source-read design note, updated 2026-05-09

Scope: design the next source-fidelity slice for normal trail cadence and the
later trail-gap/collision checks. This note does not change code or scenarios.

## Source And Docs Read

Source files:

- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseTrail.js`
- `third_party/curvytron-reference/src/server/model/Avatar.js`
- `third_party/curvytron-reference/src/server/model/Trail.js`
- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/core/Island.js`
- `third_party/curvytron-reference/src/server/core/AvatarBody.js`
- `third_party/curvytron-reference/src/server/core/Body.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `third_party/curvytron-reference/src/client/model/Avatar.js`
- `third_party/curvytron-reference/src/client/model/Trail.js`
- `third_party/curvytron-reference/src/client/repository/GameRepository.js`
- `third_party/curvytron-reference/src/server/controller/GameController.js`

Current docs:

- `docs/working/environment/coverage_tracker.md`
- `docs/working/environment/probe_backlog.md`
- `docs/working/environment/print_manager_investigation.md`
- `docs/working/environment/body_trail_investigation.md`

## Current Checkpoint

Verified slices already cover:

- `source-body-canary`: seeded opponent body tangent/overlap, own-body latency
  at delta `3` and `4`, and same-frame point materialization.
- `source-print-manager-canary`: forced print-to-hole, hole-to-print, and active
  no-toggle PrintManager behavior.
- `source-trail-cadence-canary`: normal point insertion and below-radius
  no-point behavior.
- `source-trail-gap-canary`: forced hole-space safety,
  stored-body-in-visual-hole kill, print-to-hole boundary kill, and
  hole-to-print boundary kill.

The normal `Avatar.isTimeToDraw()` threshold and the four first gap/collision
controls are now pinned. Keep the boundary details in
[trail_gap_collision_probe_plan.md](trail_gap_collision_probe_plan.md), and use
the broader rebuild map for the next source-fidelity slice.

## Exact Source Facts

Normal server trail cadence:

- Default avatar speed is `16` units per second:
  `BaseAvatar.prototype.velocity = 16`
  (`BaseAvatar.js:35-40`).
- Default avatar radius is `0.6`:
  `BaseAvatar.prototype.radius = 0.6`
  (`BaseAvatar.js:49-54`).
- Server `Avatar.update(step)` updates angle, updates position, then adds a
  normal point only when `printing` is true and `isTimeToDraw()` is true
  (`server/model/Avatar.js:23-31`).
- `isTimeToDraw()` returns true when `trail.lastX === null`. Otherwise it
  returns true only when distance from `trail.lastX/lastY` to current position is
  greater than `avatar.radius`, not greater-than-or-equal
  (`server/model/Avatar.js:40-47`).
- `BaseTrail.addPoint(x, y)` appends `[x, y]` and updates `lastX/lastY`
  (`BaseTrail.js:25-30`).

At the source framerate of about `1000 / 60` ms, straight base movement is
`16 * (1000 / 60) / 1000 = 0.2666666667` units per tick. From an existing trail
point, the first two frames stay below radius, and the third frame reaches about
`0.8`, so it draws. With `step_ms = 100`, the avatar moves `1.6`, so it draws on
each printed tick unless the forced trail cursor is moved near the endpoint.

World body state is separate from trail point state:

- Server `Avatar.addPoint()` first calls `BaseAvatar.addPoint()`, then emits a
  `point` event (`server/model/Avatar.js:158-162`).
- `Game.onPoint(data)` listens to avatar `point` events. If the game is started
  and the world is active, it immediately inserts
  `new AvatarBody(data.x, data.y, data.avatar)` into the world
  (`server/model/Game.js:113-118`).
- `AvatarBody` assigns `num = avatar.bodyCount++`
  (`server/core/AvatarBody.js:8-13`).
- `World.addBody()` only runs when `world.active` is true. It assigns a world
  body id, increments `world.bodyCount`, and inserts the body into islands by
  its bounding-box corners (`server/core/World.js:51-63`).
- Collision uses strict circle overlap: `distance < radiusA + radiusB`
  (`server/core/Island.js:83-90`).
- Same-avatar bodies are ignored until
  `currentBody.num - storedBody.num > avatar.trailLatency`; `trailLatency` is
  `3` (`AvatarBody.js:33-40`, `BaseAvatar.js:56-62`).

Visual/client trail state is not collision state:

- `BaseTrail.clear()` clears `trail.points`, `trail.lastX`, and `trail.lastY`
  (`BaseTrail.js:35-40`).
- Server `Trail.clear()` calls `BaseTrail.clear()` and emits a trail `clear`
  event (`server/model/Trail.js:15-18`).
- No source path connects `Trail.clear()` to `World.removeBody()`. Old collision
  bodies inserted by `Game.onPoint -> World.addBody(new AvatarBody(...))`
  remain until a world clear path such as `Game.clearTrails()` or
  `World.clear()` runs (`server/model/Game.js:198-203`,
  `server/core/World.js:357-365`).
- Therefore a visual gap is not deletion of collision bodies. It means no normal
  per-radius point bodies are added while `printing` is false, while old bodies
  from earlier printed sections still exist.
- Client visuals have their own trail handling. Client `setPositionFromServer`
  adds visual trail points while client `printing` is true
  (`client/model/Avatar.js:74-82`). Client `Trail.addPoint()` asks for a visual
  clear when the next point jumps by more than tolerance `1`
  (`client/model/Trail.js:58-67`). Server `GameController.onPoint()` forwards
  only important point events to clients (`server/controller/GameController.js:328-333`).

Update order risk:

- `Game.update(step)` iterates avatars in reverse order. For each alive avatar,
  it runs `avatar.update(step)`, checks border, checks body collision, then only
  if still alive runs `avatar.printManager.test()` and bonus catch
  (`server/model/Game.js:44-73`).
- Normal `isTimeToDraw()` point bodies are inserted before body collision in the
  same avatar update.
- Print-manager gap toggles happen after body collision. A print-to-hole toggle
  cannot save an avatar from a collision already checked in that frame.

## First Trail Cadence Fixtures

Implemented names:

- `source_trail_normal_point_step`
- `source_trail_no_point_below_radius_step`

Batch:

- `source_trail_batch.json`
- `source_trail_cadence_batch.json` is an alias manifest with the same two
  scenarios.

Status:

- JS oracle tests pin both raw traces.
- Python/common-trace parity passes through `source-trail-cadence-canary`.
- Promoted batch claim: `source_trail_batch.json` with
  `--python-runner source-trail-cadence-canary` protects the two named trail
  cadence fixtures.

Important implementation detail:

- The scenario can force the source draw cursor with `trail.last_x/last_y`
  without forcing a visible trail point. The JS oracle's `lastTrailPoint`
  reflects `trail.points`, not the hidden draw cursor.
- Python runner state must therefore keep two concepts separate: visible trail
  point state for common trace, and hidden draw cursor state for
  `isTimeToDraw()`.

What the fixtures prove:

- `source_trail_normal_point_step`: `printing: true`, forced cursor `(20,40)`,
  `step_ms: 100`, endpoint `(21.6,40)`, one non-important point, one new body,
  `trailPointCount: 1`, `bodyCount: 1`, `worldBodyCount: 1`.
- `source_trail_no_point_below_radius_step`: `printing: true`, forced cursor
  `(20,40)`, `step_ms: 25`, endpoint `(20.4,40)`, no point, no new body,
  `trailPointCount: 0`, `bodyCount: 0`, `worldBodyCount: 0`, visible
  `lastTrailPoint: null`.

Do not use literal exact-distance `0.6` as the next threshold fixture without a
separate floating-point plan. In JS, `20 + 16 * 37.5 / 1000` can land slightly
above `0.6`.

## First Gap Fixtures

Implemented names:

- `source_trail_gap_hole_space_safe_step`
- `source_trail_gap_stored_body_still_kills_step`
- `source_trail_gap_print_to_hole_boundary_kills_step`
- `source_trail_gap_hole_to_print_boundary_kills_step`

Batch:

- `source_trail_gap_batch.json`

Status:

- JS oracle tests pin all four raw traces.
- Python/common-trace parity passes through `source-trail-gap-canary`.
- Promoted batch claim: `source_trail_gap_batch.json` with
  `--python-runner source-trail-gap-canary` protects the four named forced
  trail-gap fixtures.

What the fixtures prove:

- No normal per-radius point bodies are added while `printing` is false, so a
  crossing through the empty hole space is safe when no old body is there.
- Old bodies from the printed section before the visual clear still exist. A
  player crossing one of those old bodies should still die.
- The print-to-hole boundary body is inserted before the later lower-index
  player updates, even though the visual trail is cleared.
- The hole-to-print boundary body is also inserted before the later lower-index
  player updates, but unlike print-to-hole it leaves the newly visible trail
  point in place. The fixture checks the important point event, the printing
  property event with value `true`, `p1.trailPointCount: 1`,
  `p1.lastTrailPoint: [41.6, 40]`, `p1.printManager.distance: 39`, and p0's
  same-frame death to that emitted p1 body.

## Python Runner And Common Trace Requirements

The cadence runner exists as `source-trail-cadence-canary`; the first gap runner
exists as `source-trail-gap-canary`. Keep any boundary expansion inside that
runner narrow and named.

The Python runner must compare these source rules:

- Straight source movement and current `step_ms`.
- Printing state.
- `isTimeToDraw()` with `lastX === null` first-point behavior and strict
  `distance > radius` later behavior.
- Normal point insertion during `Avatar.update()`, before border/body collision.
- Synchronous point-to-body insertion:
  `Game.onPoint -> World.addBody(new AvatarBody(...))`.
- `AvatarBody.num`, live `body.num`, avatar `bodyCount`, and world body count.
- Own-body latency using point numbers.
- For print-to-hole boundary and later gap fixtures, `setPrinting(false)` must
  add the important boundary point, clear `trail.points/lastX/lastY`, and leave
  existing world bodies in place.
- The symmetric `setPrinting(true)` boundary must add the important point,
  keep that point visible in trail state, reset the print distance from the
  source random stream, and leave the emitted body immediately collidable.
- For later gap fixtures, collision must run before `PrintManager.test()`.

Common trace should include:

- Per player: `x`, `y`, `angle`, `alive`, `score`, `roundScore`.
- Per player for this slice: `printing`, `trailPointCount`, `lastTrailPoint`,
  `bodyNum`, and `bodyCount`.
- Per frame: `worldBodyCount`.
- Ordered events when `comparison.include_events` is true: at least `position`,
  `point`, `property`, `die`, `score:round`, `score`, and `round:end`.
- For later gap collision, add selected world body summaries if a count and
  death event are not enough to explain a mismatch: owner id, `num`, `x`, `y`,
  and radius.

Avoid comparing browser pixels for this slice. State and event parity should
prove gameplay first. Pixel or replay checks can come later to confirm visual
rendering.

## Risks

- The source loop is reverse player order. A higher-index avatar can emit a
  point that kills a lower-index avatar later in the same `Game.update(step)`.
- Normal point insertion happens before collision. Print-manager toggles happen
  after collision.
- Setup can accidentally call `printManager.start()` and insert an important
  point if a fixture sets `printing: true` without forced trail or print-manager
  runtime state. The first two fixtures should force `trail.points`.
- `Trail.clear()` clears only trail cursor/visual state. It does not remove
  world bodies.
- `worldBodyCount` is the source `World.bodyCount` monotonic insert counter. In
  these fixtures there is no removal, so it also equals inserted body count.
- Exact-threshold floating-point cases are useful later, but not as the first
  cadence fixtures.
