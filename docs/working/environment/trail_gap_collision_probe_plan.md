# Trail Gap Collision Probe Plan

Status: all three narrow gap fixtures verified through `source-trail-gap-canary`,
2026-05-09.

- Goal: after normal trail cadence parity lands, pin the server gameplay meaning
  of gaps: no normal point bodies inside a hole, stored bodies still collide, and
  print-to-hole boundary points are immediate world bodies.
- Use three players for all first gap/collision fixtures. Player array order is
  `p0`, `p1`, `p2`; source update order is therefore `p2`, `p1`, `p0`.
- Make `p1` the gap owner and `p0` the crossing victim/control. This avoids
  same-avatar trail latency hiding a body from its owner.
- Use a 3P map-95 setup, `step_ms: 100`, no bonuses, active world, started game,
  and fixed `Math.random() = 0.5`. At base speed `16`, each straight step moves
  `1.6` units and default collision radius is `0.6`.
- Common geometry: `p1` starts at `(40,40)`, heading `0`, and reaches
  `(41.6,40)`. `p0` starts at `(43.2,40)`, heading `pi`, and also reaches
  `(41.6,40)`. `p2` starts at `(80,20)`, heading `pi`, and only keeps the round
  non-terminal when `p0` dies.
- Implemented in this order: forced-hole safe crossing, stored-body kill in
  visual hole, then print-to-hole boundary same-frame kill.
- Current runner note: new `source_trail_gap_*` scenario ids need explicit
  `source-trail-gap-canary` allowlist entries before Python/common-trace parity
  is meaningful.

Source and runner files read:

- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/model/Avatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseTrail.js`
- `third_party/curvytron-reference/src/server/model/Trail.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/core/Island.js`
- `third_party/curvytron-reference/src/server/core/AvatarBody.js`
- `third_party/curvytron-reference/src/server/core/Body.js`
- `third_party/curvytron-reference/src/shared/Collection.js`
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`
- `tools/reference_oracle/scenario_runner.js`
- `src/curvyzero/fidelity/source_runners.py`
- `src/curvyzero/env/scenario_schema.py`
- `src/curvyzero/env/trace_compare.py`
- `tools/run_fidelity_batch.py`
- `tools/run_fidelity_loop.py`
- Existing source trail, body, and print-manager scenario JSON fixtures.

## Fixture 1: Forced Hole Interior Is Safe

Name: `source_trail_gap_hole_space_safe_step`

Status: verified through `source-trail-gap-canary`.

Purpose:

- Prove a higher-index avatar in a hole does not emit a normal same-frame body
  at its endpoint, so a lower-index player can occupy that endpoint safely.
- Keep one stored body at the gap start to prove the fixture is not just an empty
  world.

Setup:

- `player_count: 3`, `source_setup.map_size: 95`, `game.started: true`,
  `in_round: true`, `borderless: false`, `world_active: true`.
- One tick with `step_ms: 100`; all moves are `0`.
- `p0`: initial `(43.2,40)`, `angle_rad: pi`, `printing: false`,
  `trail.points: []`.
- `p1`: initial `(40,40)`, `angle_rad: 0`, `printing: false`,
  `trail.points: []`, `body_count: 1`,
  `print_manager: {active: true, distance: 10, last_x: 40, last_y: 40}`.
- `p2`: initial `(80,20)`, `angle_rad: pi`, `printing: false`,
  `trail.points: []`.
- `initial_state.world_bodies`: one p1 body at `(40,40)`, radius `0.6`, num `0`.

Expected events:

- `position(p2, 78.4, 20)`
- `position(p1, 41.6, 40)`
- `position(p0, 41.6, 40)`
- No `point`, `property`, `die`, score, or `round:end` events.

Expected final state:

- All players alive.
- `worldBodyCount: 1`.
- `p1.printing: false`, `p1.trailPointCount: 0`,
  `p1.lastTrailPoint: null`, `p1.bodyNum: 1`, `p1.bodyCount: 1`.
- `p1.printManager.active: true`, `distance: 8.4`,
  `lastX: 41.6`, `lastY: 40`.
- `p0.trailPointCount: 0`, `p0.bodyNum: 0`, `p0.bodyCount: 0`.

What it proves:

- Hole interior has no normal per-radius point body when `printing` is false.
- The live p1 head is not itself a collision body.
- Endpoint distance from the gap-start stored body is `1.6`, safely greater than
  radius sum `1.2`; this is not a tangent probe.

## Fixture 2: Stored Body Still Kills In A Visual Hole

Name: `source_trail_gap_stored_body_still_kills_step`

Status: verified through `source-trail-gap-canary`.

Purpose:

- Prove `Trail.clear()` and `printing: false` do not remove stored world bodies.
- Use p0 as the crossing player so p1's own-body latency cannot hide the body.

Setup:

- Same common setup as fixture 1.
- `p1` again starts in a forced hole with active manager distance `10` and
  `body_count: 1`.
- `initial_state.world_bodies`: one p1 body at `(41.6,40)`, radius `0.6`,
  num `0`.

Expected events:

- `position(p2, 78.4, 20)`
- `position(p1, 41.6, 40)`
- `position(p0, 41.6, 40)`
- `point(p0, 41.6, 40, important=false)` from the death side effect.
- `die(p0, killer=p1, old=false)`
- `score:round(p0, score=0, roundScore=0)`
- No `round:end`, because p1 and p2 remain alive.

Expected final state:

- `p0.alive: false`; `p1.alive: true`; `p2.alive: true`.
- `worldBodyCount: 2`: seeded p1 body plus p0 death point body.
- `p1.printing: false`, `p1.trailPointCount: 0`,
  `p1.lastTrailPoint: null`, `p1.bodyNum: 1`, `p1.bodyCount: 1`.
- `p1.printManager.active: true`, `distance: 8.4`,
  `lastX: 41.6`, `lastY: 40`.
- `p0.trailPointCount: 1`, `p0.lastTrailPoint: [41.6, 40]`,
  `p0.bodyNum: 0`, `p0.bodyCount: 1`.

What it proves:

- A visually cleared or forced-empty trail does not imply an empty collision
  world.
- A stored p1 body can be ignored by p1 through owner latency and still kill p0.
- `old=false` is expected here; this fixture does not prove wall-clock old-body
  age.

## Fixture 3: Print-To-Hole Boundary Body Kills Later Player

Name: `source_trail_gap_print_to_hole_boundary_kills_step`

Status: verified through `source-trail-gap-canary`.

Purpose:

- Prove the transition into a hole, not just a forced hole state.
- Prove the important boundary point from `setPrinting(false)` is inserted into
  the world before the lower-index avatar updates, even though the visual trail
  cursor is cleared.

Setup:

- Same common 3P geometry and one tick.
- No seeded `world_bodies`.
- `p1`: initial `(40,40)`, `angle_rad: 0`, `printing: true`,
  `trail: {points: [], last_x: 41.6, last_y: 40}` to suppress a normal
  `Avatar.update` point at the endpoint,
  `print_manager: {active: true, distance: 1, last_x: 40, last_y: 40}`.
- `p0`: initial `(43.2,40)`, `angle_rad: pi`, `printing: false`,
  `trail.points: []`.
- `p2`: initial `(80,20)`, `angle_rad: pi`, `printing: false`,
  `trail.points: []`.

Expected events:

- `position(p2, 78.4, 20)`
- `position(p1, 41.6, 40)`
- `point(p1, 41.6, 40, important=true)`
- `property(p1, printing=false)`
- `position(p0, 41.6, 40)`
- `point(p0, 41.6, 40, important=false)`
- `die(p0, killer=p1, old=false)`
- `score:round(p0, score=0, roundScore=0)`

Expected final state:

- `p0.alive: false`; `p1.alive: true`; `p2.alive: true`.
- `worldBodyCount: 2`: p1's boundary body plus p0's death point body.
- `p1.printing: false`, `p1.trailPointCount: 0`,
  `p1.lastTrailPoint: null`, `p1.bodyNum: 0`, `p1.bodyCount: 1`.
- `p1.printManager.active: true`, `distance: 5.25`,
  `lastX: 41.6`, `lastY: 40`.
- `p0.trailPointCount: 1`, `p0.lastTrailPoint: [41.6, 40]`,
  `p0.bodyNum: 0`, `p0.bodyCount: 1`.

What it proves:

- `PrintManager.test()` runs after p1's collision check, toggles printing, and
  `setPrinting(false)` emits an important point before the property event.
- `Trail.clear()` leaves p1 with no visual trail points, but the just-emitted
  boundary body remains in the world and can kill p0 later in the same
  `Game.update`.

## Traps And Non-Claims

- Do not test hole crossing with the owner alone; own-body latency can hide a
  real stored body.
- Do not infer collision body absence from `trailPointCount: 0`; always assert
  `worldBodyCount`, ordered point events, and second-player survival/death.
- Do not claim swept collision. These fixtures use endpoint checks, matching the
  source `Game.update` behavior.
- Do not claim exact tangent behavior here. Existing body canaries cover strict
  overlap and tangent safety.
- Do not let fixture 3 emit a normal p1 point before the print-manager toggle;
  the forced trail cursor must be the post-step endpoint.
- Do not expect a print-manager toggle after an avatar dies. `PrintManager.test`
  only runs if the avatar is still alive after border/body collision.
- Do not rely on `source_setup.map_size` to force the JS map size. The JS oracle
  derives size from player count; three players gives source size `95`, so keep
  Python setup aligned to `95`.
- Do not assert a trail `clear` event unless the oracle starts recording trail
  emitter events. For now, assert cleared trail state.
- Treat `worldBodyCount` as the source world insert counter. There is no removal
  in these fixtures, so it also equals inserted bodies here.
