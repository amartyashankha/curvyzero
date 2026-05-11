# CurvyTron Collisions, Trails, And World Source Map

Status: source deep dive, 2026-05-08

Scope: server source behavior for `World`, `Island`, `Body`, `AvatarBody`,
`PrintManager`, `Trail`, and `Game.onPoint/update`. Client trail rendering is
not the rule source, but it is noted where it explains visual gaps.

## Main Findings

- The server uses endpoint circle collision, not swept collision. Each tick calls
  `avatar.update(step)`, which moves the avatar and may emit a trail point, then
  checks the border and stored world bodies. See `Game.prototype.update` in
  `third_party/curvytron-reference/src/server/model/Game.js:37-80`.
- Normal borders kill. Borderless mode wraps after the body center crosses the
  arena edge. The checks are strict `<` and `>`, so exact equality is safe. See
  `World.prototype.getBoundIntersect` in
  `third_party/curvytron-reference/src/server/core/World.js:276-295`.
- Bodies are circles. A collision needs `distance < radiusA + radiusB`; exact
  tangent distance is safe. See `Island.prototype.bodiesTouch` in
  `third_party/curvytron-reference/src/server/core/Island.js:83-90`.
- Same-avatar collision is delayed by trail point number, not wall-clock time:
  a stored own body only matches when `currentBody.num - storedBody.num > 3`.
  See `AvatarBody.prototype.match` in
  `third_party/curvytron-reference/src/server/core/AvatarBody.js:33-40` and
  `BaseAvatar.prototype.trailLatency` in
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:56-62`.
- `AvatarBody.oldAge = 2000` only feeds the emitted death event's `old` flag. It
  does not decide whether a body collides. See
  `third_party/curvytron-reference/src/server/core/AvatarBody.js:19-24`,
  `third_party/curvytron-reference/src/server/core/AvatarBody.js:47-50`, and
  `Avatar.prototype.die` in
  `third_party/curvytron-reference/src/server/model/Avatar.js:180-188`.
- Trail holes are distance-based. `PrintManager` alternates printed distance and
  hole distance, with random multipliers, and toggles at `distance <= 0`. See
  `third_party/curvytron-reference/src/server/manager/PrintManager.js:17-29`,
  `third_party/curvytron-reference/src/server/manager/PrintManager.js:53-60`,
  and `third_party/curvytron-reference/src/server/manager/PrintManager.js:90-103`.

## Game Tick And Point Storage

`Game` creates one `World`, binds `onPoint`, clears each avatar, and subscribes
to each avatar's `point` event in
`third_party/curvytron-reference/src/server/model/Game.js:6-27`.

`Game.prototype.update(step)` runs avatars in reverse collection order. For each
alive avatar it:

1. Calls `avatar.update(step)`, which may emit a normal trail point before the
   collision checks.
2. Calls `world.getBoundIntersect(avatar.body, this.borderless ? 0 : avatar.radius)`.
3. If there is a border hit, borderless mode calls `world.getOposite(...)` and
   `avatar.setPosition(...)`; normal mode calls `kill(...)`.
4. If there is no border hit and the avatar is not invincible, calls
   `world.getBody(avatar.body)` and kills on a match.
5. If the avatar is still alive, calls `avatar.printManager.test()` and then
   `bonusManager.testCatch(avatar)`.

The source lines are
`third_party/curvytron-reference/src/server/model/Game.js:37-80`.

Point timing matters. Normal distance-spaced trail points come from
`Avatar.prototype.update` before border/body collision:
`third_party/curvytron-reference/src/server/model/Avatar.js:23-33`. Hole toggles
from `PrintManager.prototype.test` happen only after the avatar survives:
`third_party/curvytron-reference/src/server/model/Game.js:70-73` and
`third_party/curvytron-reference/src/server/manager/PrintManager.js:90-103`.
Because `point` events are synchronous, a point emitted by one avatar can be in
the world before later avatars in the same reverse-order `Game.update` loop.

`Game.prototype.onPoint(data)` is the only normal trail-to-world bridge. If the
game is started and the world is active, each emitted point becomes a new
`AvatarBody` and is inserted into the world:
`third_party/curvytron-reference/src/server/model/Game.js:113-118`.

The world becomes active in `Game.prototype.onStart`, after scheduling
`avatar.printManager.start` for 3000 ms later:
`third_party/curvytron-reference/src/server/model/Game.js:257-268`.
`Game.prototype.clearTrails` clears world bodies, reactivates the world, and emits
`clear`: `third_party/curvytron-reference/src/server/model/Game.js:198-203`.

## Borders And Borderless

Default `borderless` is false in
`third_party/curvytron-reference/src/shared/model/BaseGame.js:76-82`. A new round
resets it to that default in
`third_party/curvytron-reference/src/shared/model/BaseGame.js:167-181`.
`BaseGame.prototype.setBorderless` coerces to boolean in
`third_party/curvytron-reference/src/shared/model/BaseGame.js:294-302`, and the
server `Game.prototype.setBorderless` emits the change in
`third_party/curvytron-reference/src/server/model/Game.js:292-303`.

Normal mode passes `avatar.radius` as the border margin, so the avatar dies when
its circle crosses the square. Borderless mode passes margin `0`, so the center
must cross the square before wrapping. This branch is in
`third_party/curvytron-reference/src/server/model/Game.js:51-58`.

`World.prototype.getBoundIntersect` checks left, right, top, bottom in that order:
`third_party/curvytron-reference/src/server/core/World.js:276-295`. A diagonal
corner exit therefore resolves one axis first, not a true two-axis torus step.
`World.prototype.getOposite` maps `[0, y]` to `[size, y]`, `[size, y]` to
`[0, y]`, `[x, 0]` to `[x, size]`, and `[x, size]` to `[x, 0]`:
`third_party/curvytron-reference/src/server/core/World.js:297-324`.

Borderless mode is a teleport rule. Collision lookup does not look across
opposite edges as if the arena were a continuous torus.

## World Grid, Islands, And Body Storage

`World(size, islands)` builds a square spatial grid. If `islands` is not passed,
it uses `Math.round(size / 40)`, where `40` is `islandGridSize`. It stores islands
in a `Collection` under IDs like `x:y`:
`third_party/curvytron-reference/src/server/core/World.js:4-20` and
`third_party/curvytron-reference/src/server/core/World.js:22-28`.

`World.prototype.getIslandByPoint` maps a point to
`floor(point / islandSize)` on each axis:
`third_party/curvytron-reference/src/server/core/World.js:29-44`.

`World.prototype.addBody` only works when `world.active` is true. It assigns a
monotonic world body ID, then inserts the body by the four corners of its circle
bounding box:
`third_party/curvytron-reference/src/server/core/World.js:46-63`.
`addBodyByPoint` ignores out-of-world points because missing islands return null:
`third_party/curvytron-reference/src/server/core/World.js:65-79`.

Each `Island` owns a `Collection` of bodies:
`third_party/curvytron-reference/src/server/core/Island.js:9-18`. Adding a body to
an island also records the island on the body:
`third_party/curvytron-reference/src/server/core/Island.js:20-30`. The base
`Body` shape is just `x`, `y`, `radius`, `data`, `islands`, and `id`:
`third_party/curvytron-reference/src/server/core/Body.js:9-17`.
If two bounding-box corners land in the same island, `Collection.add` prevents a
duplicate body entry for that island:
`third_party/curvytron-reference/src/shared/Collection.js:53-79`.

Collision lookup mirrors insertion. `World.prototype.getBody` checks the four
corners of the query body's bounding box and returns the first colliding body:
`third_party/curvytron-reference/src/server/core/World.js:97-126`. `Island` first
checks whether the query body overlaps the island bounds, then scans bodies in
that island:
`third_party/curvytron-reference/src/server/core/Island.js:62-73` and
`third_party/curvytron-reference/src/server/core/Island.js:103-109`.

`World.prototype.clear` deactivates the world, resets `bodyCount`, and clears all
islands: `third_party/curvytron-reference/src/server/core/World.js:354-365`.
`World.prototype.activate` only sets `active = true`:
`third_party/curvytron-reference/src/server/core/World.js:367-373`.

## Circle Overlap And Self Delay

All stored avatar trail bodies are `AvatarBody` instances. The constructor uses
the avatar's current radius, stores the avatar in `data`, assigns a per-avatar
`num`, and records a wall-clock birth time:
`third_party/curvytron-reference/src/server/core/AvatarBody.js:8-14`.

The live avatar body is created in `Avatar(player)`:
`third_party/curvytron-reference/src/server/model/Avatar.js:6-13`. Every
`Avatar.prototype.setPosition` keeps the live body at the avatar's current
position and sets `body.num = bodyCount`:
`third_party/curvytron-reference/src/server/model/Avatar.js:55-64`.

For self-collision, `Island.prototype.bodiesTouch(storedBody, currentBody)` calls
`storedBody.match(currentBody)`. `AvatarBody.prototype.match` ignores same-avatar
bodies until `currentBody.num - storedBody.num > avatar.trailLatency`:
`third_party/curvytron-reference/src/server/core/Island.js:83-90`,
`third_party/curvytron-reference/src/server/core/AvatarBody.js:33-40`, and
`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:56-62`.

Other-avatar bodies match immediately. Plain `Body.prototype.match` always
returns true:
`third_party/curvytron-reference/src/server/core/Body.js:19-29`.

## Trail Printing And Holes

`Avatar.prototype.update(step)` updates angle and position. If `printing` is true
and `isTimeToDraw()` is true, it adds a point:
`third_party/curvytron-reference/src/server/model/Avatar.js:18-33`.

`Avatar.prototype.isTimeToDraw` emits the first point immediately when the trail
has no `lastX`, then emits later points only when distance from the last trail
point is greater than the avatar radius:
`third_party/curvytron-reference/src/server/model/Avatar.js:35-47`.

`Avatar.prototype.addPoint` calls `BaseAvatar.addPoint` and emits a `point` event:
`third_party/curvytron-reference/src/server/model/Avatar.js:151-162`.
`BaseAvatar.addPoint` appends to the trail:
`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:108-117`.
`BaseTrail.addPoint` stores `[x, y]` and updates `lastX/lastY`:
`third_party/curvytron-reference/src/shared/model/BaseTrail.js:19-30`.

`PrintManager` has base `printDistance = 60` and `holeDistance = 5`:
`third_party/curvytron-reference/src/server/manager/PrintManager.js:17-29`.
When printing, the random distance is `60 * (0.3 + random * 0.7)`. When not
printing, it is `5 * (0.8 + random * 0.5)`:
`third_party/curvytron-reference/src/server/manager/PrintManager.js:53-60`.

`PrintManager.start` activates, records the avatar's current position, and turns
printing on:
`third_party/curvytron-reference/src/server/manager/PrintManager.js:62-73`.
`PrintManager.test` subtracts straight-line distance from its last sampled point
to the current avatar position, updates `lastX/lastY`, and toggles when the
remaining distance is `<= 0`:
`third_party/curvytron-reference/src/server/manager/PrintManager.js:87-103`.
Overshoot is not carried into the next print or hole distance because
`setPrinting` replaces `distance` with a fresh random distance:
`third_party/curvytron-reference/src/server/manager/PrintManager.js:39-46`.

`BaseAvatar.prototype.setPrinting` always adds the current point when the
printing flag changes. If it turns printing off, it then clears the avatar's
trail object:
`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:292-310`.
On the server, `Trail.prototype.clear` clears `BaseTrail` and emits `clear`:
`third_party/curvytron-reference/src/server/model/Trail.js:12-19`.

Death has an extra print-manager side effect. `Avatar.prototype.die` calls
`BaseAvatar.prototype.die`, then `this.printManager.stop()`, then emits `die`:
`third_party/curvytron-reference/src/server/model/Avatar.js:178-188`.
For an active manager on an avatar that is still printing, this means the death
position first emits a normal non-important point from `BaseAvatar.die`; then
`PrintManager.stop` emits an important `setPrinting(false)` point and a
`property printing=false` event before the `die` event; then `PrintManager.clear`
resets `active`, `distance`, `lastX`, and `lastY` to zero. The promoted
`source_print_manager_active_stop_on_death_step` fixture pins that ordering.

That means a hole has no normal per-radius trail points inside it, but it does
have boundary point emissions when printing toggles off and on. Existing world
bodies are not removed by `avatar.trail.clear()`. They are removed only by world
clear paths such as `Game.prototype.clearTrails`.

Client rendering has an extra gap helper: client `Trail.prototype.addPoint`
clears the visible segment if the next point jumps by more than tolerance `1` on
either axis:
`third_party/curvytron-reference/src/client/model/Trail.js:16-22` and
`third_party/curvytron-reference/src/client/model/Trail.js:52-67`. That is visual
handling, not server collision logic.

## Scenarios And Probes Needed Next

1. Border edge probe.
   Check normal mode at exactly `radius`, just inside, and just outside each wall.
   Borderless exact-edge and corner-axis behavior is covered by
   `source_borderless_exact_edge_corner_axis_step`; add a next-frame second-axis
   check only if another feature needs it.

2. Circle overlap probe.
   Seed one stored body and move an avatar body to distances `radiusSum`,
   `radiusSum - epsilon`, and `radiusSum + epsilon`. This locks the strict
   `<` overlap rule and catches any accidental tangent kill.

3. Self-collision latency probe.
   Seed own `AvatarBody` values around the threshold: `current - stored == 3`
   should be safe; `current - stored == 4` should kill. Also force an old body
   by controlling time and confirm `old` changes only the death event flag.

4. Trail print toggle probe.
   Force `printManager.active`, `distance`, `lastX`, and `lastY`. Step less than,
   equal to, and past the remaining distance. Check `property` and `point` events,
   `world.bodyCount`, and that overshoot does not carry over.

5. Trail hole probe.
   Force printing off, move through a gap, then turn printing back on. Confirm no
   regular per-radius points are emitted inside the gap, but boundary points are
   emitted on each print-state change.

6. Borderless trail probe.
   Wrap while the print manager is active. Because `PrintManager.getDistance` uses
   straight-line distance in normal coordinates, a teleport can consume a large
   distance and toggle printing immediately after wrap. Confirm source behavior
   before matching it in Python.

7. World island probe. Done in `tests/test_source_env.py` through a vendored JS
   source probe. It confirms corner insertion across island boundaries,
   duplicate prevention through `Collection.add`, missing-island tolerance for
   out-of-world corners, and lookup by the query body's four corners.

8. Tick-order probe.
   Use two or more players with collisions in the same frame. Confirm reverse
   avatar iteration, border-before-body order, no print-manager test after death,
   shared frame-start death score, and whether a newly emitted trail point from
   one avatar can kill a later avatar in the same frame.

The current JS oracle already loads these source files and calls
`game.update(step_ms)` from scenarios:
`tools/reference_oracle/scenario_runner.js:18-44` and
`tools/reference_oracle/scenario_runner.js:448-474`. For the probes above, it
needs fixture knobs for manual body seeding, print-manager internals, and
controlled time.
