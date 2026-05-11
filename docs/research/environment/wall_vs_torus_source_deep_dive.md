# Wall Vs Torus Source Deep Dive

Status: source deep dive
Date: 2026-05-08
Scope: `third_party/curvytron-reference`

## Short Answer

The reference game is not normally toroidal.

Normal play uses a square arena with hard walls. A live avatar dies when its circular body crosses the map edge. The "borderless" behavior is a timed game bonus, `BonusGameBorderless`, not the default rule. When active, it teleports the avatar to the opposite edge instead of killing it.

Even in borderless, the source is not a clean mathematical torus. It checks the avatar center against the border, throws away overshoot, wraps only one axis per check, and skips body collision for that avatar on the wrap frame.

## Files And Functions

### Arena Size

- `src/shared/model/BaseGame.js:14` sets `this.size` from `getSize(this.avatars.count())`.
- `src/shared/model/BaseGame.js:46` sets `perPlayerSize = 80`.
- `src/shared/model/BaseGame.js:230-235` computes map size as:
  `round(sqrt(80^2 + ((players - 1) * 80^2 / 5)))`.
- `src/server/model/Game.js:208-215` rebuilds the server `World` when size changes.
- `src/server/core/World.js:4-19` builds a square world of that size.

Expected source sizes:

| Players | Size |
| --- | ---: |
| 1 | 80 |
| 2 | 88 |
| 3 | 95 |
| 4 | 101 |

### Normal Wall Death

- `src/shared/model/BaseGame.js:81` defines `borderless = false`.
- `src/shared/model/BaseGame.js:170-174` resets `borderless` to false at each new round.
- `src/server/model/Game.js:37-80` is the authoritative update loop.
- `src/server/model/Game.js:51` checks border collision with:
  `this.world.getBoundIntersect(avatar.body, this.borderless ? 0 : avatar.radius)`.
- In normal mode, that margin is `avatar.radius`.
- `src/server/model/Game.js:53-59` kills the avatar when a border hit exists and `borderless` is false.
- `src/server/core/World.js:276-295` implements the border check.

Normal wall predicate:

- west wall: `body.x - radius < 0`
- east wall: `body.x + radius > world.size`
- north wall: `body.y - radius < 0`
- south wall: `body.y + radius > world.size`

Equality is safe. The code uses `< 0` and `> this.size`, not `<=` or `>=`.

Wall collision has priority over trail/body collision. In `src/server/model/Game.js:53-68`, body collision is only checked in the `else` branch when there was no border hit.

### Trail And Body Collision Near Walls

- `src/server/model/Game.js:61-66` checks `world.getBody(avatar.body)` only after the wall check passes.
- `src/server/core/World.js:104-126` looks for one colliding body by checking islands around the four corners of the current body's bounding box.
- `src/server/core/Island.js:83-90` uses strict circle overlap:
  `distance < bodyA.radius + bodyB.radius && match`.
- `src/server/core/AvatarBody.js:33-40` ignores recent own trail points until:
  `current_body.num - stored_body.num > avatar.trailLatency`.
- `src/shared/model/BaseAvatar.js:61` sets `trailLatency = 3`.

So normal wall death is not a special body in the world. It is a separate edge predicate that runs before trail collision.

## Borderless Bonus

### How It Turns On

- `src/shared/model/BaseRoomConfig.js:17-30` enables `BonusGameBorderless` by default with the other bonuses.
- `src/server/model/RoomConfig.js:19-31` maps `BonusGameBorderless` to the server bonus class.
- `src/client/model/preset/DefaultPreset.js:24-37` includes it in the "All" preset.
- `src/client/model/preset/SoloPreset.js:24-30` includes it in the "Solo" preset.
- `src/client/model/preset/EmptyPreset.js:11-17` has no bonuses, so no borderless bonus.
- `src/server/model/Bonus/BonusGameBorderless.js:20` sets duration to `10000` ms.
- `src/server/model/Bonus/BonusGameBorderless.js:27` sets probability to `0.8`.
- `src/server/model/Bonus/BonusGameBorderless.js:36-40` returns the effect `['borderless', true]`.
- `src/server/model/Bonus/Bonus.js:22-30` applies a bonus and schedules `off()` after its duration.
- `src/server/model/Bonus/BonusGame.js:40-54` adds/removes the game bonus from `game.bonusStack`.
- `src/server/model/GameBonusStack.js:40-45` applies `borderless` through `game.setBorderless(...)`.
- `src/server/model/Game.js:297-302` emits the `borderless` event when the flag changes.

### How Wrapping Works

When borderless is active:

- `src/server/model/Game.js:51` calls `getBoundIntersect` with margin `0`.
- That means the avatar wraps only after its center crosses outside `[0, size]`.
- `src/server/model/Game.js:53-57` calls `world.getOposite(...)` and `avatar.setPosition(...)`.
- `src/server/core/World.js:276-295` returns a border point such as `[0, body.y]` or `[size, body.y]`.
- `src/server/core/World.js:305-323` maps that border point to the opposite edge.

Examples:

- if `x < 0`, `getBoundIntersect` returns `[0, y]`, then `getOposite` returns `[size, y]`.
- if `x > size`, it returns `[size, y]`, then wraps to `[0, y]`.
- if `y < 0`, it wraps to `[x, size]`.
- if `y > size`, it wraps to `[x, 0]`.

Important details:

- Overshoot is lost. `x = size + 0.2` becomes exactly `x = 0`, not `x = 0.2`.
- Exact edge equality is safe. A center at exactly `0` or exactly `size` does
  not wrap because the source checks strict `<` and `>`.
- Corners are not true torus behavior. `World.getBoundIntersect` checks x before y, so a diagonal exit wraps one axis first.
- The wrap frame skips trail/body collision for that avatar, because `world.getBody(...)` is in the `else` branch after the border check in `src/server/model/Game.js:60-66`.
- If borderless is active but the avatar center has not crossed the edge, normal body collision still runs.
- The avatar can have part of its body outside the arena during borderless mode, because the margin is `0` instead of `avatar.radius`.

### Trail Side Effects On Wrap

The source has extra behavior around trails during a wrap:

- `src/server/model/Avatar.js:23-31` updates position and may add a trail point before `Game.update` checks the border.
- `src/server/model/Game.js:113-117` turns every emitted point into an `AvatarBody` while the world is active.
- `src/server/model/Game.js:70-72` runs `avatar.printManager.test()` and bonus catch after wrapping if the avatar is still alive.
- `src/server/manager/PrintManager.js:90-100` subtracts straight-line distance from its last point to the current avatar position.

That means a borderless teleport can look like a very large movement to `PrintManager`. It may toggle printing immediately after wrap.

The browser client has a visual guard:

- `src/client/model/Trail.js:58-66` clears the visible trail segment when a new point is more than tolerance `1` away from the last point on either axis.

This prevents a long visible line across the arena in many wrap cases, but it is client rendering logic. The server wrap rule is still the source of truth.

## Normal Mode Vs Bonus Mode

Normal mode:

- square bounded arena
- `borderless = false`
- edge check uses `avatar.radius`
- crossing a wall kills the avatar
- trail/body collision only happens if no wall collision happened

Borderless bonus mode:

- temporary game-wide flag
- enabled by `BonusGameBorderless`
- lasts `10000` ms unless stacked/removed through bonus stack behavior
- edge check uses margin `0`
- crossing by center teleports to the opposite edge
- no body collision is checked on the wrap frame
- not a full torus because overshoot and diagonal wrapping are not preserved

## Are Our Docs Wrong?

Mostly no. The current docs are directionally right, but some are too short.

Correct source summaries:

- `docs/research/curvytron_reference_notes.md:46` says normal border collision kills and borderless wraps.
- `docs/research/environment/curvytron_js_state_oracle.md:199-200` says normal borders use avatar radius and borderless uses margin `0`.
- `docs/design/deterministic_environment.md:54` says normal border kills and borderless wraps.

Oversimplifications:

- Those docs do not always say that borderless is a timed game bonus, not the normal mode.
- They usually say "wrap" without saying it is center-based, one-axis-at-a-time, and loses overshoot.
- They do not mention that the wrap frame skips body collision.
- They do not mention the print-manager distance side effect after teleport.

I found no current durable doc that says the reference default is toroidal. If any future note says "CurvyTron is torus by default," that would be wrong for this source. The exact statement should be: default CurvyTron has walls; the `BonusGameBorderless` game bonus temporarily changes wall death into source-specific edge teleporting.
