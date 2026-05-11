# Reference Oracle

Status: Draft

This page defines the practical headless oracle we want for comparing CurvyZero to the
original CurvyTron JS source.

## Goal

Build a minimal Node-based JS oracle that loads the original source, steps game objects
directly, and emits a simple state trace. The trace should compare cleanly against the
Python environment.

The oracle must support multiplayer cases, starting with 2 players and then 3-4 players.

Supporting research: `docs/research/environment/curvytron_js_state_oracle.md`.

Full server/browser hosting is deferred. The near-term path is direct object stepping:
`new Game(room)`, forced state, staged inputs, `game.update(step_ms)`, JSON trace.

## Source Of Truth

Use server mechanics as the rule source:

- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/model/Avatar.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/core/Island.js`
- `third_party/curvytron-reference/src/server/core/AvatarBody.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`

Use client files only to confirm input mapping. Browser protocol and rendered-game checks
are later work.

## Oracle Source-Frame Contract

A single oracle source frame is:

1. Start from a fully specified JS game state.
2. Stage one current control value for each living player: `-1`, `0`, or `1`.
3. Call `game.update(step_ms)`.
4. Capture state and source events after the call.

The input values mean:

- `-1`: left
- `0`: straight, neither, or both buttons
- `1`: right

Apply all staged player control values before calling `game.update(step_ms)`.
The server has no explicit joint action object; it reads current control state
during an elapsed-ms frame.

## Trace Fields

Each trace row should include:

- scenario id
- player count
- tick index
- `step_ms`
- map size
- round flags: `started`, `inRound`, `borderless`
- deaths count and death order
- round winner and game winner
- source events for this tick, if recorded

For each player:

- id
- x, y
- angle
- move input used this tick
- velocity and angular velocity
- radius
- alive and present
- printing flag
- score and round score
- trail point count and last trail point
- body number and body count
- active bonuses

For collision checks:

- border hit or wrap
- hit body owner, if any
- killer id, if any
- old-body flag, if available
- bonus caught, if any

## Events To Record

Record these source events when they occur. This is not a browser-hosting requirement;
the probe can attach listeners or use a tiny `GameController` stub.

- `position`
- `angle`
- `point`
- `die`
- `score`
- `score:round`
- `property`
- `bonus:pop`
- `bonus:clear`
- `bonus:stack`
- `round:new`
- `round:end`
- `clear`
- `borderless`

State traces are the priority. Compressed browser payloads from `GameController` are
deferred until browser interoperability matters.

## Multiplayer Cases

Do not stop at 1v1. The first oracle suite should cover 2, 3, and 4 players.

### 2 Players

Use 2-player cases to lock the basic source-frame contract:

- Map size is `88`.
- Force two spawn positions and headings.
- Apply same-frame left/right/straight control values.
- Check straight motion, constant turn motion, and wall death.
- Check opponent trail hit.
- Check same-frame double death.
- Check head-head or near-head behavior with a note that JS update order matters.
- Check per-player `position`, `angle`, `die`, `score:round`, and `round:end` events.

### 3 Players

Use 3-player cases to expose ordering and scoring bugs:

- Map size is `95`.
- Compare forced spawn layout first; seeded random spawn can come later.
- Apply three inputs before the tick.
- Kill one player in an earlier tick.
- Then kill two players in the same later tick.
- Both same-frame deaths should get the prior death count as round score.
- If one player survives, the survivor receives `2` extra round points at round end.
- Compare all per-player events, not just the winner.

### 4 Players

Use 4-player cases for larger death-order and score checks:

- Map size is `101`.
- Compare forced spawn layout first.
- Apply four inputs before the tick.
- Create one case with two prior deaths, then one new death.
- Create one case with two prior deaths, then two same-frame deaths.
- New deaths after two prior deaths should receive round score `2`.
- The final survivor receives `3` extra round points at round end.
- Compare death order, score events, and final resolved scores.

## Spawn Policy

The reference spawn code uses `Math.random`. For repeatable tests:

1. Prefer forced positions and angles for physics goldens.
2. Add seeded `Math.random` tests only after the forced-state oracle works.
3. For seeded spawn tests, compare:
   - map size
   - spawn positions
   - spawn angles
   - no initial overlap
   - border margin
   - direction validity

This avoids making all early physics tests depend on random-stream details.

## Scoring Policy

Compare scoring in two phases:

1. While the round continues, compare `roundScore`.
2. If the tick ends the round, compare `score:round` events and final `score`, because
   `resolveScores()` runs during that same update.

Important source behavior:

- `Game.update()` captures `deaths.count()` once at frame start.
- Every death in that frame gets that same captured value.
- Same-frame deaths therefore share score.
- The final survivor gets `max(players - 1, 1)` extra round score.

This is especially important for 3-4 player tests.

## Practical Build Plan

1. Make a Node script that loads CurvyTron server-side source files into a `vm` context.
2. Skip network-only files such as `Server.js` and `launcher.js`.
3. Stub controller pieces with event recorders.
4. Construct fake rooms and fake players.
5. Disable bonuses for the first oracle suite.
6. Force state directly instead of starting timers.
7. Call `avatar.updateAngularVelocity(move)` for each player.
8. Call `game.update(step_ms)`.
9. Emit JSON state trace rows.
10. Add Python-side comparison code later.

Exact JS calls the probe should use:

```js
var game = new Game(room);
game.world.activate();
avatar.setPosition(x, y);
avatar.setAngle(angle);
avatar.updateVelocities();
avatar.updateAngularVelocity(move);
game.update(step_ms);
```

When trail collision is needed:

```js
avatar.printManager.start();
// For a normal game object, avatar.addPoint(...) emits `point`,
// and Game.onPoint adds the AvatarBody to game.world.
avatar.addPoint(x, y, true);
```

For manual seeding, add the body directly instead:

```js
game.world.addBody(new AvatarBody(x, y, avatar));
```

When round scoring is needed:

```js
// If game.update(...) caused the terminal death, checkRoundEnd()
// and resolveScores() already ran.

// If a fixture bypassed game.update(...) and changed alive flags directly:
game.checkRoundEnd();
```

Keep the oracle script small. It should call the original functions and avoid copying
physics rules into new JS code.

## First Acceptance Set

The first useful oracle is done when it can emit JSON traces for:

1. 2-player straight movement.
2. 2-player same-frame turns.
3. 2-player wall death and score.
4. 2-player same-frame double death.
5. 3-player prior-death plus same-frame deaths.
6. 4-player death-order and final-survivor score.

Each trace should include per-player state and source events.
