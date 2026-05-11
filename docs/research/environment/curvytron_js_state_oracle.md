# CurvyTron JS State Oracle Notes

Status: Draft research note

Scope: original server-side CurvyTron v1 code in `third_party/curvytron-reference`.
This note is about a practical headless state oracle, not hosting the original game.

## Short Answer

The smallest useful oracle target is direct JS object stepping:

1. Load the needed JS source files in a minimal Node script.
2. Create a fake room with 2-4 fake players.
3. Create `new Game(room)`.
4. Force a known round state: positions, angles, controls, trail bodies, bonuses.
5. Apply all player moves for the tick.
6. Call `game.update(step_ms)`.
7. Emit a state trace and any source events recorded during the step.

This can be headless. It does not need a browser. It should not need real WebSockets.
The main friction is that the source files are old-style globals, not CommonJS modules.
The future probe should load the needed files into a Node `vm` context in dependency
order. It can skip network-only server files such as `Server.js` and `launcher.js`.

Deferred: running the original web server, connecting the browser client, screenshot
checks, and full browser protocol comparison. Those can come later after direct state
traces are useful.

## Files That Matter

Authoritative server mechanics:

- `src/server/model/Game.js`
- `src/server/model/Avatar.js`
- `src/server/core/World.js`
- `src/server/core/Island.js`
- `src/server/core/AvatarBody.js`
- `src/server/core/Body.js`
- `src/server/manager/PrintManager.js`
- `src/server/manager/BonusManager.js`
- `src/server/model/Bonus/*.js`

Shared base behavior:

- `src/shared/model/BaseGame.js`
- `src/shared/model/BaseAvatar.js`
- `src/shared/model/BaseTrail.js`
- `src/shared/model/BaseRoom.js`
- `src/shared/model/BasePlayer.js`
- `src/shared/model/BaseRoomConfig.js`
- `src/shared/model/BaseBonus*.js`
- `src/shared/manager/BaseBonusManager.js`
- `src/shared/Collection.js`

Client input mapping, only if we need to confirm how buttons become move values:

- `src/client/model/PlayerInput.js`

Full client/server wire behavior is deferred.

## What One Physics Update Means

The server tick calls `Game.prototype.update(step)`.

`step` is elapsed time in milliseconds. The loop in `BaseGame.loop()` computes:

```js
var now = new Date().getTime();
var step = now - this.rendered;
this.rendered = now;
this.onFrame(step);
```

For a deterministic oracle, skip the timer loop and call `game.update(fixed_step_ms)`
directly.

Inside one `Game.update(step)`:

1. Capture `score = this.deaths.count()` once at frame start.
2. Set `deathInFrame = false`.
3. For each avatar, in reverse collection order:
   - If alive, call `avatar.update(step)`.
   - Check border collision.
   - If not border-hit and not invincible, check body collision through `world.getBody`.
   - If still alive, run `avatar.printManager.test()`.
   - If still alive, run `bonusManager.testCatch(avatar)`.
4. If any death happened, call `checkRoundEnd()`.

`Avatar.update(step)` does only local avatar physics:

1. Update angle from current `angularVelocity`.
2. Update position from current `velocityX` and `velocityY`.
3. If alive, printing, and far enough from last trail point, add a trail point.

Position math is:

```js
velocityX = Math.cos(angle) * velocity / 1000;
velocityY = Math.sin(angle) * velocity / 1000;
x += velocityX * step;
y += velocityY * step;
```

Turn math is:

```js
angularVelocity = move * angularVelocityBase;
angle += angularVelocity * step;
```

Base values:

- speed: `16` units per second
- turn base: `2.8 / 1000` radians per millisecond
- radius: `0.6`
- self-collision trail latency: `3`

## Controls

Client controls are left and right buttons.

From `PlayerInput`:

- left only -> `-1`
- right only -> `1`
- neither -> `false`, sent as `0`
- both -> `false`, sent as `0`

The client sends:

```js
client.addEvent('player:move', {avatar: avatar_id, move: move || 0});
```

The server applies:

```js
player.avatar.updateAngularVelocity(data.move);
```

There is no per-tick input queue inside `Game.update`. For an oracle tick, apply all
planned player moves first, then call `game.update(step_ms)`. That treats the inputs as
simultaneous at the tick boundary.

## Multiplayer Details

CurvyTron is multiplayer. Do not design the oracle as a 1v1-only check.

Map size depends on player count:

```js
round(sqrt(80 * 80 + ((players - 1) * 80 * 80 / 5)))
```

Useful expected sizes:

| Players | Map size |
| --- | ---: |
| 1 | 80 |
| 2 | 88 |
| 3 | 95 |
| 4 | 101 |

Spawn is random in `Game.onRoundNew()`:

- Iterate avatars in reverse order.
- Pick a free position with `world.getRandomPosition(radius, spawnMargin)`.
- Pick a direction with `world.getRandomDirection(x, y, spawnAngleMargin)`.
- `spawnMargin` is `0.05`.
- `spawnAngleMargin` is `0.3`.

For repeatable 2-4 player cases, either patch `Math.random` with a known stream or skip
random spawning and force exact avatar positions and angles.

Update order is reverse avatar order. This matters for multi-avatar collision fixtures.
Inputs should be staged for all players first, but the JS movement and collision checks
still happen one avatar at a time.

## Collision And Death

The collision world stores circular bodies in island buckets. `World.addBody()` inserts a
body into islands touched by the four corners of its bounding box. `World.getBody()` checks
the four corners of the current body.

A body collision requires:

```js
distance < bodyA.radius + bodyB.radius && bodyA.match(bodyB)
```

Equal distance is safe.

For same-avatar trail bodies, `AvatarBody.match()` ignores recent own trail points:

```js
current_body.num - stored_body.num > avatar.trailLatency
```

Normal borders kill when the avatar body crosses the map edge using avatar radius as the
margin. Borderless mode checks with margin `0` and wraps to the opposite side.

Head-head behavior should be tested carefully. The server does not keep every current head
in the world as a separate simultaneous object. A head can collide with trail bodies that
already exist, including bodies printed earlier in the same frame by avatars updated
earlier. So head-head or multiple-death cases are order-sensitive unless the fixture uses
pre-existing bodies or makes both deaths happen through walls/trails.

Death behavior:

- `Game.kill(avatar, killer, score)` calls `avatar.die(killer)`.
- Death clears bonus stack, marks `alive=false`, adds a final trail point, and stops
  printing.
- The dead avatar is added to `game.deaths`.
- `deathInFrame` becomes true.

## Scoring

At the start of a frame, `Game.update()` captures `score = deaths.count()`.

Each avatar killed in that frame gets that same score added to `roundScore`.

This means:

- First death of a round gets `0`.
- If one player is already dead, the next death gets `1`.
- If two players are already dead, the next death gets `2`.
- Deaths in the same frame share the prior death count.

At round end, the last alive avatar gets:

```js
Math.max(avatars.count() - 1, 1)
```

Then all avatars resolve `score += roundScore` and reset `roundScore` to `0`.

For multiplayer oracle cases, compare `score:round` events and `roundScore` while the
round continues. If the frame ends the round, `resolveScores()` runs during that same
update, so compare final `score` and the emitted score events after the tick.

## Trails And Prints

`PrintManager` creates distance-based trail gaps:

- printed section base distance: `60`
- hole base distance: `5`
- printed distance uses random multiplier `0.3` to `1.0`
- hole distance uses random multiplier `0.8` to `1.3`

`PrintManager.start()` turns printing on at the current position.

`PrintManager.test()` subtracts distance traveled since the previous print-manager test.
When the remaining distance reaches `<= 0`, it toggles printing.

`BaseAvatar.setPrinting()` always adds the current point. When printing turns off, the
visible trail object is cleared, but old bodies already added to the collision world stay
there.

For deterministic probes, force `printManager.active`, `printing`, `distance`, `lastX`,
and `lastY` rather than relying on timers and randomness.

## Bonuses

A minimal motion/collision oracle can disable bonuses with room config. If bonuses are in
scope, compare:

- active bonus map entries: id, x, y, radius, type
- per-player bonus stack entries
- changed fields: velocity, radius, inverse, invincible, printing, color
- game fields such as `borderless`
- emitted bonus source events

`BonusManager.testCatch(avatar)` checks the bonus-world body against the avatar body,
removes the bonus, then applies it.

## State Fields To Compare

Per game:

- player count
- map size
- in-round flag
- started flag
- borderless flag
- death count and death order
- round winner
- game winner
- active bonus list
- world body count and selected body summaries

Per avatar:

- id, name, color
- x, y
- angle
- velocity, velocityX, velocityY
- angularVelocity and angularVelocityBase
- radius
- alive and present
- printing and print-manager state
- inverse and invincible
- score and roundScore
- body x, y, radius, num
- bodyCount
- trail points, lastX, lastY
- bonus stack

Collision/event details:

- border hit or wrapped edge
- body hit, killer avatar id, and whether hit body is old
- trail point added this tick
- bonus caught this tick
- round ended this tick

Source events to record if the probe can attach listeners:

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
- `game:start`
- `game:stop`
- `end`

## Smallest Headless Probe Sketch

Recommended first probe shape:

```js
// 1. Create a minimal vm context with:
// EventEmitter, setTimeout, clearTimeout, setInterval, clearInterval, console.
// Do not load Server.js or launcher.js for this probe.

// 2. Load the minimum source files in dependency order:
// src/shared/Collection.js
// shared base models/managers/services used below
// server service/model/core/manager files used by Game and Avatar
// server/model/Game.js

// 3. Replace GameController with a tiny source-event recorder before constructing Game:
GameController = function(game) {
  this.game = game;
  this.events = [];
};

// 4. Build a minimal room.
var room = {
  name: 'oracle',
  players: new Collection([], 'id', true),
  config: {
    getMaxScore: function() { return 10; },
    getBonuses: function() { return []; },
    getVariable: function(name) { return name === 'bonusRate' ? 0 : undefined; }
  },
  controller: { clients: new Collection() }
};

// 5. Add fake players that can return server Avatars.
function fakePlayer(name, color) {
  var player = new Player(null, name, color);
  room.players.add(player);
  return player;
}

fakePlayer('p1', '#ff0000');
fakePlayer('p2', '#00ff00');

var game = new Game(room);

// 6. Put the game into a known active state.
game.inRound = true;
game.started = true;
game.world.activate();

var a0 = game.avatars.items[0];
var a1 = game.avatars.items[1];
a0.setPosition(20, 40);
a0.setAngle(0);
a0.updateVelocities();
a0.printManager.start();

a1.setPosition(60, 40);
a1.setAngle(Math.PI);
a1.updateVelocities();
a1.printManager.start();

// 7. Apply simultaneous tick-boundary controls.
a0.updateAngularVelocity(-1);
a1.updateAngularVelocity(1);

// 8. Step.
game.update(1000 / 60);

// 9. Snapshot positions, angles, alive flags, scores, trails, deaths, and events.
```

This sketch needs small stubs for controller/socket classes. That is normal. Keep the
stubs outside the physics classes and do not change `Game`, `Avatar`, `World`, `Island`,
`AvatarBody`, or `PrintManager`.

## First Multiplayer Cases

Start simple and grow:

1. 2 players, fixed positions, both straight for 60 ticks.
2. 2 players, left/right inputs applied at the same tick boundary.
3. 2 players, wall death for one player, then round-end score resolution.
4. 2 players, same-frame double wall death.
5. 2 players, opponent trail hit.
6. 2 players, head-head or near-head fixture, with a note about update-order behavior.
7. 3 players, map size and forced spawn layout.
8. 3 players, one prior death, then two same-frame deaths; both new deaths get score `1`.
9. 4 players, two prior deaths, then one or two same-frame deaths; new deaths get score `2`.
10. 4 players, final survivor gets `3` round points at round end.

For every case, record per-player state and source events. Multiplayer bugs often hide
in ordering, not only in the final winner.
