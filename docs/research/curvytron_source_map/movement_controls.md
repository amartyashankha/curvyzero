# Movement And Controls

Status: source deep dive
Date: 2026-05-08
Scope: `third_party/curvytron-reference`

## Short Answer

The source movement model is elapsed-millisecond kinematics, not fixed Python
ticks.

For source fidelity, clone the server path as the source of truth:

1. Resolve each player's current move to `-1`, `0`, or `1`.
2. Apply all move changes with `avatar.updateAngularVelocity(move)`.
3. Call `game.update(step_ms)`.
4. Inside each avatar update, turn first, then move forward.

The base speed is `16` units per second. The base turn rate is `2.8 / 1000`
radians per millisecond. A normal move of `-1` turns left, `1` turns right, and
`0` goes straight. Inverse controls flip the sign inside angular velocity.

The current Python code has a narrow source-kinematics runner that matches the
current one-step forced movement fixtures. The main `curvyzero-v0` environment
is still a toy ruleset and should not be treated as source-faithful.

## Source Files

Client input:

- `third_party/curvytron-reference/src/client/model/Player.js:44-56`
  creates the two default controls.
- `third_party/curvytron-reference/src/client/model/Player.js:101-104`
  returns the left/right binding values.
- `third_party/curvytron-reference/src/client/model/PlayerInput.js:32`
  sets the default keyboard binding to left arrow `37` and right arrow `39`.
- `third_party/curvytron-reference/src/client/model/PlayerInput.js:121-142`
  handles keydown and keyup.
- `third_party/curvytron-reference/src/client/model/PlayerInput.js:149-176`
  handles gamepad axis and button input.
- `third_party/curvytron-reference/src/client/model/PlayerInput.js:183-209`
  handles touch input by splitting the screen into left and right halves.
- `third_party/curvytron-reference/src/client/model/PlayerInput.js:228-235`
  resolves pressed buttons into one move value.
- `third_party/curvytron-reference/src/client/model/PlayerInput.js:258-262`
  emits the local `move` event.
- `third_party/curvytron-reference/src/client/controller/GameController.js:103-110`
  listens to local avatar input.
- `third_party/curvytron-reference/src/client/controller/GameController.js:154-157`
  sends `player:move` to the server.

Server movement:

- `third_party/curvytron-reference/src/server/controller/GameController.js:142-160`
  attaches server listeners to client, avatar, and bonus events.
- `third_party/curvytron-reference/src/server/controller/GameController.js:314-320`
  receives `player:move` and calls `avatar.updateAngularVelocity(data.move)`.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:40`
  sets base velocity to `16`.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:47`
  sets base angular velocity to `2.8 / 1000`.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:68`
  sets normal controls to not inverse.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:124-132`
  computes angular velocity from move, base turn rate, and inverse state.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:169-179`
  applies angular velocity to angle.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:186-192`
  applies forward velocity to position.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:199-207`
  clamps and applies speed changes.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:212-220`
  converts speed and angle into `velocityX` and `velocityY`.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:225-231`
  updates turn rate when speed changes.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:249-254`
  applies inverse controls and refreshes angular velocity.
- `third_party/curvytron-reference/src/server/model/Avatar.js:23-33`
  updates one live avatar: angle, position, then maybe trail point.

Game loop and update order:

- `third_party/curvytron-reference/src/shared/model/BaseGame.js:39`
  sets the target loop delay to `1000 / 60` ms.
- `third_party/curvytron-reference/src/shared/model/BaseGame.js:128-139`
  computes `step = now - this.rendered` from wall-clock milliseconds.
- `third_party/curvytron-reference/src/shared/model/BaseGame.js:191-194`
  schedules the next server frame with `setTimeout`.
- `third_party/curvytron-reference/src/shared/model/BaseGame.js:210-213`
  calls `this.update(step)`.
- `third_party/curvytron-reference/src/server/model/Game.js:37-80`
  is the authoritative server update loop.
- `third_party/curvytron-reference/src/server/model/Game.js:44-75`
  iterates avatars in reverse order.
- `third_party/curvytron-reference/src/server/model/Game.js:48-73`
  updates a live avatar, checks border/body collision, then runs print and bonus
  catch logic if still alive.
- `third_party/curvytron-reference/src/server/model/Game.js:77-79`
  checks round end after the frame if any death happened.

Events:

- `third_party/curvytron-reference/src/server/model/Avatar.js:55-64`
  emits `position` after setting position.
- `third_party/curvytron-reference/src/server/model/Avatar.js:84-90`
  emits `angle` after setting angle.
- `third_party/curvytron-reference/src/server/model/Avatar.js:71-77`,
  `109-116`, `123-138`, and `145-149` emit `property` for speed, radius,
  invincible, inverse, and color changes.
- `third_party/curvytron-reference/src/server/model/Avatar.js:158-162`
  emits `point`.
- `third_party/curvytron-reference/src/server/model/Avatar.js:180-189`
  emits `die`.
- `third_party/curvytron-reference/src/server/model/Avatar.js:196-210`
  emits score events.
- `third_party/curvytron-reference/src/server/model/Game.js:113-117`
  turns an emitted avatar point into an `AvatarBody` in the collision world.
- `third_party/curvytron-reference/src/server/controller/GameController.js:328-449`
  turns source events into socket events.
- `third_party/curvytron-reference/src/server/controller/GameController.js:458-520`
  forwards game lifecycle events.
- `third_party/curvytron-reference/src/shared/core/BaseSocketClient.js:105-123`
  queues or sends one event.
- `third_party/curvytron-reference/src/shared/core/BaseSocketClient.js:131-146`
  queues or sends many events.
- `third_party/curvytron-reference/src/shared/core/BaseSocketClient.js:187-200`
  serializes events as JSON and flushes the queue.
- `third_party/curvytron-reference/src/shared/service/Compressor.js:20-34`
  compresses position and angle for wire messages.

## Input Mapping

The browser reduces controls to one move value:

| Pressed state | Source move |
| --- | ---: |
| left only | `-1` |
| right only | `1` |
| neither | `0` on the wire |
| both | `0` on the wire |

The exact source path is:

- `PlayerInput.resolve` uses
  `(active[0] !== active[1]) ? (active[0] ? -1 : 1) : false` at
  `third_party/curvytron-reference/src/client/model/PlayerInput.js:228-235`.
- `PlayerInput.setMove` emits `{avatar, move}` at
  `third_party/curvytron-reference/src/client/model/PlayerInput.js:258-262`.
- The client sends `move: e.detail.move ? e.detail.move : 0` at
  `third_party/curvytron-reference/src/client/controller/GameController.js:154-157`.
- The server applies it immediately at
  `third_party/curvytron-reference/src/server/controller/GameController.js:314-320`.

There is no input queue inside `Game.update`. For deterministic traces, stage all
planned move changes first, then call `game.update(step_ms)`.

## Angular Velocity

`BaseAvatar.updateAngularVelocity(factor)` is the key function.

- If a factor is supplied, angular velocity becomes:
  `factor * angularVelocityBase * (inverse ? -1 : 1)`.
- If no factor is supplied, the function preserves the current turn direction
  while recalculating speed/inverse-dependent base turn values.

Source refs:

- Default turn rate:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:47`.
- Core calculation:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:124-132`.
- Angle update:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:169-179`.
- Server setter:
  `third_party/curvytron-reference/src/server/model/Avatar.js:97-102`.

Normal looped turning uses elapsed milliseconds:

```text
angle = angle + angularVelocity * step_ms
```

There is also a straight-angle bonus path. If `directionInLoop` is false, the
source adds `angularVelocity` once, then sets angular velocity back to zero. That
branch is at `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:171-177`.
The bonus that sets it is `BonusEnemyStraightAngle.getEffects` at
`third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyStraightAngle.js:36-41`.

## Inverse Controls

Inverse is a movement property, not a client-side remap.

- The default is `false` at
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:68`.
- `updateAngularVelocity` multiplies the sign by `-1` when inverse is true at
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:124-132`.
- `setInverse` stores the bool and refreshes angular velocity at
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:249-254`.
- The server avatar emits a `property` event for inverse at
  `third_party/curvytron-reference/src/server/model/Avatar.js:134-138`.
- `BonusEnemyInverse.getEffects` returns `['inverse', 1]` at
  `third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyInverse.js:29-32`.
- `BonusStack.apply` turns odd inverse stack values into true at
  `third_party/curvytron-reference/src/server/model/BonusStack.js:51-53`.

Important clone detail: changing inverse while a player is already turning must
flip the active angular velocity. The source does that by calling
`updateAngularVelocity()` from `setInverse`.

## Velocity

Base speed is stored in units per second, then converted to units per
millisecond:

```text
velocity_ms = velocity / 1000
velocityX = cos(angle) * velocity_ms
velocityY = sin(angle) * velocity_ms
x = x + velocityX * step_ms
y = y + velocityY * step_ms
```

Source refs:

- Base speed `16`:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:40`.
- Position update:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:186-192`.
- Velocity vector update:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:212-220`.
- Speed floor:
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:199-207`.

Speed also changes turn rate:

- `BaseAvatar.updateVelocities` calls `updateBaseAngularVelocity` at
  `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:212-220`.
- `updateBaseAngularVelocity` uses:
  `ratio * BaseAvatar.prototype.angularVelocityBase + Math.log(1 / ratio) / 1000`
  at `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:225-231`.

Speed-affecting bonuses:

- `BonusSelfFast.getEffects` adds `0.75 * base_velocity` at
  `third_party/curvytron-reference/src/server/model/Bonus/BonusSelfFast.js:29-32`.
- `BonusEnemyFast.getEffects` does the same to enemies at
  `third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyFast.js:29-32`.
- `BonusSelfSlow.getEffects` subtracts `base_velocity / 2` at
  `third_party/curvytron-reference/src/server/model/Bonus/BonusSelfSlow.js:22-25`.
- `BonusEnemySlow.getEffects` does the same to enemies at
  `third_party/curvytron-reference/src/server/model/Bonus/BonusEnemySlow.js:22-25`.
- `BonusStack.apply` applies the final speed through `setVelocity` at
  `third_party/curvytron-reference/src/server/model/BonusStack.js:48-50`.

## Time Step

The live server targets 60 Hz, but physics uses measured elapsed time:

- `BaseGame.prototype.framerate = 1/60 * 1000` at
  `third_party/curvytron-reference/src/shared/model/BaseGame.js:39`.
- Each loop computes `step = now - this.rendered` at
  `third_party/curvytron-reference/src/shared/model/BaseGame.js:128-139`.
- The next loop is scheduled by `setTimeout(this.loop, this.framerate)` at
  `third_party/curvytron-reference/src/shared/model/BaseGame.js:191-194`.
- The frame calls `this.update(step)` at
  `third_party/curvytron-reference/src/shared/model/BaseGame.js:210-213`.

For probes, fixed `1000 / 60` is useful, but it is a trace contract choice. The
source itself accepts the actual elapsed milliseconds.

## Update Order

One server frame does this:

1. `Game.update(step)` captures `score = deaths.count()` once at frame start.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:37-40`.
2. It clears `deathInFrame`.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:42`.
3. It loops avatars from the last index down to zero.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:44-75`.
4. For each live avatar, it calls `avatar.update(step)`.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:48-50`.
5. `Avatar.update` turns first, then moves, then maybe emits a point.
   Source: `third_party/curvytron-reference/src/server/model/Avatar.js:23-33`.
6. The game checks normal wall or borderless wrap.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:51-59`.
7. If there was no border hit and the avatar is not invincible, it checks body
   collision.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:60-68`.
8. If still alive, it runs print manager and bonus catch.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:70-73`.
9. After all avatars, it checks round end if any death happened.
   Source: `third_party/curvytron-reference/src/server/model/Game.js:77-79`.

This order matters. It is not simultaneous collision resolution. Later-index
avatars move and can emit trail bodies before earlier-index avatars move.

## Event Emission

Movement emits source events during the update, before the frame is done:

- `setAngle` emits `angle` during `avatar.update(step)`.
- `setPosition` emits `position` during `avatar.update(step)`.
- `addPoint` emits `point`; `Game.onPoint` immediately adds a collision body if
  the world is active.
- `die` emits `die` during collision handling.
- property changes from bonuses emit `property`.

The server controller listens to these source events and sends socket events:

- Listener setup:
  `third_party/curvytron-reference/src/server/controller/GameController.js:150-160`.
- `position` wire event:
  `third_party/curvytron-reference/src/server/controller/GameController.js:340-347`.
- `angle` wire event:
  `third_party/curvytron-reference/src/server/controller/GameController.js:354-360`.
- `die` wire event:
  `third_party/curvytron-reference/src/server/controller/GameController.js:367-374`.
- important `point` wire event:
  `third_party/curvytron-reference/src/server/controller/GameController.js:328-333`.
- `property` wire event:
  `third_party/curvytron-reference/src/server/controller/GameController.js:426-433`.
- bonus stack wire event:
  `third_party/curvytron-reference/src/server/controller/GameController.js:440-449`.

Wire values are not always raw floats. Position and angle use `Compressor`, which
rounds to centi-units with `(0.5 + value * 100) | 0`.
Source: `third_party/curvytron-reference/src/shared/service/Compressor.js:11-34`.

The headless source oracle records raw event order before compression in
`tools/reference_oracle/scenario_runner.js:170-216` and records each tick after
staging moves and calling `game.update(stepMs)` at
`tools/reference_oracle/scenario_runner.js:448-473`.

## What Must Be Cloned For Source-Fidelity Kinematics

Minimum kinematics clone:

- Source move values: `-1`, `0`, `1`.
- Stage all moves before one update.
- `updateAngularVelocity` exactly, including inverse sign.
- Base speed `16` units/s.
- Base angular velocity `2.8 / 1000` radians/ms.
- Speed-to-vector conversion with `cos(angle)` and `sin(angle)`.
- Turn first, move second.
- Elapsed millisecond step, not a fixed Python tick size hidden in config.
- Speed-dependent turn-rate recalculation.
- Straight-angle bonus behavior, if any bonus-modified movement is in scope.
- Floating-point comparison tolerances in trace diffs.

To go beyond pure kinematics, the clone also needs:

- Reverse avatar update order.
- Source wall and body collision order.
- Trail point emission and `Game.onPoint` body insertion.
- Print manager distance state and holes.
- Bonus stack apply/remove timing.
- Event order and, for wire fidelity, compressor rounding.

## Current Python Status

Implemented:

- `CurvyTronReferenceDefaults` stores source-derived constants for docs and
  future rulesets. See `src/curvyzero/env/config.py:13-68`.
- The main `CurvyTronEnv` is a deterministic 2-player toy env. It says this in
  `src/curvyzero/env/core.py:1-5` and enforces two players at
  `src/curvyzero/env/core.py:32-36`.
- Toy-v0 action mapping is left/straight/right as actions `0/1/2`, not source
  wire moves. See `src/curvyzero/env/core.py:136-149`.
- Toy-v0 movement uses fixed per-tick `speed` and `turn_rate_radians`.
  See `src/curvyzero/env/core.py:96-134`.
- The scenario runner has a normal toy-v0 path and marks it as not source
  fidelity. See `src/curvyzero/env/scenarios.py:27-38` and
  `src/curvyzero/env/scenarios.py:90-107`.
- A narrow source-kinematics runner exists for the current forced movement
  fixtures. See `src/curvyzero/env/scenarios.py`.
- That runner implements source turn-then-move math for fixed-step movement.
- The matching test checks angle and position for straight, left, right, and
  forced two-player movement fixtures. See `tests/test_env_scenarios.py`.

Known gaps:

- It is not a full game. Its own message says collisions, trails, bonuses, and
  full rules are missing.
- Toy-v0 is still grid-based and fixed-tick. It does not clone elapsed-ms
  kinematics, circular body collision, source trail holes, event ordering, bonus
  stacks, or multiplayer source update behavior.
- The source scenario files for straight, left, and right micro-steps exist, but
  their comparison blocks still mark source-fidelity required as false. See
  `scenarios/environment/source_kinematics_straight_step.json:84-94`,
  `scenarios/environment/source_kinematics_left_turn_step.json:84-94`, and
  `scenarios/environment/source_kinematics_right_turn_step.json:84-94`.
- The open question on fixed `1000 / 60` vs recorded elapsed milliseconds is
  still open at `docs/research/curvytron_source_map/open_questions.md:9-11`.
