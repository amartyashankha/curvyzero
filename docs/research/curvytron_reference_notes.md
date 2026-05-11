# CurvyTron Reference Notes

Local source mining of `third_party/curvytron-reference`; no network used. This is scoped to mechanics needed for a Python simulator.

## Runtime Map

- Authoritative mechanics are server-side: `src/server/model/Game.js`, `src/server/model/Avatar.js`, `src/server/manager/PrintManager.js`, `src/server/core/World.js`, `src/server/core/Island.js`, `src/server/core/AvatarBody.js`, `src/server/manager/BonusManager.js`, and `src/server/model/Bonus/*.js`.
- Shared constants and state live in `src/shared/model/BaseGame.js`, `src/shared/model/BaseAvatar.js`, `src/shared/model/BaseRoomConfig.js`, `src/shared/model/BaseBonus*.js`, and `src/shared/manager/BaseBonusManager.js`.
- Config defaults: server port sample is `8080` (`third_party/curvytron-reference/config.json.sample:1-11`); launcher also falls back to port `8080` and inspector disabled (`third_party/curvytron-reference/src/server/launcher.js:1-13`).

## Timing And Rounds

- Server tick loop targets `1/60 * 1000` ms and reschedules with `setTimeout(this.loop, this.framerate)` (`third_party/curvytron-reference/src/shared/model/BaseGame.js:34-40`, `third_party/curvytron-reference/src/shared/model/BaseGame.js:128-139`, `third_party/curvytron-reference/src/shared/model/BaseGame.js:191-203`).
- Per-frame `step` is elapsed wall-clock milliseconds: `now - this.rendered` (`third_party/curvytron-reference/src/shared/model/BaseGame.js:132-138`).
- Round warmup is `3000` ms and warmdown is `5000` ms (`third_party/curvytron-reference/src/shared/model/BaseGame.js:49-60`). `newRound()` starts after warmup; `endRound()` stops after warmdown (`third_party/curvytron-reference/src/shared/model/BaseGame.js:318-340`).
- After `game:start`, server delays `avatar.printManager.start` another `3000` ms, so avatars move before trails begin (`third_party/curvytron-reference/src/server/model/Game.js:257-268`).
- Map size is `round(sqrt(80^2 + ((players - 1) * 80^2 / 5)))` (`third_party/curvytron-reference/src/shared/model/BaseGame.js:41-47`, `third_party/curvytron-reference/src/shared/model/BaseGame.js:223-236`).

## Movement And Controls

- Base avatar constants: velocity `16`, angular velocity base `2.8/1000` radians/ms, radius `0.6`, trail latency `3`, `inverse=false`, `invincible=false`, `directionInLoop=true` (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:35-82`).
- Position integration: `x += velocityX * step`, `y += velocityY * step`; components are `cos(angle) * velocity/1000` and `sin(angle) * velocity/1000` (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:186-230`).
- Speed setter clamps to at least `BaseAvatar.velocity / 2` (`8`) and recomputes movement/turn velocity (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:194-230`).
- Turn input sets angular velocity with `factor * angularVelocityBase`, flipped by inverse controls (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:119-132`).
- Continuous turns add `angularVelocity * step`; straight-angle mode adds `angularVelocity` once and then resets angular velocity to `0` (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:169-179`).
- Client default controls are left/right arrow key codes `[37, 39]`; input resolves to `-1` for left-only, `1` for right-only, and false for neither/both (`third_party/curvytron-reference/src/client/model/PlayerInput.js:27-33`, `third_party/curvytron-reference/src/client/model/PlayerInput.js:228-234`).
- Client sends falsey moves as `0`; server applies `player.avatar.updateAngularVelocity(data.move)` (`third_party/curvytron-reference/src/client/controller/GameController.js:154-157`, `third_party/curvytron-reference/src/server/controller/GameController.js:314-320`).

## Trail Gaps / PrintManager

- Printed trail and hole lengths are distance-based, not tick-count based: print base `60`, hole base `5` (`third_party/curvytron-reference/src/server/manager/PrintManager.js:17-29`).
- Random print distance is `60 * (0.3 + random*0.7)`; random hole distance is `5 * (0.8 + random*0.5)` (`third_party/curvytron-reference/src/server/manager/PrintManager.js:48-60`).
- `PrintManager.start()` activates printing at the avatar's current position; `stop()` sets printing false and clears manager state (`third_party/curvytron-reference/src/server/manager/PrintManager.js:62-84`).
- Each active print-manager tick subtracts traveled distance and toggles printing when remaining distance reaches `<= 0` (`third_party/curvytron-reference/src/server/manager/PrintManager.js:87-103`).
- Server avatar update adds trail points only when alive, printing, and distance from last point is greater than avatar radius (`third_party/curvytron-reference/src/server/model/Avatar.js:23-47`).
- `BaseAvatar.setPrinting()` always adds the current point; when printing turns off it then clears the trail object, producing a visual gap while leaving already-added world bodies in place (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:297-309`, `third_party/curvytron-reference/src/server/model/Trail.js:12-19`).
- Every emitted point becomes an `AvatarBody` in the collision world while game/world are active (`third_party/curvytron-reference/src/server/model/Game.js:113-117`).

## Collision And Death

- The server update order is: avatar motion, border check, trail/body collision check, then print-manager/bonus catch if still alive (`third_party/curvytron-reference/src/server/model/Game.js:37-80`).
- World stores circular bodies in island buckets by the four corners of their bounding box (`third_party/curvytron-reference/src/server/core/World.js:46-79`).
- Collision lookup probes the current body corners and asks islands for colliding bodies (`third_party/curvytron-reference/src/server/core/World.js:97-126`).
- Body collision is strict: `distance < bodyA.radius + bodyB.radius` and `bodyA.match(bodyB)` (`third_party/curvytron-reference/src/server/core/Island.js:62-90`).
- Same-avatar self collision is ignored until `currentBody.num - storedBody.num > avatar.trailLatency` (`third_party/curvytron-reference/src/server/core/AvatarBody.js:33-40`).
- Normal border collision kills when `body.x/y +/- avatar.radius` crosses `[0, size]`; borderless mode uses margin `0` and wraps through `World.getOposite()` (`third_party/curvytron-reference/src/server/model/Game.js:51-58`, `third_party/curvytron-reference/src/server/core/World.js:276-323`).
- `Game.kill()` calls `avatar.die(killer)`, adds captured round score, records death, and marks `deathInFrame=true` (`third_party/curvytron-reference/src/server/model/Game.js:82-94`).
- Death clears bonuses, marks `alive=false`, adds a final point, stops printing, and emits killer/old-body metadata (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:282-290`, `third_party/curvytron-reference/src/server/model/Avatar.js:176-189`).

## Scoring And Win Conditions

- At frame start, `score = this.deaths.count()` is captured once; all deaths in that same frame receive that same score (`third_party/curvytron-reference/src/server/model/Game.js:37-94`).
- Round ends when no more than one avatar remains alive (`third_party/curvytron-reference/src/server/model/Game.js:151-170`).
- Round winner is the last alive avatar, or the only avatar in a one-player game; winner gets `max(avatars.count() - 1, 1)` extra round score (`third_party/curvytron-reference/src/server/model/Game.js:175-188`).
- All avatars then resolve `score += roundScore` and reset `roundScore=0` (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:312-351`, `third_party/curvytron-reference/src/server/model/Game.js:190-193`).
- Default max score is `max(1, (room.players.count() - 1) * 10)` (`third_party/curvytron-reference/src/shared/model/BaseRoomConfig.js:159-179`).
- Game ends if no present avatars remain, originally-multiplayer game has one present avatar, or exactly one present avatar has a unique score at/above max score; tied leaders continue (`third_party/curvytron-reference/src/server/model/Game.js:121-146`).

## Items / Bonuses

- Default enabled bonuses are `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`, `BonusSelfMaster`, `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`, `BonusEnemyInverse`, `BonusEnemyStraightAngle`, `BonusGameBorderless`, `BonusAllColor`, and `BonusGameClear` (`third_party/curvytron-reference/src/shared/model/BaseRoomConfig.js:13-30`, `third_party/curvytron-reference/src/server/model/RoomConfig.js:19-31`).
- Room variable `bonusRate` defaults to `0` and must stay in `[-1, 1]` (`third_party/curvytron-reference/src/shared/model/BaseRoomConfig.js:13-15`, `third_party/curvytron-reference/src/shared/model/BaseRoomConfig.js:75-85`).
- Base bonus radius is `3`, default duration `5000` ms, default probability `1` (`third_party/curvytron-reference/src/shared/model/BaseBonus.js:19-45`).
- Bonus application sets target, schedules `off()` for nonzero duration, then calls `on()` (`third_party/curvytron-reference/src/server/model/Bonus/Bonus.js:19-31`).
- Targets: self means catcher, enemy means alive non-catchers, all means all alive avatars, game means game object (`third_party/curvytron-reference/src/server/model/Bonus/BonusSelf.js:17-54`, `third_party/curvytron-reference/src/server/model/Bonus/BonusEnemy.js:17-55`, `third_party/curvytron-reference/src/server/model/Bonus/BonusAll.js:17-55`, `third_party/curvytron-reference/src/server/model/Bonus/BonusGame.js:17-55`).
- Spawn cap is `20`; base pop time is `3000`, adjusted to `3000 - 1500*bonusRate`; next delay is adjusted base times `[1, 2)` (`third_party/curvytron-reference/src/shared/manager/BaseBonusManager.js:19-38`, `third_party/curvytron-reference/src/server/manager/BonusManager.js:6-16`, `third_party/curvytron-reference/src/server/manager/BonusManager.js:157-160`).
- Bonus spawn position must be clear in both trail world and bonus world, with margin `BaseBonus.radius + bonusPopingMargin * world.size` (`third_party/curvytron-reference/src/server/manager/BonusManager.js:76-99`).
- Catching a bonus removes it from map/world and applies it to the avatar/game (`third_party/curvytron-reference/src/server/manager/BonusManager.js:101-150`).
- Weighted random type selection uses each bonus type's `getProbability(game)` and skips nonpositive probabilities (`third_party/curvytron-reference/src/server/manager/BonusManager.js:167-197`).
- Bonus stack application: radius uses `baseRadius * 2^value`; velocity is assigned then clamped by avatar; inverse is odd/even; invincible is boolean; printing starts/stops PrintManager; color is set directly; straight-angle fields overwrite rather than sum (`third_party/curvytron-reference/src/shared/model/BaseBonusStack.js:52-83`, `third_party/curvytron-reference/src/server/model/BonusStack.js:42-110`).
- Game bonus stack applies `borderless` through `game.setBorderless()` (`third_party/curvytron-reference/src/server/model/GameBonusStack.js:40-50`).
- Key effects:
  - SelfFast: 4000 ms, `velocity += 0.75 * baseVelocity` (`third_party/curvytron-reference/src/server/model/Bonus/BonusSelfFast.js:15-32`).
  - SelfSlow: 5000 ms default, `velocity -= baseVelocity/2` (`third_party/curvytron-reference/src/server/model/Bonus/BonusSelfSlow.js:15-25`).
  - SelfSmall: 7500 ms, radius exponent `-1` (`third_party/curvytron-reference/src/server/model/Bonus/BonusSelfSmall.js:15-32`).
  - SelfMaster: 7500 ms, invincible and printing stopped (`third_party/curvytron-reference/src/server/model/Bonus/BonusSelfMaster.js:15-35`).
  - EnemyFast: 6000 ms, enemies speed up (`third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyFast.js:15-32`).
  - EnemySlow: 5000 ms default, enemies slow down (`third_party/curvytron-reference/src/server/model/Bonus/BonusEnemySlow.js:15-25`).
  - EnemyBig: 7500 ms, enemies radius exponent `+1` (`third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyBig.js:15-32`).
  - EnemyInverse: 5000 ms default, probability `0.8`, enemies invert controls (`third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyInverse.js:15-32`).
  - EnemyStraightAngle: 5000 ms, probability `0.6`, one-shot 90-degree turns (`third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyStraightAngle.js:15-42`).
  - GameBorderless: 10000 ms, probability `0.8`, border wrap (`third_party/curvytron-reference/src/server/model/Bonus/BonusGameBorderless.js:15-41`).
  - GameClear: instant, clears trails; probability decreases once dead/present ratio reaches `0.5` (`third_party/curvytron-reference/src/server/model/Bonus/BonusGameClear.js:15-46`).
  - AllColor: 7500 ms, rotates alive avatar colors (`third_party/curvytron-reference/src/server/model/Bonus/BonusAllColor.js:17-71`).
- `BonusSelfGodzilla` exists but is absent from default config, server mapping, and client sprite list; treat as unreachable unless hidden config is found (`third_party/curvytron-reference/src/server/model/Bonus/BonusSelfGodzilla.js:1-30`, `third_party/curvytron-reference/src/server/model/RoomConfig.js:19-31`, `third_party/curvytron-reference/src/client/manager/BonusManager.js:33-46`).

## Golden-Test Candidates

1. Map size formula for 1, 2, 4, and 8 players (`third_party/curvytron-reference/src/shared/model/BaseGame.js:223-236`).
2. Fixed-step movement: angle `0`, velocity `16`, step `1000` ms advances x by `16` (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:186-230`).
3. Turn input mapping: left/right/none/both produce `-1`, `1`, `0`, `0`, and inverse flips sign (`third_party/curvytron-reference/src/client/model/PlayerInput.js:228-234`, `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:124-132`).
4. Velocity clamp and angular-base recomputation for slow/fast stacked effects (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:194-230`, `third_party/curvytron-reference/src/server/model/BonusStack.js:42-110`).
5. Straight-angle mode turns once by `Math.PI/2` then resets angular velocity (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:169-179`, `third_party/curvytron-reference/src/server/model/Bonus/BonusEnemyStraightAngle.js:36-42`).
6. Trail printing adds points only after traveled distance exceeds radius (`third_party/curvytron-reference/src/server/model/Avatar.js:23-47`).
7. Hole toggle adds endpoint and then stops adding bodies until printing resumes (`third_party/curvytron-reference/src/shared/model/BaseAvatar.js:297-309`, `third_party/curvytron-reference/src/server/model/Game.js:113-117`).
8. Self-collision latency threshold `current.num - stored.num > 3` (`third_party/curvytron-reference/src/server/core/AvatarBody.js:33-40`).
9. Collision strictness: distance equal to sum radii is safe, slightly less kills (`third_party/curvytron-reference/src/server/core/Island.js:83-90`).
10. Wall collision kills; borderless collision wraps to opposite edge (`third_party/curvytron-reference/src/server/model/Game.js:51-58`, `third_party/curvytron-reference/src/server/core/World.js:276-323`).
11. Scoring order: first death gets 0, later deaths get prior death count, same-frame deaths share score, winner gets `players-1` (`third_party/curvytron-reference/src/server/model/Game.js:37-94`, `third_party/curvytron-reference/src/server/model/Game.js:175-193`).
12. Round ends at <=1 alive; game win only on unique leader at/above max score (`third_party/curvytron-reference/src/server/model/Game.js:125-170`).
13. Spawn position and direction respect border/collision margins (`third_party/curvytron-reference/src/server/model/Game.js:242-248`, `third_party/curvytron-reference/src/server/core/World.js:157-231`).
14. Bonus pop timing/rate/cap formula (`third_party/curvytron-reference/src/shared/manager/BaseBonusManager.js:19-38`, `third_party/curvytron-reference/src/server/manager/BonusManager.js:6-16`, `third_party/curvytron-reference/src/server/manager/BonusManager.js:157-160`).
15. Weighted bonus choice and `BonusGameClear` probability as death ratio changes (`third_party/curvytron-reference/src/server/manager/BonusManager.js:167-197`, `third_party/curvytron-reference/src/server/model/Bonus/BonusGameClear.js:29-38`).
16. Bonus-stack composition for radius, velocity, inverse, printing, color, straight-angle, and borderless reset (`third_party/curvytron-reference/src/shared/model/BaseBonusStack.js:52-83`, `third_party/curvytron-reference/src/server/model/BonusStack.js:42-110`, `third_party/curvytron-reference/src/server/model/GameBonusStack.js:40-50`).
17. Bonus catch removes from map before applying effect (`third_party/curvytron-reference/src/server/manager/BonusManager.js:101-150`).
18. Transport compression if simulator mirrors wire protocol: `(0.5 + value*100)|0` and `/100`, including JS bitwise truncation (`third_party/curvytron-reference/src/shared/service/Compressor.js:6-35`).

## Ambiguities / Watchouts

- Avatars move for 3000 ms after `game:start` before printing starts; preserve or consciously override this (`third_party/curvytron-reference/src/server/model/Game.js:257-268`).
- Collision is endpoint-circle based, not swept; large simulator steps can tunnel unless matching reference behavior is desired (`third_party/curvytron-reference/src/server/model/Game.js:37-80`, `third_party/curvytron-reference/src/server/core/World.js:104-126`).
- `Game.update()` writes undeclared, unused `dead = false`; ignore unless reproducing JS globals (`third_party/curvytron-reference/src/server/model/Game.js:39-47`).
- Use server files as authoritative where client/server classes share names (`Game`, `Avatar`, `BonusStack`).
