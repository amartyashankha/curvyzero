# CurvyTron Rounds, Scoring, And Multiplayer

Status: source-mined with narrow source fixtures and broader gaps called out

This page records how the original CurvyTron source handles round lifecycle,
map size, player presence, death order, and multiplayer scoring. Server-side
files are the source of truth for gameplay. Client files are only cited when
they explain display or local replay behavior.

Line refs below are from the current checkout under
`third_party/curvytron-reference`.

Current proof boundary: the named JS oracle and `CurvyTronSourceEnv` fixtures
prove a narrow source slice for 1P/2P/3P/4P wall scoring, selected lifecycle
paths, active-round leave, spawn order, present/absent handling, and match/tie
edges. They do not prove broad public env parity, trainer rewards, replay,
visual behavior, bonuses, warmdown leave, or every 4P match lifecycle variant.
Use [coverage_tracker.md](../../working/environment/coverage_tracker.md) for the
current promoted slice and
[no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md](../../working/environment/no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md)
for the remaining no-bonus multiplayer fixture gaps.

## Main Source Files

| Area | File/functions | Line refs |
| --- | --- | --- |
| Shared game state | `src/shared/model/BaseGame.js`: constructor, loop, round lifecycle, size, avatar filters | `BaseGame` 6-29, `loop` 128-139, `onStop` 154-165, `getSize` 230-235, `getAliveAvatars` 263-266, `getPresentAvatars` 273-276, `newRound` 321-329, `endRound` 335-341 |
| Server game rules | `src/server/model/Game.js`: update/death/scoring/round reset | `update` 37-80, `kill` 89-94, `checkRoundEnd` 151-170, `resolveScores` 175-193, `onRoundEnd` 221-225, `onRoundNew` 230-251, `onStart` 257-268, `onStop` 274-289 |
| Collections and order | `src/shared/Collection.js` | constructor 8-21, `count` 38-41, `add` 61-79, `map` 229-238, `filter` 247-258, `match` 267-278 |
| Avatar state | `src/shared/model/BaseAvatar.js`, `src/server/model/Avatar.js` | `BaseAvatar` fields 6-28, `updatePosition` 186-192, `die` 285-290, `addScore` 317-320, `resolveScore` 327-331, `clear` 366-392, `destroy` 397-402, server `Avatar.update` 23-33, `setPosition` 55-64, `die` 180-189, `setRoundScore` 207-210 |
| World/collision | `src/server/core/World.js`, `Island.js`, `AvatarBody.js` | `World.getBody` 104-126, `getRandomPosition` 165-176, `getBoundIntersect` 276-295, `getOposite` 305-324, `Island.bodiesTouch` 83-90, `AvatarBody.match` 33-40 |
| Room and match score | `src/shared/model/BaseRoom.js`, `src/shared/model/BaseRoomConfig.js` | room player collection 8-10, `addPlayer` 44-47, `removePlayer` 76-79, `newGame` 94-106, `closeGame` 111-125, `getMaxScore` 164-167, `getDefaultMaxScore` 176-178 |

## Round Lifecycle

`BaseGame.newRound(time)` sets `started = true`, sets `inRound = true`, calls
`onRoundNew()`, then starts the game loop after `warmupTime` unless a custom
time is passed. `warmupTime` is 3000 ms (`BaseGame.js` 53, 321-329).

`Game.onRoundNew()` emits `round:new`, calls the shared reset, clears the world,
clears `deaths`, clears bonus state, and then spawns only `present` avatars. A
non-present avatar is not spawned. It is added to `deaths` instead (`Game.js`
230-251).

`Game.onStart()` emits `game:start`, schedules each avatar's print manager to
start 3000 ms later, activates the world, and then starts shared timers
(`Game.js` 257-268). It loops over all avatars, not only `present` avatars, so
leave/disconnect cases must stay fixture-backed before broader claims.

`Game.checkRoundEnd()` ends the round when there are zero or one alive avatars.
It only runs after a death in `Game.update()` or after `removeAvatar()` calls it
(`Game.js` 77-79, 101-106, 151-170).

`Game.onRoundEnd()` calls `resolveScores()` first, then emits `round:end` with
the round winner if one exists (`Game.js` 221-225). `BaseGame.endRound()` then
schedules `stop()` after `warmdownTime`, which is 5000 ms (`BaseGame.js` 60,
335-341).

After `stop()`, `Game.onStop()` checks `isWon()`. If the match is not won, it
starts a new round. If the match is won, it ends and cleans up (`Game.js`
274-289).

On a non-winning stop, the next `round:new` is synchronous inside
`Game.onStop()`: `game:stop` emits first, shared stop cleanup runs, `isWon()`
returns null, and `newRound()` immediately emits the next `round:new` before
the next warmup `game:start` timer (`Game.js` 274-289; `BaseGame.js` 321-329).

`RoomController.launch()` and ready handling create the `Game`, but the first
round starts only after `GameController.checkReady()` calls `game.newRound()`
(`RoomController.js` 320-349, 527-539; `GameController.js` 282-287).

`Game.onStart()` emits `game:start`, schedules every avatar's
`printManager.start` for 3000 ms later in reverse avatar order, activates the
world, then calls the shared start path (`Game.js` 257-268). With bonuses
enabled, shared start calls `BonusManager.start()`, which immediately consumes a
random value for the first bonus pop timeout (`BaseGame.js` 144-149;
`BonusManager.js` 24-33, 157-160). With `bonuses=[]`, that bonus-time random
call is absent.

`BaseGame.stop()` only runs `onStop()` when a frame timer exists (`BaseGame.js`
117-123). A headless lifecycle fixture must therefore exercise `BaseGame.start()`
or model the frame timer explicitly; calling `Game.onStart()` alone is enough
for delayed PrintManager timer facts, but not enough to prove `game:stop` and
next-round behavior.

## Natural Spawn And RNG

Natural spawn happens inside `Game.onRoundNew()` after `round:new` is emitted,
after shared avatar clearing, and after `world.clear()` resets `active=false`
and empties all island bodies (`Game.js` 230-251; `World.js` 357-365).

For each present avatar, in reverse avatar order, source spawn calls
`World.getRandomPosition(avatar.radius, this.spawnMargin)`, emits a position via
`avatar.setPosition(...)`, calls
`World.getRandomDirection(avatar.x, avatar.y, this.spawnAngleMargin)`, then
emits an angle if the accepted angle differs from the cleared angle
(`Game.js` 242-248; `Avatar.js` 55-64, 84-89).

`getRandomPosition()` consumes x then y through `getRandomPoint(margin)`, where
`margin = radius + border * size`; with default radius `0.6` and
`spawnMargin = 0.05`, the margin is `0.6 + 0.05 * size` (`BaseGame.js` 67;
`BaseAvatar.js` 54; `World.js` 165-176, 263-265). Because the round just cleared
the world and spawn does not insert bodies, natural spawn does not currently
reject overlap against other newly spawned avatars. It would only loop if
pre-existing world bodies remained, which the normal `onRoundNew()` path clears.

`getRandomDirection()` consumes one or more random angles via
`getRandomAngle()`. Rejection depends on `isDirectionValid(angle, x, y,
tolerance * size)`, with `spawnAngleMargin = 0.3` (`BaseGame.js` 74;
`World.js` 187-197, 209-231, 251-253).

Non-present avatars consume no spawn RNG and are added to `deaths` instead
(`Game.js` 242-250).

## Collection And Player Order

Room players live in a `Collection([], 'id', true)`, so missing ids are assigned
by `Collection.setId()` when added (`BaseRoom.js` 8-10, `Collection.js`
124-134). The normal player order is insertion order in `Collection.items`.

`Collection.map()` and `Collection.filter()` loop from the end to the start,
push into a temporary array, then construct a new `Collection`, whose constructor
also loops from the end to the start. The double reverse preserves the original
order in the returned collection (`Collection.js` 8-21, 229-258).

`BaseGame` builds avatars from room players with `room.players.map(...)`, so
avatar collection order follows room player order (`BaseGame.js` 13).

`Game.update()` loops avatars from the last item down to the first item
(`Game.js` 44-75). In a 4-player room added as P0, P1, P2, P3, update order is
P3, P2, P1, P0. This matters for same-frame collisions because the game does
not move all players first and resolve collisions second.

`Collection.match()` scans from first to last and returns the first match
(`Collection.js` 267-278). `resolveScores()` uses this to find the surviving
winner when there is more than one avatar in the collection (`Game.js`
179-183). Since the round only ends with zero or one alive avatar, this should
normally mean the sole survivor.

## Map Size And Max Score

Initial game size uses total avatar count: `this.size =
this.getSize(this.avatars.count())` (`BaseGame.js` 13-14). On stop, the source
recomputes size from `getPresentAvatars().count()` and rebuilds the world if the
size changed (`BaseGame.js` 154-165, 218-221; `Game.js` 208-216).

The arena formula is:

```text
size = round(sqrt(80 * 80 + ((players - 1) * 80 * 80 / 5)))
```

Source refs: `BaseGame.perPlayerSize = 80` at `BaseGame.js` 46 and
`BaseGame.getSize(players)` at `BaseGame.js` 230-235.

Expected sizes:

| Players | Size |
| --- | ---: |
| 1 | 80 |
| 2 | 88 |
| 3 | 95 |
| 4 | 101 |

Default match max score is `max(1, (room.players.count() - 1) * 10)`, based on
room player count, not present avatar count (`BaseRoomConfig.js` 164-178).

Expected defaults:

| Players | Max score |
| --- | ---: |
| 1 | 1 |
| 2 | 10 |
| 3 | 20 |
| 4 | 30 |

## Present And Alive

Each avatar starts with `alive = true`, `present = true`, `score = 0`, and
`roundScore = 0` (`BaseAvatar.js` 22-28).

`BaseAvatar.clear()` resets round state and sets `alive = true`, but it does not
set `present = true` (`BaseAvatar.js` 366-392). `BaseAvatar.destroy()` calls
`clear()`, then sets `present = false` and `alive = false` (`BaseAvatar.js`
397-402).

`BaseGame.getAliveAvatars()` filters only `alive`; `getPresentAvatars()` filters
only `present` (`BaseGame.js` 263-276). Most scoring logic works from
`avatars.count()`, not present count. This is important:

- `resolveScores()` gives winner bonus using total `this.avatars.count()`
  (`Game.js` 185-187).
- `onRoundNew()` adds non-present avatars to `deaths`, which can change death
  scores in later rounds (`Game.js` 242-250).
- `isWon()` uses present count for game-over checks before max-score checks
  (`Game.js` 125-145).

Leaving during a game does not remove the avatar from `game.avatars`.
`Game.removeAvatar()` calls the shared `removeAvatar()`, which makes the avatar
die and destroy itself, emits `player:leave`, and checks round end (`BaseGame.js`
95-100; `Game.js` 101-106). It does not add that avatar to `this.deaths` in the
current round. The narrow active 2P case is Python/oracle verified by
`source_lifecycle_mid_round_remove_avatar_2p.json` in `tests/test_source_env.py`:
avatar 2 leaves after delayed PrintManager start, emits `player:leave`, leaves
`deaths=[]`, and the round ends because only avatar 1 remains alive.

## Update And Death Handling

At the start of each server update, `Game.update(step)` captures
`score = this.deaths.count()` once (`Game.js` 37-42). Every kill in that update
gets that same score value, even though `this.deaths.add(avatar)` happens after
each kill (`Game.js` 89-94). This is the key same-frame scoring rule.

For each alive avatar, in reverse avatar order:

1. `avatar.update(step)` updates angle, then position, then maybe emits a trail
   point if printing and far enough from the last point (`Avatar.js` 23-33).
2. `Game.update()` checks border collision with margin `avatar.radius` unless
   borderless is active (`Game.js` 48-59; `World.js` 276-295).
3. If borderless is active, crossing an edge wraps to the opposite edge instead
   of killing the avatar (`Game.js` 51-57; `World.js` 305-324).
4. If not borderless and the border check hits, `kill(avatar, null, score)` runs
   (`Game.js` 53-59).
5. If no border hit and the avatar is not invincible, the game checks
   `world.getBody(avatar.body)` for trail/body collision (`Game.js` 60-67;
   `World.js` 104-126).
6. If still alive, print-manager hole/line toggles and bonus catches happen
   (`Game.js` 70-73).

`Game.kill()` calls `avatar.die(killer)`, then `avatar.addScore(score)`, then
adds the avatar to `deaths`, and marks `deathInFrame = true` (`Game.js` 89-94).
Server `Avatar.die()` calls shared `die()`, stops the print manager, and emits a
death event with killer data if a body caused the death (`Avatar.js` 180-189).

If the print manager is active when `Avatar.die()` runs, `PrintManager.stop()`
sets printing false before the `die` event. That emits the important
point/property side effects and consumes one hole-distance random value through
`PrintManager.setPrinting(false)` (`PrintManager.js` 78-84, 24-30; `Avatar.js`
166-174, 180-189).

Shared `BaseAvatar.die()` clears bonuses, sets `alive = false`, and adds a point
at the current position (`BaseAvatar.js` 285-290). If the print manager was
active, stopping it can also add a printing-state point before clearing the
visual trail (`PrintManager.js` 78-84; `BaseAvatar.js` 297-309). These death
point side effects need probe-backed traces because they affect collision bodies
and client trail display differently.

Collision is endpoint-circle based. `World.getBody()` checks the four corner
sample points around the query body (`World.js` 104-126). `Island.bodiesTouch()`
uses strict distance `< radius sum` and the stored body's `match()` result
(`Island.js` 83-90). For self-collision, `AvatarBody.match()` ignores recent own
trail bodies until `current_body_num - trail_body_num > trailLatency`
(`AvatarBody.js` 33-40). `trailLatency` is 3 (`BaseAvatar.js` 61).

## Same-Frame Deaths

Source scoring treats all deaths in one `Game.update()` as tied for score. They
all receive the same `score` captured at the start of the frame (`Game.js`
37-42, 89-94).

Source collision resolution is still order-sensitive. The server updates and
checks one avatar at a time in reverse order (`Game.js` 44-75). A trail point
emitted by an earlier-updated avatar can be added to the world before a
later-updated avatar checks collision (`Avatar.js` 23-33; `Game.js` 113-118).
The earlier avatar does not re-check against trail points created later in that
same frame.

So the rule to test is two-part:

- Same update frame: tied deaths get the same death score.
- Same physical setup: collision outcomes may depend on reverse avatar update
  order, especially for head-head or near-head trail interactions.

## Scoring Model

There are two score layers:

- `roundScore`: temporary score inside the current round.
- `score`: match score after `resolveScores()`.

On death, the avatar gets `roundScore += frame_start_deaths_count`
(`Game.js` 37-42, 89-94; `BaseAvatar.js` 317-320; `Avatar.js` 207-210).

On round end, if there is a winner, the winner gets
`roundScore += max(this.avatars.count() - 1, 1)` (`Game.js` 175-188).

Then every avatar runs `resolveScore()`, which adds `roundScore` into `score`
and resets `roundScore` to zero (`Game.js` 190-192; `BaseAvatar.js` 327-331).

This means the source implements rank-like round scoring:

- First death in a present 3-player or 4-player round gets 0.
- A later solo death gets the number of avatars already dead at the start of
  that frame.
- Multiple deaths in the same frame get the same score.
- The sole survivor gets `players - 1`, using total avatar count.
- If everyone dies in the same ending frame, there is no winner bonus.

One-player edge: `resolveScores()` treats the only avatar as winner even if the
round ended with that avatar dead, because it checks `this.avatars.count() === 1`
before looking for an alive avatar (`Game.js` 179-183). The winner bonus is
`max(0, 1) = 1` (`Game.js` 185-187). This is now covered by a focused
`CurvyTronSourceEnv` check in `tests/test_source_env.py`.

## Cases And Current Status

These are source expectations from the code. Some narrow variants are already
fixture-backed; broader variants should stay open until a named fixture proves
them.

### 2 Players

| Case | Setup | Expected source score delta | Why |
| --- | --- | --- | --- |
| 2P solo death | P0 dies, P1 survives | P0 +0, P1 +1 | Narrow wall/lifecycle fixtures pin this. Broader event/wire/public-env variants remain separate. |
| 2P same-frame double death | P0 and P1 die in the same `Game.update()` | P0 +0, P1 +0 | Source scoring rule is pinned in narrow same-frame slices; broader event-order variants remain separate. |
| 2P reverse-order head/trail ambiguity | Force both heads near each other or near same-frame emitted trail | Probe outcome, not assume symmetry | A promoted reverse-order single-death fixture pins one case. Broader geometries remain open. |
| 2P player leaves mid-round | P0 leaves or disconnects while P1 remains | Python/oracle verified for one active 2P fixture | `removeAvatar()` destroys P0, emits `player:leave`, does not add P0 to `deaths`, and ends the round when one avatar remains. Warmdown/terminal leave remains open. |

### 3 Players

| Case | Setup | Expected source score delta | Why |
| --- | --- | --- | --- |
| 3P ordered deaths | P0 dies, later P1 dies, P2 survives | P0 +0, P1 +1, P2 +2 | A direct source-env scoring slice pins ordered 3P wall scoring. Broader lifecycle/public variants remain separate. |
| 3P two die together, one survives | P0 and P1 die in same frame, P2 survives | P0 +0, P1 +0, P2 +2 | Narrow wall and survivor-scoring fixtures pin this shape. Broader event-order variants remain open. |
| 3P one dies, then two die together | P0 dies first; P1 and P2 die in one later frame | P0 +0, P1 +1, P2 +1 | Pinned for a focused tie-at-max/lifecycle scoring path. Broader geometries remain open. |
| 3P all die together | P0, P1, P2 die in one frame | All +0 | Pinned for a focused all-dead next-round fixture. Broader reset/autoreset variants remain open. |
| 3P map/reset canary | Start a 3-player game | Size 95, max score 20, reverse update order P2, P1, P0 | Focused spawn order, warmup, next-round, present/absent, survivor, match-end, tie, and multi-round fixtures are pinned. Broad public lifecycle parity remains open. |

### 4 Players

| Case | Setup | Expected source score delta | Why |
| --- | --- | --- | --- |
| 4P ordered deaths | P0 dies, then P1, then P2, P3 survives | P0 +0, P1 +1, P2 +2, P3 +3 | Direct wall canaries pin a narrow source scoring slice. Broader lifecycle/public variants remain separate. |
| 4P first pair tied | P0 and P1 die together, later P2 dies, P3 survives | P0 +0, P1 +0, P2 +2, P3 +3 | Expected from source rule; keep broad variants open unless a named fixture pins the exact setup. |
| 4P middle pair tied | P0 dies first; P1 and P2 die together; P3 survives | P0 +0, P1 +1, P2 +1, P3 +3 | Expected from source rule; broader fixture coverage is still listed in the gap doc. |
| 4P three die together | P0, P1, P2 die together, P3 survives | P0 +0, P1 +0, P2 +0, P3 +3 | Narrow survivor next-round fixture covers selected 4P survivor behavior; broader tie geometries remain open. |
| 4P all die together | All four die in one frame | All +0 | Focused all-dead next-round and terminal draw canaries exist. Broader reset/autoreset variants remain open. |
| 4P map/reset canary | Start a 4-player game | Size 101, max score 30, reverse update order P3, P2, P1, P0 | Focused first-round spawn order, all-dead next-round, survivor next-round, tied max-score, and active leave continuation are pinned. Broader 4P match lifecycle and present/absent variants remain open. |

## Current Python Gap Notes

The current `curvyzero-v0` Python environment is intentionally not full source
fidelity yet.

- It rejects any config where `players != 2` (`src/curvyzero/env/core.py`
  32-35).
- It uses one terminal reward, `+1/-1/0`, instead of source roundScore/match
  score (`src/curvyzero/env/core.py` 188-197).
- It moves all alive players first, then checks deaths, then applies deaths,
  which differs from source's per-avatar update/check loop
  (`src/curvyzero/env/core.py` 96-134; source `Game.js` 37-80).
- It skips drawing death-frame segments for dead players, while source `die()`
  emits death-position points (`src/curvyzero/env/core.py` 156-162; source
  `BaseAvatar.js` 285-290).
- It already stores source-derived arena and max-score formulas as reference
  metadata (`src/curvyzero/env/config.py` 57-67), but the v0 simulator uses a
  fixed 64 by 64 arena by default (`src/curvyzero/env/config.py` 91-104).

## Probe Priority

1. Keep the promoted 1P/2P/3P/4P wall, same-frame scoring, and lifecycle slices
   green when runners change.
2. Add only missing variants that isolate a new rule, starting with the exact
   gaps in
   [no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md](../../working/environment/no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md).
3. Add broader player-leave/present-continuation probes before copying general
   present/non-present behavior into public env, replay, or trainer contracts.
4. Keep source scoring proof separate from trainer reward design.
