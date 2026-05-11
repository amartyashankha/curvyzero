# Round Lifecycle, Natural Spawn, And Row-Local RNG Plan

Status: Python direct parity for the current pinned lifecycle/spawn/RNG slice

Worker E scope: natural round lifecycle, natural spawn, and RNG stream order for
source-fidelity planning. This does not change reset, observation, LightZero, or
speed code.

## Source Files Read

- `third_party/curvytron-reference/src/shared/model/BaseRoom.js`: `newGame`
  creates `Game` and emits `game:new` at lines 94-105.
- `third_party/curvytron-reference/src/server/controller/RoomController.js`:
  launch/ready flow calls `room.newGame()` at lines 320-349, 527-539, and
  emits `room:game:start` at lines 703-706.
- `third_party/curvytron-reference/src/server/controller/GameController.js`:
  player ready flow calls `game.newRound()` at lines 282-287; socket event
  forwarding for `game:start`, `game:stop`, `round:new`, and `round:end` is at
  lines 458-490.
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`: warmup is
  `3000` ms at line 53; warmdown is `5000` ms at line 60; spawn margins are
  lines 67 and 74; lifecycle methods are `start` lines 106-112, `stop` lines
  117-123, `onStart` lines 144-149, `onStop` lines 154-165, `newRound` lines
  321-329, and `endRound` lines 335-341.
- `third_party/curvytron-reference/src/server/model/Game.js`: update/death
  loop is lines 37-80; round end scoring/event is lines 221-225; natural spawn
  is lines 230-251; game start/stop are lines 257-289.
- `third_party/curvytron-reference/src/server/core/World.js`: spawn position
  and heading helpers are lines 165-197; random angle/point calls are lines
  251-265; `clear` deactivates the world at lines 357-365 and `activate` is
  lines 370-373.
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`: start
  and random print/hole distances are lines 53-72; stop/test are lines 78-103.
- `third_party/curvytron-reference/src/server/manager/BonusManager.js`: start
  timeout and pop flow are lines 24-33 and 59-74; bonus position/type/time RNG
  is lines 84-99 and 157-197.

## Exact Source Facts

1. Room launch does not itself start a round. `RoomController.launch()` calls
   `room.newGame()`, which constructs a `Game` and emits `game:new`. The first
   natural round starts later when `GameController.checkReady()` sees every
   present avatar ready and calls `game.newRound()`.

2. `BaseGame.newRound(time)` immediately sets `started=true`, then, if no round
   is active, sets `inRound=true`, calls `onRoundNew()`, and schedules
   `start()` after `time` or the default `warmupTime=3000`.

3. Server `Game.onRoundNew()` emits `round:new` before shared clearing and
   before natural spawn. It then calls `BaseGame.onRoundNew()`, resets
   `roundWinner`, clears `world`, clears `deaths`, clears `bonusStack`, and
   loops avatars in reverse collection order.

4. Natural spawn random-call order is per present avatar, reverse avatar order:
   `getRandomPosition()` draws x then y through `World.getRandomPoint()`, then
   `getRandomDirection()` draws one or more random angles through
   `World.getRandomAngle()` until `isDirectionValid()` accepts the heading.
   Non-present avatars consume no spawn RNG and are added to `deaths`.

5. Natural spawn currently has no inter-avatar body rejection after
   `Game.onRoundNew()` clears the world. `World.clear()` sets
   `active=false` and clears all island bodies. Spawn calls `testBody()`, but no
   spawn body is inserted into the world; the first normal spawn body is added
   later by point emission while `game.started && world.active`.

6. `Game.onStart()` emits `game:start`, schedules every avatar's
   `printManager.start` for 3000 ms later in reverse avatar order, activates
   the world, then calls `BaseGame.onStart()`. With bonuses enabled,
   `BonusManager.start()` immediately consumes one random value for the first
   bonus pop timeout. With `bonuses=[]`, that bonus-time RNG call is absent.

7. Delayed `PrintManager.start()` timers with the same due time fire in the
   order they were scheduled: reverse avatar order. Each start sets
   `active=true`, copies current avatar position to `lastX/lastY`, calls
   `setPrinting(true)`, emits the important point/property side effects, and
   consumes one random value for the initial print distance.

8. During `Game.update(step)`, avatars update in reverse order. Any
   `PrintManager.test()` toggle RNG is therefore also reverse-avatar ordered for
   same-frame toggles. Existing fixtures already pin this for PrintManager only:
   `source_print_manager_random_call_order_step.json` and
   `source_print_manager_random_cadence_multistep.json`.

9. Round termination order is death events and death `score:round` events inside
   the update, then `checkRoundEnd()`, then `endRound()`. `Game.onRoundEnd()`
   resolves winner/score events first and emits `round:end` afterward.
   `BaseGame.endRound()` then schedules `stop()` after `warmdownTime=5000`.

10. When the warmdown timer fires, `BaseGame.stop()` only runs `onStop()` if a
    frame timer exists. In the live source loop this is true because
    `BaseGame.start()` called `loop()` and `newFrame()`. A fixture that only
    calls `Game.onStart()` is not enough to prove `game:stop` or next-round
    behavior.

11. `Game.onStop()` emits `game:stop`, runs shared stop cleanup, then calls
    `isWon()`. If the match is won it calls `end()`; otherwise it immediately
    calls `newRound()`, so the next `round:new` happens synchronously after
    `game:stop`.

12. If an active `PrintManager` is stopped by death, source
    `PrintManager.stop()` calls `setPrinting(false)`, emits the important
    point/property side effects through `Avatar.setPrinting()`, and consumes one
    hole-distance random value before `Avatar.die()` emits `die`.

13. A source `Game.update()` that kills both 2P avatars in one frame leaves no
    round winner. Death handling happens in reverse avatar order, then
    `resolveScores()` emits zero-score `score` events in reverse avatar order,
    then `round:end` emits with `winner=null`.

14. The focused 3P all-dead continuation is now pinned narrowly: forced
    same-frame wall deaths produce `round:end` winner null at `3000 ms`, then
    `game:stop` and next `round:new` at `8000 ms`, followed by the next natural
    3P spawn RNG/order. This fixture itself does not prove 3P survivor scoring,
    broader present/non-present variants, or the focused all-present 3P
    `max_score: 2` match-end path that is now covered separately.

15. The focused 3P present/non-present continuation is now pinned narrowly:
    avatar 2 is non-present, the two present avatars die in one elapsed-ms
    update, `game:stop` resizes the arena from size 95 to present-player size
    88, and the next `round:new` re-adds avatar 2 to deaths while spawning only
    avatars 3 and 1. This does not prove multi-round match, broader 4P
    lifecycle, or broader present/non-present variants. The focused
    all-present 3P `max_score: 2` match-end path is covered separately.

## Row-Local RNG Policy Needed

Use one source-style random stream per environment row and record every draw in
source chronological order. The minimum call-site labels needed before stronger
source-faithful training claims are:

- `spawn.position_x`, `spawn.position_y`, `spawn.angle_attempt_n`
- `bonus.start_timeout`, `bonus.pop_timeout`, `bonus.type`,
  `bonus.position_x`, `bonus.position_y`
- `print_manager.start_distance`, `print_manager.stop_distance`,
  `print_manager.toggle_distance`

For the no-bonus training subset, the bonus labels can be absent by rule, but
the scenario metadata must say bonuses are disabled. Do not share one RNG stream
across vector rows; row-local tape/state/cursor must be part of reset metadata
and replay references.

## First Runnable Fixture Landed

Worker H added `tools/reference_oracle/lifecycle_oracle.js` and
`scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_2p.json`.
The proof constructs the source `Game`, leaves `started=false/inRound=false`,
attaches listeners, calls `game.newRound(0)`, and advances controlled timers.
It pins this first source lifecycle order:

- `round:new`;
- reverse-avatar spawn RNG for position x/y and angle, with source
  `position`/`angle` events;
- `game:start` through the real `BaseGame.start()` path;
- delayed reverse-avatar `PrintManager.start()` calls at 3000 ms, including
  important point/property side effects and start-distance RNG.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_2p.json
```

Worker L added a focused pytest source-claim gate for the same fixture:

```text
uv run --extra dev pytest tests/test_lifecycle_oracle.py
```

The oracle uses a controlled `Date`, controlled timers, and labeled
`Math.random` tape. This first fixture intentionally stops before terminal
death, warmdown, `game:stop`, and next `round:new`.

## Second Runnable Fixture Landed

Worker K extended `tools/reference_oracle/lifecycle_oracle.js` and added
`scenarios/environment/source_lifecycle_spawn_rng_2p_next_round.json`.
The proof starts the real source loop with `game.newRound(0)`, advances through
delayed `PrintManager.start()`, then uses the smallest harness action support to
place both avatars at opposite walls and call source `Game.update(100)`.

It pins this additional source lifecycle order:

- terminal update death handling in reverse avatar order at 3000 ms;
- active `PrintManager.stop()` point/property side effects and
  `print_manager.stop_distance` RNG before each `die`;
- zero-score double-death resolution and `round:end` with `winner=null`;
- warmdown `game:stop` 5000 ms after `round:end`;
- synchronous next `round:new` from `Game.onStop()`;
- next-round reverse-avatar spawn RNG/position/angle at 8000 ms.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_spawn_rng_2p_next_round.json
```

## Third Runnable Fixture Landed

`scenarios/environment/source_lifecycle_spawn_heading_rejection_retry_2p.json`
pins one rejected spawn heading attempt followed by the accepted retry. This
keeps heading rejection inside the current 2P source claim, not in the missing
list for that slice.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_spawn_heading_rejection_retry_2p.json
```

## Fourth Runnable Fixture Landed

`scenarios/environment/source_lifecycle_spawn_rng_order_3p.json` pins only the
3P first-round natural spawn order and RNG labels. At 0 ms, avatars spawn in
reverse order 3, 2, 1, and each avatar consumes `position_x`, `position_y`, then
`angle_attempt_0`.

It does not prove 3P survivor scoring, 3P present/absent continuation, broader
present/non-present variants, bonuses, optimized/vector lifecycle, or
trainer/replay final observation. Separate fixtures now prove focused 3P
warmup/delayed PrintManager start, focused 3P all-dead next-round continuation,
and one focused all-present 3P `max_score: 2` match-end path.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_spawn_rng_order_3p.json
```

## Fifth Runnable Fixture Landed

`scenarios/environment/source_lifecycle_present_absent_3p_round_new.json` pins
one 3P first-round `Game.onRoundNew()` present/non-present case. Avatar 2 is
non-present, source skips avatar 2 for natural spawn RNG, spawns avatar 3 then
avatar 1, and adds avatar 2 to `game.deaths`.

The snapshot pins avatar 2 as `alive=false`, `present=false`, at `(0.6, 0.6)`,
with `deathCount=1` and `deaths=[2]`.

It does not prove 3P survivor scoring, 3P present/absent continuation, bonuses,
optimized/vector lifecycle, or trainer/replay final observation. Separate
fixtures now prove focused 3P warmup/delayed PrintManager start, focused 3P
all-dead next-round continuation, and one focused all-present 3P `max_score: 2`
match-end path.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_present_absent_3p_round_new.json
```

## Sixth Runnable Fixture Landed

`scenarios/environment/source_lifecycle_match_end_at_max_score_2p.json` pins
one 2P max-score match-end path only. With `max_score: 1`, avatar 2 dies,
avatar 1 reaches score 1, source emits `round:end` with winner 1 at 3000 ms,
then emits `game:stop` and `end` at 8000 ms. It does not immediately emit
another `round:new`.

The final snapshot pins `started=false`, `inRound=false`, cleared world fields,
and no avatars.

It does not prove multi-round match, 3P/4P lifecycle, bonuses,
reset/autoreset, optimized/vector lifecycle, or trainer/replay final
observation.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_match_end_at_max_score_2p.json
```

## Seventh Runnable Fixture Landed

`scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_3p.json`
pins focused 3P warmup and delayed PrintManager start. It proves first-round
spawn order/RNG labels, `game:start`, and `print_manager:start` order/random
calls after 3000 ms. The source order is avatar 3, then 2, then 1.

It does not prove 3P survivor scoring, 3P present/absent continuation, bonuses,
optimized/vector lifecycle, or trainer/replay final observation. One focused
all-present 3P `max_score: 2` match-end path is covered separately.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_3p.json
```

## Eighth Runnable Fixture Landed

`scenarios/environment/source_lifecycle_spawn_rng_order_4p.json` pins only the
4P first-round natural spawn order and RNG labels. At 0 ms, avatars spawn in
reverse order 4, 3, 2, 1, and each avatar consumes `position_x`, `position_y`,
then `angle_attempt_0`.

It does not prove broader 4P match lifecycle beyond the separately promoted
all-dead and survivor next-round fixtures, match
end, bonuses, optimized/vector lifecycle, or trainer/replay final observation.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_spawn_rng_order_4p.json
```

## Ninth Runnable Fixture Landed

`scenarios/environment/source_lifecycle_spawn_rng_3p_next_round.json` pins only
the focused 3P all-dead warmdown/next-round path. After delayed PrintManager
start, all three avatars die on forced same-frame wall collisions. Source emits
`round:end` with `winner=null` at 3000 ms, then `game:stop` and the next
`round:new` at 8000 ms, and consumes the next natural 3P spawn RNG/order.

It does not prove 3P survivor scoring, 3P present/absent continuation, broader
4P lifecycle, bonuses, optimized/vector lifecycle, or trainer/replay final
observation. One focused all-present 3P `max_score: 2` match-end path is
covered separately.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_spawn_rng_3p_next_round.json
```

## Focused 3P Match-End Fixture Landed

`scenarios/environment/source_lifecycle_match_end_at_max_score_3p.json` pins
only one all-present 3P `max_score: 2` match-end path. After delayed
PrintManager start, avatars 3 then 2 die on forced wall collisions, avatar 1
reaches score 2, source emits `round:end` with winner 1 at 3000 ms, then emits
`game:stop` and `end` at 8000 ms with no immediate next `round:new`.

It does not prove multi-round match, broader 4P match lifecycle, broader
present/non-present variants, vector lifecycle, or trainer/replay/final
observation.

Runnable check:

```text
node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_match_end_at_max_score_3p.json
```

## Python Direct Parity Landed

The narrow Python lifecycle runner now matches the JS oracle for all 24 pinned
lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`,
`source_lifecycle_survivor_score_4p_next_round`,
`source_lifecycle_present_absent_3p_survivor_score_round_end`, and
`source_lifecycle_multi_round_match_end_3p`: three 2P core lifecycle fixtures,
the focused 3P spawn-order fixture, the focused 3P warmup/PrintManager-start
fixture, the focused 4P first-round spawn-order fixture, the focused 4P
all-present all-dead warmdown/next-round fixture, the focused 4P survivor
warmdown/next-round fixture, the focused 3P present/non-present first-round
fixture, the focused 3P present/non-present survivor-scoring fixture, the
focused 3P present/non-present warmdown/next-round fixture, the focused 2P
max-score match-end fixture, the
focused all-present 3P `max_score: 2` match-end fixture, the focused
all-present 3P multi-round match-end fixture, the focused 3P
all-dead warmdown/next-round fixture, the focused 3P survivor-scoring
`round:end` fixture, the focused 3P survivor warmdown/next-round fixture, and
the focused 3P tie-at-max continuation fixture.
`tests/test_source_lifecycle_runner.py`
compares Python events, `randomCalls`, snapshots, timer advances, expectations,
and lifecycle action metadata against `tools/reference_oracle/lifecycle_oracle.js`.

Focused check:

```text
uv run --extra dev pytest tests/test_source_lifecycle_runner.py -q
```

This is only the current pinned slice. It proves 2P warmup/start,
2P terminal-to-next-round, one 2P heading-rejection retry, focused 3P
first-round spawn-order/RNG labels, focused 3P warmup/PrintManager start,
focused 4P first-round spawn-order/RNG labels, focused 4P all-dead and
survivor next-round paths, one focused 3P first-round present/non-present
`onRoundNew()` case with avatar 2 skipped for spawn RNG and added to deaths,
one focused 3P present/non-present survivor-scoring case, plus one focused 2P
max-score match-end case with
`game:stop`/`end` and no immediate next `round:new`, plus one focused
all-present 3P `max_score: 2` match-end path with avatars 3 then 2 dying,
avatar 1 reaching score 2, `game:stop`/`end`, and no immediate next
`round:new`. The separate focused 3P all-dead continuation proves
`round:end` winner null, `game:stop`, next
`round:new`, and next natural spawn RNG/order. The focused 3P survivor fixture
proves survivor scoring through `round:end`; the separate focused survivor
warmdown fixture proves avatar 1 keeps moving and dies at 4150 ms, then
`game:stop` and next `round:new` emit at 8000 ms with next natural spawn
RNG/order. The focused 3P present/non-present continuation proves one absent
avatar across warmdown, stop-time resize, and next natural spawn. The focused
3P tie-at-max fixture proves tied leaders continue to next round, not broad
multi-round. The focused 3P all-present multi-round match-end fixture proves
only `source_lifecycle_multi_round_match_end_3p`: score 2 carries through
`game:stop` and `round:new`, then score 4 emits `game:stop` and `end` with no
later `round:new`. This does not prove broader 4P match lifecycle, broader
present/non-present variants, bonuses, production reset/autoreset,
optimized/vector lifecycle, or trainer/replay/final observation.

## Next Exact Source-Fidelity Step

Add one broader source fixture only when it isolates a missing lifecycle rule.
The highest-value next cases are broader 4P match lifecycle only
if it isolates a new rule, or broader present/non-present
variants only if a specific rule needs it. Promote vector row-local RNG arrays
only after the reset/timer shape is clear and the vector path can compare back
to the 21 promoted lifecycle source fixtures without claiming more than they
prove.
