# Print Manager Investigation

Status: source-read plus delayed-start and random call-order fixtures,
2026-05-09

Scope: deterministic `PrintManager` behavior in the original CurvyTron source
and the smallest next fixtures for CurvyZero source-fidelity work. This note is
working memory for the promoted source-fidelity fixtures.

## Source Files Read

- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `third_party/curvytron-reference/src/server/model/Avatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseTrail.js`
- `third_party/curvytron-reference/src/server/model/Trail.js`
- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`
- `third_party/curvytron-reference/src/server/controller/GameController.js`
- `third_party/curvytron-reference/src/server/model/BonusStack.js`
- `tools/reference_oracle/scenario_runner.js`
- `src/curvyzero/env/trace_compare.py`

## Source Facts

`PrintManager` owns four runtime fields:

- `active`: starts `false`.
- `lastX`: starts `0`.
- `lastY`: starts `0`.
- `distance`: starts `0`.

Prototype constants:

- `printDistance = 60`.
- `holeDistance = 5`.

`start()`:

- Runs only when `active` is false.
- Sets `active = true`.
- Sets `lastX = avatar.x` and `lastY = avatar.y`.
- Calls `setPrinting(true)`.

`stop()`:

- Runs only when `active` is true.
- Sets `active = false`.
- Calls `setPrinting(false)`.
- Calls `clear()`, which resets `active=false`, `distance=0`, `lastX=0`,
  and `lastY=0`.

`test()`:

- Does nothing when `active` is false.
- Subtracts the Euclidean distance from the previous manager point to the
  avatar's current point:
  `distance -= sqrt((lastX - avatar.x)^2 + (lastY - avatar.y)^2)`.
- Then updates `lastX` and `lastY` to the current avatar position.
- Then toggles printing only if `distance <= 0`.
- There is no carryover of overshoot. A toggle resets `distance` to a fresh
  random print or hole distance.

`togglePrinting()` calls `setPrinting(!avatar.printing)`.

`setPrinting(printing)`:

- Calls `avatar.setPrinting(printing)`.
- Then sets `distance = getRandomDistance()` based on the avatar's new
  `printing` value.

Random distance formulas:

```text
if avatar.printing:
  distance = 60 * (0.3 + Math.random() * 0.7)
else:
  distance = 5 * (0.8 + Math.random() * 0.5)
```

So the source ranges are:

- Printing span: `[18, 60)` because `Math.random()` is `[0, 1)`.
- Hole span: `[4, 6.5)`.

When the oracle's deterministic `Math.random()` returns `0.5`:

- New printing distance: `60 * (0.3 + 0.5 * 0.7) = 39`.
- New hole distance: `5 * (0.8 + 0.5 * 0.5) = 5.25`.

A constant random value is enough for single-threshold toggle fixtures, but it
cannot prove random stream call order. Same-frame PrintManager toggles need a
non-constant random tape so the final per-avatar distances reveal which source
object consumed each value.

At base speed `16` units/s with `step_ms = 1000 / 60`, straight movement is
`16 * (1000 / 60) / 1000 = 0.2666666667` units per tick. With deterministic
random:

- A fresh print span of `39` toggles to a hole on the 147th straight tick.
- A fresh hole span of `5.25` toggles back to printing on the 20th straight
  tick.

For fixtures, do not wait 147 ticks. Force `print_manager.distance` near the
threshold.

## Update Order

`Game.update(step)` iterates avatars in reverse player-array order.

For each alive avatar:

1. `avatar.update(step)`.
2. Border check.
3. Body collision check, if no border death and not invincible.
4. If still alive, `avatar.printManager.test()`.
5. Then `bonusManager.testCatch(avatar)`.

For PrintManager random calls, this means same-frame `test()` toggles consume
`Math.random()` in reverse avatar order. The random draw happens inside
`PrintManager.setPrinting()`, after `avatar.setPrinting()` emits the boundary
point/property side effects and before `setPrinting()` returns.

Inside `avatar.update(step)`:

1. `updateAngle(step)`.
2. `updatePosition(step)`.
3. If `avatar.printing` and `isTimeToDraw()` are true, `addPoint(x, y)`.

`isTimeToDraw()` is true when `trail.lastX === null`, otherwise when the
distance from the last trail point to the current position is greater than
`avatar.radius` (`0.6` by default).

Important consequence: normal printed points from `avatar.update()` happen
before border/body collision. Print-manager toggles happen after collision.
A print-manager hole toggle cannot prevent a body collision that has already
been checked in that frame.

## Event And State Side Effects

`Avatar.setPrinting(printing)` always emits a server-side `property` event after
calling `BaseAvatar.setPrinting()`, even if the boolean did not change.

`BaseAvatar.setPrinting(printing)` only mutates trail state when the boolean
changes:

- It coerces `printing` to boolean.
- If the value changes, it assigns `avatar.printing`.
- It calls `addPoint(avatar.x, avatar.y, true)`.
- If the new value is false, it then clears the trail.

Because server `Avatar.addPoint()` overrides the base method:

- It first appends to `avatar.trail.points` and updates `trail.lastX/lastY`.
- Then it emits a `point` event with `{avatar, x, y, important}`.

When printing changes from true to false:

1. The avatar emits an important `point` at the current position.
2. `BaseAvatar.setPrinting()` clears the trail.
3. Server `Trail.clear()` emits a `clear` event on the trail object.
4. `Avatar.setPrinting()` emits `property` with `property: "printing"` and
   `value: false`.

The game does not listen to `Trail.clear()`. It listens to avatar `point`.
`Game.onPoint()` immediately inserts an `AvatarBody` into the world when
`game.started && game.world.active` are true. Clearing the visual trail does
not remove already inserted world bodies.

When printing changes from false to true:

1. The avatar emits an important `point` at the current position.
2. The trail is not cleared.
3. `Avatar.setPrinting()` emits `property` with `property: "printing"` and
   `value: true`.

Normal `Avatar.update()` trail points are not important. Print-manager boundary
points are important.

`GameController.onPoint()` forwards only important points to clients, but
`Game.onPoint()` inserts world bodies for all avatar point events while the
world is active.

Death-time order is now pinned for an active, currently-printing manager,
including both normal-wall and seeded body-collision deaths:

1. `Game.kill()` calls `avatar.die(...)`.
2. `BaseAvatar.die()` marks the avatar dead and emits a normal non-important
   point at the death position.
3. `Avatar.die()` calls `printManager.stop()`.
4. `PrintManager.stop()` sets `active = false`, calls `setPrinting(false)`, and
   then `clear()`.
5. Because p0 was printing, `setPrinting(false)` emits an important stop point,
   clears the visual trail, and emits `property printing=false`.
6. `clear()` resets `active=false`, `distance=0`, `lastX=0`, and `lastY=0`.
7. Only then does `Avatar.die()` emit the `die` event. For body collisions,
   that event carries the body owner as `killer` and the body age flag as `old`.

`PrintManager.test()` does not run after a death because `Game.update()` only
calls it inside `if (avatar.alive)` after border/body collision checks.

## Setup Side Effects

The source round lifecycle is delayed:

- `BaseGame.newRound()` calls `Game.onRoundNew()` and schedules `start()` after
  `warmupTime = 3000`.
- `Game.onStart()` emits `game:start`, schedules each `avatar.printManager.start`
  after `3000`, activates the world, then calls base start logic.

So source trail printing starts 3000 ms after `game:start`, not immediately at
round creation.

The JS oracle now has a controlled timer queue. Normal forced scenarios do not
fire queued callbacks unless the fixture explicitly advances fake time. For
forced scenarios, the important setup behavior is still in `applyForcedState()`:

- If state has no `print_manager`, no `printManager`, and no `trail`, then
  `printing: false` calls `avatar.printManager.stop()`, while any other case
  calls `avatar.printManager.start()`.
- If state has `print_manager`, `printManager`, or `trail`, then setup avoids
  `start()`/`stop()` and directly assigns `avatar.printing` only when
  `state.printing` is boolean.

This means a scenario with `printing: true` and no forced print/trail runtime
state can create setup-time side effects:

- `printManager.start()` sets `active`, `lastX`, `lastY`, and random distance.
- It emits a setup-time important point and property event.
- If the oracle has already set `game.started = true` and `world.active = true`,
  `Game.onPoint()` inserts a world body during setup.
- The oracle clears `events` before the first tick, but world/trail/body side
  effects remain.

Use forced `trail` or forced `print_manager` state in print-manager fixtures to
avoid accidental setup-time `start()`/`stop()` behavior.

## Delayed Start Rule

`Game.onStart()` schedules `avatar.printManager.start` once per avatar with
`setTimeout(..., 3000)`, then activates the world. The scheduled
`PrintManager.start()` behavior is:

1. If the manager is inactive, set `active = true`.
2. Copy the current avatar position into `lastX` and `lastY`.
3. Call `setPrinting(true)`.
4. `setPrinting(true)` emits an important point if printing changes from false
   to true, emits `property printing=true`, and sets a fresh print distance.

With deterministic `Math.random() = 0.5`, the fresh print distance is `39`.

The promoted fixture is
`scenarios/environment/source_print_manager_delayed_start_timer_step.json`. It
invokes `Game.onStart()` after forced state, advances the controlled oracle
timer queue by `2999 ms`, then by `1 ms`, and captures both ticks with
`step_ms = 0`.

Pinned trace:

- At `2999 ms`: `printing=false`, `printManager.active=false`,
  `distance=0`, `lastX=0`, `lastY=0`, and no trail/body point.
- At `3000 ms`: the timer fires before the zero-step update. Event order is
  important point, `property printing=true`, then the source zero-step
  `position` event. State is `printing=true`, `trailPointCount=1`,
  `lastTrailPoint=[20, 40]`, `bodyNum=1`, `bodyCount=1`,
  `worldBodyCount=1`, and
  `printManager={active:true,distance:39,lastX:20,lastY:40}`.

## JS Oracle Support

`tools/reference_oracle/scenario_runner.js` already supports deterministic
print-manager setup:

- It replaces `Math.random()` with a function that returns `0.5`.
- It can instead consume an exhausting tape from
  `source_setup.random.math_random_sequence`; each value must be in `[0, 1)`.
- It writes a top-level `randomCalls` log with `{index, value}` entries in
  consumption order. The log intentionally does not name the consumer; final
  source state and event order attribute the value.
- It supports opt-in timer advancement through
  `source_setup.game.invoke_on_start` and
  `time_policy.timer_advance_ms_sequence`.
- Raw avatar snapshots include:
  `printing`, `trailPointCount`, `lastTrailPoint`, `bodyNum`, `bodyCount`,
  and `printManager.active/distance/lastX/lastY`.
- `players[].initial.print_manager` or `printManager` can force:
  `active`, `distance`, `last_x`/`lastX`, and `last_y`/`lastY`.
- `players[].initial.trail.points` can force the trail points. Optional
  `last_x`/`lastX` and `last_y`/`lastY` override the inferred last point.
- Forced state order is: basic avatar state for all players, then forced trail
  state, then forced print-manager state, then world bodies, then body counters.

Fixture warning: when forced runtime state is present, `printing` must still be
set explicitly if the fixture needs true printing. Otherwise the avatar stays at
the default false value from `clear()`.

## Current Common Trace Status

Confirmed from `src/curvyzero/env/trace_compare.py`:

- Common trace now has a narrow print-manager opt-in for
  `source_print_manager_*` scenarios and `source-print-manager-canary`.
- Per player, that opt-in compares `printing`, `trailPointCount`,
  `lastTrailPoint`, `bodyNum`, `bodyCount`, and `printManager.active`,
  `printManager.distance`, `printManager.lastX`, and `printManager.lastY`.
- Per frame, it compares `worldBodyCount`.
- When `comparison.include_events` is true, `property` events now include
  `player_id`, `property`, and `value`.

Do not add a common `trail.clear` event yet unless a later fixture records trail
clear directly. The current JS oracle does not subscribe to trail `clear`; final
trail state and point/property order are enough for the current print-manager
fixtures.

## Implemented Fixture Checkpoint

The main promoted batch uses forced manager state, active world,
`source_setup.game.started = true`, `source_setup.game.in_round = true`, and
`Math.random() = 0.5`. Toggle and delayed-start fixtures stay one-player;
death-stop fixtures use three players to avoid round-end noise.

Verified through `source-print-manager-canary`:

1. `source_print_manager_print_to_hole_step`
   - Starts `printing: true`, manager distance `1`, last manager point `(20,40)`,
     and trail last point `(21.6,40)` to suppress normal drawing.
   - Moves to `(21.6,40)`, toggles to hole, emits one important point and one
     `property printing=false`, clears the visual trail, sets distance `5.25`,
     and leaves one world body.

2. `source_print_manager_hole_to_print_step`
   - Starts `printing: false`, manager distance `1`, last manager point
     `(20,40)`.
   - Moves to `(21.6,40)`, toggles to print, emits one important point and one
     `property printing=true`, keeps one trail point, sets distance `39`, and
     leaves one world body.

3. `source_print_manager_exact_zero_toggle_step`
   - Starts `printing: false`, manager distance `0`, and last manager point
     equal to the current avatar position `(20,40)`.
   - Uses a `0 ms` step with no seeded bodies, so collision stays clear and
     `PrintManager.test()` subtracts `0`.
   - The source `distance <= 0` branch toggles to print, emits one important
     point and one `property printing=true`, keeps one trail point, sets
     distance `39`, and leaves one world body.

4. `source_print_manager_no_toggle_control_step`
   - Starts `printing: false`, manager distance `10`, last manager point
     `(20,40)`.
   - Moves to `(21.6,40)`, subtracts distance to `8.4`, updates `lastX/lastY`,
     emits only position, and leaves no trail point or world body.

5. `source_print_manager_delayed_start_timer_step`
   - Starts `printing: false` with inactive manager state and invokes
     `Game.onStart()` after forced state.
   - At `2999 ms`, no print-manager start has fired.
   - At `3000 ms`, `PrintManager.start()` sets `active=true`, copies
     `(20,40)` into `lastX/lastY`, emits an important point and
     `property printing=true`, sets distance `39`, and leaves one world body.

6. `source_print_manager_active_stop_on_death_step`
   - Three-player fixture to avoid round-end noise.
   - p0 starts `printing: true` with active manager distance `10`, crosses the
     normal right wall, and dies at `(95.5,47.5)`.
   - Event order is p2 position, p1 position, p0 position, p0 non-important
     death point, p0 important stop point, `property printing=false`, `die`, and
     `score:round`.
   - Final p0 state is `alive: false`, `printing: false`, empty visible trail,
     `bodyCount: 2`, and cleared manager
     `{active:false,distance:0,lastX:0,lastY:0}`.
   - `worldBodyCount` is `2`; the stop point remains a collision body even
     though the visual trail was cleared.

7. `source_print_manager_active_hole_stop_on_death_step`
   - Three-player fixture to avoid round-end noise.
   - p0 starts `printing: false` with active manager distance `10`, crosses the
     normal right wall, and dies at `(95.5,47.5)`.
   - Event order is p2 position, p1 position, p0 position, p0 non-important
     death point, `property printing=false`, `die`, and `score:round`.
   - Because p0 was already in a hole, `PrintManager.stop()` emits no important
     stop point and does not clear the death trail point.
   - Final p0 state is `alive: false`, `printing: false`, one visible trail
     point at the death position, `bodyCount: 1`, and cleared manager
     `{active:false,distance:0,lastX:0,lastY:0}`.
   - `worldBodyCount` is `1`.

8. `source_print_manager_body_collision_stop_on_death_step`
   - Three-player fixture to avoid round-end noise.
   - p0 starts `printing: true` with an active manager distance `10`, a trail
     cursor at `(20,20)`, and a seeded overlapping p1 `AvatarBody` at
     `(21.19,20)`.
   - The `0 ms` step emits only position events before collision; the trail
     cursor suppresses a normal pre-collision point.
   - Body collision kills p0 before `PrintManager.test()`, so manager distance
     and last point are not updated by the test path.
   - Event order is p2 position, p1 position, p0 position, p0 non-important
     death point, p0 important stop point, `property printing=false`,
     `die(killer=p1, old=false)`, and `score:round`.
   - Final p0 state is `alive: false`, `printing: false`, empty visible trail,
     `bodyCount: 2`, and cleared manager
     `{active:false,distance:0,lastX:0,lastY:0}`.
   - The seeded p1 body plus p0's death and stop points leave
     `worldBodyCount` at `3`.

Main verified batch:
`uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-delayed-start-batch`
protects the deterministic PrintManager toggle, delayed-start, and death-stop
claim.

Single delayed-start loop:
`uv run python tools/run_fidelity_loop.py scenarios/environment/source_print_manager_delayed_start_timer_step.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-delayed-start-loop`
matches the source/common trace for the delayed-start fixture.

Targeted tests:
`uv run --extra dev pytest tests/test_env_scenarios.py tests/test_run_fidelity_batch.py`
act as regression hygiene around the scenario and batch harness.

Random call-order probe:

- `source_print_manager_random_call_order_step`
  - Two-player fixture with no timers, no round lifecycle, no bonuses, and
    `step_ms = 0`.
  - p0 and p1 both start in a hole with active PrintManagers at exact-zero
    distance.
  - `Game.update()` visits p1 before p0, so p1 toggles first and consumes tape
    value `0.1`; p0 consumes `0.9`.
  - The source print-distance formula makes this visible as p1 distance `22.2`
    and p0 distance `55.8`.
  - Raw JS events are p1 position, p1 important point, p1
    `property printing=true`, then the same three events for p0.
  - Python `source-print-manager-canary` now consumes
    `source_setup.random.math_random_sequence` for PrintManager distance calls,
    reports top-level `randomCalls`, and matches JS through common trace.

Separate random batch:
`uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_random_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-random-batch`
previously covered the single call-order row with `1` pass, `0` fail, `0`
blocked.

Random cadence probe:

- `source_print_manager_random_cadence_multistep`
  - One-player, four-tick fixture with no timers, no round lifecycle, no
    bonuses, and `step_ms = 1000`.
  - p0 starts in a hole with an active PrintManager at exact-zero distance.
  - Tape value `0.0` creates a real source print distance of `18` at `x=26`;
    the next two ticks spend `16 + 16` units from that taped distance.
  - The print-to-hole tick emits the normal printed point, then the important
    PrintManager boundary point/property event, clears the visual trail, and
    consumes tape value `0.5` for hole distance `5.25`.
  - The final hole tick spends that real hole distance, toggles back to
    printing, and consumes tape value `0.25` for the next print distance
    `28.5`.
  - Python `source-print-manager-canary` reports random calls
    `[0, 0.5, 0.25]`, preserves body/trail counters, and matches JS through
    common trace.

Current random batch:
`uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_random_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-random-cadence-batch`
with `2` pass, `0` fail, `0` blocked.

Focused local tests:
`uv run --extra dev pytest tests/test_env_scenarios.py::test_js_scenario_runner_pins_print_manager_random_cadence_multistep tests/test_env_scenarios.py::test_source_print_manager_random_runner_matches_js_common_trace tests/test_run_fidelity_batch.py::test_loads_source_print_manager_random_batch_manifest`
with `4` passed.

## Risks And Gaps

- Setup-time `printManager.start()` can silently add world bodies if a fixture
  sets `printing: true` without forced trail or print-manager runtime state.
- A print-to-hole toggle emits a point before clearing the trail. Raw trace must
  check both the important point event and the final empty trail.
- World bodies are not removed by `Trail.clear()`. Visual holes and collision
  bodies are different state.
- Print-manager toggles run after collision, so hole fixtures should not be used
  to explain away a same-frame body collision.
- Active stop-on-death is covered for `printing: true` normal-wall death,
  already-hole normal-wall death, and seeded body-collision death with two
  survivors. It does not yet cover natural emitted-body collision death.
- The delayed `PrintManager.start()` callback is covered in isolation. Full
  round lifecycle timers (`newRound` warmup, `game:start`, `round:end`,
  warmdown, `game:stop`, and next round) are still separate.
- Timed bonus spawn/expiry and `AvatarBody.oldAge` controlled-clock
  `old:true` death metadata are separate timer fixtures.
- Common trace can prove the current print-manager toggle slice, but only through
  the narrow opt-in fields for `source_print_manager_*`.
- The main eight-case PrintManager batch still uses constant
  `Math.random() = 0.5`; the random-tape call-order and cadence fixtures are
  intentionally separated so batch cadence changes stay explicit.
- The promoted cadence fixture covers one straight-line print-to-hole-to-print
  cycle from real taped distances. Broader timer/lifecycle-driven cadence is
  still separate.
- Browser/replay messages are still out of scope. State/event parity comes
  first.
