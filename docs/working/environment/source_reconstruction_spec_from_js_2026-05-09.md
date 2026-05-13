# CurvyTron Source Reconstruction Spec From JS

Status: working implementation spec
Date: 2026-05-09
Scope: original CurvyTron JS source under `third_party/curvytron-reference`,
existing source-map docs under `docs/research/curvytron_source_map/`, and
current JS/Python oracle fixtures under `scenarios/environment/`.

This is the practical rebuild map. It lists the source mechanics we need to
reproduce, the proof status today, and the missing fixtures that still block
broader claims.

`rg --files -g '*.map'` did not find vendored sourcemap files in this checkout.
Here, "source maps" means the curated source-map notes and fixture inventory in
this repo, especially:

- `docs/research/curvytron_source_map/*.md`
- `docs/research/curvytron_reference_notes.md`
- `docs/working/environment/source_feature_inventory.md`

## Proof Labels

- `source-read`: read directly from JS source or source-map docs, but no
  dedicated oracle fixture is pinned.
- `JS-pinned`: original JS oracle/probe pins the behavior.
- `Python-verified`: Python source-env or runner matches the JS oracle for the
  named slice.
- `narrow`: the proof is real but intentionally small.
- `deferred`: outside the source gameplay reconstruction lane for now.
- `unknown`: the source rule or comparison contract is not settled.

## Source Truth

Use the server files as gameplay authority. Client files explain input mapping,
wire payloads, visual trail rendering, room UI, and browser rendering, but the
client is not authoritative for collisions, scoring, or lifecycle.

Primary source files:

- `src/server/model/Game.js`
- `src/server/model/Avatar.js`
- `src/server/manager/PrintManager.js`
- `src/server/core/World.js`
- `src/server/core/Island.js`
- `src/server/core/AvatarBody.js`
- `src/server/manager/BonusManager.js`
- `src/server/model/Bonus/*.js`
- `src/shared/model/BaseGame.js`
- `src/shared/model/BaseAvatar.js`
- `src/shared/model/BaseRoomConfig.js`
- `src/shared/model/BaseBonus*.js`
- `src/shared/manager/BaseBonusManager.js`
- `src/shared/core/BaseSocketClient.js`
- `src/shared/service/Compressor.js`

## Core Tick Skeleton

Source rule:

1. The live loop targets `1000 / 60` ms, but each frame uses measured elapsed
   milliseconds.
2. `Game.update(step)` captures `score = deaths.count()` once at frame start.
3. Avatars update in reverse collection order.
4. For one alive avatar, source does: turn and move, maybe emit a trail point,
   check border, maybe wrap or die, check body collision if not invincible, then
   if still alive run `PrintManager.test()` and `BonusManager.testCatch()`.
5. If any death happened, `checkRoundEnd()` runs after the avatar loop.

Proof status: `Python-verified` for movement order, border/body priority,
same-frame point insertion, PrintManager post-collision timing, and focused
multiplayer scoring/order fixtures.

Missing fixtures:

- A 3P or 4P same-frame emitted-trail stress case only if it isolates a new
  rule.
- Broader live event traces that include movement, body, print, bonus, and score
  events in one scenario.

## Movement And Control

Source rules:

- Browser input resolves to move values `-1`, `0`, and `1`.
- Left-only is `-1`; right-only is `1`; both or neither becomes false locally
  and `0` on the wire.
- The server applies move changes immediately with
  `avatar.updateAngularVelocity(data.move)`. There is no input queue inside
  `Game.update`.
- Base speed is `16` units per second.
- Base angular velocity is `2.8 / 1000` radians per ms.
- Base radius is `0.6`.
- Trail latency is `3`.
- One avatar update turns first, then moves forward.
- Position uses:

```text
velocityX = cos(angle) * velocity / 1000
velocityY = sin(angle) * velocity / 1000
x = x + velocityX * step_ms
y = y + velocityY * step_ms
```

- Normal turning uses `angle += angularVelocity * step_ms`.
- Inverse controls are server avatar state. They flip the sign of active
  angular velocity.
- Speed changes recompute turn rate:

```text
ratio = velocity / 16
angularVelocityBase = ratio * (2.8 / 1000) + log(1 / ratio) / 1000
```

- `setVelocity()` clamps velocity to at least `8`.
- Straight-angle mode sets `directionInLoop=false`; the source adds one angular
  step, then clears angular velocity.

Proof status:

- `Python-verified`: base movement, turn-first ordering, elapsed-ms movement,
  varied elapsed steps, input value mapping.
- `source-read`: speed-modified turn rate, inverse while already turning, and
  straight-angle timing under active input.

Missing fixtures:

- `inverse_while_turning`
- `speed_bonus_turn_rate`
- `straight_angle_existing_turn`
- Browser/gamepad/touch fixtures only if browser input parity becomes a target.

## World, Bodies, And Trails

Source rules:

- The arena is square.
- Size formula is:

```text
round(sqrt(80^2 + (players - 1) * 80^2 / 5))
```

- Expected sizes: 1P `80`, 2P `88`, 3P `95`, 4P `101`.
- `World` defaults to `round(size / 40)` islands unless an island count is
  passed.
- Bodies are inserted only when `world.active` is true.
- A body is stored in island buckets by the four corners of its circle bounding
  box.
- Out-of-world corners are ignored.
- Duplicate insertion into the same island is prevented by `Collection.add`.
- Collision lookup also checks the four bounding-box corners and returns the
  first colliding stored body found.
- Stored bodies are circles. Collision is strict:

```text
distance < radiusA + radiusB
```

- Exact tangent is safe.
- Live heads are not collision bodies by themselves.
- A point event inserts a new `AvatarBody` immediately when
  `game.started && world.active`.
- Same-avatar stored bodies match only when:

```text
currentBody.num - storedBody.num > avatar.trailLatency
```

- Opponent bodies match immediately.
- `AvatarBody.oldAge = 2000` affects only the emitted death event's `old` flag.
  It does not affect collision.
- Normal border mode passes avatar radius as the margin and kills when the
  avatar circle crosses the square.
- Borderless mode passes margin `0`, so the center must cross the square.
- Borderless exact edge equality is safe.
- Borderless corner exits resolve in left, right, top, bottom check order. This
  is a teleport, not torus collision lookup.
- `Trail.clear()` clears visual trail cursor state only. Existing world bodies
  remain until the world is cleared.
- Normal trail cadence emits the first point when no last point exists, then
  only when distance from the last point is strictly greater than avatar radius.

Proof status:

- `Python-verified`: map sizes in source scenarios, reverse update order, body
  strictness, tangent safety, own latency at delta `3` and `4`, opponent body
  collisions, old-body metadata at `2000` ms, same-frame point materialization,
  world island insertion/lookup shape, borderless exact-edge/corner-axis/wrap
  checks, normal trail threshold, forced trail gap bodies.

Missing fixtures:

- Normal-wall exact edge, just-inside, and just-outside for every wall if the
  existing border batch is not enough for a later implementation.
- More island-boundary collision lookup cases in full game context.
- Natural emitted own-body loop, not only seeded body canaries.
- Natural emitted-body death with active PrintManager if needed.
- Borderless second-axis next-frame wrap only if a later feature depends on it.

## PrintManager

Source rules:

- `Game.onStart()` schedules each avatar's `printManager.start` after `3000` ms.
- It schedules all avatars in reverse avatar order, including non-present
  avatars.
- `PrintManager.start()` sets `active=true`, copies current `x/y` into
  `lastX/lastY`, and calls `setPrinting(true)`.
- `PrintManager.setPrinting(value)` always replaces its remaining distance with
  a fresh print or hole distance after calling `avatar.setPrinting(value)`.
- Server `Avatar.setPrinting(value)` always emits a `property printing` event.
- Shared `BaseAvatar.setPrinting(value)` emits a current point only when the
  boolean changes. When the change is from printing to not-printing, it then
  clears the visual trail cursor.
- Printed distance is:

```text
60 * (0.3 + Math.random() * 0.7)
```

- Hole distance is:

```text
5 * (0.8 + Math.random() * 0.5)
```

- `PrintManager.test()` only runs after the avatar survives border/body checks.
- It subtracts straight-line distance since the last sampled point, then toggles
  when `distance <= 0`.
- Overshoot is not carried into the next printed or hole span.
- `PrintManager.stop()` sets printing false if active, then clears manager
  state to inactive with zero distance and zero last position.
- On death, `BaseAvatar.die()` clears bonuses, sets `alive=false`, and adds a
  non-important death point before `PrintManager.stop()` runs. If the manager is
  active and printing, stop then emits an important stop point and
  `property printing=false` before the `die` event. If already in a hole, stop
  emits the property but no important stop point.

Proof status:

- `Python-verified`: delayed start at 3000 ms, random start-distance order,
  print-to-hole and hole-to-print toggles, exact-zero toggle, no-toggle control,
  active wall death stop, active already-hole death stop, body-collision
  stop-on-death, borderless wrap affecting PrintManager distance.

Missing fixtures:

- Broader natural multi-step print/hole cadence with finite fixture-tape
  exhaustion checks.
- Natural emitted-body death while the manager is active, only if the existing
  seeded body fixture is too narrow for implementation.
- Same-time timer ordering when PrintManager start/stop collides with other
  timers.

## Lifecycle, Scoring, And Timers

Source rules:

- `BaseGame.newRound(time)` sets `started=true`, then if not already in a round
  sets `inRound=true`, calls `onRoundNew()`, and schedules `start()` after
  `time` or `warmupTime`.
- `warmupTime = 3000` ms.
- `warmdownTime = 5000` ms.
- `Game.onRoundNew()` emits `round:new`, resets base round state, clears world,
  clears deaths, clears bonus stack, and spawns present avatars in reverse
  order.
- Non-present avatars do not spawn and are added to `deaths`.
- Natural spawn position consumes x then y random calls. Spawn heading consumes
  one or more random angle attempts until the border-margin check accepts.
- The normal first-round spawn path does not insert spawn bodies, so freshly
  spawned avatars do not reject each other after `world.clear()`.
- `Game.onStart()` emits `game:start`, schedules delayed PrintManager starts,
  activates the world, then starts bonus/FPS shared state.
- `BaseGame.stop()` only calls `onStop()` if a frame timer exists.
- `Game.onStop()` emits `game:stop`, stops shared state, recomputes map size
  from present avatars, then either ends the game or starts the next round.
- On a non-winning stop, the next `round:new` is synchronous inside `onStop()`.
- On match end, `end()` clears avatars/world and emits `end`.
- Death scoring uses the frame-start death count captured once at the beginning
  of `Game.update`.
- Every death in the same frame gets the same death score.
- On round end, the sole survivor gets `max(total avatars - 1, 1)` round score.
- In a one-player round, `resolveScores()` treats the only avatar as winner even
  if it just died.
- Then all avatars resolve `score += roundScore` and reset `roundScore=0`.
- Default max score is:

```text
max(1, (room.players.count() - 1) * 10)
```

- Expected max scores: 1P `1`, 2P `10`, 3P `20`, 4P `30`.
- A match winner must be a unique present avatar at or above max score. Tied
  leaders continue.

Proof status:

- `Python-verified`: focused 2P warmup/print-start, 2P next-round, 2P heading
  rejection retry, 2P match-end, 3P spawn order, 3P warmup/print-start, 3P
  all-dead next-round, 3P survivor round-end, 3P survivor next-round, 3P
  present/non-present round-new and next-round, 3P max-score match-end, 3P
  tie-at-max continuation, 3P multi-round match-end, 4P spawn order, 4P
  next-round, 1P wall-death scoring, and long narrow 1v1 no-bonus wall-round
  behavior when node is available.

Missing fixtures:

- Broader 4P lifecycle beyond the current focused paths.
- More present/non-present variants, especially with different avatars absent.
- Broader multi-round match variants with mixed prior scores.
- Timer ordering when bonus expiry, PrintManager, warmdown, and new-round timers
  share a timestamp.
- Production reset/autoreset and final-observation timing; these are not source
  JS concepts and need a wrapper contract.

## Multiplayer, Presence, And Leave

Source rules:

- Room player insertion order becomes avatar collection order.
- `Game.update()` runs from last avatar to first.
- `alive` and `present` are separate.
- `BaseAvatar.clear()` resets `alive=true` but does not set `present=true`.
- `BaseAvatar.destroy()` clears the avatar, then sets `present=false` and
  `alive=false`.
- `getAliveAvatars()` filters by `alive`.
- `getPresentAvatars()` filters by `present`.
- `Game.removeAvatar()` calls shared removal, which dies and destroys the
  avatar, emits `player:leave`, and checks round end.
- Mid-round leave does not add that avatar to current `deaths`.
- Non-present avatars are added to `deaths` during the next `onRoundNew()`.
- `onStart()` still schedules PrintManager starts for all avatars, including
  non-present avatars.
- Size is initialized from total avatars and recomputed on stop from present
  avatars.
- Several scoring paths use total avatar count, not present count.

Proof status:

- `Python-verified`: reverse order in movement/collision/scoring, focused
  3P/4P order fixtures, one active 2P mid-round leave case, non-present
  round-new behavior, non-present next-round behavior, non-present delayed
  PrintManager start.

Missing fixtures:

- Leave/disconnect variants for different player counts and timing points.
- Broader present/non-present scoring after multiple rounds.
- Spectator join/catch-up only if wire parity becomes active.

## Bonuses

Source rules:

- Default `bonusRate` is `0` and must stay in `[-1, 1]`.
- Default enabled bonuses are `BonusSelfSmall`, `BonusSelfSlow`,
  `BonusSelfFast`, `BonusSelfMaster`, `BonusEnemySlow`, `BonusEnemyFast`,
  `BonusEnemyBig`, `BonusEnemyInverse`, `BonusEnemyStraightAngle`,
  `BonusGameBorderless`, `BonusAllColor`, and `BonusGameClear`.
- `BonusSelfGodzilla` exists as a class but is not normally selectable because
  it is not in the server config map.
- Base bonus radius is `3`.
- Base bonus duration is `5000` ms.
- Base bonus probability is `1`.
- Active bonus cap is `20`.
- Bonus base pop time is:

```text
3000 - ((3000 / 2) * bonusRate)
```

- Each scheduled pop delay is `base_pop_time * (1 + Math.random())`.
- `BonusManager.start()` clears bonus state, activates the bonus world, and
  schedules a pop only if at least one bonus type is enabled.
- `popBonus()` reschedules the next pop before checking cap/type/position.
- Type selection is weighted by `bonusType.prototype.getProbability(game)` and
  skips nonpositive probabilities.
- Spawn position must be free in both the main game world and the bonus world.
  It uses margin `BaseBonus.radius + 0.01 * game.world.size`.
- Catch checks run after movement, border/body collision, and PrintManager, only
  for avatars still alive.
- Catch uses the same strict circle overlap rule through the bonus world.
- Removing a map bonus clears its body and emits `bonus:clear` before applying
  the effect.
- `Bonus.applyTo()` resolves target, schedules `off()` if duration is truthy,
  then calls `on()`.
- Targets:
  - self: alive catcher only;
  - enemy: alive avatars except catcher;
  - all: all alive avatars;
  - game: the `Game` object.
- Stack resolution resets affected properties to defaults, applies all active
  effects, then calls source setters.
- Normal numeric effects add.
- `directionInLoop`, `angularVelocityBase`, and `color` override.
- Radius stack value is an exponent:

```text
radius = 0.6 * 2 ** stackedRadiusValue
```

- Inverse is true when the stacked inverse value is odd.
- Printing effects call `printManager.start()` for positive values and
  `printManager.stop()` otherwise.
- `BonusGameBorderless` applies through `game.setBorderless()`.
- `BonusGameClear` has duration `0` and immediately calls
  `game.clearTrails()`, which clears and reactivates the main world and emits
  `clear`.
- `BonusGameClear` probability is dynamic:

```text
ratio = 1 - aliveAvatars / presentAvatars
if ratio < 0.5: return 1
else: return round((1 - ratio) * 10) / 10
```

Bonus effects to reproduce:

| Bonus | Target | Duration | Probability | Effect |
| --- | --- | ---: | ---: | --- |
| `BonusSelfSmall` | self | `7500` | `1` | radius exponent `-1` |
| `BonusSelfSlow` | self | `5000` | `1` | velocity `-8` |
| `BonusSelfFast` | self | `4000` | `1` | velocity `+12` |
| `BonusSelfMaster` | self | `7500` | `1` | invincible, printing stopped |
| `BonusEnemySlow` | enemies | `5000` | `1` | velocity `-8` |
| `BonusEnemyFast` | enemies | `6000` | `1` | velocity `+12` |
| `BonusEnemyBig` | enemies | `7500` | `1` | radius exponent `+1` |
| `BonusEnemyInverse` | enemies | `5000` | `1` | inverse `+1` |
| `BonusEnemyStraightAngle` | enemies | `5000` | `1` | straight-angle mode, `pi/2` turn base |
| `BonusGameBorderless` | game | `10000` | `1` | borderless true |
| `BonusAllColor` | all | `7500` | `1` | rotate alive avatar colors |
| `BonusGameClear` | game | `0` | dynamic | clear trails immediately |

The `BonusEnemyInverse`, `BonusEnemyStraightAngle`, and `BonusGameBorderless`
classes declare subclass prototype probabilities, but source
`BaseBonus.getProbability()` reads `BaseBonus.prototype.probability`, so their
effective type-selection probability is the base `1`.

Proof status:

- `Python-verified narrow`: active `BonusSelfSmall` catch, tangent no-catch,
  wall-death no-catch, one one-type spawn/type/position path, one game-world
  spawn retry, one `BonusSelfSmall` expiry/restore, and forced immediate
  `BonusGameClear`.
- `Python-verified narrow`: default multi-type weight/type RNG.
- `source-read`: cap behavior, bonus-world retry,
  speed/radius collision interactions, inverse stacking, straight-angle, color,
  borderless expiry, same-timestamp expiry ordering, and hidden Godzilla
  unreachability.

Missing fixtures:

- Bonus cap behavior.
- Default multi-type weighted selection with dynamic `BonusGameClear`
  probability.
- Bonus-world retry separate from main-world retry.
- Speed bonus turn-rate effect.
- Radius change collision effect beyond expiry restore.
- Inverse double-cancel and inverse while already turning.
- Straight-angle while already turning.
- Borderless bonus expiry and collision/wrap side effects.
- `BonusAllColor` snapshot and restore.
- Multi-stack expiry and same-timestamp timer ordering.
- Death interactions for bonus types beyond the current small catch/death slice.

## Randomness

Source rules:

- The original source uses `Math.random()` directly.
- Random sites include spawn x/y, spawn angle attempts, PrintManager
  print/hole distances, bonus start/pop delays, bonus type choice, bonus
  position x/y retries, room passwords, and random player color.
- Natural spawn order is reverse present avatar order.
- For each spawned avatar, the normal call order is position x, position y, then
  one or more angle attempts.
- With bonuses enabled, `Game.onStart()` starts the bonus manager and consumes a
  bonus-delay random value before the delayed PrintManager starts fire.
- PrintManager start/stop/toggle calls consume their own distance random values.
- Bonus pop consumes next delay before type and position.
- The JS source has no seed API. Our deterministic implementation needs a
  row-local random tape or generator that preserves source call order and labels.

Proof status:

- `Python-verified partial`: spawn RNG labels/order for focused 2P, 3P, and 4P
  lifecycle fixtures; one heading-retry path; PrintManager start/stop distance
  calls in lifecycle/PrintManager fixtures; narrow bonus spawn/type/position and
  one spawn retry.

Missing fixtures:

- Row-local reset/spawn RNG policy.
- Generated random tape extension plus strict fixture/direct finite-tape
  exhaustion policy.
- Bonus default-weight selection and cap behavior.
- More heading rejection paths near each border.
- Random call history in replay/debug metadata.

## Events And Wire

Source rules:

- Source events are synchronous `EventEmitter` events.
- `Game.onPoint()` listens to avatar `point` and inserts bodies synchronously
  while game/world are active.
- The server socket protocol sends JSON batches of array events.
- `BaseSocketClient.addEvent()` creates `[name]`, `[name, data]`, or
  `[name, data, callbackId]`.
- Callback responses use numeric ids.
- With no socket interval, events send immediately. With an interval, they queue
  until flush unless forced.
- Wire compression for position and angle is:

```text
compressed = (0.5 + value * 100) | 0
decompressed = compressed / 100
```

- Gameplay wire events include `position`, `angle`, important `point`, `die`,
  `property`, `score`, `score:round`, `bonus:pop`, `bonus:clear`,
  `bonus:stack`, `round:new`, `round:end`, `game:start`, `game:stop`, `clear`,
  `borderless`, `game:leave`, and `end`.
- Only important points are sent as `point` over the gameplay wire. Non-important
  points still affect server trail/world state.
- Spectator catch-up sends current positions/properties, dead state, active map
  bonuses if in round, `round:end` if between rounds, and spectator count.

Proof status:

- `source-read`: socket batch shape, callback shape, compression, event names,
  spectator catch-up.
- `Python-verified narrow`: event ordering in state/lifecycle/common-trace
  fixtures, not exact socket batching.

Missing fixtures:

- `wire_event_single_tick`: one compressed event batch for movement/death/score.
- Important versus non-important point wire fixture.
- Spectator catch-up fixture.
- Wire replay fixture if replay means replaying socket batches.

## Observations And Replay Boundaries

Source rules:

- Original CurvyTron does not define learned observations.
- Original CurvyTron does not define trainer rewards, action ids, replay rows,
  final observations, autoreset, or vector batch semantics.
- Browser pixels and client trail visuals are render outputs, not gameplay
  authority.
- Training observations must be built from source-backed simulator state.
- Debug/global observations can expose privileged state, but trainer
  observations need schema ids, hashes, channel order, scalar order, masks, and
  hidden-state leak checks.
- Rewards must be wrapper rules over source round/match outcomes and must carry
  reward schema ids/hashes.
- Replay rows must store rules hash, observation schema hash, action schema,
  reward schema, player count, episode id, reset seed/source, terminal reason,
  terminated/truncated/done flags, final observation policy, ego id, joint
  action, and optional event/state refs.

Proof status:

- `deferred`: learned observations and production replay are wrapper work, not
  JS source mechanics.
- `narrow`: one source-state movement observation canary exists for the current
  ray schema, and replay-v0 shape/hash plumbing exists for a 1v1 no-bonus
  training chunk path.

Missing fixtures:

- Source-backed observation fixtures for turn perspective, trail gaps, stored
  bodies, boundary bodies, borderless wrap, same-frame collision death, and
  normal-wall terminal states.
- Terminal/final-observation policy tied to source lifecycle.
- Production replay integration with source-backed observations and rewards.
- Autoreset contract that preserves terminal transition data before reset.
- Wire replay contract, if needed, separate from trainer replay.

## Fast Runtime Implications

Implementation implications:

- Do not use a fixed tick count as physics. The runtime must accept elapsed-ms
  steps and reproduce source kinematics.
- Do not use swept collision unless a separate non-source mode is explicitly
  named. Source collision is endpoint-circle collision, so large steps can
  tunnel.
- Preserve reverse avatar order. Moving all avatars first and resolving
  collisions later changes outcomes.
- Preserve synchronous point insertion. A point emitted by one avatar can kill a
  later avatar in the same source update.
- Keep visible trail state separate from collision world bodies.
- Preserve PrintManager timing after collision and before bonus catch.
- Preserve timer ordering with warmup, warmdown, delayed PrintManager starts,
  bonus pop timers, and bonus expiry timers.
- Use row-local RNG state in batched runtimes. Shared global random streams will
  break spawn, PrintManager, and bonus reproducibility.
- Track random call labels and indices for debug/replay.
- Keep world/island behavior source-shaped, or prove an optimized lookup returns
  the same body in every promoted fixture.
- Vector reset is a boundary, not natural lifecycle proof. Natural spawn,
  timers, row-local RNG, final observation, and autoreset must be explicit.
- Optimized/vector code should only claim the named source slices it matches.
- Modal/JAX/Mctx jobs are coarse training/search jobs, not per-step JS env
  calls.
- Speed measurements are useful only with the ruleset, player counts, bonuses,
  observation packing, reset policy, and replay policy stated next to them.

Current fast-runtime proof status:

- `narrow`: scalar source-env covers focused source claims and has local scout
  speed numbers for a narrow no-bonus rollout.
- `deferred`: production batched lifecycle with timers, bonuses, row-local RNG,
  observations, final observations, autoreset, and replay.

## Current Missing Fixture Queue

Highest-value next fixtures:

1. `inverse_while_turning`
2. `speed_bonus_turn_rate`
3. `straight_angle_existing_turn`
4. Bonus default weighted selection and `BonusGameClear` dynamic probability.
5. Bonus cap behavior.
6. Bonus-world position retry.
7. Borderless bonus expiry.
8. `BonusAllColor` snapshot/restore.
9. Same-timestamp bonus expiry with PrintManager or warmdown timer.
10. Source-backed observation terminal/wall fixture.
11. Source-backed trail-gap observation fixture.
12. Wire compressed single-tick event fixture.
13. Row-local RNG/reset fixture for production reset.
14. Broader 4P lifecycle and present/non-present variants.

## Minimal Reconstruction Order

1. Keep movement/control, reverse avatar order, border/body collision, point
   insertion, PrintManager, scoring, and lifecycle boring before adding more
   bonuses.
2. Add deterministic RNG labels before natural spawn, long PrintManager holes,
   or random bonuses.
3. Add bonuses by type and timing slice, not as one large block.
4. Add wire checks only when state event order already passes.
5. Add observations and replay only from source-backed states.
6. Move to fast/vector runtime one named source slice at a time.
