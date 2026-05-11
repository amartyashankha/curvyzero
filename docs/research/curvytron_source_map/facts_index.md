# CurvyTron Compact Facts Index

Status: compact index over deeper source-map docs

Use this as the quick orientation page. Treat the linked docs as the deeper
sources. If a fact is marked probe-needed, do not use it as final behavior until
a headless JS trace or trace diff proves it.

## Movement

Source doc: [movement_controls.md](movement_controls.md)

- Input is resolved to move values `-1`, `0`, `1`. Left is `-1`, right is `1`,
  both/neither becomes `0` on the wire.
- Physics uses elapsed milliseconds. The live loop targets `1000 / 60` ms, but
  each update receives measured `step_ms`.
- Base speed is `16` units per second. Base turn rate is `2.8 / 1000` radians
  per millisecond.
- One avatar update turns first, then moves forward, then may emit a trail
  point.
- Inverse controls are server-side avatar state. Changing inverse while turning
  flips the active angular velocity.
- Speed changes also change turn rate through the source formula in
  `BaseAvatar.updateBaseAngularVelocity`.
- Straight-angle bonus mode adds one angular step and then zeros angular
  velocity.

Verified narrow slice:

- The trace contract supports both fixed-rate and recorded elapsed movement
  checks. `source_kinematics_batch.json` includes varied elapsed-ms multi-step
  movement. The broader source-claim status is tracked in
  [coverage_tracker.md](../../working/environment/coverage_tracker.md).

Probe-needed:

- Confirm straight-angle timing when an avatar is already turning.

## World And Border

Source docs: [collisions_trails_world.md](collisions_trails_world.md),
[rounds_scoring_multiplayer.md](rounds_scoring_multiplayer.md)

- The arena is square. Size scales by player count: 1P `80`, 2P `88`, 3P `95`,
  4P `101`.
- Normal borders kill when the avatar circle crosses the arena edge.
- Borderless mode wraps after the avatar center crosses the edge. Exact equality
  is safe because bound checks are strict.
- If a borderless move crosses both x and y bounds, the first x-axis hit wraps
  and the other axis is left for a later check.
- Border checks run before body collision checks.
- The world uses island buckets. Bodies are inserted and queried by the four
  corners of each circle bounding box.
- Borderless wrap is a teleport rule. Collision lookup does not act like a
  continuous torus across opposite edges.

Probe-needed:

- Broader wall edge sweeps if exact-edge variants become important beyond the
  promoted normal-wall fixtures.
- Borderless next-frame second-axis wrap, only if another feature needs it.
- Broader spawn/reset variants beyond the pinned 2P heading retry and focused
  3P/4P spawn-order/lifecycle fixtures.
- Broader borderless wrap with active trail printing beyond the pinned
  PrintManager wrap/toggle case, because teleport distance may affect later
  hole state.

## Trails

Source doc: [collisions_trails_world.md](collisions_trails_world.md)

- Trail printing starts 3000 ms after `game:start`.
- Normal trail points are distance-spaced: the first point is immediate when the
  trail has no last point, then later points require distance greater than the
  avatar radius.
- Print holes are distance-based. Printed distance and hole distance use random
  multipliers.
- Hole overshoot is not carried into the next print/hole span.
- Toggling printing always emits a current point. Turning printing off then
  clears the avatar trail object, but existing world collision bodies remain.
- Client trail gaps are visual handling. Server collision bodies are the rule
  source.

Verified narrow slice:

- Delayed PrintManager start at `3000 ms` is pinned in the lifecycle fixtures.
- The deterministic PrintManager batch covers the current toggle/death-stop
  slice, and the source env pins exact/epsilon trail draw thresholds.

Probe-needed:

- Broader PrintManager toggle timing at less than, equal to, and greater than
  remaining distance.
- Hole body behavior: no regular per-radius points inside the gap, but boundary
  points on print-state changes.
- Broader death-frame point side effects beyond the promoted stop-on-death
  canaries.

## Collision

Source docs: [collisions_trails_world.md](collisions_trails_world.md),
[rounds_scoring_multiplayer.md](rounds_scoring_multiplayer.md)

- Collision is endpoint-circle collision, not swept collision.
- A stored body collides when `distance < radiusA + radiusB`. Exact tangent
  distance is safe.
- Self-collision is delayed by trail point number. Own bodies match only when
  `currentBody.num - storedBody.num > 3`.
- `oldAge = 2000` only affects the emitted death event's `old` flag. It does
  not decide collision.
- One avatar is updated and checked at a time in reverse avatar order. Earlier
  updates in the same frame can create trail bodies before later avatars check.
- If an avatar dies, print-manager testing and bonus catch do not run for it.
- Active PrintManager stop-on-death is ordered before the `die` event. For a
  currently-printing avatar, death emits a non-important death point, then an
  important stop point, then `property printing=false`, then `die`; the manager
  ends cleared to zero.
- For an active PrintManager when the dying avatar is already in a hole, death
  emits the non-important death point, then `property printing=false`, then
  `die`. No important stop point is emitted, and the death trail point remains.

Probe-needed:

- Broader body-collision death variants beyond the pinned active PrintManager
  stop-on-death slice.
- Broader head-head and same-frame collision variants beyond the promoted
  reverse-order single-death fixture.
- Broader killer-field comparisons beyond the current wall/body/old metadata
  canaries.

## Scoring

Source doc: [rounds_scoring_multiplayer.md](rounds_scoring_multiplayer.md)

- There are two score layers: `roundScore` for the current round and `score`
  for the match.
- At frame start, `Game.update` captures `deaths.count()`. Every death in that
  frame gets that same death score.
- On round end, the sole winner gets `max(total avatars - 1, 1)` round points.
- Then each avatar resolves `roundScore` into match `score` and resets
  `roundScore`.
- If everyone dies in the ending frame, there is no survivor bonus.
- Default max score is `max(1, (room players - 1) * 10)`: 1P `1`, 2P `10`,
  3P `20`, 4P `30`.

Probe-needed:

- Broader wall-death and emitted score-event variants beyond
  `source_normal_wall_multiplayer_batch.json` and the direct 3P/4P wall
  canaries.
- Broader same-frame double-death event order beyond the pinned same-frame
  source-env scoring slices.
- Broader 3P and 4P tied-death/lifecycle cases. Named fixture gaps are listed in
  [no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md](../../working/environment/no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md).
- Training reward/ruleset copying remains open; source score proofs are not a
  trainer-ready reward contract.

## Multiplayer

Source doc: [rounds_scoring_multiplayer.md](rounds_scoring_multiplayer.md)

- Room player insertion order becomes avatar collection order.
- `Game.update` loops from last avatar to first. For players added P0, P1, P2,
  P3, update order is P3, P2, P1, P0.
- `getAliveAvatars()` filters `alive`; `getPresentAvatars()` filters `present`.
  These are separate states.
- Non-present avatars are not spawned on new round and are added to `deaths`.
- Leaving mid-round destroys the avatar and checks round end, but does not add
  that avatar to current-round `deaths`.

Probe-needed:

- Broader 3P and 4P reset/lifecycle canaries beyond the pinned spawn, survivor,
  all-dead, match/tie, and present/absent slices.
- Player leave/disconnect variants beyond the narrow active-round source
  fixtures and metadata bridge.
- Present/alive edge cases before treating total-avatar scoring as broad public
  environment or replay parity.

## Bonuses

Source doc: [bonuses_config.md](bonuses_config.md)

- Default `bonusRate` is `0`. Default enabled bonuses are the normal self,
  enemy, game, and all-player bonuses listed in the bonus doc.
- Spawn timing base is `3000 - ((3000 / 2) * bonusRate)`, then multiplied by
  `1 + Math.random()`.
- Active bonus cap is `20`. Bonus spawn checks both the main game world and the
  bonus world.
- Catch checks run after movement, border/body collision, and print-manager
  testing.
- Bonus catch collision uses the same strict circle overlap rule.
- Stack math resets to defaults, applies active effects, then calls source
  setters. Normal numeric effects add; `directionInLoop`, `angularVelocityBase`,
  and `color` override.
- Radius effects are exponents: effective radius is `0.6 * 2 ** stackValue`.
- Borderless and clear-trails are game-level effects. Every new round resets
  borderless and clears bonus state. One forced `BonusGameClear` catch proof is
  pinned for immediate clear/reactivation; reduced and full-probability natural
  `BonusGameClear` type-selection edges are pinned separately.
- `BonusSelfGodzilla` exists but is not normally selectable.

Verified narrow slice:

- One minimal `BonusSelfSmall` one-type spawn/type/position fixture, default
  multi-type weight/type fixtures including the reduced and full-probability
  `BonusGameClear` edges, game-world and bonus-world rejected-position retry
  fixtures, and one cap-at-20 skip fixture are pinned through JS/Python
  source-env checks. The bonus-world retry proof is narrow source proof only,
  not fast-runtime natural spawn, public bonus env, bonus replay, or broad
  bonus effects.

Probe-needed:

- Broader deterministic `Math.random` policy for caps/probability, trail holes,
  and production reset streams.
- One speed, radius collision, color, and borderless trace. The forced immediate
  clear trace and narrow natural type-selection edges are already pinned; broad
  `BonusGameClear` probability/effect coverage is still open.
- Same-frame stack expiry, multi-stack math, and special override properties.
- Broader bonus catch order in frames with other bonus types or death
  interactions. The active `BonusSelfSmall` catch/no-catch/death-order slice is
  already pinned.

## Networking, Rendering, And Build

Source doc: [network_render_build.md](network_render_build.md)

- Server game state is the authority. The browser sends move changes and mirrors
  server events.
- Socket messages are JSON batches of array events.
- Position and angle wire values are compressed to integer hundredths with
  `(0.5 + value * 100) | 0`, then decompressed by dividing by `100`.
- Gameplay events to preserve for traces include `position`, `angle`, `point`,
  `die`, `property`, `score`, `score:round`, `round:new`, `round:end`,
  `game:start`, `game:stop`, `borderless`, and `clear`.
- Rendering uses stacked canvases and local visual interpolation. It is not the
  source of gameplay truth.
- Trackers feed Inspector/Influx observability. They are not game semantics.
- The old app uses Gulp 3, Bower, generated `bin/` and `web/js/` files, and
  defaults to port `8080` when `config.json` is missing.

Probe-needed:

- Wire-event compression only when comparing socket messages.
- Per-player/spectator event views if networking fidelity becomes a goal.
- Raw old-app build in a disposable or pinned environment before relying on
  browser screenshots.

## Highest-Risk Facts To Probe First

1. Elapsed-ms trace contract: fixed `1000 / 60`, recorded elapsed steps, or both.
2. Normal wall death and borderless wrap edge behavior.
3. Strict circle overlap and self-collision latency.
4. Trail printing delay, hole toggles, and death-frame point emissions.
5. Reverse avatar update order in same-frame collisions.
6. Same-frame scoring for 2P, then tied-death scoring for 3P and 4P.
7. Broader random stream policy before production trail-hole, spawn, or bonus
   traces.
8. Broader bonus stack/effect order after the current `BonusSelfSmall`
   catch/spawn/expiry slices.
