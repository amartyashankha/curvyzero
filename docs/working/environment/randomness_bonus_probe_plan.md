# Randomness And Bonus Probe Plan

Status: source-read plan plus tiny JS-oracle proofs, 2026-05-09. The current
runtime fixtures cover seeded active `BonusSelfSmall` catch/no-catch/death-order,
one natural one-type bonus spawn/type/position RNG proof, natural game-world
and bonus-world spawn-position retry proofs, reduced and full-probability
default `BonusGameClear` type-selection edges, one timed `BonusSelfSmall`
expiry/restore proof, and one forced `BonusGameClear` immediate clear proof.
Python source-env parity exists for those same narrow slices.

Scope: JS oracle and future environment-fidelity work for source randomness,
bonus spawning, bonus catch order, bonus stack effects, and expiry. This plan
does not change the current simplified `curvyzero-v0` no-bonus ruleset.

The bonus-world retry proof is narrow source proof only:
`source_bonus_spawn_bonus_world_retry_step.json`, guarded in
`tests/test_env_scenarios.py` and `tests/test_source_env.py`. It does not
promote fast-runtime natural bonus spawn, public bonus env support, bonus
replay, or broad bonus effects.

## Priority

Start with forced bonus state, not natural spawn timers. The safest first proof
is a forced map bonus caught after one safe movement step, using a self-targeted
radius bonus. That exercises source catch ordering, bonus-world removal, stack
addition, and one gameplay-critical property without depending on random spawn
placement, weighted type selection, or timeout expiry.

Recommended first runtime fixture:

1. `source_bonus_forced_catch_self_small_step`

Recommended first five bonus fixtures:

1. `source_bonus_forced_catch_self_small_step`
2. `source_bonus_dead_avatar_skips_catch_step`
3. `source_bonus_self_small_expiry_restore_step`
4. `source_bonus_game_clear_immediate_step` (promoted)
5. `source_bonus_self_slow_turn_rate_step`

Natural bonus spawning now has seven minimal JS/Python source-env proofs. The
first, `source_bonus_spawn_type_position_rng_step.json`, enables only
`BonusSelfSmall`, uses `bonus_rate=1` so the first pop fires at 1500 ms before
source PrintManager starts, and pins five labeled draws:
`bonus.start_delay`, `bonus.next_delay_after_pop`,
`bonus.type.BonusSelfSmall`, `bonus.position.x`, and `bonus.position.y`.
It proves `bonus:pop` before the zero-elapsed source update's position events,
`bonusCount=1`, `bonusWorldBodyCount=1`, type `BonusSelfSmall`, and spawned
position `(23.94, 64.06)`.

The second, `source_bonus_spawn_game_world_retry_step.json`, keeps the same
one-type and pre-PrintManager timing but seeds a main game-world body exactly at
the first candidate `(23.94, 64.06)`. It proves that the source rejects that
candidate, draws one retry pair labeled `bonus.position.retry_1.x/y`, emits one
`bonus:pop` for the accepted position `(68.072, 19.928)`, and leaves
`worldBodyCount=1`, `bonusCount=1`, and `bonusWorldBodyCount=1`.

The third, `source_bonus_spawn_bonus_world_retry_step.json`, seeds one active
`BonusSelfSmall` after `BonusManager.start()` so the first natural candidate
collides with the bonus world instead of the main world. It proves that the
source rejects that candidate, draws one retry pair labeled
`bonus.position.retry_1.x/y`, emits one new `bonus:pop` for id `2` at the
accepted position `(68.072, 19.928)`, and leaves `worldBodyCount=0`,
`bonusCount=2`, and `bonusWorldBodyCount=2`.

The fourth, `source_bonus_default_weights_type_rng_step.json`, enables the full
default source bonus order, forces two of four present avatars dead before the
natural pop, and pins `bonus.type.BonusAllColor` from the original JS oracle.
That draw lands in `BonusAllColor` only with `BonusGameClear`'s reduced dynamic
probability included. It proves the default configured order/weights through
type selection and position `(27.255, 73.745)` only. These fixtures still do not
prove catch beyond the existing active `BonusSelfSmall` and `BonusGameClear`
slices, broader stack/effects, broader expiry ordering, or vector/runtime
support.

The fifth, `source_bonus_spawn_cap_twenty_step.json`, seeds 20 active map
bonuses after `BonusManager.start()`, advances to the first natural pop, and
proves the source draws `bonus.next_delay_after_pop` before the cap check, then
does not draw bonus type or position, emits no new `bonus:pop`, and leaves
`bonusCount=20` plus `bonusWorldBodyCount=20`.

The sixth, `source_bonus_default_weights_select_game_clear_step.json`, keeps
the two-dead-of-four setup and pins the paired reduced-probability edge: with
`BonusGameClear` probability `0.5`, type draw `0.965` selects the final
`BonusGameClear` bucket and emits a natural `bonus:pop` at `(27.255, 73.745)`.

The seventh,
`source_bonus_default_weights_game_clear_full_probability_step.json`, forces
only one of four present avatars dead before the pop. Because the source ratio
is below `0.5`, `BonusGameClear` probability remains `1`; with default total
weight `11.2`, type draw `0.93` selects `BonusGameClear`. This draw would miss
that bucket under the reduced `0.5` probability, so the proof pins the
full-probability side without claiming catch, clear effects, cap, retry, or
vector/runtime support.

Timed `BonusSelfSmall` expiry now has one minimal JS/Python source-env proof:
`source_bonus_self_small_expiry_restore_step.json`. It starts from the same
forced catch setup, advances the source timeout by `7500` ms, and proves radius
restore from `0.3` to `0.6`, `bonus:stack remove`, no second `bonus:clear`, and
expiry events before the following zero-elapsed position events. It does not
prove multi-stack math, same-frame expiry with other timers, other effects, or
vector/runtime support.

`BonusGameClear` now has one minimal JS/Python source-env proof:
`source_bonus_game_clear_immediate_step.json`. It seeds one main-world body far
from p0, catches a forced `BonusGameClear` after one safe 100 ms update, emits
`bonus:clear` then `clear`, resets `worldBodyCount` to `0`, leaves the world
active, keeps both avatars alive with no active avatar bonuses, and emits no
`bonus:stack` or avatar `property`. It does not prove natural selection,
dynamic probability, borderless, speed, inverse, color, broader stack timing,
or vector/runtime support.

Defer `BonusAllColor` until gameplay effects are pinned; it matters for
trace/wire/UI parity, but not collision, movement, scoring, or trail bodies.

## Files Read

- `docs/research/curvytron_source_map/bonuses_config.md`
- `docs/research/curvytron_source_map/movement_controls.md`
- `docs/working/environment/source_feature_inventory.md`
- `docs/working/environment/probe_backlog.md`
- `third_party/curvytron-reference/src/shared/model/BaseRoomConfig.js`
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`
- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `third_party/curvytron-reference/src/server/manager/BonusManager.js`
- `third_party/curvytron-reference/src/shared/manager/BaseBonusManager.js`
- `third_party/curvytron-reference/src/shared/model/BaseBonus.js`
- `third_party/curvytron-reference/src/shared/model/BaseBonusStack.js`
- `third_party/curvytron-reference/src/server/model/BonusStack.js`
- `third_party/curvytron-reference/src/server/model/GameBonusStack.js`
- `third_party/curvytron-reference/src/server/model/Bonus/*.js`

## RNG Call Sites

The source uses global `Math.random()`. A source-faithful JS oracle should first
observe and lock that one global call order before a Python/vector ruleset
splits state into named explicit streams.

Gameplay-critical random calls:

| Area | Source path | Draws | Why it matters |
| --- | --- | --- | --- |
| Natural spawn position | `Game.onRoundNew()` -> `World.getRandomPosition()` -> `World.getRandomPoint()` | Two draws for initial `x,y`, plus two per rejection retry. Avatars are processed in reverse collection order. | Determines round-start positions and can alter immediate overlap/body-free checks. |
| Natural spawn heading | `Game.onRoundNew()` -> `World.getRandomDirection()` -> `World.getRandomAngle()` | One draw for initial angle, plus one per direction-rejection retry. | Determines movement trajectory and must satisfy source border-angle margin. |
| Print section length | `PrintManager.start()` and `PrintManager.togglePrinting()` while printing becomes true | One draw: `60 * (0.3 + random * 0.7)`. | Controls natural printed segment length and future hole timing. |
| Hole section length | `PrintManager.togglePrinting()` and active `PrintManager.stop()` while printing becomes false | One draw: `5 * (0.8 + random * 0.5)`. | Controls gap length; stop-on-bonus/death can draw even when visual trail is cleared. |
| Bonus schedule delay | `BonusManager.start()` and `BonusManager.popBonus()` -> `getRandomPopingTime()` | One draw per scheduled pop: `bonusPopingTime * (1 + random)`. Base is `3000 - (1500 * bonusRate)`. | Controls natural bonus timing and must be separated from catch/stack fixtures. |
| Bonus type selection | `BonusManager.popBonus()` -> `getRandomBonusType()` | One draw: `random * totalWeight` after dynamic probability filtering. | Chooses bonus class; must account for `BonusGameClear` dynamic probability and default bonus order. |
| Bonus spawn position | `BonusManager.getRandomPosition()` -> `World.getRandomPoint()` | Two draws for initial `x,y`, plus two per retry. Tests both game world and bonus world. | Determines active bonus map body and future catch point. |

Environment-adjacent random calls to isolate or pin:

| Area | Source path | Policy |
| --- | --- | --- |
| Room password | `BaseRoomConfig.generatePassword()` | Not a step-dynamics call. Give it a separate pregame stream only for room-config tests. |
| Player fallback color | `BasePlayer.getRandomColor()` | Avoid by setting explicit player colors in oracle fixtures. Needed only for color/UI parity. |
| Room name / random room choice | `RoomNameGenerator`, `RoomRepository`, `Collection.getRandomItem()` from client room flows | Out of scope for environment physics; do not let these consume scenario RNG. |
| Client tips/explosions | `MessageTip`, `Explode` | Render/UI-only; exclude from state oracle fixtures. |

## Deterministic JS Random Policy

Use three modes, declared per oracle scenario:

| Mode | Meaning | Expected behavior |
| --- | --- | --- |
| `forbidden` | Fully forced-state fixtures. | Patch `Math.random` to throw. Use this for catch order, stack math, clear, and manual expiry fixtures. |
| `constant_0_5` | Legacy physics fixtures that already rely on neutral randomness. | Keep only where existing canaries need it; do not use for new call-order claims. |
| `scripted_stream` | Natural spawn, print/hole, and bonus spawn fixtures. | Patch `Math.random` to pop explicit floats from the scenario manifest and record every draw. Fail on underflow or unused draws. |

For `scripted_stream`, store a draw list with both values and intended labels:

```json
[
  {"label": "spawn.p1.x", "value": 0.25},
  {"label": "spawn.p1.y", "value": 0.75},
  {"label": "spawn.p1.angle.retry0", "value": 0.01},
  {"label": "spawn.p1.angle.accept", "value": 0.40},
  {"label": "bonus.delay.initial", "value": 0.50},
  {"label": "bonus.type.0", "value": 0.10},
  {"label": "bonus.position.0.x", "value": 0.50},
  {"label": "bonus.position.0.y", "value": 0.50}
]
```

The JS oracle should emit an `rngCalls` trace with index, value, source stack or
classified call site, and optional expected label. Compare the call log before
state fields. A state match with a shifted call log is still a fidelity failure.

Do not use a seed-only golden for source-call-order tests. A seeded generator is
fine for creating candidate values, but promoted scenarios should store the
concrete draw stream so retry paths and weighted thresholds are reviewable.

Timer policy:

- Do not use real wall-clock timers in deterministic fixtures.
- For bonus pop timing, call `getRandomPopingTime()` or use a fake scheduler
  that records the requested delay.
- For natural pop, call `bonusManager.popBonus()` directly after the scheduled
  delay is captured.
- For expiry, use manual `bonus.off()` or a fake clock after stack-add basics
  are proven. Keep expiry order separate from spawn RNG.

Future Python/vector policy:

- Preserve the JS oracle's global draw log as the source parity artifact.
- The array environment can later expose explicit per-environment RNG state and
  stream ids such as `spawn`, `print`, `bonus_delay`, `bonus_type`, and
  `bonus_position`, but it must have a compatibility mode that reproduces the
  JS oracle's serialized global draw order.

## Gameplay-Critical Bonus Effects

These effects change transition dynamics and should be covered before UI-only
pieces:

| Effect group | Bonuses | Gameplay surface |
| --- | --- | --- |
| Catch/removal/targeting | All map bonuses through `BonusManager.testCatch()` | Catch happens after movement, border/body collision, and `PrintManager.test()`, only while the avatar is still alive. |
| Radius | `BonusSelfSmall`, `BonusEnemyBig` | Changes collision body radius using exponent math, and changes future wall/body/catch interactions. |
| Velocity and turn-rate coupling | `BonusSelfSlow`, `BonusSelfFast`, `BonusEnemySlow`, `BonusEnemyFast` | Changes movement speed and recomputes `angularVelocityBase`. |
| Inverse controls | `BonusEnemyInverse` | Flips active turn direction through `setInverse()`; two active inverse effects cancel by parity. |
| Invincible and printing | `BonusSelfMaster` | Suppresses body death and stops print-manager output; stop can create trail/printing side effects. |
| Straight-angle mode | `BonusEnemyStraightAngle` | Sets `directionInLoop=false` and `angularVelocityBase=pi/2`; turn is applied once then cleared. |
| Borderless | `BonusGameBorderless` | Converts wall death into source wrap and resets on new round/expiry. |
| Clear trails | `BonusGameClear` | Immediately clears and reactivates the collision world; duration is `0`, no normal stack timer. |
| Stack expiry | Timed self/enemy/all/game bonuses | Removal resets touched properties to defaults, reapplies remaining stack entries, and may emit property/borderless/stack events. |

## Render/UI Or Color-Only Pieces

Keep these out of the first gameplay proof loop:

| Piece | Source behavior | Probe priority |
| --- | --- | --- |
| `BonusAllColor` | Rotates alive avatar colors from a snapshot and restores player colors on expiry. | Later trace/UI fixture. It changes `avatar.color` and property events, but not movement, collision, scoring, print bodies, or wall behavior. |
| Client map bonus sprites | Client `BonusManager`, `MapBonus`, and `StackedBonus` render icons for active bonuses. | Browser/render parity only. |
| Socket compression/display payloads | `GameController` forwards `bonus:pop`, `bonus:clear`, `bonus:stack`, `property`, `clear`, and `borderless`. | Wire fixture after state parity. |
| Explosion/tip randomness | Client visual effects and UI text. | Exclude from environment oracle. |

## First Five Fixture Specs

### 1. Forced Catch, Self Small

Name: `source_bonus_forced_catch_self_small_step`

Purpose:

- Prove a forced active map bonus is caught after safe movement.
- Prove `BonusManager.remove()` emits clear before stack application.
- Prove self-target stack add changes radius from `0.6` to `0.3`.

Setup sketch:

- Two-player map `88`; `started=true`, `in_round=true`,
  `world_active=true`, `borderless=false`.
- Disable natural timers and use `rng_mode: forbidden`.
- Seed one `BonusSelfSmall` in `bonusManager` and bonus world at p0's
  post-step endpoint, with no game-world body collision.
- Keep print managers inactive.
- One tick with p0 moving safely into strict overlap and p1 far away.

Expected claims:

- p0 remains alive.
- Bonus is removed from active map bonuses.
- Events include `bonus:clear`, `bonus:stack add`, and radius `property`.
- p0 radius is `0.3`; p1 radius remains `0.6`.

### 2. Dead Avatar Skips Catch

Name: `source_bonus_dead_avatar_skips_catch_step`

Purpose:

- Prove catch only runs in the `if (avatar.alive)` branch after collision.
- Avoid accidental implementations that apply a bonus to an avatar killed on
  the same movement step.

Setup sketch:

- Same forced bonus mechanics as fixture 1.
- Put p0's post-step endpoint both outside the normal wall or overlapping a
  seeded body and overlapping the forced bonus.
- Use `rng_mode: forbidden`.

Expected claims:

- p0 dies from wall/body collision.
- The bonus remains active in the bonus manager and bonus world.
- No `bonus:clear`, `bonus:stack`, radius, velocity, or other bonus property
  event fires for p0.

### 3. Manual Expiry, Self Small

Name: `source_bonus_self_small_expiry_restore_step`

Purpose:

- Prove stack removal resets a touched property to the source default.
- Prove the real source `setTimeout` duration for `BonusSelfSmall`.

Setup sketch:

- Start from a caught `BonusSelfSmall` on p0 or call `bonus.applyTo(p0, game)`
  directly.
- Assert active stack and radius `0.3`.
- Advance the source timer by exactly `7500` ms before a zero-elapsed update.
- Use `rng_mode: forbidden`.

Expected claims:

- Events include `bonus:stack remove` and radius `property`.
- p0 radius returns to `0.6`.
- Other players and game-level flags are unchanged.

### 4. Immediate Game Clear

Name: `source_bonus_game_clear_immediate_step`

Purpose:

- Prove duration-`0` game bonus bypasses normal stack timing and immediately
  clears the collision world.
- Cover a gameplay-critical game-level effect without RNG.

Setup sketch:

- Seed a few existing trail bodies in `game.world`.
- Force catch of `BonusGameClear`, or apply it directly after fixture 1 proves
  catch mechanics.
- Use `rng_mode: forbidden`.

Expected claims:

- `clear` event fires.
- Game world is cleared and active afterward.
- No timed stack entry remains for `BonusGameClear`.
- Avatar visual trail state should be asserted only if the source runner already
  exposes it; the key gameplay claim is collision-world clear/reactivation.

Status: promoted by `source_bonus_game_clear_immediate_step.json` through the
JS oracle and `CurvyTronSourceEnv`. The exact promoted claim is seeded
`BonusGameClear` catch after safe movement, event order `bonus:clear` then
`clear`, `worldActive=true`, `worldBodyCount=0`, no `bonus:stack`, no avatar
property change, and no active avatar bonuses. Natural selection/probability and
other bonus effects remain separate.

### 5. Self Slow Speed And Turn Rate

Name: `source_bonus_self_slow_turn_rate_step`

Purpose:

- Prove velocity bonuses affect both forward speed and turn-rate base.
- Catch implementations that update `velocity` but forget
  `updateBaseAngularVelocity()`.

Setup sketch:

- Force p0 to catch `BonusSelfSlow`, or apply directly after fixture 1.
- p0 should be actively turning before application so angular velocity refresh
  is observable.
- Use `rng_mode: forbidden`.

Expected claims:

- p0 velocity becomes `8`.
- `velocityX/Y` are recomputed from current angle.
- `angularVelocityBase` uses the source speed-coupled formula.
- Active `angularVelocity` is refreshed to the same turn sign with the new base.

## Follow-On Fixture Order

After the first five:

1. `source_bonus_enemy_inverse_active_turn_flip_step`
2. `source_bonus_enemy_inverse_double_cancel_step`
3. `source_bonus_game_borderless_catch_wrap_expiry_step`
4. `source_bonus_self_master_invincible_print_stop_step`
5. `source_bonus_enemy_straight_angle_existing_turn_step`
6. `source_bonus_spawn_delay_type_position_stream_step`
7. `source_bonus_spawn_position_retry_game_world_step`
8. `source_bonus_spawn_position_retry_bonus_world_step`
9. `source_bonus_all_color_snapshot_restore_step`

## Natural Bonus Spawn Probe Shape

Use this only after `scripted_stream` is available:

- Enable a narrow bonus list, ideally one or two classes at first.
- Capture the initial schedule delay from `bonusManager.start()`.
- Manually trigger `popBonus()` through the fake scheduler.
- Assert the second schedule delay is drawn before cap/type/position work,
  matching source `popBonus()` order.
- Assert weighted type selection call and chosen class.
- Assert position draw call count and retry behavior.
- Assert `bonus:pop` state: id, x, y, radius, type, active bonus-world body.

## Traps And Non-Claims

- Do not use `Math.random = () => 0.5` for new RNG-order fixtures; it hides
  shifted call order and retry loops.
- Do not test stack math through natural spawn first. Forced catch/apply makes
  failures local.
- Do not rely on real `setTimeout` for bonus duration or pop timing in oracle
  fixtures.
- Do not infer gameplay from client sprite or stacked-icon code.
- Do not treat `BonusAllColor` as movement/collision fidelity.
- Do not let fallback player-color generation consume the scenario RNG stream.
- Do not claim browser wire payload parity until state events are pinned.
