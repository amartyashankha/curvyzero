# CurvyTron Bonuses And Config Source Map

Status: source-mined; narrow JS/Python checks exist for selected bonus slices

Scope: original CurvyTron v1 source under `third_party/curvytron-reference`.
Server files are the source of truth for behavior. Client files are useful for
room UI, sprites, and wire handling.

## Main Source Files

| Area | Source refs | What they say |
| --- | --- | --- |
| Room defaults | `src/shared/model/BaseRoomConfig.js:4`, `src/shared/model/BaseRoomConfig.js:13`, `src/shared/model/BaseRoomConfig.js:17` | Room config starts with `bonusRate: 0` and the default enabled bonus set. |
| Server bonus class map | `src/server/model/RoomConfig.js:19`, `src/server/model/RoomConfig.js:58` | Enabled bonus names become server constructors through `bonusTypes`; `getBonuses()` returns constructors. |
| Game wiring | `src/shared/model/BaseGame.js:6`, `src/shared/model/BaseGame.js:19` | `BaseGame` creates `new BonusManager(this, room.config.getBonuses(), room.config.getVariable('bonusRate'))`. |
| Bonus manager | `src/server/manager/BonusManager.js:6`, `src/server/manager/BonusManager.js:59`, `src/server/manager/BonusManager.js:106`, `src/server/manager/BonusManager.js:167` | Server spawn, catch, event, and weighted type selection logic. |
| Base bonus values | `src/shared/model/BaseBonus.js:31`, `src/shared/model/BaseBonus.js:38`, `src/shared/model/BaseBonus.js:45`, `src/shared/model/BaseBonus.js:71` | Base radius is `3`, default duration is `5000` ms, default probability is `1`. |
| Bonus stack math | `src/shared/model/BaseBonusStack.js:22`, `src/shared/model/BaseBonusStack.js:52`, `src/server/model/BonusStack.js:42`, `src/server/model/BonusStack.js:76`, `src/server/model/BonusStack.js:97` | Effects are stacked, reset to source defaults when removed, then applied to avatar fields. |
| Game bonus stack | `src/server/model/GameBonusStack.js:40` | Game-level effects currently only special-case `borderless`. |
| Client display | `src/client/manager/BonusManager.js:33`, `src/client/model/bonus/MapBonus.js:10`, `src/client/model/bonus/StackedBonus.js:8` | Client code maps bonus names to sprite/display state. It is not the gameplay authority. |

## Function Index

Use these as the exact jump points for future implementation work:

| Function or constructor | Source ref |
| --- | --- |
| `BaseRoomConfig(room)` | `src/shared/model/BaseRoomConfig.js:4` |
| `BaseRoomConfig.prototype.setVariable()` | `src/shared/model/BaseRoomConfig.js:75` |
| `BaseRoomConfig.prototype.getDefaultMaxScore()` | `src/shared/model/BaseRoomConfig.js:176` |
| `RoomConfig.prototype.getBonuses()` | `src/server/model/RoomConfig.js:58` |
| `BaseGame(room)` | `src/shared/model/BaseGame.js:6` |
| `BaseGame.prototype.onRoundNew()` | `src/shared/model/BaseGame.js:170` |
| `BonusManager(game, bonuses, rate)` | `src/server/manager/BonusManager.js:6` |
| `BonusManager.prototype.start()` | `src/server/manager/BonusManager.js:24` |
| `BonusManager.prototype.popBonus()` | `src/server/manager/BonusManager.js:59` |
| `BonusManager.prototype.getRandomPosition()` | `src/server/manager/BonusManager.js:84` |
| `BonusManager.prototype.testCatch()` | `src/server/manager/BonusManager.js:106` |
| `BonusManager.prototype.getRandomPopingTime()` | `src/server/manager/BonusManager.js:157` |
| `BonusManager.prototype.getRandomBonusType()` | `src/server/manager/BonusManager.js:167` |
| `Bonus.prototype.applyTo()` | `src/server/model/Bonus/Bonus.js:22` |
| `BaseBonusStack.prototype.resolve()` | `src/shared/model/BaseBonusStack.js:52` |
| `BaseBonusStack.prototype.append()` | `src/shared/model/BaseBonusStack.js:115` |
| `BonusStack.prototype.apply()` | `src/server/model/BonusStack.js:42` |
| `BonusStack.prototype.getDefaultProperty()` | `src/server/model/BonusStack.js:76` |
| `BonusStack.prototype.append()` | `src/server/model/BonusStack.js:97` |
| `GameBonusStack.prototype.apply()` | `src/server/model/GameBonusStack.js:40` |
| `Game.prototype.update()` | `src/server/model/Game.js:37` |
| `Game.prototype.clearTrails()` | `src/server/model/Game.js:198` |
| `Game.prototype.setBorderless()` | `src/server/model/Game.js:297` |
| `World.prototype.getOposite()` | `src/server/core/World.js:305` |
| `Island.prototype.bodiesTouch()` | `src/server/core/Island.js:83` |
| `BaseAvatar.prototype.updateAngularVelocity()` | `src/shared/model/BaseAvatar.js:124` |
| `BaseAvatar.prototype.updateAngle()` | `src/shared/model/BaseAvatar.js:169` |
| `BaseAvatar.prototype.setVelocity()` | `src/shared/model/BaseAvatar.js:199` |
| `BaseAvatar.prototype.updateBaseAngularVelocity()` | `src/shared/model/BaseAvatar.js:225` |
| `BaseAvatar.prototype.setRadius()` | `src/shared/model/BaseAvatar.js:239` |
| `GameController.prototype.onBonusPop()` | `src/server/controller/GameController.js:381` |
| `GameController.prototype.onBonusClear()` | `src/server/controller/GameController.js:396` |
| `GameController.prototype.onBonusStack()` | `src/server/controller/GameController.js:440` |
| `GameController.prototype.onBorderless()` | `src/server/controller/GameController.js:508` |

## Room Defaults And Presets

`BaseRoomConfig` initializes one config variable: `bonusRate: 0`
(`src/shared/model/BaseRoomConfig.js:13`). `setVariable()` parses a float and
rejects values outside `[-1, 1]` (`src/shared/model/BaseRoomConfig.js:75`).
The room UI exposes this as a range slider with `min="-1"`, `max="1"`, and
`step="0.1"` (`src/client/views/rooms/parameters.html:23`,
`src/client/views/rooms/parameters.html:28`).

All normal bonuses are enabled by default:

- `BonusSelfSmall`
- `BonusSelfSlow`
- `BonusSelfFast`
- `BonusSelfMaster`
- `BonusEnemySlow`
- `BonusEnemyFast`
- `BonusEnemyBig`
- `BonusEnemyInverse`
- `BonusEnemyStraightAngle`
- `BonusGameBorderless`
- `BonusAllColor`
- `BonusGameClear`

This list appears in `BaseRoomConfig` (`src/shared/model/BaseRoomConfig.js:17`)
and in the client default preset named `All`
(`src/client/model/preset/DefaultPreset.js:17`,
`src/client/model/preset/DefaultPreset.js:24`). The server constructor map has
the same playable set (`src/server/model/RoomConfig.js:19`).

Other client presets only change which bonuses are toggled:

| Preset | Source refs | Enabled bonus names |
| --- | --- | --- |
| `All` | `src/client/model/preset/DefaultPreset.js:17`, `src/client/model/preset/DefaultPreset.js:24` | All default bonuses above. |
| `Speed of light` | `src/client/model/preset/SpeedPreset.js:17`, `src/client/model/preset/SpeedPreset.js:24` | `BonusSelfFast`, `BonusEnemyFast`. |
| `Super size me` | `src/client/model/preset/SizePreset.js:17`, `src/client/model/preset/SizePreset.js:24` | `BonusEnemyBig`. |
| `Solo` | `src/client/model/preset/SoloPreset.js:17`, `src/client/model/preset/SoloPreset.js:24` | Self bonuses plus `BonusGameBorderless` and `BonusGameClear`. |
| `No bonuses` | `src/client/model/preset/EmptyPreset.js:4`, `src/shared/model/Preset.js:11` | Empty inherited `Preset.prototype.bonuses`. |
| `Custom` | `src/client/model/preset/CustomPreset.js:4` | UI label only; the chosen toggle set is the real config. |

`BonusSelfGodzilla` exists as a class
(`src/server/model/Bonus/BonusSelfGodzilla.js:7`) but is not in the default
toggle list or server `bonusTypes` map (`src/shared/model/BaseRoomConfig.js:17`,
`src/server/model/RoomConfig.js:19`). Treat it as hidden/unreachable source
state unless a later probe finds another path to it.

## Bonus Manager

`BonusManager(game, bonuses, rate)` keeps its own one-island `World` for active
bonus bodies (`src/server/manager/BonusManager.js:6`,
`src/server/manager/BonusManager.js:10`). Its base constants come from
`BaseBonusManager`: max active bonuses `20`, base pop time `3000` ms, and spawn
margin factor `0.01` (`src/shared/manager/BaseBonusManager.js:24`,
`src/shared/manager/BaseBonusManager.js:31`,
`src/shared/manager/BaseBonusManager.js:38`).

`bonusRate` changes the base pop time with:

```text
bonusPopingTime = 3000 - ((3000 / 2) * rate)
```

That is from `src/server/manager/BonusManager.js:13`. Then each scheduled spawn
uses `bonusPopingTime * (1 + Math.random())`
(`src/server/manager/BonusManager.js:157`). So the source ranges are:

| `bonusRate` | Base pop time | Actual delay from `getRandomPopingTime()` |
| ---: | ---: | ---: |
| `-1` | `4500` ms | `4500` to `<9000` ms |
| `0` | `3000` ms | `3000` to `<6000` ms |
| `1` | `1500` ms | `1500` to `<3000` ms |

Spawn flow:

1. `start()` clears state, activates the bonus world, and schedules a timeout
   only if there is at least one enabled bonus type
   (`src/server/manager/BonusManager.js:24`,
   `src/server/manager/BonusManager.js:30`).
2. `popBonus()` reschedules first, checks the cap, picks a weighted class, finds
   a free position, creates the bonus, then calls `add()`
   (`src/server/manager/BonusManager.js:59`). At the cap, the reschedule has
   already happened and the function returns before type or position RNG.
3. `getRandomPosition(radius, border)` uses a body radius of
   `radius + border * game.world.size`, and retries until both the game world
   and bonus world are free (`src/server/manager/BonusManager.js:84`,
   `src/server/manager/BonusManager.js:86`,
   `src/server/manager/BonusManager.js:93`).
4. `add()` puts the bonus body into the bonus world and emits `bonus:pop`
   (`src/server/manager/BonusManager.js:123`).

Catch flow:

1. `Game.update()` calls `bonusManager.testCatch(avatar)` after movement,
   border/body collision checks, and print-manager testing
   (`src/server/model/Game.js:48`, `src/server/model/Game.js:70`,
   `src/server/model/Game.js:72`).
2. `testCatch()` looks up the avatar body in the bonus world, removes the bonus,
   then applies it (`src/server/manager/BonusManager.js:106`,
   `src/server/manager/BonusManager.js:112`).
3. `remove()` removes the bonus body and emits `bonus:clear`
   (`src/server/manager/BonusManager.js:140`).
4. Circle overlap is strict `<`, not `<=`, through `Island.bodiesTouch()`
   (`src/server/core/Island.js:83`, `src/server/core/Island.js:89`).

Type selection is weighted by `bonusType.prototype.getProbability(game)`.
`getRandomBonusType()` builds a cumulative pot and samples with
`Math.random() * totalWeight` (`src/server/manager/BonusManager.js:167`,
`src/server/manager/BonusManager.js:178`,
`src/server/manager/BonusManager.js:188`).
The focused fixtures `source_bonus_default_weights_type_rng_step.json` and
`source_bonus_default_weights_select_game_clear_step.json` pin this for the full
default enabled order: with two of four present avatars already dead,
`BonusGameClear` contributes `0.5`, the default total weight is `11.5`, type
draw `0.945` still selects `BonusAllColor`, and type draw `0.965` selects
`BonusGameClear` before the position draws.

`source_bonus_default_weights_game_clear_full_probability_step.json` pins the
other side of that dynamic probability branch. With only one of four present
avatars dead, the source ratio is below `0.5`, `BonusGameClear` contributes its
full probability `1`, the default total weight is `12.0`, and type draw `0.93`
selects `BonusGameClear` before the same `(27.255, 73.745)` position draws.

`source_bonus_spawn_cap_twenty_step.json` pins the source cap branch: after
`BonusManager.start()` schedules the first pop, the fixture seeds 20 active map
bonuses, advances the fake clock to the pop, observes the next-delay draw, and
then sees no type draw, no position draw, and no new `bonus:pop`.

Fast-runtime promotion note: `vector_runtime.bonus_type_selection_metadata`
mirrors only the weighted type-selection metadata for already eligible rows.
The caller must provide the row-local `bonus.type` draw plus `alive`/`present`
arrays; the helper returns selected type, total weight, weighted draw, and
`BonusGameClear` dynamic probability. It does not schedule bonus timers, own the
`bonus.start_delay`/`bonus.next_delay_after_pop`/position RNG sequence, enforce
the cap-at-20 branch, mutate optional bonus arrays, or claim public bonus-env
support.

Fast-runtime cap note: `vector_runtime.bonus_spawn_cap_metadata` mirrors only
the row-local source cap gate for already-due pop rows after the caller has
handled timer eligibility and the next-delay draw. Rows with caller-supplied
`bonus_count >= 20` are returned as capped and excluded from type selection.
It does not schedule natural bonus timers, draw the next delay, draw type or
position RNG, spawn bonuses, mutate optional bonus arrays, enforce public-env
cap behavior, or claim replay support.

`source_bonus_spawn_bonus_world_retry_step.json` pins the bonus-world side of
the same source spawn-position check. After `BonusManager.start()` schedules the
first pop, the fixture seeds one active `BonusSelfSmall` at the first natural
candidate position. The source rejects that bonus-world collision, consumes one
retry x/y pair, and emits the new `bonus:pop` for the accepted position
`(68.072, 19.928)`. `tests/test_env_scenarios.py` guards the JS oracle output,
and `tests/test_source_env.py` guards the matching `CurvyTronSourceEnv` path.
This proves only the bonus-world retry branch. It does not promote fast-runtime
natural spawn, a public bonus env, bonus replay, or broad bonus effects.

Bonus ids are assigned by `Collection` because the manager uses
`new Collection([], 'id', true)` (`src/shared/manager/BaseBonusManager.js:11`).
`Collection.setId()` increments missing ids (`src/shared/Collection.js:124`,
`src/shared/Collection.js:132`).

## Targets And Timers

`Bonus.applyTo()` resolves a target, starts a timeout only when `duration` is
truthy, then calls `on()` (`src/server/model/Bonus/Bonus.js:22`,
`src/server/model/Bonus/Bonus.js:26`). `BonusSelf`, `BonusEnemy`, `BonusAll`,
and `BonusGame` differ mainly by `getTarget()`, `on()`, and `off()`:

| Base type | Source refs | Target behavior |
| --- | --- | --- |
| Self | `src/server/model/Bonus/BonusSelf.js:22`, `src/server/model/Bonus/BonusSelf.js:32`, `src/server/model/Bonus/BonusSelf.js:40` | Alive catcher only; add/remove this bonus to that avatar stack. |
| Enemy | `src/server/model/Bonus/BonusEnemy.js:22`, `src/server/model/Bonus/BonusEnemy.js:32`, `src/server/model/Bonus/BonusEnemy.js:40` | Alive avatars except the catcher. |
| All | `src/server/model/Bonus/BonusAll.js:22`, `src/server/model/Bonus/BonusAll.js:32`, `src/server/model/Bonus/BonusAll.js:40` | All alive avatars. |
| Game | `src/server/model/Bonus/BonusGame.js:22`, `src/server/model/Bonus/BonusGame.js:32`, `src/server/model/Bonus/BonusGame.js:40` | The `Game` object through `game.bonusStack`. |

`BaseBonusStack.resolve()` starts from default properties, applies all active
effects, then writes the final values (`src/shared/model/BaseBonusStack.js:52`,
`src/shared/model/BaseBonusStack.js:65`,
`src/shared/model/BaseBonusStack.js:78`). Most properties add their effect value
(`src/shared/model/BaseBonusStack.js:115`). `BonusStack.append()` overrides
`directionInLoop`, `angularVelocityBase`, and `color` by assignment instead of
addition (`src/server/model/BonusStack.js:97`).

Death clears an avatar's bonus stack without resolving properties back to
defaults (`src/shared/model/BaseAvatar.js:78`). The fixture
`source_bonus_self_fast_stack_death_late_expiry_step.json` pins the important
2P safety case: three stacked `BonusSelfFast` catches raise velocity to `52`,
normal-wall death empties the active stack but leaves the dead avatar's velocity
as-is, and later timeout removals emit `bonus:stack remove` events without
property restore or state resurrection.

`source_bonus_self_fast_expiry_then_wall_death_same_tick_step.json` pins the
other focused 2P timer ordering: one `BonusSelfFast` timeout is drained before
the same source-runner step calls `Game.update`. The timeout restores velocity
to `16`; the following wall-death movement uses that restored speed, p0 reaches
`x=-1.4`, dies, p1 scores, and the stack remains empty. This is source runner
and public vector timer ordering, not browser event-loop or pixel proof.

`test_source_env_4p_bonus_targets_skip_dead_and_absent_avatars` and the matching
public vector tests pin the first 4P target-filter proof: enemy bonuses target
other alive avatars only, all-avatar bonuses target alive avatars only, absent
seats are skipped because source reset/remove marks them not alive, and game
bonuses still apply to global game state. This is a focused source/public
targeting proof, not broad 3P/4P bonus stack/death replay coverage.

`source_bonus_enemy_slow_4p_stack_wall_death_terminal_step.json` pins the
focused 4P terminal stack/death case against the JS oracle: p0 catches
`BonusEnemySlow`, p1/p2/p3 receive slowed stack entries, the slowed targets hit
the normal wall before the 5000 ms expiry, death clears each dead target stack
without restoring velocity, and p0 wins the round. The public vector mirror now
uses the same fixture; trainer/replay has the matching packaging proof. This is
still a focused case, not broad browser event-loop, render, or every-bonus proof.

## Bonus Table

Unless noted, probability is the base `1` from `BaseBonus`
(`src/shared/model/BaseBonus.js:45`) and duration is the base `5000` ms from
`BaseBonus` (`src/shared/model/BaseBonus.js:38`).

| Bonus | Target | Duration | Probability | Effects and source refs |
| --- | --- | ---: | ---: | --- |
| `BonusSelfSmall` | Self | `7500` | `1` | `['radius', -1]`: `src/server/model/Bonus/BonusSelfSmall.js:20`, `src/server/model/Bonus/BonusSelfSmall.js:29`. |
| `BonusSelfSlow` | Self | `5000` | `1` | `['velocity', -BaseAvatar.prototype.velocity/2]`: `src/server/model/Bonus/BonusSelfSlow.js:22`. |
| `BonusSelfFast` | Self | `4000` | `1` | `['velocity', 0.75 * BaseAvatar.prototype.velocity]`: `src/server/model/Bonus/BonusSelfFast.js:20`, `src/server/model/Bonus/BonusSelfFast.js:29`. |
| `BonusSelfMaster` | Self | `7500` | `1` | `['invincible', true]` and `['printing', -1]`: `src/server/model/Bonus/BonusSelfMaster.js:20`, `src/server/model/Bonus/BonusSelfMaster.js:29`. |
| `BonusEnemySlow` | Enemies | `5000` | `1` | `['velocity', -BaseAvatar.prototype.velocity/2]`: `src/server/model/Bonus/BonusEnemySlow.js:22`. |
| `BonusEnemyFast` | Enemies | `6000` | `1` | `['velocity', 0.75 * BaseAvatar.prototype.velocity]`: `src/server/model/Bonus/BonusEnemyFast.js:20`, `src/server/model/Bonus/BonusEnemyFast.js:29`. |
| `BonusEnemyBig` | Enemies | `7500` | `1` | `['radius', 1]`: `src/server/model/Bonus/BonusEnemyBig.js:20`, `src/server/model/Bonus/BonusEnemyBig.js:29`. |
| `BonusEnemyInverse` | Enemies | `5000` | `1` | `['inverse', 1]`: `src/server/model/Bonus/BonusEnemyInverse.js:20`, `src/server/model/Bonus/BonusEnemyInverse.js:29`. |
| `BonusEnemyStraightAngle` | Enemies | `5000` | `1` | `['directionInLoop', false]` and `['angularVelocityBase', Math.PI/2]`: `src/server/model/Bonus/BonusEnemyStraightAngle.js:20`, `src/server/model/Bonus/BonusEnemyStraightAngle.js:27`, `src/server/model/Bonus/BonusEnemyStraightAngle.js:36`. |
| `BonusGameBorderless` | Game | `10000` | `1` | `['borderless', true]`: `src/server/model/Bonus/BonusGameBorderless.js:20`, `src/server/model/Bonus/BonusGameBorderless.js:27`, `src/server/model/Bonus/BonusGameBorderless.js:36`. |
| `BonusAllColor` | All | `7500` | `1` | Rotates each alive avatar to the next alive avatar's color: `src/server/model/Bonus/BonusAllColor.js:22`, `src/server/model/Bonus/BonusAllColor.js:32`, `src/server/model/Bonus/BonusAllColor.js:54`, `src/server/model/Bonus/BonusAllColor.js:66`. |
| `BonusGameClear` | Game | `0` | Dynamic | Clears trails immediately; probability depends on alive/present ratio: `src/server/model/Bonus/BonusGameClear.js:20`, `src/server/model/Bonus/BonusGameClear.js:29`, `src/server/model/Bonus/BonusGameClear.js:43`. |
| `BonusSelfGodzilla` | Hidden self class | `5000` | `1` | Not normally selectable. Effects are invincible, printing `100`, radius `10`, velocity `6`: `src/server/model/Bonus/BonusSelfGodzilla.js:22`. |

`BonusGameClear.getProbability(game)` computes:

```text
ratio = 1 - aliveAvatars / presentAvatars
if ratio < 0.5: return 1
else: return round((1 - ratio) * 10) / 10
```

Source refs: `src/server/model/Bonus/BonusGameClear.js:29`,
`src/server/model/Bonus/BonusGameClear.js:31`,
`src/server/model/Bonus/BonusGameClear.js:33`,
`src/server/model/Bonus/BonusGameClear.js:37`.

## Speed And Turn Effects

Source base speed is `16` units per second and base angular velocity is
`2.8 / 1000` radians per millisecond
(`src/shared/model/BaseAvatar.js:40`,
`src/shared/model/BaseAvatar.js:47`). Velocity components are recomputed as:

```text
velocityX = cos(angle) * velocity / 1000
velocityY = sin(angle) * velocity / 1000
```

Source refs: `src/shared/model/BaseAvatar.js:212`,
`src/shared/model/BaseAvatar.js:214`,
`src/shared/model/BaseAvatar.js:216`.

Speed bonus effects are additive deltas on top of default velocity. So one slow
effect gives `16 - 8 = 8`; one fast effect gives `16 + 12 = 28`.
`setVelocity()` clamps to at least `8` and then recomputes velocity vectors
(`src/shared/model/BaseAvatar.js:199`, `src/shared/model/BaseAvatar.js:201`,
`src/shared/model/BaseAvatar.js:203`).

Changing speed also changes turn feel. `updateVelocities()` calls
`updateBaseAngularVelocity()` (`src/shared/model/BaseAvatar.js:212`,
`src/shared/model/BaseAvatar.js:219`). The source formula is:

```text
ratio = velocity / 16
angularVelocityBase = ratio * (2.8 / 1000) + log(1 / ratio) / 1000
```

Source refs: `src/shared/model/BaseAvatar.js:225`,
`src/shared/model/BaseAvatar.js:228`,
`src/shared/model/BaseAvatar.js:229`.

Player inputs call `updateAngularVelocity(factor)`, which applies inverse
controls and multiplies by the current `angularVelocityBase`
(`src/shared/model/BaseAvatar.js:124`,
`src/shared/model/BaseAvatar.js:131`,
`src/server/controller/GameController.js:314`). Normal turning updates angle by
`angularVelocity * step`; straight-angle mode adds `angularVelocity` once and
then zeroes it (`src/shared/model/BaseAvatar.js:169`,
`src/shared/model/BaseAvatar.js:172`,
`src/shared/model/BaseAvatar.js:175`).

Future fidelity note: `BonusEnemyStraightAngle` changes `angularVelocityBase`
and `directionInLoop` through stack application. A probe should confirm exactly
when the new turn base affects an already-turning avatar versus the next
`player:move` event.

## Radius Effects

Base avatar radius is `0.6` (`src/shared/model/BaseAvatar.js:54`).
Radius stack values are exponents, not raw radii:

```text
effectiveRadius = 0.6 * 2 ** stackedRadiusValue
```

Source refs: `src/server/model/BonusStack.js:45`,
`src/server/model/BonusStack.js:46`. So `BonusSelfSmall` with `-1` gives
`0.3`, and `BonusEnemyBig` with `1` gives `1.2`. `setRadius()` clamps to at
least `0.6 / 8 = 0.075` (`src/shared/model/BaseAvatar.js:239`,
`src/shared/model/BaseAvatar.js:241`). The server avatar also updates the
collision body radius and emits a `property` event
(`src/server/model/Avatar.js:109`, `src/server/model/Avatar.js:113`,
`src/server/model/Avatar.js:114`).

## Color Effects

`BonusAllColor` snapshots alive targets and their current colors, then gives
each target the next color in that snapshot
(`src/server/model/Bonus/BonusAllColor.js:32`,
`src/server/model/Bonus/BonusAllColor.js:36`,
`src/server/model/Bonus/BonusAllColor.js:40`,
`src/server/model/Bonus/BonusAllColor.js:66`,
`src/server/model/Bonus/BonusAllColor.js:70`). `BonusStack.append()` treats
`color` as an override, not an additive value
(`src/server/model/BonusStack.js:97`, `src/server/model/BonusStack.js:102`).
The default color on removal is the player color
(`src/server/model/BonusStack.js:83`). `Avatar.setColor()` emits a `property`
event (`src/server/model/Avatar.js:145`, `src/server/model/Avatar.js:148`).

## Borderless Bonus

`BonusGameBorderless` lasts `10000` ms, has probability `1`, and contributes
`['borderless', true]` to the game stack
(`src/server/model/Bonus/BonusGameBorderless.js:20`,
`src/server/model/Bonus/BonusGameBorderless.js:27`,
`src/server/model/Bonus/BonusGameBorderless.js:36`).

`GameBonusStack.apply()` maps this to `game.setBorderless(true/false)`
(`src/server/model/GameBonusStack.js:40`,
`src/server/model/GameBonusStack.js:43`). `Game.setBorderless()` emits the
`borderless` event when the value changes (`src/server/model/Game.js:297`,
`src/server/model/Game.js:301`), and `GameController` sends that to clients
(`src/server/controller/GameController.js:508`,
`src/server/controller/GameController.js:510`).

At movement time, normal border checks use the avatar radius as margin. When
`borderless` is true, border checks use margin `0`; hitting the bound wraps to
the opposite edge instead of killing the avatar
(`src/server/model/Game.js:51`, `src/server/model/Game.js:54`,
`src/server/model/Game.js:55`, `src/server/model/Game.js:58`).
`World.getOposite()` maps left to right, right to left, top to bottom, and
bottom to top (`src/server/core/World.js:305`,
`src/server/core/World.js:307`, `src/server/core/World.js:311`,
`src/server/core/World.js:315`, `src/server/core/World.js:319`).

Every new round resets `borderless` to the base `false` value and clears active
bonuses (`src/shared/model/BaseGame.js:170`,
`src/shared/model/BaseGame.js:172`,
`src/shared/model/BaseGame.js:174`).

## Clear, Printing, Inverse, And Invincible Effects

`BonusGameClear` is immediate because its duration is `0`
(`src/server/model/Bonus/BonusGameClear.js:20`). It does not use the normal
game stack on/off path; its `on()` directly calls `game.clearTrails()`
(`src/server/model/Bonus/BonusGameClear.js:43`,
`src/server/model/Bonus/BonusGameClear.js:45`). `Game.clearTrails()` clears and
reactivates the collision world and emits `clear` (`src/server/model/Game.js:198`,
`src/server/model/Game.js:200`, `src/server/model/Game.js:202`).
`source_bonus_game_clear_immediate_step.json` now pins this as a forced
JS/Python source-env proof only: seeded catch after safe movement, `bonus:clear`
then `clear`, `worldActive=true`, `worldBodyCount=0`, no `bonus:stack`, no
avatar property change, and no active avatar bonuses. It does not prove natural
spawned `BonusGameClear` catch/clear coupling. Natural `BonusGameClear`
probability/type selection is covered separately by the default-weight fixtures
and by the metadata-only `bonus_type_selection_metadata` helper for
caller-supplied type draws.

`BonusSelfMaster` sets invincible and stops printing through the avatar stack
(`src/server/model/Bonus/BonusSelfMaster.js:29`). `BonusStack.apply()` maps
`invincible` to `setInvincible()`, and `printing` to
`printManager.start()` or `printManager.stop()` depending on whether the final
value is positive (`src/server/model/BonusStack.js:54`,
`src/server/model/BonusStack.js:57`). The default stack value for `printing` is
`1`, so the master effect `-1` resolves to `0` and stops printing
(`src/server/model/BonusStack.js:79`).

`BonusEnemyInverse` adds `1` to the `inverse` stack. The final inverse flag is
`value % 2 !== 0`, so two active inverse effects cancel each other
(`src/server/model/Bonus/BonusEnemyInverse.js:29`,
`src/server/model/BonusStack.js:51`,
`src/server/model/BonusStack.js:52`). `BaseAvatar.setInverse()` updates
angular velocity when inverse changes (`src/shared/model/BaseAvatar.js:249`,
`src/shared/model/BaseAvatar.js:253`).

## Wire And Trace Events

Important server events for bonus fidelity:

| Event | Source refs | Payload |
| --- | --- | --- |
| `bonus:pop` | `src/server/controller/GameController.js:381` | `[bonus.id, compressed x, compressed y, bonus.constructor.name]`. |
| `bonus:clear` | `src/server/controller/GameController.js:396` | `bonus.id`. |
| `bonus:stack` | `src/server/controller/GameController.js:440` | `[avatar.id, add/remove, bonus.id, bonus.constructor.name, bonus.duration]`. |
| `property` | `src/server/controller/GameController.js:426` | `[avatar.id, property, value]`, including velocity, radius, invincible, inverse, color, printing. |
| `borderless` | `src/server/controller/GameController.js:508` | Boolean borderless state. |
| `clear` | `src/server/controller/GameController.js:498` | No payload. |

The client turns these events into map icons, local stack icons, and local
game state (`src/client/repository/GameRepository.js:249`,
`src/client/repository/GameRepository.js:267`,
`src/client/repository/GameRepository.js:282`,
`src/client/repository/GameRepository.js:329`).

## v0 Ignore List

These source facts can be ignored by the current `curvyzero-v0` ruleset, as long
as the ruleset stays clearly labeled as simplified no-bonus behavior:

- Bonus spawning, `bonusRate`, weighted probabilities, bonus ids, and random
  bonus positions.
- Bonus catch bodies and active bonus world state.
- Stack timers and expiration ordering beyond the one pinned
  `BonusSelfSmall` expiry/restore source-env slice.
- All avatar bonus effects: speed, radius, inverse, invincible, printing, turn
  mode, and color.
- Game bonus effects beyond the forced `BonusGameClear` source-env proof and
  natural `BonusGameClear` type-selection proof: borderless wrap and natural
  spawned-clear catch/effect coupling.
- Bonus wire events and client sprite/display behavior.
- `BonusSelfGodzilla`, because it is not reachable from normal room config.

This matches current code: `CurvyTronEnv` is explicitly a "Minimal deterministic
1v1 no-bonus environment" (`src/curvyzero/env/core.py:29`,
`src/curvyzero/env/core.py:30`). `CurvyTronReferenceDefaults` stores source
metadata for future work and says `curvyzero-v0` does not consume it directly
(`src/curvyzero/env/config.py:14`, `src/curvyzero/env/config.py:17`).

## Future Source-Fidelity Requirements

A future `curvytron-v1-reference` or equivalent ruleset should match these
source behaviors:

- Room config must expose the default enabled bonus list, `bonusRate`, and
  max-score defaults from `BaseRoomConfig`.
- Bonus manager must use source spawn cap, source pop-time formula, source
  weighted probability selection, and source random stream policy.
- Spawn placement must test both the main game world and bonus world with the
  same margin body used by `BonusManager.getRandomPosition()`.
- Catch timing must run after movement, border/body collision, and print-manager
  testing, in source avatar update order.
- Stack math must reset removed properties to defaults, add normal numeric
  effects, override special properties, and update avatar setters in the same
  places as the source.
- Speed bonuses must also reproduce source turn-rate coupling through
  `updateBaseAngularVelocity()`.
- Radius bonuses must use exponent math and update collision body radius.
- Color bonus must rotate alive players' colors from a snapshot and restore
  player colors on expiry.
- Borderless must use margin `0`, wrap through `World.getOposite()`, emit
  `borderless`, and reset at round start.
- Clear bonus must immediately clear and reactivate the collision world.
- Trace output must include active bonuses, bonus stack entries, changed avatar
  properties, game `borderless`, and emitted bonus events.

Open probes:

- Determine a deterministic `Math.random` policy for spawn timing, spawn
  placement, and probability selection.
- Prove one speed bonus, one radius collision case beyond the existing
  `BonusSelfSmall` restore proof, one color bonus, and one borderless wrap with
  headless JS traces. The first forced clear trace and natural `BonusGameClear`
  probability/selection metadata are pinned; natural spawned clear catch/effect
  coupling remains open.
- Extend stacked expiration/death probes beyond the promoted 2P
  `BonusSelfFast` wall-death fixtures, especially other bonus types, larger
  stacks, and special override properties.
- Test `BonusEnemyStraightAngle` timing for already-turning avatars.
- Decide whether to keep `BonusSelfGodzilla` only as a hidden source note or add
  a non-default debug scenario for it.
