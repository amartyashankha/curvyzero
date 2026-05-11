# CurvyTron Reference

Access date: 2026-05-08

## Source

- URL: https://github.com/Curvytron/curvytron
- Local path: `third_party/curvytron-reference`
- Local commit inspected initially: `8fec14c`
- License file present: `third_party/curvytron-reference/LICENSE`

## Role

This repository is source/reference material for rule mining, provenance, possible browser smoke tests, and golden-test inspiration. It is not the main training runtime.

Server-side files are authoritative for simulator mechanics when client and server models
overlap. The main files inspected for source-fidelity promotion are:

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

## Promoted Findings

The detailed working notes live in `docs/research/curvytron_reference_notes.md`. These
source-derived findings have been promoted into simulator design docs and config metadata:

- Runtime tick target is 60 Hz, with elapsed wall-clock milliseconds used as each physics
  step. Round warmup is `3000` ms and warmdown is `5000` ms.
- Avatar trail printing starts `3000` ms after `game:start`, so avatars move before trails
  begin.
- Map size formula is `round(sqrt(80^2 + ((players - 1) * 80^2 / 5)))`.
- Base avatar constants are velocity `16`, angular velocity base `2.8 / 1000` radians/ms,
  radius `0.6`, trail latency `3`, `inverse=false`, `invincible=false`, and
  `directionInLoop=true`.
- Movement integrates with elapsed milliseconds: velocity components are
  `cos(angle) * velocity / 1000` and `sin(angle) * velocity / 1000`.
- Trail gaps are distance-based through `PrintManager`: base print distance `60` and base
  hole distance `5`, with randomized multipliers.
- Collision is endpoint-circle based after motion. Body overlap is strict:
  `distance < radius_a + radius_b`. Same-avatar bodies only collide when the point number
  delta is greater than trail latency `3`.
- Server update order is avatar motion, border check, trail/body collision, then
  print-manager and bonus catch for avatars still alive.
- On death, `BaseAvatar.die()` emits the death point before
  `PrintManager.stop()`. For an active printing manager, stop emits the
  important stop point and `property printing=false` before the `die` event.
- Same-frame deaths share the same frame-start death-count score. The last alive avatar
  receives `max(players - 1, 1)` extra round score.
- Default max score is `max(1, (players - 1) * 10)`. A game winner must be a unique leader
  at or above max score; tied leaders continue.
- Default enabled bonuses are `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`,
  `BonusSelfMaster`, `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`,
  `BonusEnemyInverse`, `BonusEnemyStraightAngle`, `BonusGameBorderless`, `BonusAllColor`,
  and `BonusGameClear`.
- Bonus defaults include radius `3`, default duration `5000` ms, spawn cap `20`, and base
  pop time `3000` ms adjusted by `bonusRate`.

## Current Handling

The local clone is ignored by the main CurvyZero git repository to avoid accidentally committing a nested repository. If the project later wants tighter provenance, choose deliberately between a submodule, subtree, or vendored snapshot.

## Supported Decisions

- `docs/decisions/0001-investigation-first-repo-structure.md`
- `docs/design/rulesets.md`
- `docs/research/curvytron_reference_notes.md`
