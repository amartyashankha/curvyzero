# CurvyTron Source Map

Status: working index

This folder is the map for reverse-engineering the original CurvyTron source.
It should help the next pass pick a subsystem, find the right files, and write
one small proof. It is not meant to be a full source audit by itself.

Start with:

- [Compact facts index](facts_index.md)
- [Environment source claim tracker](../../working/environment/coverage_tracker.md)
- [No-bonus multiplayer source fixture gaps](../../working/environment/no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md)
- [CurvyTron reference source record](../../sources/curvytron_reference.md)
- [CurvyTron reference notes](../curvytron_reference_notes.md)
- [Environment design index](../../design/environment/README.md)
- [Environment research index](../environment/README.md)
- [Environment fidelity handoff](../../handoffs/2026-05-08-environment-fidelity-handoff.md)

## Working Rule

Use server-side files as the source of truth when client and server behavior
overlap. Use the client mainly for input mapping, display, and wire behavior.
If a source/probe result is missing, mark it `pending` instead of guessing.
For Environment Reconstruction, there is one intended product/runtime path: the
fast shared CurvyTron environment. `VectorMultiplayerEnv` is the current
no-bonus multiplayer path, while strict `VectorTrainerEnv1v1NoBonus` is only
the older proven 1v1 boundary. JS probes and `CurvyTronSourceEnv` are proof
tools, not alternate product environments.

## Source Hierarchy

| Area | Main source files | What this covers | Current status |
| --- | --- | --- | --- |
| World and borders | `src/shared/model/BaseGame.js`, `src/server/model/Game.js`, `src/server/core/World.js`, `src/server/core/Island.js` | Map size, coordinate bounds, island buckets, spawn space, normal-wall death, borderless wrap. | Narrow source-env/oracle slices are pinned for normal wall, selected borderless branches, island lookup, and 2P/3P/4P spawn/order facts. Broader natural reset and edge sweeps remain open. |
| Movement | `src/shared/model/BaseAvatar.js`, `src/server/model/Avatar.js`, `src/client/model/PlayerInput.js`, `src/server/controller/GameController.js` | Speed, turn rate, elapsed-ms integration, input values, inverse controls, straight-angle hooks. | Source-mined. One forced two-player movement trace now matches through the source-kinematics runner. |
| Trails | `src/server/manager/PrintManager.js`, `src/server/model/Avatar.js`, `src/server/model/Trail.js`, `src/shared/model/BaseTrail.js`, `src/shared/model/BaseAvatar.js` | Trail start delay, point emission, distance-based holes, final death point, visual gaps versus collision bodies. | Narrow PrintManager, trail cadence, trail-gap, and death-stop slices are pinned. Broader natural multi-step variants remain open. |
| Collisions | `src/server/model/Game.js`, `src/server/core/World.js`, `src/server/core/Island.js`, `src/server/core/AvatarBody.js`, `src/server/core/Body.js` | Update order, endpoint-circle overlap, strict `<`, self-collision latency, current-head behavior, multi-player ordering. | Narrow wall, body, PrintManager death-stop, and collision-order slices are pinned. Broader natural emitted-body and 3P/4P order stress remain open. |
| Scoring | `src/server/model/Game.js`, `src/shared/model/BaseAvatar.js`, `src/shared/model/BaseRoomConfig.js` | Death score, same-frame score, winner bonus, match score, max score, tied leaders. | Narrow 1P/2P/3P/4P wall and selected lifecycle scoring slices are pinned. Broader present/non-present, match lifecycle, and public/trainer parity remain open. |
| Rounds | `src/shared/model/BaseGame.js`, `src/server/model/Game.js` | Warmup, warmdown, `game:start`, delayed printing, round end, game end. | The `source-lifecycle-v25` slice is pinned for 28 named 2P/3P/4P cases including focused 4P all-present multi-round match end, focused 4P present/absent first-round/survivor/next-round facts, focused 3P/4P present/absent tie-at-max source proofs, one focused 3P warmdown leave source proof, and `source_lifecycle_remove_avatar_to_single_present_3p.json`. Focused public metadata parity is green for 4P present/absent, 4P all-present multi-round, 3P/4P present/absent tie-at-max, 3P staged match-mode warmdown leave, and the single-present 3P active leave edge. Broad public warmdown leave, broader leave edge variants, broader public lifecycle parity, replay, and trainer observations remain open. |
| Bonuses | `src/shared/model/BaseBonus*.js`, `src/shared/manager/BaseBonusManager.js`, `src/server/manager/BonusManager.js`, `src/server/model/Bonus/*.js`, `src/server/model/BonusStack.js`, `src/server/model/GameBonusStack.js` | Spawn timing, catch radius, enabled set, targets, durations, stacking, movement/control/trail/game effects. | Source-mined. Narrow `BonusSelfSmall` catch/spawn/expiry traces are pinned; broad implementation remains deferred. |
| Networking | `src/server/controller/GameController.js`, `src/client/controller/GameController.js`, `src/shared/service/Compressor.js`, `src/server/trackers/*.js`, `src/server/core/SocketClient.js` | Move messages, state/event messages, compression, per-player perspectives, trackers. | Deferred. Not needed for the first state oracle, except input mapping. |
| Rendering | `src/client/model/Game.js`, `src/client/model/Avatar.js`, `src/client/model/Trail.js`, `src/client/core/Canvas.js`, `src/client/manager/BonusManager.js`, `src/sass/**` | Canvas scale, visible trails, bonus sprites, UI state, screenshots/videos. | Deferred. Pixels are later human review after state traces. |
| Config and build | `config.json.sample`, `src/server/launcher.js`, `src/server/dependencies.js`, `gulpfile.js`, `bower.json`, `package.json` | Default port/config, source load order, old build output, browser assets, run strategy. | Partly known. Raw app build is blocked in this checkout by missing generated files and old dependencies. |

## Reverse-Engineering Order

1. Lock movement and time first.
2. Add world and border goldens.
3. Add collision and trail goldens.
4. Add scoring and round lifecycle cases.
5. Add bonuses only after the core round is steady.
6. Add networking and rendering after state/event traces are useful.
7. Keep config/build work separate from gameplay proofs.

## Docs That Should Exist

These are the docs this source map expects over time. Do not create all of them
at once. Create a page when a subsystem has enough proof to be useful.

| Doc | Covers | Current status |
| --- | --- | --- |
| `README.md` | This hierarchy, source roots, doc plan, and work order. | Exists. |
| `facts_index.md` | Short facts index across movement, world/borders, trails, collision, scoring, multiplayer, bonuses, networking/rendering/build, plus high-risk probe targets. | Exists. |
| `open_questions.md` | Questions that block faithful reconstruction or need a probe. | Exists. |
| `movement_controls.md` | Input values, elapsed-ms integration, speed, turn rate, inverse, straight-angle behavior, update order, and current Python gaps. | Exists. First narrow movement trace is proven. |
| `collisions_trails_world.md` | Map/world grid, borders, borderless wrap, trail points, print holes, collision bodies, self latency, and collision probe plan. | Exists. Named source fixtures pin narrow wall/body/trail slices; broader variants remain open. |
| `rounds_scoring_multiplayer.md` | Warmup/warmdown, map size, present/alive state, deaths, round end, winner score, same-frame scoring, and 2/3/4-player cases. | Exists. Named source fixtures now prove a narrow multiplayer slice; broader variants are still listed as open. |
| `bonuses_config.md` | Room config, enabled bonuses, spawn/catch, durations, targets, stack rules, game effects, wire events, and future fidelity requirements. | Exists. First narrow `BonusSelfSmall` catch/spawn/expiry traces are pinned; broad implementation remains deferred. |
| `network_render_build.md` | Socket protocol, controllers, client mirrors, rendering path, trackers, old build shape, and run notes. | Exists. Deferred except input/event/compression facts. |

## Evidence Status Terms

- `source-mined`: the source files were read and a fact is recorded.
- `probe-backed`: a headless JS trace or runnable check confirms the behavior.
- `python-matched`: Python source-fidelity behavior matches the JS trace.
- `deferred`: useful later, but not needed for the next fidelity loop.
- `pending`: known gap. Do not treat it as solved.
