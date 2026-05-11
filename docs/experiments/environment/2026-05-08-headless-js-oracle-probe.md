# 2026-05-08 Headless JS Oracle Probe

## Question

Can we load enough original CurvyTron server-side JS to step game state headlessly with
`avatar.updateAngularVelocity(move)` and `game.update(step_ms)`?

## Setup

- Working tree: `/Users/shankha/curvy`
- Reference clone: `third_party/curvytron-reference`
- Probe script: `tools/reference_oracle/headless_probe.js`
- Node: `v25.9.0`
- No dependency install was run in the reference clone.

## Source Read

I checked the existing oracle notes and the original server files for the direct-step path:

- `docs/research/environment/curvytron_js_state_oracle.md`
- `docs/design/environment/reference_oracle.md`
- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`
- `third_party/curvytron-reference/src/server/model/Avatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/core/Island.js`
- `third_party/curvytron-reference/src/server/core/AvatarBody.js`

The relevant behavior matches the notes:

- `Game.update(step)` is a synchronous state step.
- `step` is milliseconds.
- Move input is applied before the tick with `avatar.updateAngularVelocity(move)`.
- `move` values are `-1`, `0`, and `1`.
- The old source is global/concatenation-style JS, so direct `require(...)` is not enough.

## Command

```sh
node tools/reference_oracle/headless_probe.js
```

## Output

```json
{
  "rawDependencyRequire": {
    "ok": false,
    "target": "third_party/curvytron-reference/src/server/dependencies.js",
    "name": "Error",
    "code": "MODULE_NOT_FOUND",
    "message": "Cannot find module 'faye-websocket'\nRequire stack:\n- /Users/shankha/curvy/third_party/curvytron-reference/src/server/dependencies.js\n- /Users/shankha/curvy/tools/reference_oracle/headless_probe.js",
    "requireStack": [
      "third_party/curvytron-reference/src/server/dependencies.js",
      "tools/reference_oracle/headless_probe.js"
    ]
  },
  "vmProbe": {
    "scenario": "forced_two_player_turn_step",
    "feasible": true,
    "stepMs": 16.666667,
    "game": {
      "size": 88,
      "started": true,
      "inRound": true,
      "borderless": false,
      "deathCount": 0,
      "deaths": [],
      "roundWinner": null,
      "gameWinner": null,
      "worldBodyCount": 2
    },
    "avatars": [
      {
        "id": 1,
        "name": "p0",
        "move": -1,
        "x": 20.266376,
        "y": 39.98756,
        "angle": -0.046667,
        "velocity": 16,
        "velocityX": 0.015983,
        "velocityY": -0.000746,
        "angularVelocity": -0.0028,
        "radius": 0.6,
        "alive": true,
        "present": true,
        "printing": true,
        "score": 0,
        "roundScore": 0,
        "trailPointCount": 1,
        "lastTrailPoint": [
          20,
          40
        ],
        "bodyNum": 1,
        "bodyCount": 1,
        "printManager": {
          "active": true,
          "distance": 38.733333,
          "lastX": 20.266376,
          "lastY": 39.98756
        }
      },
      {
        "id": 2,
        "name": "p1",
        "move": 1,
        "x": 59.733624,
        "y": 39.98756,
        "angle": 3.188259,
        "velocity": 16,
        "velocityX": -0.015983,
        "velocityY": -0.000746,
        "angularVelocity": 0.0028,
        "radius": 0.6,
        "alive": true,
        "present": true,
        "printing": true,
        "score": 0,
        "roundScore": 0,
        "trailPointCount": 1,
        "lastTrailPoint": [
          60,
          40
        ],
        "bodyNum": 1,
        "bodyCount": 1,
        "printManager": {
          "active": true,
          "distance": 38.733333,
          "lastX": 59.733624,
          "lastY": 39.98756
        }
      }
    ],
    "events": [
      {
        "event": "angle",
        "data": {
          "avatar": 2,
          "angle": 3.188259
        }
      },
      {
        "event": "position",
        "data": {
          "avatar": 2,
          "x": 59.733624,
          "y": 39.98756
        }
      },
      {
        "event": "angle",
        "data": {
          "avatar": 1,
          "angle": -0.046667
        }
      },
      {
        "event": "position",
        "data": {
          "avatar": 1,
          "x": 20.266376,
          "y": 39.98756
        }
      }
    ]
  },
  "loadedSources": [
    "src/shared/Collection.js",
    "src/shared/service/BaseFPSLogger.js",
    "src/server/service/FPSLogger.js",
    "src/shared/service/Compressor.js",
    "src/shared/model/BaseBonus.js",
    "src/shared/model/BaseBonusStack.js",
    "src/shared/model/BaseTrail.js",
    "src/server/model/Trail.js",
    "src/shared/model/BaseAvatar.js",
    "src/server/model/BonusStack.js",
    "src/shared/model/BasePlayer.js",
    "src/shared/manager/BaseBonusManager.js",
    "src/server/manager/BonusManager.js",
    "src/server/manager/PrintManager.js",
    "src/server/core/Body.js",
    "src/server/core/Island.js",
    "src/server/core/World.js",
    "src/server/core/AvatarBody.js",
    "src/server/core/SocketGroup.js",
    "src/server/controller/GameController.js",
    "src/server/model/GameBonusStack.js",
    "src/shared/model/BaseGame.js",
    "src/server/model/Avatar.js",
    "src/server/model/Player.js",
    "src/server/model/Game.js"
  ]
}
```

## Result

Direct headless stepping is feasible.

The raw server dependency entry cannot be loaded without installed dependencies. The exact
first missing dependency is:

```text
Cannot find module 'faye-websocket'
```

But a small Node `vm` loader can skip `src/server/dependencies.js`, provide Node's built-in
`EventEmitter`, load the original model/core/manager files in order, create fake players,
force state, apply moves, and call `game.update(step_ms)`.

## Blocker

No blocker for direct state stepping.

The blocker is only for loading the raw bundled server dependency entry or running the real
server: `faye-websocket` is missing, and likely other old dependencies would be needed after
that. Those should still not be installed into `third_party/curvytron-reference`.

## Next Step

Grow the probe into a tiny trace suite:

- 2-player straight movement
- 2-player turn movement
- wall death and score
- same-frame double death
- 3-player and 4-player death-order scoring
