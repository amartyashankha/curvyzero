# 2026-05-08 JS Scenario Runner

## Question

Can the JS reference oracle run the shared environment scenario fixture directly?

## Scope

- Runner: `tools/reference_oracle/scenario_runner.js`
- Fixture: `scenarios/environment/forced_two_player_turn_step.json`
- Full server/browser hosting stayed out of scope.

## Result

Yes. The runner loads the original CurvyTron server-side JS through `vm`, applies forced
player state and moves, calls `game.update(step_ms)`, and prints JSON trace output.

With no CLI argument, it reads `scenarios/environment/forced_two_player_turn_step.json`
if present. If that file is absent, it runs the built-in forced two-player turn scenario.
It also accepts an explicit scenario path.

## Commands

```sh
node --check tools/reference_oracle/scenario_runner.js
```

Result: exit 0, no output.

```sh
node tools/reference_oracle/scenario_runner.js scenarios/environment/forced_two_player_turn_step.json
```

Output:

```json
{
  "scenario": "forced_two_player_turn_step",
  "playerCount": 2,
  "trace": [
    {
      "scenario": "forced_two_player_turn_step",
      "playerCount": 2,
      "tick": 0,
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
          "activeBonuses": [],
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
          "activeBonuses": [],
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
    }
  ],
  "source": "scenarios/environment/forced_two_player_turn_step.json",
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

## Notes

- The runner accepts the current fixture shape: `players[].initial`, `angle_rad`,
  `avatar_id`, and step move objects with `player_id`.
- The runner also accepts a simpler inline shape with `players[].state` and moves as an
  object or array.
- No blockers for this scenario-runner smoke test.
