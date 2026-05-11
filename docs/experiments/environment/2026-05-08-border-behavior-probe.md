# 2026-05-08 Border Behavior Probe

## Question

What does the original CurvyTron server code do when an avatar crosses the right
border?

## Setup

- Working tree: `/Users/shankha/curvy`
- Reference clone: `third_party/curvytron-reference`
- Probe script: `tools/reference_oracle/border_probe.js`
- Node: `v25.9.0`
- No dependency install was run.

The probe follows the existing headless oracle pattern from
`tools/reference_oracle/headless_probe.js` and `tools/reference_oracle/scenario_runner.js`:
it loads the original server-side source files into a Node `vm`, creates a small fake
room, forces avatar 1 near the right edge, and calls `game.update(step_ms)`.

The forced state uses a two-player game, target avatar 1, arena size 88, radius 0.6,
start x 87.35, y 44, angle 0, and a 100 ms step. At source speed 16 units/s, the
step moves the target to x 88.95 before wall handling.

## Commands

```sh
node --check tools/reference_oracle/border_probe.js
```

Result: exit 0, no output.

```sh
node tools/reference_oracle/border_probe.js
```

Output:

```json
{
  "probe": "border_behavior",
  "target": "avatar 1 crosses the right boundary",
  "cases": [
    {
      "mode": "normal",
      "result": "dies",
      "stepMs": 100,
      "startGap": 0.05,
      "game": {
        "size": 88,
        "started": true,
        "inRound": false,
        "borderless": false,
        "deathCount": 1,
        "deaths": [
          1
        ],
        "roundWinner": 2,
        "gameWinner": null,
        "worldBodyCount": 1
      },
      "targetBefore": {
        "id": 1,
        "name": "p0",
        "x": 87.35,
        "y": 44,
        "angle": 0,
        "velocityX": 0.016,
        "velocityY": 0,
        "radius": 0.6,
        "alive": true,
        "present": true,
        "printing": false,
        "trailPointCount": 0,
        "bodyNum": 0,
        "bodyCount": 0
      },
      "targetAfter": {
        "id": 1,
        "name": "p0",
        "x": 88.95,
        "y": 44,
        "angle": 0,
        "velocityX": 0.016,
        "velocityY": 0,
        "radius": 0.6,
        "alive": false,
        "present": true,
        "printing": false,
        "trailPointCount": 1,
        "bodyNum": 0,
        "bodyCount": 1
      },
      "events": [
        {
          "event": "position",
          "data": {
            "avatar": 1,
            "x": 88.95,
            "y": 44
          }
        },
        {
          "event": "die",
          "data": {
            "avatar": 1,
            "killer": null,
            "old": null
          }
        },
        {
          "event": "round:end",
          "data": {
            "winner": 2
          }
        }
      ]
    },
    {
      "mode": "borderless",
      "result": "wraps",
      "stepMs": 100,
      "startGap": 0.05,
      "game": {
        "size": 88,
        "started": true,
        "inRound": true,
        "borderless": true,
        "deathCount": 0,
        "deaths": [],
        "roundWinner": null,
        "gameWinner": null,
        "worldBodyCount": 0
      },
      "targetBefore": {
        "id": 1,
        "name": "p0",
        "x": 87.35,
        "y": 44,
        "angle": 0,
        "velocityX": 0.016,
        "velocityY": 0,
        "radius": 0.6,
        "alive": true,
        "present": true,
        "printing": false,
        "trailPointCount": 0,
        "bodyNum": 0,
        "bodyCount": 0
      },
      "targetAfter": {
        "id": 1,
        "name": "p0",
        "x": 0,
        "y": 44,
        "angle": 0,
        "velocityX": 0.016,
        "velocityY": 0,
        "radius": 0.6,
        "alive": true,
        "present": true,
        "printing": false,
        "trailPointCount": 0,
        "bodyNum": 0,
        "bodyCount": 0
      },
      "events": [
        {
          "event": "borderless",
          "data": {
            "value": true
          }
        },
        {
          "event": "position",
          "data": {
            "avatar": 1,
            "x": 88.95,
            "y": 44
          }
        },
        {
          "event": "position",
          "data": {
            "avatar": 1,
            "x": 0,
            "y": 44
          }
        }
      ]
    }
  ],
  "loadedSourceCount": 25
}
```

## Answer

Normal mode kills the avatar after the step crosses the wall. The avatar remains at
the crossed position, x 88.95, and emits `die`.

Borderless mode wraps the avatar. It first emits the crossed position, x 88.95, then
sets the avatar to the opposite side at x 0. The avatar stays alive.
