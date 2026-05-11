# Collision Order Probe Plan

Status: death-point and head-head fixtures Python-promoted
Date: 2026-05-09

Scope: narrow collision-order runner support, promoted batch membership, tests,
and docs.

## Files Read

- `docs/working/environment/source_feature_inventory.md`
- `docs/working/environment/coverage_tracker.md`
- `docs/research/curvytron_source_map/collisions_trails_world.md`
- `docs/research/curvytron_source_map/rounds_scoring_multiplayer.md`
- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/model/Avatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js`
- `third_party/curvytron-reference/src/server/core/AvatarBody.js`
- `third_party/curvytron-reference/src/server/core/Island.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `tools/reference_oracle/scenario_runner.js`
- `scenarios/environment/source_collision_death_point_kills_later_player_step.json`
- `scenarios/environment/source_collision_head_head_reverse_order_single_death_step.json`

## Source Rules

- `Game.update(step)` captures `score = deaths.count()` once at frame start and
  visits avatars in reverse order.
- One avatar is fully processed before the next: move, optional normal point,
  wall check, body check, then print-manager/bonus checks only if still alive.
- `Game.onPoint()` synchronously inserts point bodies while the game is started
  and the world is active.
- Live avatar heads are not inserted into `World` as collision bodies by
  themselves. Same-endpoint head-head-looking collisions depend on emitted
  point bodies.
- `Game.kill()` calls `avatar.die(killer)` before `avatar.addScore(score)`.
  `BaseAvatar.die()` emits a point before the server emits `die`.
- Round end is checked after the full reverse-order loop, not immediately after
  the first kill.

Use source radius `0.6`, velocity `16` units/s, and `step_ms=100`, so straight
movement distance is `1.6`. Two-player map size is `88`.

## First Fixture

Candidate id: `source_collision_death_point_kills_later_player_step`

JS pin status: fixture added at
`scenarios/environment/source_collision_death_point_kills_later_player_step.json`.
The command
`node tools/reference_oracle/scenario_runner.js scenarios/environment/source_collision_death_point_kills_later_player_step.json`
matches the expected event and final-state plan. Raw oracle avatar ids are `2`
for `p1` and `1` for `p0`.

Python/common-trace status: promoted through the narrow
`source-body-canary` runner path and
`scenarios/environment/source_collision_order_batch.json`. The focused loop
`uv run python tools/run_fidelity_loop.py scenarios/environment/source_collision_death_point_kills_later_player_step.json --python-runner source-body-canary --fail-on-mismatch`
returns `diff_status=pass`.

Why first: it is the smallest open rule. It proves a death point from an
earlier-updated avatar can kill a later-updated avatar in the same source frame,
and it catches implementations that end the round immediately after the first
death.

Setup:

| Field | Value |
| --- | --- |
| player_count | `2` |
| map_size | `88` |
| game | `started=true`, `in_round=true`, `borderless=false`, `world_active=true` |
| seeded world bodies | one `p0` body at `(44, 44)`, `radius=0.6`, `num=0` |
| p0 | `(45.6, 44)`, `angle_rad=Math.PI`, `printing=false`, forced `trail.points=[]` |
| p1 | `(42.4, 44)`, `angle_rad=0`, `printing=false`, forced `trail.points=[]` |
| step | `step_ms=100`, moves `p0=0`, `p1=0` |

Both players move to `(44, 44)`. `p1` updates first and dies to the seeded `p0`
body. `p0` updates later and should die to `p1`'s just-emitted death point, not
to its own seeded body.

Expected events:

1. `position p1` at `(44, 44)`.
2. `point p1` at `(44, 44)`, `important=false`.
3. `die p1`, `killer=p0`, `old=false`.
4. `score:round p1` with delta `0`.
5. `position p0` at `(44, 44)`.
6. `point p0` at `(44, 44)`, `important=false`.
7. `die p0`, `killer=p1`, `old=false`.
8. `score:round p0` with delta `0`.
9. `score p1`, then `score p0`.
10. `round:end`, `winner=null`.

Expected final state:

| Player | alive | death killer | score |
| --- | --- | --- | ---: |
| p0 | `false` | `p1` | `0` |
| p1 | `false` | `p0` | `0` |

Expected game/body state: `deathCount=2`, deaths `[p1, p0]`,
`roundWinner=null`, `worldBodyCount=3` (`1` seeded body plus `2` death points).

Observed JS facts on 2026-05-09:

- `game.size=88`, `inRound=false`, `deathCount=2`, deaths `[2, 1]`,
  `roundWinner=null`, `worldBodyCount=3`.
- Event order is position/point/die/score:round for `p1`, then the same for
  `p0`, then `score p1`, `score p0`, and `round:end winner=null`.
- Both death events have `old=false`; `p1`'s killer is `p0`, and `p0`'s killer
  is `p1`.
- Both players finish at `(44, 44)`, dead, with `score=0` and `roundScore=0`.

Observed JS/Python common-trace facts on 2026-05-09:

- `source-body-canary` emits the same event order, including `die p1
  killer=p0 old=false` before `die p0 killer=p1 old=false`.
- Common trace includes `worldBodyCount=3`, player body counters
  `p0 bodyNum=1 bodyCount=2` and `p1 bodyNum=0 bodyCount=1`, and both final
  `trailPointCount=1`.
- Scores and `roundScore` stay `0`; `round:end` has `winner_id=null`.

Traps:

- Keep `p1.printing=false`; otherwise a normal point can be emitted before
  `p1` dies.
- Keep the seeded body owned by `p0` with `num=0`; if `p0`'s current body number
  is forced too far ahead, `p0` can die to its own body.
- Do not advance controlled time past `2000` ms; this spec expects `old=false`.
- Both deaths must score `0`; `p0` getting `1` means the implementation used the
  updated death count instead of the frame-start count.

## Second Fixture

Candidate id: `source_collision_head_head_reverse_order_single_death_step`

JS pin status: fixture added at
`scenarios/environment/source_collision_head_head_reverse_order_single_death_step.json`.
The command
`node tools/reference_oracle/scenario_runner.js scenarios/environment/source_collision_head_head_reverse_order_single_death_step.json`
was run locally on 2026-05-09 and matched the expected event and final-state
facts stored in the fixture. Raw oracle avatar ids are `2` for `p1` and `1` for
`p0`.

Python/common-trace status: promoted through the narrow `source-body-canary`
runner path and `scenarios/environment/source_collision_order_batch.json`. The
focused loop
`uv run python tools/run_fidelity_loop.py scenarios/environment/source_collision_head_head_reverse_order_single_death_step.json --python-runner source-body-canary --fail-on-mismatch`
returns `diff_status=pass`.

Why backup: it is the cleanest head-head/reverse-order probe after the
death-point side-effect fixture. It proves symmetric same-endpoint movement is
order-asymmetric in source because only emitted point bodies collide; live heads
do not collide by themselves.

Setup:

| Field | Value |
| --- | --- |
| player_count | `2` |
| map_size | `88` |
| game | `started=true`, `in_round=true`, `borderless=false`, `world_active=true` |
| seeded world bodies | none |
| p0 | `(42.4, 44)`, `angle_rad=0`, `printing=true`, forced `trail.points=[]`, print manager inactive |
| p1 | `(45.6, 44)`, `angle_rad=Math.PI`, `printing=true`, forced `trail.points=[]`, print manager inactive |
| step | `step_ms=100`, moves `p0=0`, `p1=0` |

Both players move to `(44, 44)`. `p1` updates first, emits a normal point, and
survives its own fresh body due own-trail latency. `p0` updates second and dies
to `p1`'s point.

Expected events:

1. `position p1` at `(44, 44)`.
2. `point p1` at `(44, 44)`, `important=false`.
3. `position p0` at `(44, 44)`.
4. `point p0` at `(44, 44)`, `important=false`.
5. `point p0` at `(44, 44)`, `important=false`.
6. `die p0`, `killer=p1`, `old=false`.
7. `score:round p0` with delta `0`.
8. `score:round p1` with survivor bonus `1`.
9. `score p1`, then `score p0`.
10. `round:end`, `winner=p1`.

Expected final state:

| Player | alive | death killer | score |
| --- | --- | --- | ---: |
| p0 | `false` | `p1` | `0` |
| p1 | `true` | none | `1` |

Expected game/body state: `deathCount=1`, deaths `[p0]`, `roundWinner=p1`,
`worldBodyCount=3` (`p1` normal point, `p0` normal point, `p0` death point).

Observed JS facts on 2026-05-09:

- `game.size=88`, `inRound=false`, `deathCount=1`, raw deaths `[1]`,
  raw `roundWinner=2`, `worldBodyCount=3`.
- Event order is `position/point` for `p1`, then `position/point/point/die` for
  `p0`, then `score:round p0`, `score:round p1`, `score p1`, `score p0`, and
  `round:end winner=p1`.
- `p0` finishes dead at `(44, 44)` with `trailPointCount=2`,
  `bodyNum=0`, `bodyCount=2`, `score=0`, and `roundScore=0`.
- `p1` finishes alive at `(44, 44)` with `trailPointCount=1`,
  `bodyNum=0`, `bodyCount=1`, `score=1`, and `roundScore=0`.
- `p0`'s death event has `killer=p1` and `old=false`.

Observed JS/Python common-trace facts on 2026-05-09:

- `source-body-canary` emits the same event order, including `p1`'s normal point
  before `p0` moves, `p0`'s normal point, `p0`'s death point, and then `die p0
  killer=p1 old=false`.
- Common trace includes `worldBodyCount=3`, player body counters
  `p0 bodyNum=0 bodyCount=2` and `p1 bodyNum=0 bodyCount=1`, and final
  `trailPointCount` values `p0=2`, `p1=1`.
- Final scores are `p0=0`, `p1=1`; `round:end` has `winner_id=p1`.

Traps:

- Keep forced trail runtime state; setup-time `printManager.start()` would add
  contaminating point bodies.
- If `p1.printing=false`, this becomes a live-head control and both players
  survive.
- Collision lookup must continue after `p0`'s own fresh body fails own-latency
  matching; the killing body is `p1`'s earlier point.
- This does not prove mutual head-head death. The expected source behavior is
  intentionally order-asymmetric.
- Do not read final printing cleanup from this fixture. Both print managers are
  forced inactive to isolate normal point insertion from `PrintManager.stop()`
  side effects.

## Parked

The first two PrintManager death-stop gaps are now covered by
`source_print_manager_active_stop_on_death_step` and
`source_print_manager_active_hole_stop_on_death_step`. The remaining nearby
variant is body-collision death; keep it separate from the existing
collision-order batch unless it proves a new update-order rule.
