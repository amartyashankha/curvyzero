# Borderless Trail Probe Plan

Status: fixture 3 Python/common-trace verified, 2026-05-09.
`source_borderless_print_manager_wrap_toggle_step` is JS-oracle pinned and
Python/common-trace verified through `source-border-rules` as part of the
promoted `source_border_batch.json` border claim.
`source_borderless_wrap_skips_destination_body_then_next_frame_kills` is
JS-oracle pinned and Python/common-trace verified through `source-border-rules`.
`source_borderless_exact_edge_corner_axis_step` is also JS-oracle pinned and
Python/common-trace verified through `source-border-rules`.

## Priority

Borderless-with-trails was done before exact-edge/corner controls. Verified borderless
evidence now includes the plain `source_borderless_wrap_step`, the first
PrintManager/trail wrap fixture, and the destination-body skip on the wrap frame
followed by next-frame collision after teleport.
Exact edge/corner behavior is now covered by the first small control after the
trail/body branch.

Run this slice before new head-head/death-frame fixtures. It extends already
verified wall/body/PrintManager mechanics; head-head adds a separate collision
class and should not carry unresolved borderless assumptions.

Recommended order:

1. `source_borderless_print_manager_wrap_toggle_step` - implemented
2. `source_borderless_wrap_skips_destination_body_then_next_frame_kills` -
   implemented
3. `source_borderless_exact_edge_corner_axis_step` - implemented

## Files Read

- `third_party/curvytron-reference/src/server/model/Game.js:37-80`
- `third_party/curvytron-reference/src/server/model/Game.js:113-118`
- `third_party/curvytron-reference/src/server/model/Avatar.js:23-33`
- `third_party/curvytron-reference/src/server/core/World.js:276-324`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js:90-103`
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:292-310`
- `third_party/curvytron-reference/src/server/core/World.js:46-63`
- `third_party/curvytron-reference/src/server/core/World.js:97-126`
- `third_party/curvytron-reference/src/server/core/Island.js:83-90`

Source facts used: movement and normal point printing run before border check;
borderless uses margin `0`; strict edge equality is safe; body lookup is only in
the no-border `else` branch; `PrintManager.test()` runs after a survivor wraps;
print toggles emit an important point before a false toggle clears visual trail;
strict edge equality is safe, and corner exits resolve the first border axis
found.

## Fixture 1: PrintManager Wrap Toggle

Name: `source_borderless_print_manager_wrap_toggle_step`

Setup:

- One-player map `80`; `borderless: true`, `started: true`,
  `in_round: true`, `world_active: true`, no bonuses.
- One tick, `step_ms: 20`, move `0`.
- `p0` starts `(79.8, 40)`, angle `0`, `printing: true`,
  `body_count: 0`, `body_num: 0`.
- Force `trail: {points: [], last_x: 79.4, last_y: 40}`.
- Force `print_manager: {active: true, distance: 1, last_x: 79.8,
  last_y: 40}`.

Expected wrap/collision:

- Movement reaches `(80.12, 40)`.
- Borderless wraps to `(0, 40)`.
- p0 remains alive; no body collision is checked on the wrap branch.

Expected point/body/PrintManager state:

- Events: pre-wrap `position(80.12, 40)`, normal
  `point(80.12, 40, important=false)`, wrapped `position(0, 40)`,
  important `point(0, 40, important=true)`, then
  `property(printing=false)`.
- `worldBodyCount: 2`.
- Final p0: `printing: false`, `trailPointCount: 0`,
  `lastTrailPoint: null`, `bodyCount: 2`, `bodyNum: 1`.
- Final PrintManager: `active: true`, `distance: 5.25`, `lastX: 0`,
  `lastY: 40`.

Traps and non-claims:

- This proves post-wrap PrintManager distance, not destination-body collision.
- Do not claim client trail rendering or absence of a visible cross-arena line.
- Do not infer torus behavior; the pre-wrap point and wrapped head are separate
  source positions.

Implementation status:

- Added fixture in `scenarios/environment/source_borderless_print_manager_wrap_toggle_step.json`.
- Promoted it into `scenarios/environment/source_border_batch.json`.
- Verified JS raw oracle events/state and Python common trace parity.
- The destination-body-skip fixture is now promoted below.

## Fixture 2: Wrap Skips Destination Body

Name: `source_borderless_wrap_skips_destination_body_then_next_frame_kills`

Setup:

- Three-player map `95`; `borderless: true`, `started: true`,
  `in_round: true`, `world_active: true`, no bonuses.
- Two ticks: tick 0 `step_ms: 20`, tick 1 `step_ms: 0`; all moves `0`.
- `p0` starts `(94.8, 44)`, angle `0`, `printing: false`,
  `body_count: 0`, `body_num: 0`.
- `p1` starts `(20, 20)`, angle `0`, `printing: false`.
- `p2` starts `(80, 20)`, angle `pi`, `printing: false`.
- Seed one p1 world body at `(0, 44)`, radius `0.6`, num `0`.

Expected wrap/collision:

- Tick 0: p0 reaches `(95.12, 44)`, wraps to `(0, 44)`, overlaps the seeded p1
  body, and stays alive because the border branch skips body lookup.
- Tick 1: p0 remains at `(0, 44)`; exact edge equality is safe, no border is
  hit, body lookup runs, and p0 dies to p1.

Expected point/body/PrintManager state:

- Initial `worldBodyCount: 1`.
- After tick 0: still `1`, no point event from p0.
- After tick 1: p0 death emits `point(0, 44, important=false)`;
  `worldBodyCount: 2`.
- Final p0: `alive: false`, `trailPointCount: 1`,
  `lastTrailPoint: [0, 44]`, `bodyCount: 1`, `bodyNum: 0`.
- Print managers stay inactive.

Traps and non-claims:

- This proves skipped body lookup on the wrap frame and normal lookup next
  frame.
- It does not prove torus collision across edges; the seeded body is at the
  actual destination.
- It does not prove head-head, same-frame death order, or old-body aging.

Implementation status:

- Added fixture in
  `scenarios/environment/source_borderless_wrap_skips_destination_body_then_next_frame_kills.json`.
- Promoted it into `scenarios/environment/source_border_batch.json`.
- JS oracle command:
  `node tools/reference_oracle/scenario_runner.js scenarios/environment/source_borderless_wrap_skips_destination_body_then_next_frame_kills.json`.
- Python/common-trace batch command:
  `uv run python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-border-regression`.
- Observed JS facts on 2026-05-09: the draft spec was correct.
- Tick 0 source events are `position(p2, 79.68, 20)`,
  `position(p1, 20.32, 20)`, `position(p0, 95.12, 44)`, then
  `position(p0, 0, 44)`. `deathCount: 0`, `worldBodyCount: 1`, and p0 is
  alive at `(0, 44)` with `trailPointCount: 0`, `bodyCount: 0`, `bodyNum: 0`.
- Tick 1 source events are `position(p2, 79.68, 20)`,
  `position(p1, 20.32, 20)`, `position(p0, 0, 44)`,
  `point(p0, 0, 44, important=false)`, `die(p0, killer=p1, old=false)`, and
  `score:round(p0, score=0, roundScore=0)`. Final state is `deathCount: 1`,
  `deaths: [p0]`, `worldBodyCount: 2`; p0 is dead at `(0, 44)` with
  `trailPointCount: 1`, `lastTrailPoint: [0, 44]`, `bodyCount: 1`,
  `bodyNum: 0`.
- Python/common-trace parity is verified. It matches the border branch
  body-lookup skip, strict-edge-safe next frame, death point insertion, killer
  `p1`, `old=false`, score-round side effect, and `worldBodyCount` transition
  `1 -> 2`. Print managers remain inactive throughout.

## Fixture 3: Exact Edge And Corner Control

Name: `source_borderless_exact_edge_corner_axis_step`

Setup:

- Two-player map `88`; `borderless: true`, `started: true`,
  `in_round: true`, `world_active: true`, no bonuses.
- One tick, `step_ms: 100`, all moves `0`.
- `p0` starts `(87.35, 87.35)`, angle `pi / 4`, `printing: false`,
  `body_count: 0`, `body_num: 0`.
- `p1` starts `(86.4, 20)`, angle `0`, `printing: false`,
  `body_count: 0`, `body_num: 0`.

Expected wrap/collision:

- p1 moves to exactly `(88, 20)` and does not wrap; strict `>` makes equality
  safe.
- p0 moves to about `(88.481371, 88.481371)`. The source detects the right edge
  first and wraps to `(0, 88.481371)`. The y-axis remains outside until a later
  border check.
- Both players remain alive.

Expected point/body/PrintManager state:

- No point events.
- `worldBodyCount: 0`.
- Both trails remain empty and print managers inactive.

Traps and non-claims:

- This is not a torus claim. It proves one-axis corner wrapping.
- It does not prove next-frame second-axis wrap.
- It does not cover trails or stored bodies; fixtures 1 and 2 own those.

Implementation status:

- Added fixture in
  `scenarios/environment/source_borderless_exact_edge_corner_axis_step.json`.
- Promoted it into `scenarios/environment/source_border_batch.json`.
- JS oracle command:
  `node tools/reference_oracle/scenario_runner.js scenarios/environment/source_borderless_exact_edge_corner_axis_step.json`.
- Python/common-trace single command:
  `uv run python tools/run_fidelity_loop.py scenarios/environment/source_borderless_exact_edge_corner_axis_step.json --python-runner source-borderless-wrap --artifact-root /private/tmp/curvy-borderless-exact-edge-corner`.
- Batch command:
  `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-border-regression`.
- Observed JS facts on 2026-05-09: p1 lands at exactly `(88, 20)` and does not
  wrap; p0 moves to `(88.481371, 88.481371)` then wraps only x to
  `(0, 88.481371)`; both players stay alive, `worldBodyCount` remains `0`, and
  the only events are the p1 position, p0 pre-wrap position, and p0 wrapped
  position.
- Python/common-trace parity is verified with the existing borderless runner.
