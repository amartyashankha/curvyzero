# Body And Trail Investigation Note

Status: updated after print-manager death-stop Python/common-diff verification, 2026-05-09

Scope: plan the next local source-fidelity slice after movement, normal-wall,
borderless, and narrow multiplayer scoring/death-order checks. This note records
that the seeded opponent-body, own-body latency, and same-frame point
materialization canaries now pass through the narrow `source-body-canary`
runner. Deterministic print-manager toggle basics and active printing
stop-on-death now pass through the narrow `source-print-manager-canary` runner.

## Source Map Links

- [CurvyTron source map index](../../research/curvytron_source_map/README.md)
- [Compact facts index](../../research/curvytron_source_map/facts_index.md)
- [Collisions, trails, and world source map](../../research/curvytron_source_map/collisions_trails_world.md)
- [Source map open questions](../../research/curvytron_source_map/open_questions.md)
- [Next mismatch plan after kinematics](../../research/environment/next_mismatch_plan.md)
- [Environment fidelity handoff](../../handoffs/2026-05-08-environment-fidelity-handoff.md)

## Current Verified Checkpoint

Use active lanes and the mismatch plan for the terse checkpoint; this note keeps
the body/trail fixture detail. The last recorded local source-fidelity state is:

- Movement one-step fixtures match through the source-kinematics runner.
- Normal-wall death fixtures match through the source normal-wall path.
- The mixed wall/border batch is recorded as passing through
  `source-border-rules`.
- The narrow 3P/4P normal-wall scoring, death-order, and terminal-draw slice is
  recorded as passing through `source-border-rules`.
- The source-body-canary batch now verifies opponent tangent-safe and opponent
  overlap-kill cases:
  `source_body_opponent_tangent_safe_step.json` and
  `source_body_opponent_overlap_kills_step.json`.
- The source-body-canary batch also verifies own stored-body latency with
  `source_body_own_delta3_safe_step.json` and
  `source_body_own_delta4_kills_step.json`: both seed a p0 body num `0` at
  `(20,20)`, force live p0 to `(20,20)` at `step_ms: 0` with printing off, and
  use a 3P map-95 setup to avoid terminal ambiguity. Delta `3` is safe with
  `worldBodyCount: 1`; delta `4` kills p0 with killer p0, `old: false`, and
  `worldBodyCount: 2`.
- The source-body-canary batch now also verifies same-frame point
  materialization with `source_body_same_frame_point_kills_step.json` and
  `source_body_same_frame_point_control_safe_step.json`: the positive case
  emits a p1 point before p0's collision check and kills p0 with
  `killer_id: p1`; the control case keeps p1 non-printing and leaves all
  players alive with `worldBodyCount: 0`.
- `CurvyTronSourceEnv` now stores source-shaped `SourceBodyState`,
  `SourceIslandState`, and `SourceWorldState`. It inserts a body when source
  point emission occurs while the game has started and the world is active, and
  `worldBodyCount` comes from the mutable world body count.
- `tests/test_source_env.py` has focused direct source-env tests for opponent
  strict overlap versus tangent safety, own-trail latency delta `3` safe and
  delta `4` kill, old metadata at 2000 ms, wall priority, same-frame
  reverse-order point insertion killing a lower-index avatar, and the active
  already-hole PrintManager stop property event. It also has direct borderless
  branch checks: wrap skips destination-body lookup until the next frame,
  exact-edge/corner first-axis behavior, and borderless PrintManager
  wrap/toggle behavior. The file is now at 25 focused direct source-env tests
  after scalar checks for 3P ordered death scoring, 3P same-frame wall-death
  scoring, a 3P absent-player scoring corner, tie-at-max next-round
  behavior, timer-drain large advance across
  `game:stop -> round:new -> game:start`, an infinite-loop guard for zero-delay
  timer loops, normal trail exact-threshold and epsilon behavior,
  non-present delayed PrintManager starts, and 1P wall-death scoring.
  `advance_timers` now drains every due timer up to the target time, including
  newly scheduled due timers.
  Focused checks already passed: `uv run pytest tests/test_source_env.py -q`
  and `uv run ruff check src/curvyzero/env/source_env.py tests/test_source_env.py`.
  Integrated focused status now pairs those 29 direct source-env pytest cases with the
  18-test vector runtime API shell: the combined focused command over
  `tests/test_source_env.py`, `tests/test_vector_runtime.py`,
  `tests/test_source_lifecycle_runner.py`, `tests/test_lifecycle_oracle.py`,
  and `tests/test_env_reference_defaults.py` is the current focused command. Ruff
  and the doc guard also passed.
- The source-print-manager batch verifies print-to-hole, hole-to-print, active
  no-toggle control, active printing stop-on-death, and active already-hole
  stop-on-death through
  `source-print-manager-canary`. It checks forced active manager state, distance
  subtraction, `<= 0` toggles, important point side effects, property payloads,
  new deterministic distances `5.25` and `39`, active stop order before `die`,
  the already-hole no-important-stop-point branch, and final trail/body
  counters.
- Common-trace diff is the default local comparison path; raw diff is only a
  debug path.
- The verified wall/border and multiplayer slices plus direct source-env body
  tests still do not cover broad multiplayer body rollouts, broader trail
  cadence/gaps, bonuses, browser messages, or replay payloads.
- The Python-verified body slice is narrow: it covers seeded opponent stored-body
  strict overlap, exact-tangent safety, and own stored-body latency at the
  `> 3` point-number gate, plus the two direct same-frame point materialization
  fixtures only. Print-manager toggle basics are verified separately, but
  broader trail storage/cadence/gaps, bonuses, browser messages, and replay
  payloads remain pending.

The tangent-safe fixture intentionally seeds the opponent body at
`21.200000000000003` rather than literal `21.2`: in Node, `21.2 - 20` rounds
slightly below the strict `1.2` radius-sum threshold, so a decimal that is just
outside the threshold is needed to prove equal-distance safety instead of
accidentally proving overlap death.

## Why Body/Trail Is Next

Body and trail behavior is the next high-leverage local mechanic because it sits
directly after the already verified border branch in `Game.update(step)`. The
source order is movement, border check, body collision, print-manager test, then
bonus catch. Movement and border behavior are now narrow but useful anchors; the
next mismatch should therefore come from stored bodies and trail printing, not
from broader browser, Modal, bonus, or training-interface work.

The body/trail slice also closes several source facts that affect later claims:
strict circle overlap, self-collision latency by trail point number, immediate
opponent-trail collision, death-frame point side effects, and the difference
between visual trail gaps and world collision bodies.

## Source-Read Findings

Scout source reads added the following constraints for the first body/trail
canaries:

- Endpoint collision: `Game.update(step)` checks body collisions after movement
  and border handling, using the avatar endpoint after the current step.
- Strict overlap: `Island.bodiesTouch` uses strict `<`, so exact tangent bodies
  are safe and only distances below the summed radii collide.
- Opponent bodies immediate: `AvatarBody.match` allows opponent bodies to kill
  without any point-number latency gate.
- Own latency: `AvatarBody.match` permits a stored own body only when
  `live_body.num - stored_body.num > trailLatency`; `trailLatency` is `3`, so
  stored num `0` is safe against live num `3` and kills against live num `4`.
- Point insertion is synchronous: `Avatar.update` can call `Avatar.addPoint`
  during the same update, and that new point inserts a collision body before
  the later body-collision phase.
- Print-manager order: `PrintManager.test` runs after body collision, so print
  hole toggles cannot prevent a body collision already checked in that frame.
- Same-frame materialization source order: `Game.update(step)` iterates avatars
  in reverse order, calls `avatar.update(step)`, then checks border/body
  collision. `Avatar.update` can call `Avatar.addPoint`, and `Game.onPoint`
  immediately inserts an `AvatarBody` into the world while `world.active` is
  true. A higher-index avatar can therefore print a point that kills a lower-
  index avatar later in the same `Game.update(step)`.

## Open Questions

- How should print-manager randomness be controlled before testing print holes:
  patched `Math.random`, an injected stream, or a deterministic source-runner
  hook?
- What is the smallest deterministic print-manager fixture that proves the
  print/hole toggle without depending on natural randomness?
- What is the smallest opponent-trail fixture that avoids head-head ambiguity and
  reverse-update-order ambiguity after print-manager behavior is pinned?
- Which trail-gap behavior belongs in common trace: visible trail points,
  collision body count, point events, or a smaller state projection?
- Which bonus, replay-message, or observation projections need state evidence
  after broader trails are named?

## JS Oracle Pinned: Same-Frame Point Materialization

The JS oracle now proves that same-frame point insertion is observable without
relying on browser rendering or replay payloads.

Use a 3-player map-95 setup with player array order `p0`, `p1`, `p2`, so source
update order is `p2`, `p1`, then `p0`. Keep all moves straight.

Positive fixture: `source_body_same_frame_point_kills_step`

- `step_ms: 100`.
- `p0` victim starts at `(40, 40)`, angle `0`, printing `false`.
- `p1` printer starts at `(40, 40)`, angle `0`, printing `true`.
- Forced `trail.points: []` makes setup assign `avatar.printing` directly and
  avoids a setup-time `printManager.start()` body.
- `p2` bystander starts at `(80, 20)`, angle `pi`, printing `false`.
- Verified event order: p2 position, p1 position, p1 point with
  `important: false`, p0 position, p0 death point, p0 die with
  `killer_id: p1` and `old: false`, then p0 `score:round`.
- Expected final state: p0 dead, p1/p2 alive, `worldBodyCount: 2`.

Control fixture: `source_body_same_frame_point_control_safe_step`

- Same geometry, but `p1` starts with printing `false`.
- Expected events: positions only.
- Expected final state: all alive and `worldBodyCount: 0`.
- Purpose: prove that overlapping the live p1 head is not enough; p0 dies only
  when p1's same-frame point materializes into the world.

JS oracle hooks pinned for this slice:

- `players[].initial.print_manager` supports direct `active`, `distance`,
  `last_x`, and `last_y` assignment.
- `players[].initial.trail.points` seeds or clears trail points directly.
  Optional `last_x` and `last_y` override the inferred last point.
- When explicit print-manager or trail state is present, setup assigns
  `avatar.printing` directly instead of calling `printManager.start()` or
  `.stop()`.

Python support is now verified through `source-body-canary`. The implementation
is intentionally direct for these fixtures only: it reads
`players[].initial.trail.points` and `players[].initial.printing`, emits a point
when `isTimeToDraw` is true, inserts that body into the mutable world body list
immediately, and then runs the existing wall/body collision checks.

The scalar source env has now moved beyond a fixture-only body stub. It keeps
source-shaped body/island/world state, carries `borderless` on reset/source game
state, uses source-shaped border detection, wraps on the first axis to the
opposite position, skips destination-body lookup on the wrap frame, inserts
source point bodies only when the game has started and the world is active, and
reports `worldBodyCount` from the world. It now also carries focused scalar
3P scoring and timer-drain guards, including the zero-delay loop guard. This is
still not full environment fidelity: multiplayer beyond the current narrow
proofs, broader present/alive leave, broader non-present continuation,
world/island boundaries, and moving source semantics into the fast runtime
remain open. The focused live movement event trace is promoted separately.

## TODO Checklist

- [x] Source-body canary:
  Add an opponent tangent-safe fixture proving exact tangent distance is safe
  under the source's strict `<` body-overlap check.
- [x] Source-body canary:
  Add an opponent overlap-kill fixture proving opponent stored bodies kill
  immediately when overlap is strict and no head-head ambiguity is present.
- [x] JS oracle:
  Add own-latency fixtures proving own stored body difference `3` is safe and
  difference `4` kills.
- [x] Canary:
  Implement Python-runner support for the own-latency fixtures proving
  own stored body difference `3` is safe and difference `4` kills.
- [x] JS oracle:
  Add a same-frame point-insert fixture proving point events insert collision
  bodies synchronously before the later body-collision phase in the same
  `Game.update(step)`.
- [x] JS oracle:
  Add forced print-manager and trail-state hooks needed for the same-frame point
  materialization fixtures.
- [x] Canary:
  Add Python runner support for `source_body_same_frame_point_kills_step` and
  `source_body_same_frame_point_control_safe_step`.
- [x] Common trace:
  Decide the smallest common-trace projection needed for body/trail comparison:
  alive state, positions, death events, killer identity, round-end state, and any
  body/trail counters that are needed to explain mismatches.
- [x] Diff:
  Run JS and Python through common-trace diff first. Record the first mismatch
  before changing Python behavior.
- [ ] Deterministic print-manager:
  Add the smallest forced or seeded print-manager fixture and matching Python
  source-runner behavior.
- [ ] Test:
  Add focused regression coverage for the selected print-manager fixture and
  rerun the existing verified wall/border, multiplayer, and body-canary batches.
- [ ] Broader trails:
  After print-manager behavior is pinned, add the next opponent-trail or trail-gap
  canary with source evidence.
- [ ] Later mechanics:
  Keep bonuses, replay/server messages, observations, and optimization behind
  the deterministic print-manager and broader-trail work.
