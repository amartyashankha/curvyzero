# CurvyTron Source Map Open Questions

Status: working list

These questions should be closed with source reads, headless JS probes, or Python
trace diffs. Do not close them from memory.

For the current promoted narrow source slices, see
[coverage_tracker.md](../../working/environment/coverage_tracker.md). For the
remaining no-bonus multiplayer fixture gaps, see
[no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md](../../working/environment/no_bonus_multiplayer_source_fixture_gaps_2026-05-11.md).

## Highest Priority

| Question | Why it matters | How to close it | Status |
| --- | --- | --- | --- |
| Should the source-fidelity runner use fixed `1000 / 60` ms steps, recorded elapsed-ms steps, or support both? | Time handling changes movement, collisions, trails, and scoring. | Keep the promoted movement batch green after runner edits. | Closed for core movement: fixed-step and varied elapsed-ms movement are promoted in `source_kinematics_batch.json`. Reopen only for a new mechanic that uses timers differently. |
| What are the exact normal-wall death trace fields and score events? | Wall death is the next basic collision proof. | Keep the promoted border and multiplayer scoring batches green after runner edits. | Closed for current wall/scoring claims: `source_border_batch.json` and `source_normal_wall_multiplayer_batch.json` are promoted. Broader lifecycle scoring remains separate. |
| What are the exact borderless-wrap trace fields? | Source supports borderless wrap, and docs must not call all borders fixed walls. | Keep the promoted border batch green after borderless edits. | Closed for current borderless claims: plain wrap, PrintManager wrap, destination-body skip, and exact-edge/corner-axis behavior are promoted. Next-frame second-axis behavior is only needed if a later feature depends on it. |
| How should random behavior be controlled? | Spawn positions, trail holes, and bonus spawns depend on randomness. | Use row-local `Math.random` tape/cursor/logs and compare each named source claim before optimized parity. | Partial: PrintManager random call order/cadence and 28 pinned lifecycle fixtures through focused 4P survivor next-round, focused 3P/4P present/absent survivor scoring, next-round, and tie-at-max facts, one focused 3P warmdown leave source proof, the focused 3P single-present active leave edge, and focused 3P/4P all-present multi-round match-end are promoted. Focused public metadata is green for 4P present/absent reset/survivor/next-round, 4P all-present multi-round, 3P/4P present/absent tie-at-max, 3P staged match-mode warmdown leave, and `source_lifecycle_remove_avatar_to_single_present_3p.json`. `source_bonus_spawn_type_position_rng_step.json` adds the one-type bonus spawn/type proof, `source_bonus_default_weights_type_rng_step.json` adds the default multi-type weight/type proof, and `source_bonus_spawn_game_world_retry_step.json` plus `source_bonus_spawn_bonus_world_retry_step.json` add narrow natural bonus retry proofs. The vector seeder rejects lifecycle fixtures honestly and records RNG metadata for future reset/spawn work; `vector_spawn.py` covers only promoted first-round spawn facts. Still open for broader present/alive leave cases, broad public warmdown leave beyond the focused 3P staged metadata proof, broad bonus probability/runtime support, and production reset/replay seed history. |
| Which current Python differences must close before the first serious learning run? | `curvyzero-v0` is useful, but it is not source-faithful. | Choose a minimum fidelity bar for training claims. | Open. |

## World And Borders

- What broader spawn positions, heading retries, present masks, and reset
  streams are still needed beyond the pinned 2P heading retry and focused 3P/4P
  spawn-order/lifecycle fixtures?
- What broader retry controls, if any, are needed beyond the pinned 2P heading
  rejection fixture?
- Does the source ever place initial trail bodies at spawn before printing starts,
  or only after print manager activation and movement?
- What should the Python representation of island buckets be: exact source-style
  buckets, or a separate deterministic backend with matching outcomes?
- How should borderless wrap interact with trail bodies near both edges?

## Movement And Controls

- Should the scenario schema model source input as left/right buttons or as the
  resolved move value `-1`, `0`, `1`?
- What tolerance is acceptable for position and angle when JS and Python use the
  same elapsed-ms formula?
- Should straight-angle behavior live in movement docs, bonus docs, or both?
- Are there source cases where move changes happen mid-frame, or can every trace
  stage all moves before `game.update(step_ms)`?

## Trails

- Which broader timer/lifecycle traces are still useful beyond the pinned
  3000 ms post-`game:start` PrintManager start fixtures?
- When printing turns off for a hole, which bodies remain collidable and which
  trail points are only visual state?
- How should a trace show the difference between a visual gap and existing world
  collision bodies?
- What small scenario proves that distance-based holes match the source?

## Collisions

- How should head-head collision be documented when update order can make it
  order-sensitive?
- What is the smallest same-frame double-death probe that avoids ambiguity?
- What 3-player case best proves frame-start death-count scoring?
- Should Python reproduce endpoint-only collision exactly, including tunneling,
  for `curvytron-v1-reference`?
- How should collision events name the killer for wall, self-trail, other-trail,
  and unclear same-frame cases?

## Scoring And Rounds

- What are the exact emitted events when a round ends in the same frame as one or
  more deaths?
- How should traces represent `roundScore` before and after `resolveScores()`?
- What remaining 3/4-player scenarios best prove broader winner score, tied
  leaders, and max-score continuation beyond the named narrow fixtures?
- Does a one-player game need a dedicated score/winner golden?
- Which round lifecycle fields belong in common traces and which stay runner-only?

## Bonuses

- Which bonus should be implemented first as a proof: speed, radius, inverse,
  borderless, clear, or printing stop? Current answer: active `BonusSelfSmall`
  catch/no-catch/death-order, one expiry/restore case, one forced
  `BonusGameClear` immediate clear case, and narrow natural `BonusGameClear`
  type-selection edges are pinned. Speed, inverse, borderless, printing stop,
  color, and broad `BonusGameClear` probability/effect coverage are still open.
- How should bonus timers be represented in deterministic traces? Current
  partial answer: `source_bonus_self_small_expiry_restore_step.json` advances
  the real source timeout by `7500` ms and proves radius restore before the
  next zero-elapsed movement events. Same-frame expiry with other timers and
  multi-stack expiry are still open.
- What happens when multiple stack effects expire on the same frame?
- Should unreachable `BonusSelfGodzilla` be ignored forever, or kept as a hidden
  source note only?
- What broader scenario proves bonus spawn collision checks against both trail
  world and bonus world? Current partial answer:
  `source_bonus_spawn_game_world_retry_step` and
  `source_bonus_spawn_bonus_world_retry_step` prove one main game-world retry
  and one bonus-world retry. Public bonus env, replay, and broad natural spawn
  behavior are still open.

## Networking

- Which server messages are needed for fidelity after state traces pass?
- Does compression need exact JS bitwise truncation in Python if we compare wire
  messages?
- Are per-player message views different enough to need player-perspective trace
  fixtures?
- Which network/controller files can be skipped by the headless oracle forever?

## Rendering

- What canvas scale and camera behavior should a later renderer match?
- Which visual states matter for gameplay review: trail gaps, deaths, bonuses,
  colors, score, or all of them?
- When should pixel or video checks start? Current answer: after state and event
  traces are boring.

## Config And Build

- Can the old app build in a disposable local copy with pinned old dependencies?
- Is a pinned Modal image a better place to test the old Gulp/Bower build?
- Which generated files are missing from this checkout, and should any be
  committed, ignored, or rebuilt only in artifacts?
- Should the local reference source remain an ignored nested clone, become a
  submodule, or become a vendored snapshot?
