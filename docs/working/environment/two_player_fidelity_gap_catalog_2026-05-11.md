# Two-Player Fidelity Gap Catalog - 2026-05-11

Scope: 2P CurvyTron environment fidelity only. Ignore 3P/4P as proof, except where this doc names a gap caused by relying on non-2P evidence.

This is a remaining-gap catalog, not a status victory lap. A hole is closed only when a 2P source/original fixture or probe and a CurvyZero public/runtime test both exist.

## Already Covered For 2P

- Product runtime direction is `VectorMultiplayerEnv`. `VectorTrainerEnv1v1NoBonus` remains a strict proof/profile surface, not the product gameplay environment.
- Source/original rules have been reconstructed from the server JS around `Game`, `BaseGame`, `Avatar`, `BaseAvatar`, `PrintManager`, `BonusManager`, and bonus stacks.
- 2P source lifecycle fixtures exist for warmup/PrintManager start, next round after draw, survivor movement/death during warmdown before the next round, spawn heading rejection retry, max-score match end, active mid-round `removeAvatar`, and long 1v1 no-bonus wall terminal.
- Public/runtime no-bonus 2P checks cover long reset-to-terminal wall rollout, terminal metadata-only final rows, autoreset preservation of previous final metadata, draw warmdown into next round, survivor movement/death during warmdown with no rescore before next round, unique max-score match-end metadata, and active leave immediate survivor scoring.
- 2P collision-order canaries cover death-point ordering and same-frame head/head reverse-order behavior.
- Focused 2P seeded and natural bonus paths now cover all source-default effect families: self small/slow/fast/master, enemy slow/fast/big/inverse/straight-angle, game borderless, game clear, and all color. SelfMaster body/wall behavior and AllColor overlap have focused tests.
- Speed-changing bonuses now update turn rate with the source formula and
  restore it on expiry. Focused runtime/public tests cover self/enemy slow/fast
  catch and expiry.
- Natural bonus generation has seed/generated tape extension, fixture strictness, generated-row position retry, and no artificial callback cap in the public environment path.
- Current multiplayer replay packaging can carry metadata-only public rows and seeded bonus audit metadata. It explicitly is not a full trainer replay or full replay array surface.
- Source-state raw 64x64 visual observations exist and are backed by vector
  source state. The original 2P source arena is 88 units from
  `CurvyTronReferenceDefaults.arena_size_for_players(2)`; 64x64 is the learned
  observation raster, not the game size. These observations explicitly are not
  original browser/canvas pixel parity.
- `scripts/compare_2p_raw_visual_observation.py` now compares a 2P
  source-env snapshot raster against the `VectorMultiplayerEnv` raw gray64
  raster. On 2026-05-11, the core 2P source-state gray64 suite passed across
  34 scenarios: long wall terminal, movement traces, normal wall/draw cases,
  collision-order cases, borderless wrap cases, `BonusSelfSmall`
  catch/no-catch/expiry/wall-death cases, `BonusGameClear`, and
  `BonusGameBorderless`, plus the four natural bonus spawn/retry/cap fixtures
  and eight programmatic source-snapshot stress cases:
  printing trail-point emission, explicit 2P survivor warmdown-frame movement
  and death, opponent tangent/overlap body collision, own-body latency delta
  3/4,
  PrintManager trail-gap boundary emission,
  `BonusSelfMaster` blocks a body collision before later wall death,
  `BonusGameBorderless` expires before later wall death, and `BonusGameClear`
  clears a body before a later collision probe.
  The long no-bonus wall scenario matched exactly for 112 frames through
  terminal (`max_abs_diff=0`, `mismatch_pixels=0`), and the suite-level result
  is exact (`max_abs_diff=0`, `mismatch_pixels=0`). This is the current raw
  visual gate.
- Fixture accounting: 26 total 2P step fixtures exist. The `core2p` visual
  suite covers 25 of those step fixtures plus the long no-bonus wall rollout,
  plus 8 programmatic source-snapshot stress cases, for 34 visual scenarios
  total. `source_print_manager_random_call_order_step` is intentionally outside
  gray64 because it proves RNG/event order, not a distinct rendered state.
- The visual harness now has two intentional mismatch canaries:
  one removes a visible world body from the vector state, and one removes a
  visible map bonus. Both are expected to fail and prove the harness really
  catches visible missing geometry.
- The current trainer-facing visual path is source-state browser-like RGB64
  reduced to grayscale `64x64`. The separate bonus64 v1 gate now checks active
  map bonus mask/type planes for all 12 source-default bonus types and
  post-catch self/other/game status planes against source-derived facts. It is
  a diagnostic/proof surface, not the product trainer default. It does not
  encode `BonusAllColor` color rotation or a post-catch `BonusGameClear` status
  plane; those remain explicit limits rather than hidden claims.
- Runtime/body coverage now has true 2P source-pinned checks for opponent
  tangent safe, opponent overlap death, own-body delta3 safe, own-body delta4
  death, trail-gap hole safe, stored body death, print-to-hole boundary death,
  and hole-to-print boundary death. The visual harness also has matching 2P
  geometry/consequence probes for the most important body/trail cases.

## Next-Work Checklist

The current source-state visual gate passes: 34/34 `core2p` canvas-gray64
scenarios match exactly. The PrintManager RNG canary is intentionally not a
visual case; it proves random/event ordering, not a distinct rendered state.
The one-line visual command is now:

```bash
uv run python scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain
```

Latest result:
`PASS full_2p_source_state_visual_gate canvas_gray64=34/34 typed_bonus=12/12 canaries=2/2 mismatch_pixels=0 max_abs_diff=0.0 expected_canary_mismatch_pixels=81`.
This is a canvas-gray64 visual proof plus a diagnostic bonus64 proof, not a
full trainer/replay proof.

1. Add source/original fixture parity for the bonus stack/death stress cases
   that are currently programmatic source-env probes.
2. Promote final/replay bonus state beyond metadata-only audit rows.
3. Keep bonus64/rich tensors as diagnostic proof for hidden bonus facts; do not
   make them the trainer default.
4. Keep browser/canvas pixel checks as later debug evidence, not the current
   blocker.

## Remaining Holes

### Closed - Typed Bonus Visual And Status Sufficiency

Status: closed for the narrow 2P source-default bonus identity/status gate.

Source-state raw 64x64 comparison is now real and the current 34/34
`core2p` gray64 gate passes exactly. The only intentionally excluded 2P step
fixture is `source_print_manager_random_call_order_step`, because gray64 does
not encode PrintManager random-call order or event order.

`run_typed_bonus_visual_status_gate()` now checks the bonus64 v1 tensor without
replacing gray64. It covers all 12 source-default bonus types:

- active map bonus mask/type planes before catch
- post-catch self/other/game status planes after applying the source effect
- source-derived values compared to `VectorMultiplayerEnv` bonus64 v1 output

Explicit limits:

- `BonusAllColor` post-catch color rotation is not encoded in bonus64 v1 status
  planes.
- `BonusGameClear` has no typed post-catch status plane. Its clear geometry is
  still covered by gray64/runtime gates.
- Bonus64/rich tensor propagation as a trainer observation is not the product
  path.

How to test against source/original:

- Use existing JS/source-state fixtures and `CurvyTronSourceEnv` snapshots as
  the source truth for training-observation fidelity.
- Render the source snapshot to gray64 and compare it to `VectorMultiplayerEnv`
  gray64 for the same state and scripted actions.
- Keep browser/canvas pixels as optional later human/debug evidence only. They
  are not a blocker for the current source-state training observation.
- Expand diagnostic visual/status gates before claiming hidden natural-bonus
  source-fact coverage.

Measurement policy:

- Use exact checks for short source-state probes: same input state, same tick,
  same dimensions, same semantic body/head/trail/final-frame fields.
- If browser/canvas checks are added later, use tolerant checks. Small
  antialiasing, device-scale, or canvas backend noise should be measured, not
  treated as automatic failure.
- For long rollouts, expect tiny physics/rendering differences to compound.
  Report trajectory divergence instead of claiming frame-for-frame visual parity
  after the first divergent tick.
- When useful, compare from resync checkpoints: restart both renderers from the
  same trusted source state every fixed cadence, then measure local visual drift.
- Save artifacts for every failure: source state, browser frame, CurvyZero frame,
  diff image, metrics JSON, seed/random cursor, tick, and fixture id.

Future comparison metrics checklist:

- `max_abs_diff`
- `mismatch_pixels`
- tolerant pixel threshold and mask policy
- centroid/body/state drift
- first divergent tick
- resync cadence
- artifact saving paths

Current source-state gate:

- Command: `uv run python scripts/compare_2p_raw_visual_observation.py --suite core2p --format plain`
- Latest result: exact source-vs-vector gray64 match across 34 core 2P
  scenarios, including the long no-bonus wall rollout through terminal, the
  four natural bonus spawn/retry/cap fixtures, and the eight programmatic
  source-snapshot stress cases.
- Step-fixture coverage: 25 of 26 total 2P step fixtures are in `core2p`,
  alongside 8 programmatic source-snapshot stress cases. The remaining fixture
  is `source_print_manager_random_call_order_step`, which is verified by
  source/event-order tests rather than gray64.
- What this proves: the learned raw source-state raster can be regenerated from
  both source-shaped state and fast vector state for these covered fixtures.
- What this does not prove: original browser/canvas pixels, antialiasing,
  sprite colors, viewport scaling, or hidden bonus-status proof.
- Natural bonus spawn/retry/cap now uses a separate reset/tape path in the
  visual harness, because the ordinary forced-state seeding path consumes the
  wrong RNG.

Natural-bonus policy sufficiency:

- Gray64 v0 is the current clean visual observation path: browser-like/
  source-state canvas-like pixels reduced to grayscale `64x64`. It proves
  source-shaped and vector rasters agree for covered body/head/trail/bonus
  occupancy states.
- Gray64 v0 remains the geometry gate: 34/34 `core2p` gray64 scenarios, exact
  match, one intentionally excluded PrintManager RNG canary. Do not replace
  this trainer path with a rich tensor.
- Bonus64 v1 is the separate bonus-aware diagnostic gate, not a mutation of
  v0 gray64 and not a trainer default. It is `float32[22,64,64]`, CHW,
  source-state backed, normalized to `[0,1]`, and explicitly not
  browser/canvas pixels.

Likely code area:

- `src/curvyzero/env/vector_visual_observation.py`
- `scripts/compare_2p_raw_visual_observation.py`
- `tests/test_vector_visual_observation.py` or a new browser-pixel parity test file

### P1 - Trainer/Learned Observation And Final Observation Contract

Hole: `VectorMultiplayerEnv` still exposes debug metadata-only observations. The fixed-opponent source-state visual LightZero wrapper is not two-seat self-play, and source-state raw 64x64 is not browser pixel parity. Final observation semantics are proven for public metadata rows, not for a promoted trainer/visual observation and replay surface.

How to test against source/original:

- Add 2P observation manifests for wall terminal, collision-order terminal, body/trail gap cases after 2P fixtures exist, borderless wrap, bonus catch/expiry, and natural bonus terminal paths.
- Compare semantic observation fields to trusted source state before testing raster output.
- Assert terminal rows carry final observation before autoreset, with schema, player/ego mapping, mask/reward metadata, and replay identifiers.
- For visual final observations, compare source-state raw 64x64 before
  autoreset; browser/canvas pixels are only a later optional debug check.

Likely code area:

- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- Future two-seat visual trainer/replay adapter

### P1 - Bonus Replay And Final-State Facts

Hole: focused bonus behavior is supported, but replay remains metadata/audit only. Final rows do not yet carry a complete replayable bonus state: spawned bonus identity, catch/remove event, stack contents, expiry times, effective properties, random cursor/history, and terminal/final observation state.

How to test against source/original:

- Create 2P seeded and natural fixtures where bonus state matters on or near terminal frames:
  - SelfMaster prevents body death, then later wall death.
  - AllColor overlap selects the older stack until expiry.
  - Borderless wrap occurs before terminal wall behavior after expiry.
  - GameClear clears trails before a would-be body collision.
  - speed/radius/inverse/straight-angle effects affect movement or collision before terminal.
- Compare original JS/source-env events and state against `VectorMultiplayerEnv` final rows.
- Require replay records to include enough bonus state to reconstruct the final transition, or keep the `full_replay_arrays_claim=false` audit label explicit.

Likely code area:

- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/training/multiplayer_replay_v0.py`

### P1 - Bonus Timer, Stack, And Death Stress

Hole: bonus effect families are covered in focused paths, but broad 2P stress around simultaneous timers, PrintManager, death, warmdown, and stacking is still missing.

2026-05-11 public/runtime update: `tests/test_vector_multiplayer_env.py` now
adds two 2P seeded stress guards: `BonusGameClear` caught by the later-ordered
player clears a body before the earlier player collision pass, and
`BonusGameBorderless` expiry is applied before same-tick wall death/final-info
metadata. These reduce the public-env regression surface, but this P1 hole
stays open until matching source/original fixtures or probes exist.

2026-05-11 visual harness update: `core2p` now also includes eight
programmatic source-snapshot stress comparisons: printing trail-point emission,
explicit 2P survivor warmdown-frame movement/death,
opponent tangent/overlap body collision, own-body latency delta 3/4,
PrintManager trail-gap boundary emission,
`BonusSelfMaster` body-hit protection followed by wall death,
`BonusGameBorderless` expiry followed by wall death, and `BonusGameClear`
clearing a future collision body. These are source-env snapshot comparisons,
not new JS/original fixture files. They prove gray64 source-vs-vector
geometry/consequence parity for those states, but gray64 still hides the
important stack/status facts: PrintManager RNG/event order, invincibility,
borderless-active/expired status, and the clear event/bonus identity are not
visible channels.

How to test against source/original:

- Add 2P source/original scenarios for:
  - bonus expiry at the same timestamp as PrintManager start/stop
  - bonus catch and wall/body death in the same frame
  - death while boosted clears stack and later expiry callbacks become no-ops
  - inverse/double-inverse while turning
  - longer speed-bonus movement/collision consequences after the turn-rate fix
  - straight-angle applied while already turning
  - JS/original fixture parity for SelfMaster body-hit protection followed by
    wall death
  - JS/original fixture parity for borderless expiry followed by wall death
  - JS/original fixture parity for game clear before body collision
- Assert event order, stack order, remaining timers, effective avatar/game properties, death order, and round/match state.

Likely code area:

- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/source_env.py`
- Scenario/oracle tooling under `scenarios/environment`

### Closed - 2P Warmdown-Frame Survivor Lifecycle

Status: closed for the focused continuing-round 2P case.

What is now covered:

- `scenarios/environment/source_lifecycle_survivor_score_2p_next_round.json`
  pins the source/original behavior: avatar 2 dies at round end, avatar 1
  keeps moving during warmdown, avatar 1 dies at 4150 ms, source emits no
  second `round:end`, and `game:stop -> round:new` still fires at 8000 ms.
- `tests/test_lifecycle_oracle.py` and `tests/test_source_lifecycle_runner.py`
  prove the fixture against the original JS oracle and `CurvyTronSourceEnv`.
- `tests/test_vector_multiplayer_env.py` proves the public
  `advance_warmdown_frame(...)` bridge for 2P match-mode metadata: no rescore,
  death order `[1, 0]`, score stays `[1, 0]`, warmdown timer continues from
  3850 ms, and the next round consumes the same source RNG cursor through 16.

Still not claimed:

- A separate match-ending survivor-warmdown variant.
- Visual/browser pixel parity for warmdown frames.
- Trainer/replay final-observation promotion for this lifecycle shape.

### Closed - 2P Body/Trail Collision Canaries Beyond Collision Order

Status: closed for the focused 2P source-pinned runtime and public probes.

What is now covered:

- `tests/test_vector_runtime.py` has 2P fixture-slice checks for opponent
  tangent safe, opponent overlap death, own-body delta3 safe, own-body delta4
  death, trail-gap hole safe, stored body death, print-to-hole boundary death,
  and hole-to-print boundary death.
- `tests/test_2p_trail_gap_source_public_parity.py` creates true 2P probes from
  the original JS trail-gap scenarios and checks JS source output against
  `VectorMultiplayerEnv` public state: alive, printing, score, world body count,
  death player, hit owner, cause, and winner.
- The gray64 visual suite also has 2P programmatic body/trail geometry stress
  cases for opponent tangent/overlap, own-body latency, and PrintManager
  boundary emission.

Still not claimed:

- `old=true` body age metadata in a true 2P probe.
- Broad natural multi-step loop coverage beyond the focused probes.
- Browser/canvas pixel parity.

Likely code area:

- `scenarios/environment`
- `tests/test_env_scenarios.py`
- `tests/test_source_env.py`
- `tests/test_vector_multiplayer_env.py`
- `src/curvyzero/env/vector_runtime.py` if any parity failure appears

### P2 - Row-Local RNG And Replay History

Hole: reset provenance carries seed/source/cursor/draw-count facts, but replay is not yet complete from seed alone. Natural bonus and PrintManager traces need row-local random history or a referenced random artifact for exact replay.

How to test against source/original:

- Record a 2P natural-bonus trace with multiple random sites: reset spawn, PrintManager distances, bonus pop delay, bonus type, bonus position retries, catch, expiry, and terminal.
- Rebuild the trace from replay alone or from replay plus an explicitly referenced random-history artifact.
- Assert source-env and public env reproduce the same spawn, timer, bonus, movement, death, and final observation sequence.

Likely code area:

- `src/curvyzero/env/vector_source_random.py`
- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/training/multiplayer_replay_v0.py`
- Replay manifest/storage code

### P3 - Fully Blocked Generated Bonus-Position Policy

Hole: generated natural bonus position retries can extend the random tape, but a fully blocked generated map policy is not yet a named product behavior. The original can effectively loop while searching for a valid point, so CurvyZero needs an explicit guard policy rather than a silent hang.

How to test against source/original:

- Create a 2P generated-row scenario with no valid bonus position.
- Assert CurvyZero exits with a named truncation/error/diagnostic policy and does not claim source parity for that guard.
- Keep fixture-tape strictness separate from generated-row safety policy.

Likely code area:

- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_runtime.py`
- Public truncation/error metadata tests

### P3 - True Two-Seat Training/Replay Surface

Hole: the fixed-opponent source-state visual wrapper is explicitly not two-seat self-play. The bounded two-seat smoke path is useful plumbing evidence, but not a full environment fidelity claim for a trainer/replay surface.

How to test against source/original:

- Collect 2P episodes where both seats are policy-controlled and both receive observations from the same source state.
- Assert native action mapping, per-seat rewards, masks, final observations, and replay rows for both seats.
- Verify source-state visual and raw-pixel gates independently before advertising learned visual fidelity.

Likely code area:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- Future two-seat replay writer

## Priority Order For Fixing

1. P1 bonus timer, stack, and death stress with JS/original fixtures.
2. P1 bonus replay and final-state facts.
3. P1 trainer/learned observation and final observation contract for the clean
   visual image path.
4. P2 row-local RNG and replay history.
5. P3 fully blocked generated bonus-position policy.
6. P3 true two-seat training/replay surface.

## Practical Acceptance Pattern

For each hole, use this closure pattern:

1. Add or identify a 2P source/original fixture or probe.
2. Assert the fixture against original/source-env behavior.
3. Assert vector runtime/public env behavior against the same facts.
4. Assert terminal/final observation and replay metadata for the same episode.
5. Only then update status docs from "gap" to "covered".
