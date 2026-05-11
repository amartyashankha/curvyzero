# Two-Player Fidelity Gap Catalog - 2026-05-11

Scope: 2P CurvyTron environment fidelity only. Ignore 3P/4P as proof, except where this doc names a gap caused by relying on non-2P evidence.

This is a remaining-gap catalog, not a status victory lap. A hole is closed only when a 2P source/original fixture or probe and a CurvyZero public/runtime test both exist.

## Already Covered For 2P

- Product runtime direction is `VectorMultiplayerEnv`. `VectorTrainerEnv1v1NoBonus` remains a strict proof/profile surface, not the product gameplay environment.
- Source/original rules have been reconstructed from the server JS around `Game`, `BaseGame`, `Avatar`, `BaseAvatar`, `PrintManager`, `BonusManager`, and bonus stacks.
- 2P source lifecycle fixtures exist for warmup/PrintManager start, next round after draw, spawn heading rejection retry, max-score match end, active mid-round `removeAvatar`, and long 1v1 no-bonus wall terminal.
- Public/runtime no-bonus 2P checks cover long reset-to-terminal wall rollout, terminal metadata-only final rows, autoreset preservation of previous final metadata, draw warmdown into next round, unique max-score match-end metadata, and active leave immediate survivor scoring.
- 2P collision-order canaries cover death-point ordering and same-frame head/head reverse-order behavior.
- Focused 2P seeded and natural bonus paths now cover all source-default effect families: self small/slow/fast/master, enemy slow/fast/big/inverse/straight-angle, game borderless, game clear, and all color. SelfMaster body/wall behavior and AllColor overlap have focused tests.
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
  26 scenarios: long wall terminal, movement traces, normal wall/draw cases,
  collision-order cases, borderless wrap cases, `BonusSelfSmall`
  catch/no-catch/expiry/wall-death cases, `BonusGameClear`, and
  `BonusGameBorderless`, plus the four natural bonus spawn/retry/cap fixtures.
  The long no-bonus wall scenario matched exactly for 112 frames through
  terminal (`max_abs_diff=0`, `mismatch_pixels=0`), and the suite-level result
  is exact (`max_abs_diff=0`, `mismatch_pixels=0`). This is the current raw
  visual gate.
- Fixture accounting: 26 total 2P step fixtures exist. The `core2p` visual
  suite covers 25 of those step fixtures plus the long no-bonus wall rollout,
  for 26 visual scenarios total. `source_print_manager_random_call_order_step`
  is intentionally outside gray64 because it proves RNG/event order, not a
  distinct rendered state.
- Current gray64 values distinguish 2P player trails and heads, but all active
  map bonuses collapse to one value (`208`). That is acceptable for the current
  geometry/fidelity gate, but it is not a full visual policy signal for natural
  bonus play because the model cannot see bonus type before contact. The model
  tensor also has no explicit ego bonus stack/status channels.

## Next-Work Checklist

The current source-state visual gate passes: 26/26 `core2p` gray64 scenarios
match exactly. The PrintManager RNG canary is intentionally not a gray64 case;
it proves random/event ordering, not a distinct rendered state.

1. Decide and prove typed bonus visual/status sufficiency beyond gray64 v0.
2. Add bonus stack/death stress across timers, PrintManager, and terminal frames.
3. Add 2P survivor-movement warmdown source/public coverage.
4. Promote final/replay bonus state beyond metadata-only audit rows.
5. Add broader 2P trail/body canaries if they are still open after the current
   collision-order and visual coverage audit.

## Remaining Holes

### P0 - Typed Bonus Visual And Status Sufficiency

Hole: source-state raw 64x64 comparison is now real and the current 26/26
`core2p` gray64 gate passes exactly. The only intentionally excluded 2P step
fixture is `source_print_manager_random_call_order_step`, because gray64 does
not encode PrintManager random-call order or event order. The remaining visual
promotion question is typed bonus visual/status sufficiency: gray64 v0 proves
covered geometry/occupancy parity, but it does not expose active bonus type or
ego stack/status needed for natural-bonus policy decisions. Warmdown survivor
movement, bonus stack/death stress, final observation/replay handoff, and
broader trail/body canaries also remain outside the current gate.

How to test against source/original:

- Use existing JS/source-state fixtures and `CurvyTronSourceEnv` snapshots as
  the source truth for training-observation fidelity.
- Render the source snapshot to gray64 and compare it to `VectorMultiplayerEnv`
  gray64 for the same state and scripted actions.
- Keep browser/canvas pixels as optional later human/debug evidence only. They
  are not a blocker for the current source-state training observation.
- Expand the visual/status schema before claiming natural-bonus visual
  sufficiency.

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
- Latest result: exact source-vs-vector gray64 match across 26 core 2P
  scenarios, including the long no-bonus wall rollout through terminal and the
  four natural bonus spawn/retry/cap fixtures.
- Step-fixture coverage: 25 of 26 total 2P step fixtures are in `core2p`.
  The remaining fixture is `source_print_manager_random_call_order_step`, which
  is verified by source/event-order tests rather than gray64.
- What this proves: the learned raw source-state raster can be regenerated from
  both source-shaped state and fast vector state for these covered fixtures.
- What this does not prove: original browser/canvas pixels, antialiasing,
  sprite colors, viewport scaling, or bonus-type visual sufficiency.
- Natural bonus spawn/retry/cap now uses a separate reset/tape path in the
  visual harness, because the ordinary forced-state seeding path consumes the
  wrong RNG.

Natural-bonus policy sufficiency:

- Gray64 v0 is a geometry/fidelity gate. It proves source-shaped and vector
  rasters agree for covered body/head/trail/bonus occupancy states.
- Gray64 v0 remains the current gate: 26/26 `core2p` gray64 scenarios, exact
  match, one intentionally excluded PrintManager RNG canary. Do not silently
  replace this gate with a new tensor. Any richer tensor must get its own schema
  id, hashes, comparison command, fixture list, and promotion note.
- It is not enough to promote natural-bonus visual training. All active map
  bonuses look identical (`208`), and the learned tensor does not expose ego
  stack/status such as speed, radius, inverse, straight-angle, borderless,
  invincibility, color ownership, or expiry timing.
- v1 should be a separate bonus-aware observation proposal, not a mutation of
  v0 gray64. The simple first proposal is `float32[22,64,64]`, CHW, source-state
  backed, normalized to `[0,1]`, and explicitly not browser/canvas pixels.
- Draft 22-channel budget:
  - 4 geometry channels: ego trail/body, opponent trail/body, ego live head,
    opponent live head.
  - 12 active map-bonus channels, one for each source-default type:
    `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`, `BonusSelfMaster`,
    `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`,
    `BonusEnemyInverse`, `BonusEnemyStraightAngle`, `BonusGameBorderless`,
    `BonusAllColor`, and `BonusGameClear`.
  - 6 broadcast status channels for the effective current state needed by a
    policy: ego speed, opponent speed, ego radius, opponent radius, ego control
    or protection flags, and opponent control or protection flags. These six
    channels are a proposal, not proof; if inverse, straight-angle,
    invincibility/printing, borderless, color remap, or expiry timing cannot be
    represented without lossy packing, widen or revise v1 rather than hiding
    the loss.
- v1 tests are required before promotion. At minimum they need schema/hash
  checks, source-snapshot-vs-vector channel parity, one typed map-bonus mask
  check for every source-default bonus type, catch-frame status checks,
  expiry/restore checks, `BonusGameClear` clear coupling, `BonusGameBorderless`
  active/expired boundary behavior, `BonusAllColor` overlap/restore behavior,
  terminal/final observation before autoreset, and replay/LightZero wrapper
  propagation of the same channels.
- Until those tests pass, this gap should block claims of source-default
  bonus-faithful training observations. It should not block no-bonus training
  work, no-bonus replay/final-observation plumbing, or clearly labeled limited
  v0 gray64 source-state plumbing.
- Promotion tests must include natural spawn/retry/cap fixtures whose rendered
  observation exposes active bonus identity and position, catch frames that show
  the ego status/stack effect, expiry/restore frames, terminal/final observation
  before autoreset, and replay/trainer schema checks that preserve the same
  channels through the LightZero wrapper.

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

How to test against source/original:

- Add 2P source/original scenarios for:
  - bonus expiry at the same timestamp as PrintManager start/stop
  - bonus catch and wall/body death in the same frame
  - death while boosted clears stack and later expiry callbacks become no-ops
  - inverse/double-inverse while turning
  - speed bonus with turn-rate and collision consequences
  - straight-angle applied while already turning
  - borderless expiry followed by wall death
  - game clear before body collision
- Assert event order, stack order, remaining timers, effective avatar/game properties, death order, and round/match state.

Likely code area:

- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/source_env.py`
- Scenario/oracle tooling under `scenarios/environment`

### P2 - 2P Warmdown-Frame Survivor Lifecycle

Hole: 2P has public checks for draw warmdown, max-score match end, active leave, and long wall terminal. It does not yet have a 2P source-backed survivor-moving-during-warmdown fixture. Existing broader survivor warmdown evidence outside 2P should not close this gap.

How to test against source/original:

- Create a 2P source/original scenario with `max_score > 1` where one player dies, the survivor continues during warmdown, then hits a wall/body before `game:stop`.
- Verify source behavior: no second scoring event, no second `round:end`, final round winner unchanged, next-round transition unchanged.
- Match `VectorMultiplayerEnv.advance_warmdown_frame` and `advance_warmdown` metadata to that source trace.
- Test both continuing-round and match-ending variants for final observation policy.

Likely code area:

- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_lifecycle.py`
- `tests/test_vector_multiplayer_env.py`

### P2 - 2P Body/Trail Collision Canaries Beyond Collision Order

Hole: the public body/trail gap canaries for basic own-body, opponent-body, tangent, and trail-gap behavior are currently not all 2P canaries. For 2P, the strong public canaries are collision-order focused. That is not enough for a clean 2P-only body/trail claim.

How to test against source/original:

- Add 2P source/original variants for opponent overlap, tangent near-miss, own-body collision at the source delta boundary, same-frame point materialization safe/kill behavior, trail-gap hole safe passage, and stored trail boundary kill.
- Run each through original/source-env, vector runtime, and `VectorMultiplayerEnv`.
- Assert body owner, death hit owner, death point materialization, `old_bodies` metadata where relevant, death order, score, and final observation.

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

1. P0 typed bonus visual/status sufficiency.
2. P1 bonus timer, stack, and death stress.
3. P2 2P warmdown-frame survivor lifecycle.
4. P1 bonus replay and final-state facts.
5. P2 2P body/trail collision canaries beyond collision order, if still open.
6. P1 trainer/learned observation and final observation contract.
7. P2 row-local RNG and replay history.
8. P3 fully blocked generated bonus-position policy.
9. P3 true two-seat training/replay surface.

## Practical Acceptance Pattern

For each hole, use this closure pattern:

1. Add or identify a 2P source/original fixture or probe.
2. Assert the fixture against original/source-env behavior.
3. Assert vector runtime/public env behavior against the same facts.
4. Assert terminal/final observation and replay metadata for the same episode.
5. Only then update status docs from "gap" to "covered".
