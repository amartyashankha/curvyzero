# Full Fidelity Spec Matrix

Status: plain fidelity answer
Date: 2026-05-09
Owner: FULL FIDELITY SPEC

## Short Answer

No. We are not at full CurvyTron environment fidelity.

There is one runtime under hardening: `VectorMultiplayerEnv`. The JS
oracle and `CurvyTronSourceEnv` are proof/oracle tools, not product
environments. Strict `VectorTrainerEnv1v1NoBonus` is a narrow proof/profiling
boundary, not the destination.
Restricted wrappers are temporary explicit profile configs. They must not
replace source-default CurvyTron reconstruction in `VectorMultiplayerEnv`.

LightZero is a possible coach/training adapter, not the owner of CurvyTron
rules. Current LightZero work is contract and smoke coverage only, not real
training and not a fidelity claim.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

We have rebuilt and source-tested a strong core slice: movement, normal walls,
selected borderless behavior, seeded body collisions, own-body latency,
same-frame point materialization, selected collision order, scoring canaries,
PrintManager toggles, delayed PrintManager start, normal trail cadence, forced
trail gaps, one separate natural multi-step trail-gap source case, and one
old-body metadata case.

We have vector support for a narrower fixture-backed slice: the default mixed
vector comparator, one separate natural taped multi-step trail-gap fixture,
strict long 1v1/no-bonus bridge tests, direct seeded 3P/4P no-bonus
wall-scoring/order canaries, and a metadata-only
`VectorMultiplayerEnv` public surface. That is useful, but it is not
broad public multiplayer lifecycle, replay, trainer-ready observation, or
CurvyTron training coverage. A narrow 3P/4P scalar learned-observation
projection exists now, but it is projection-only.

We now also have a narrow lifecycle/spawn/RNG source slice with both JS oracle
evidence and direct Python parity: 18 fixtures including
`source_lifecycle_spawn_rng_4p_next_round`,
`source_lifecycle_survivor_score_4p_next_round`,
`source_lifecycle_present_absent_3p_survivor_score_round_end`, and
`source_lifecycle_multi_round_match_end_3p` are checked by
`tests/test_source_lifecycle_runner.py`. They are the three 2P core lifecycle
fixtures, focused 3P first-round spawn-order, focused 3P warmup/print-start,
focused 4P first-round spawn-order, focused 4P all-present all-dead
warmdown/next-round, focused 4P survivor warmdown/next-round, focused 3P
first-round present/absent, focused 3P present/absent survivor-scoring and
warmdown/next-round, focused 2P max-score match-end, one focused all-present 3P
`max_score: 2` match-end path, focused 3P all-dead warmdown/next-round, focused
3P tie-at-max-score continuation, focused 3P survivor-scoring `round:end` plus
survivor warmdown/next-round, and focused 3P all-present multi-round match-end.
This proves those pinned source claims only.

Training is not ready for full source-faithful CurvyTron. Strict
`VectorTrainerEnv1v1NoBonus` has landed narrowly as proof/profiling
infrastructure: it owns B rows, returns source-backed
`float32[B,2,106]` trainer observations, maps trainer actions to source moves,
uses natural PrintManager mode during public base steps, stages terminal
final observation/reward before autoreset, and resets/spawns/warms only selected
done rows. Horizon truncation, overflow truncation, terminal metadata, the
live-step replay recorder, terminal barrier replay policy, replay metadata
defaults from env info, truncation reason labeling, and safe timer diagnostics
have also landed for that strict proof/profiling slice. The remaining blockers are
row-local seed/RNG replay history, broad lifecycle/spawn beyond the current
pinned slice, trainer-ready 3P/4P env/observation support beyond the narrow
scalar projection, natural public reset/warmdown/replay coverage beyond the
direct seeded wall canaries and metadata-only public env, bonuses,
visuals/LightZero, speed/whole-loop measurement, broader vector semantics, and
real policy/search integration.
Browser pixels come later and should not be used as gameplay proof.

The current optimizer timing fence is proof/profiling only: strict wrapper
`VectorTrainerEnv1v1NoBonus` -> trainer `float32[B,2,106]` rays/scalars ->
replay-v0 plumbing for the strict 1v1/no-bonus slice. The optimizer may measure
that source-backed plumbing, but it cannot redefine source truth, turn plumbing
measurements into learning claims, or generalize timings to bonuses, broad
lifecycle, 3P/4P, visual LightZero, or full CurvyTron.

The latest combined focused validation after the metadata-only multiplayer env
wave was green on the touched code/test set; treat that as local guard output
for this narrow slice, not as a fidelity dashboard.

## Current Evidence Is Not Completion

Command results are hygiene only. Keep exact pass/fail output beside the command
that produced it, not as a pass-count dashboard. Full fidelity is source-claim
driven: progress means source claim -> oracle/probe -> Python parity ->
optimized parity.

The current top source claim is `natural lifecycle/spawn/RNG`. It now has JS
oracle evidence and Python direct parity for 20 pinned fixtures including
`source_lifecycle_spawn_rng_4p_next_round`,
`source_lifecycle_survivor_score_4p_next_round`,
`source_lifecycle_present_absent_3p_survivor_score_round_end`, and
`source_lifecycle_multi_round_match_end_3p`: three 2P core
fixtures, focused 3P first-round spawn-order, focused 3P warmup/print-start,
focused 4P first-round spawn-order, focused 4P all-present all-dead
warmdown/next-round, focused 4P survivor warmdown/next-round, focused 3P
present/absent first-round, focused 3P present/absent survivor-scoring and
warmdown/next-round, focused 2P max-score match-end, one focused all-present 3P
`max_score: 2` match-end path, focused 3P all-dead warmdown/next-round, focused
3P tie-at-max-score continuation, focused 3P survivor-scoring `round:end` plus
survivor warmdown/next-round, and focused 3P all-present multi-round match-end.
The 3P fixtures prove first-round spawn order, one present/absent
`onRoundNew()` case, one present/absent survivor-scoring case, one
present/absent continuation, one
warmup/PrintManager-start path, one all-three-dead forced wall-death
continuation, one focused all-present `max_score: 2` match-end path, one focused
tie-at-max-score continuation path, and one
survivor-scoring `round:end` case plus survivor continuation through warmdown.
The 4P fixtures prove first-round spawn order/RNG, all-present all-dead
warmdown/next-round, and survivor warmdown/next-round only. The 2P match-end fixture
proves only `max_score: 1`: avatar 2 dies, avatar 1 reaches score 1,
`round:end` winner 1 emits at 3000 ms, `game:stop` and `end` emit at 8000 ms,
no immediate `round:new` follows, and the final snapshot is stopped/cleared with
no avatars. The focused all-present 3P match-end fixture proves only
`max_score: 2`: avatars 3 then 2 die, avatar 1 reaches score 2, `round:end`
winner 1 emits at 3000 ms, `game:stop` and `end` emit at 8000 ms, no immediate
`round:new` follows, and the final snapshot is stopped/cleared with no avatars.
The 3P continuation proves only `round:end` winner null at 3000 ms, `game:stop`
then `round:new` at 8000 ms, and next natural 3P spawn RNG/order. The survivor
round-end fixture proves only two same-frame wall deaths, avatar 1 survivor
`roundScore=2`, reverse score resolution, winner 1, and deaths `[3, 2]`. The
survivor warmdown/next-round fixture proves only that the original JS frame loop
continues moving avatar 1 after `round:end`, avatar 1 dies at 4150 ms, then
`game:stop` and next `round:new` emit at 8000 ms with next natural 3P spawn
RNG/order. The tie-at-max-score fixture proves only `max_score: 1`: avatar 3
dies first, avatars 2 and 1 die together, scores `[1, 1, 0]` carry into the
next round, `round:end` winner null emits at 3000 ms, and `game:stop` then
`round:new` emit at 8000 ms with no `end`. The focused 3P all-present
multi-round fixture proves only `max_score: 3`: avatar 1 carries score 2 through
`game:stop` and `round:new` at 8000 ms, then reaches score 4 and emits
`game:stop` and `end` at 19000 ms with no later `round:new`. This does not prove
broader 3P lifecycle: broader present/non-present variants, bonuses, or broader
4P match lifecycle. It also does not prove vector optimized lifecycle rows,
production reset/autoreset, replay, observation, broader vector semantics, or
real LightZero integration.

## What Full Fidelity Would Mean

Full environment fidelity means every source-visible server rule we care about
has the same behavior in CurvyZero as in the original CurvyTron source:

- Same movement math, elapsed-time handling, turn order, player update order,
  collision order, death side effects, score order, timers, random calls,
  spawn behavior, bonus behavior, and event order.
- Same server state and source event stream through a JS oracle fixture and a
  Python common-trace or direct-parity runner.
- Same behavior after vectorization, including fixed event rows, overflow
  policy, B>1 batching, reset/autoreset, observations, rewards, and replay
  metadata.
- Same wire stream only when we claim network replay fidelity.
- Same browser pixels only when we claim client rendering fidelity.

The source browser is not the gameplay authority. Server state and source event
order are the first authority. Pixels are a later rendering check.

## Fidelity Ladder

Use this ladder for every mechanic. Do not skip rungs and call the result done.

| Rung | What it proves | Current status |
| --- | --- | --- |
| Source code read | We know what the original server source appears to do. | Broad source maps exist for movement, collision, trails, scoring, bonuses, and wire. Source-read alone is not proof. |
| JS oracle/common trace | A deterministic scenario runs against the original JS source and is projected into common trace, or compared directly when the narrow runner owns that shape. | Strong for core mechanics. Lifecycle/spawn/RNG has JS oracle evidence for 20 pinned fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p`. The first active `BonusSelfSmall` catch/no-catch and default multi-type bonus RNG/type selection also have JS oracle fixtures. Missing: broader lifecycle/spawn, broader bonuses, and wire payloads. |
| Python/source runner | A narrow Python runner matches the JS common trace or direct JS oracle output for that scenario. | Strong for current promoted batches. Lifecycle/spawn/RNG has direct Python parity for the 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p` through `tests/test_source_lifecycle_runner.py` when focused tests pass. `CurvyTronSourceEnv` also has the live movement event trace, mid-round 2P removeAvatar/leave proof, and the first active bonus proofs. Each runner is intentionally narrow. |
| Vector comparator | Fixture-seeded arrays match the Python source runner/common trace for supported fields and events. | Narrow fixture-backed comparators exist for the current supported slice; unsupported mechanics remain outside the claim. |
| Optimized parity | Vector and B>1 rows match scalar vector behavior for the named source-backed fixtures and can be timed without source runner calls in the hot loop. | Speed defaults cover only the named supported fixtures; delayed-start and natural multi-step random tape stay separate. |
| Trainer observation/reward/replay | Reset, done, truncation, final observation, reward, legal masks, policy rows, and replay schemas are explicit and tested. | Partial for the strict public 1v1/no-bonus vector trainer env: real `float32[B,2,106]` observations, masks, sparse terminal reward maps, final observations/rewards, autoreset staging, horizon truncation, overflow truncation, terminal metadata, live-step replay recording, terminal barrier replay policy, replay metadata defaults from env info, and truncation reason labeling exist. A narrow 3P/4P scalar projection `curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0` exists as `float32[R,27]` over `VectorMultiplayerEnv.state`, but it is not trainer-ready env support or replay support. Still missing row-local seed/RNG replay history, broad lifecycle, trainer-ready 3P/4P, bonuses, visuals, speed, and real LightZero/policy integration. |
| Pixel/browser later | Browser rendering and screenshots match after state and wire are already trusted. | Deferred. Do not use pixels as first gameplay proof. |

## What We Have Reconstructed

Reconstructed in narrow, source-tested slices:

- Source movement with elapsed milliseconds, base speed, base turn rate, fixed
  straight/turn movement, and varied elapsed multi-step movement.
- Normal wall death basics and selected same-frame wall draw behavior.
- Borderless plain wrap, PrintManager wrap toggle, destination-body skip on the
  wrap frame, and exact-edge/corner-axis behavior.
- Stored body collision with strict overlap, tangent safety, own-body latency,
  opponent immediate body collision, same-frame point insertion, and one
  old-body `old:true` metadata case.
- Death-point and head-head-looking collision-order canaries.
- Source scoring/order canaries for 3P and 4P normal-wall cases.
- PrintManager deterministic toggles, exact-zero behavior, active no-toggle,
  delayed 3000 ms start, active wall/body stop-on-death, and random
  PrintManager call order/cadence canaries.
- Normal trail cadence and forced trail-gap body behavior.
- The exact 2P lifecycle/spawn/RNG fixture slice: warmup/start,
  terminal-to-next-round, and heading-rejection retry event order, labeled RNG
  calls, delayed PrintManager starts, forced wall-death warmdown, `game:stop`,
  and next-round spawn RNG.
- One focused 3P first-round natural spawn fixture: reverse order 3, 2, 1 and
  `position_x`, `position_y`, `angle_attempt_0` labels at 0 ms only.
- One focused 3P first-round present/non-present fixture: avatar 2 is skipped
  for natural spawn RNG, added to `game.deaths`, and pinned as
  `present=false`, `alive=false`, at `(0.6, 0.6)`, with `deathCount=1` and
  `deaths=[2]`.
- One focused 2P max-score match-end fixture: with `max_score: 1`, avatar 2
  dies, avatar 1 reaches score 1, `round:end` winner 1 emits at 3000 ms,
  `game:stop` and `end` emit at 8000 ms, no immediate `round:new` follows, and
  the final snapshot is stopped/cleared with no avatars.
- One focused 3P all-dead warmdown/next-round fixture: all three avatars are
  forced into same-frame wall deaths, `round:end` winner null emits at
  3000 ms, `game:stop` then `round:new` emit at 8000 ms, and the next natural
  3P spawn RNG/order is pinned.
- JS/Python fidelity loop, common trace sidecars, batch manifests, and first
  mismatch reporting.
- Fixture-to-vector seeding, fixed event rows, a narrow mixed vector
  comparator, B>1 batch-row preflights, debug obs/reward packing, and a local
  actor-loop bridge.
- The strict public 1v1/no-bonus vector trainer env:
  `VectorTrainerEnv1v1NoBonus` composes reset/spawn/warmup,
  `vector_runtime.step_many`, natural PrintManager no-bonus stepping, real
  vector trainer observations, sparse terminal rewards, terminal
  snapshot-before-autoreset, horizon truncation, overflow truncation, terminal
  metadata, terminal barrier replay policy, replay metadata defaults from env
  info, truncation reason labeling, live-step replay recording, safe timer
  diagnostics, and selected-row reset/spawn/warmup after terminal rows.
- Modal CPU/vector smoke and Modal/JAX/Mctx boundary evidence, all still
  labeled as runtime plumbing unless they run real CurvyTron rollouts.

Not reconstructed as full source behavior:

- Full lifecycle beyond the current pinned slice: broader 3P lifecycle,
  broader 4P natural lifecycle, present/non-present behavior beyond the current
  canaries, broader multi-round paths, and broader timer paths.
- Natural spawn beyond the current pinned slice: broader spawn rejection edge
  cases beyond the first heading-retry canary.
- Bonuses beyond the current narrow proofs: caps, other catch/death
  interactions, broader stack
  effects/expiry ordering, bonus-world retry promotion beyond source proof,
  natural `BonusGameClear` probability/selection,
  color, speed/radius/inverse effects, other bonus types, and broader
  public/vector support. Source/runtime/public seeded borderless
  duration/expiry now passes narrowly, and seeded public `BonusGameBorderless`
  is supported.
- Row-local RNG policy across all reset/spawn, PrintManager, and bonus paths.
- Full wire payload and compression fixtures.
- Production training gaps beyond the strict 1v1/no-bonus env: row-local
  seed/RNG replay history, broad lifecycle, trainer-ready 3P/4P beyond the
  scalar projection, bonuses, visuals/LightZero, speed gates, and policy/search
  contracts.
- Browser rendering/pixel fidelity.

## Plain Fidelity Answer

No, we are not at full fidelity. The current implementation has strong
fixture-backed source and vector slices, but the missing pieces are still big
enough that we should not call it a real CurvyTron training target.

The real missing fidelity work is grouped below.

Source mechanics:

- Extend lifecycle/spawn/RNG beyond the current pinned fixtures: broader 3P
  match end, broader 4P match lifecycle, broader
  present/non-present lifecycle, multi-round match, warmup/warmdown edge cases,
  and broader natural round
  flows.
- Add bonuses beyond the narrow active `BonusSelfSmall`
  catch/no-catch/death-order, natural one-type spawn/type/position RNG, one
  game-world spawn retry, timed expiry/restore, default multi-type weight/type
  RNG, and forced `BonusGameClear` proofs: caps, broader stack effects/expiry ordering, natural
  clear probability/selection, borderless, color, speed, radius, inverse, other
  bonus types, vector/runtime support, and interaction with death.
- Broaden natural trail/body/collision edge cases only when they isolate a new
  source rule.

Vector and optimized paths:

- Add vector support for lifecycle/spawn/RNG from promoted source fixtures.
- Define row-local RNG arrays, cursor/exhaustion behavior, mixed player-count
  policy, timer rows, broader capacity policy, and event-row policy.
- Keep the current optimized path narrow until scalar vector parity covers the
  new source claims.

Reset and replay:

- Build beyond the strict public 1v1/no-bonus boundary with natural spawn,
  timer state, public autoreset, final-observation preservation, seed history,
  and public row lifecycle rules. Treat landed reset-to-terminal parity,
  horizon/overflow truncation, and terminal metadata as the narrow baseline, not
  the broad reset contract.
- Build on the live-step recorder with production reader policy, final
  observation policy, event/state refs, manifests, compaction, and schema/rules
  hash rejection.

Observation and reward:

- Back the pinned observation/reward contracts with trusted source-state
  fixtures.
- Wire sparse round-outcome reward, final observations, legal masks, and
  terminal info into the actual trainer surface.

Training and LightZero:

- Keep the local no-train LightZero-shaped smoke as plumbing until real
  DI-engine/LightZero registration, reset/step semantics, observation dicts,
  legal masks, scalar rewards, done/truncated info, replay, and policy/search
  row mapping are proven at the real boundary.

Wire and pixels:

- Add wire payload/compression fixtures only after state/event semantics are
  stable.
- Use browser pixels only for render checks after server state and wire traces
  pass.

## What Is Source-Tested

The source-tested surface means JS oracle plus Python/common-trace parity, or
direct Python/oracle parity for narrow fixtures that own that shape.

| Area | Source-tested now |
| --- | --- |
| Movement | `source_kinematics_batch.json`: 7 cases, including straight, turn, and varied elapsed-ms movement. |
| Border/walls | `source_border_batch.json`: 6 cases, including normal wall, same-frame wall draw, plain borderless wrap, PrintManager wrap, destination-body skip, and exact-edge/corner-axis. |
| Multiplayer scoring/order | `source_normal_wall_multiplayer_batch.json`: 3 canaries. |
| Body collisions | `source_body_canary_batch.json`: 6 canaries for strict overlap, tangent safety, own latency, and same-frame point materialization. |
| Old-body metadata | `source_body_old_metadata_batch.json`: 1 old seeded opponent-body death event. |
| Collision order | `source_collision_order_batch.json`: 2 canaries. |
| PrintManager | `source_print_manager_batch.json`: 8 deterministic canaries. |
| PrintManager random | `source_print_manager_random_batch.json`: 2 canaries. |
| Trail cadence | `source_trail_batch.json`: 2 canaries. |
| Trail gap | `source_trail_gap_batch.json`: 4 forced canaries. `source_trail_gap_natural_multistep_hole_crossing.json` is verified separately as one natural taped multi-step source loop and now has a separate scalar vector comparator pass. |
| Lifecycle/spawn/RNG | Eighteen promoted `source_lifecycle_*` fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p`: JS oracle plus direct Python parity for the focused 2P/3P/4P lifecycle slice through `tests/test_source_lifecycle_runner.py`. |

Acceptance commands for named source claims:

```bash
uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --fail-on-mismatch --artifact-root /private/tmp/curvy-source-kinematics-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-border-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_gap_batch.json --python-runner source-trail-gap-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-gap-regression
```

## What Is Vector-Supported

Vector support means fixture-seeded arrays compare back to Python source
common trace for the fields and event rows that the vector path claims. It does
not mean "the whole environment is vectorized."

Natural `source_lifecycle_*` `Game.newRound()` fixtures are still unsupported in
optimized/vector. `scripts/seed_vector_state_from_fixtures.py` now rejects them
as ordinary initial-state seeds and reports RNG contract metadata: call index,
site, avatar, value, at-ms, expected call count, and capacity pressure. That is
honest unsupported reporting for future reset/spawn work, not lifecycle
implementation. The guard must include every promoted `source_lifecycle_*`
fixture before vector lifecycle is claimed.

Acceptance command for the current narrow mixed vector claim:

```bash
python3 scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_body_canary_batch.json \
  scenarios/environment/source_borderless_wrap_step.json \
  scenarios/environment/source_normal_wall_death_step.json \
  scenarios/environment/source_print_manager_batch.json \
  scenarios/environment/source_trail_gap_batch.json \
  --body-capacity 4 \
  --format plain
```

That named fixture set covers:

- Six body canary fixtures.
- Simple borderless wrap.
- Normal-wall death.
- All eight source PrintManager deterministic fixtures, including narrow
  delayed-start timer support.
- Four forced trail-gap fixtures.

The separate natural trail-gap scalar comparator command is:

```bash
python3 scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_trail_gap_natural_multistep_hole_crossing.json \
  --body-capacity 4 \
  --format plain
```

This uses row-local Math.random tape arrays for the source PrintManager
distances, and it is still kept out of speed defaults.

Interface-contract acceptance commands for vector helpers:

```bash
uv run pytest tests/test_compare_vector_arrays_to_fidelity.py -q
uv run pytest tests/test_compare_vector_arrays_to_fidelity.py tests/test_benchmark_vector_batch_rows.py -q
uv run pytest tests/test_compare_vector_arrays_to_fidelity.py tests/test_benchmark_vector_batch_rows.py tests/test_benchmark_vector_actor_loop_bridge.py -q
```

Use these after vector, batch-row, or actor-bridge edits that touch the named
vector/interface contract.

Optimized-parity acceptance commands for the current B>1 speed default slice:

```bash
python3 scripts/benchmark_vector_batch_rows.py --batch-sizes 32 --event-modes debug-event no-event --body-capacity 4 --format plain
python3 scripts/benchmark_vector_actor_loop_bridge.py --batch-sizes 16 --rollout-steps 2 --event-modes debug-event no-event --body-capacity 4 --format plain
```

Use these after changing the supported fixture slice or event-row behavior.

Important vector limits:

- The vector comparator does not support the full source kinematics batch as a
  separate vector gate. It uses movement inside the currently supported
  fixture transitions.
- It does not claim the full border batch as one broad vector gate, but it now
  has source-pinned bridge coverage for normal wall, same-frame wall draw replay
  packing, simple borderless wrap, borderless PrintManager wrap toggle, and
  destination-body skip then next-frame kill. Exact-edge/corner coverage remains
  source-level unless explicitly promoted.
- It now supports the promoted collision-order batch as a source-pinned vector
  comparator bridge, but broader 3P/death-order stress is still open.
- It does not support PrintManager random fixtures as a vector batch.
- It supports only one separate natural multi-step trail-gap source fixture,
  outside the default mixed comparator and outside speed defaults.
- It does not support `source_trail_batch.json` as a vector trail-cadence gate.
- It does not support any of the promoted `source_lifecycle_*` fixtures as
  full lifecycle rows or ordinary vector seeds. The separate `vector_spawn.py`
  helper covers only promoted first-round spawn facts, and the strict public
  env covers only 1v1/no-bonus final observations/rewards, autoreset staging,
  horizon truncation, overflow truncation, terminal metadata, terminal barrier
  replay policy, and live-step replay recording. It does not support natural
  spawn beyond that helper, bonuses, full lifecycle, wire, or broad seed/RNG
  replay history.

## Training Boundary

The strict 1v1/no-bonus public vector trainer surface is now real, but it is a
narrow interface boundary, not a full CurvyTron training claim.

Ready as narrow plumbing:

- `curvyzero-v0` has a toy 1v1 env with `reset`, `step`, `observe(ego_player)`,
  `legal_action_mask(ego_player)`, `last_reset_info`, terminal
  `final_observation`, and refusal to step after terminal until reset.
- Debug vector observation/reward packing exists with `obs[B,P,9]`, debug
  reward, legal masks, done/truncated masks, and ego row ids.
- Actor bridge can run fixture-reset rollout blocks, debug pack, synthetic
  policy/search, action encoding, in-memory replay staging, and debug-only
  autoreset after replay staging.
- `VectorTrainerEnv1v1NoBonus` exposes the strict public vector env API for
  source-shaped 1v1/no-bonus rows: B-row reset/step, `float32[B,2,106]`
  observations, boolean and LightZero-style masks, source-move action mapping,
  sparse terminal rewards, final observation/reward arrays, natural
  PrintManager stepping, horizon truncation, overflow truncation, terminal
  metadata, terminal barrier replay policy, live-step replay recording, and
  selected-row autoreset staging.

Not ready for source-faithful training:

- Production-facing masked `reset_arrays` exists, and the public-facing
  `plan_autoreset_rows(...)` / `apply_autoreset_rows(...)` helpers stage final
  arrays before reset. The strict public env composes those helpers for 1v1
  no-bonus only. Still missing: broad lifecycle timer scheduling,
  production `reset_many`, and reset/autoreset beyond the strict slice.
- No broad public trainer environment contract beyond strict 1v1/no-bonus.
- Row-local seed/RNG replay history is still thin.
- Final trainer observation/reward schemas exist for the narrow 1v1/no-bonus
  ray/scalar path, including strict vector final-observation/reward arrays.
  Broader source-state observation coverage remains open.
- Replay-v0 writer/reader, a pure trainer replay-v0 builder, and a live-step
  recorder fed by public env batches exist for narrow chunks. Broader replay
  refs, manifests, compaction, and seed/RNG history remain open.
- Terminal info keys and strict public env terminal metadata are pinned for this
  slice, but not yet verified through a broad public env or real DI-engine
  boundary.
- Policy/search row mapping exists; real policy/search integration remains
  open.
- No CurvyZero LightZero adapter that has proven reset, step, observation,
  legal action mask, scalar reward, done/truncated, terminal info, deterministic
  opponent action filling, and sidecar metadata.
- No real CurvyTron rollout inside JAX/Mctx or Modal GPU.

## LightZero Training Boundary

The smallest LightZero target is an ego wrapper around `curvyzero-v0`.

- `reset(seed)` returns a LightZero dict with `observation`, `action_mask`, and
  `to_play=-1`, and records seed/episode metadata.
- `step(action)` maps one ego action in `{0,1,2}` to a CurvyZero wrapper action
  map by using a named deterministic opponent policy. That wrapper map is later
  converted to held source control state for elapsed-ms source frames.
- Observation starts from the pinned `curvyzero_egocentric_rays/v0` trainer
  helper with LightZero flat shape `float32[106]`. Older `float32[9]` rows are
  debug-only wiring notes, not the coach-facing target.
- Legal action mask is `np.int8` shape `(3,)` in `[left, straight, right]`
  order.
- Reward is one scalar ego reward, initially the named sparse round-outcome
  reward. Debug or shaped rewards stay in telemetry unless explicitly selected.
- Done is `terminated[ego] or truncated[ego]`; `info` preserves both fields,
  terminal reason, `eval_episode_return`, wrapper action map, opponent policy id,
  seed/episode ids, schema ids/hashes, final reward map, and trace hash/ref.
- LightZero batching means many single-agent env rows. It does not solve
  CurvyTron's multi-player control selection by itself, and it does not make
  wrapper action maps or `joint_action` sidecars native source objects.

This boundary is limited by the same fidelity gaps as the environment:
row-local seed/RNG replay history, broader lifecycle/spawn, 3P/4P beyond direct
seeded wall canaries, bonuses, visuals, speed/whole-loop measurement, broader
vector support, and policy/search row mapping.

## Mechanics Matrix

`Blocks training?` means blocks honest source-faithful CurvyTron training, not
toy plumbing.

| Mechanic | Current evidence | Current tests/commands | Vector status | Missing cases | Next action | Blocks training? |
| --- | --- | --- | --- | --- | --- | --- |
| Movement | Source movement is verified for one-step, straight/turn multi-step, and varied elapsed-ms movement. Source constants are known: speed `16`, turn `2.8/1000` rad/ms. | `source_kinematics_batch.json` with `source-kinematics`; `tests/test_env_scenarios.py` kinematics tests. | Movement math runs inside supported vector fixtures, but no standalone vector gate for the 7-case kinematics batch. | Bonus-modified movement, long drift/fuzz, straight-angle bonus, inverse/speed effects. | Rerun the kinematics source acceptance command; add bonus movement only under bonus fixtures. | No |
| Walls | Normal wall basics and same-frame wall draw are promoted in the border batch. Border checks run before body collision. | `source_border_batch.json` with `source-border-rules`; border tests in `test_env_scenarios.py`. | Vector supports normal-wall death and same-frame wall draw replay packing as source-pinned bridge coverage. | Normal-wall exact-edge/epsilon matrix and broader terminal draw stress. | Add exact inside/on/outside fixtures only if needed; keep wall vector cases source-pinned. | No |
| Borderless | Plain wrap, PrintManager wrap, destination-body skip on wrap frame, and exact-edge/corner-axis behavior are source-tested. Important rule: not torus collision lookup. | `source_border_batch.json`; JS runner borderless tests. | Vector supports simple wrap, PrintManager wrap/toggle, and destination-body skip then next-frame kill as source-pinned bridge coverage. Exact-edge/corner remains source-level unless promoted. | Vector support for exact-edge/corner-axis and next-frame second-axis behavior if needed. | Add only the exact promoted fixture needed next; do not turn borderless into a broad speed claim yet. | Yes |
| Body collision | Strict circle overlap and tangent safety are source-tested for seeded bodies. Opponent bodies collide immediately. | `source_body_canary_batch.json`; body scenario tests. | Vector supports all six body canary fixtures. | Natural emitted-body collisions, island bucket edges, young/old boundary pair. | Add emitted-trail body fixture and vector support. | Yes |
| Own-body latency | Own body latency is source-tested: own body collides only when `currentBody.num - storedBody.num > 3`. | `source_body_own_delta3_safe_step`, `source_body_own_delta4_kills_step`; body batch. | Vector supports both seeded own-latency fixtures. | Natural own-loop collision from emitted trail points. | Add natural emitted own-body loop fixture if self-collision training is claimed. | Yes |
| Same-frame points | Same-frame point materialization is source-tested; earlier reverse-order updates can create bodies before later players collide. | `source_body_same_frame_point_kills_step`, control safe fixture, collision-order batch. | Vector supports body same-frame point fixtures and the promoted collision-order batch bridge. | Death-frame side effects in more natural cases; wider 3P order stress. | Rerun the body/collision source acceptance commands after collision edits; add broader stress only if it isolates a new source rule. | Yes |
| Collision order | Death-point-kills-later-player and head-head-looking reverse-order single-death are source-tested. | `source_collision_order_batch.json` with `source-body-canary`; JS collision-order tests. | Vector supports the promoted collision-order batch as a source-pinned bridge. | Broader death order, 3P stress, ambiguous killer rules. | Keep the promoted batch as a regression guard; add 3P stress only when needed. | Yes |
| Scoring and round end | Normal-wall 3P/4P scoring/order and one-survivor terminal score are source-tested. The 2P all-dead lifecycle fixture has direct JS/Python parity for terminal score events, `round:end`, warmdown, `game:stop`, and next `round:new`. Focused 3P all-dead, survivor, present/non-present survivor, tie-at-max, match-end, and multi-round match-end fixtures are pinned. Focused 4P all-dead and survivor next-round fixtures are pinned. One focused 2P mid-round removeAvatar/leave proof verifies `player:leave`, no current-round death-list insert for the leaver, and immediate round end when one avatar remains alive. | `source_normal_wall_multiplayer_batch.json`; the 28 promoted `source_lifecycle_*` fixtures including `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p`; `source_lifecycle_mid_round_remove_avatar_2p.json`; round/scoring source-map docs. | Vector supports only the one-survivor score update needed by `source_normal_wall_death_step`; lifecycle is not optimized yet. | Broader present/alive leave cases, broader 4P match lifecycle stress. | Add new lifecycle fixtures only when they isolate one missing source rule. | Yes |
| PrintManager toggles | Print-to-hole, hole-to-print, exact-zero, active no-toggle, delayed start, wall/body death-stop are source-tested. | `source_print_manager_batch.json`; `tests/test_env_scenarios.py`; `tests/test_compare_vector_arrays_to_fidelity.py`. | Vector supports the deterministic PrintManager fixtures in the mixed comparator. B>1 speed defaults exclude delayed-start. | Natural long cadence, random-distance rows in vector, broad reset/timer integration. | Keep deterministic toggle support separate from random/lifecycle support. | Yes |
| Delayed start and timers | Narrow delayed `3000 ms` PrintManager start fixture is source-tested. Comparator models the full two-tick timer trace. | `source_print_manager_delayed_start_timer_step`; vector delayed-start tests. | Narrow vector comparator support exists. Focused B>1 helper exists, and strict 1v1 public env uses the warmup/start timer path. Speed defaults exclude it. | Warmdown, next-round timers, broader row-local timer API, timer overflow/truncation policy. | Extend from strict warmup/start into source-backed warmdown and next-round timers before broad lifecycle claims. | Yes |
| Random cadence | PrintManager random call order and one random cadence case are source-tested. | `source_print_manager_random_batch.json` with `source-print-manager-canary`. | Not vector-supported in current comparator or speed defaults. | Unified RNG policy for spawn, headings, PrintManager distances, bonus timing/type/position. | Write row-local random stream policy and call-log fixtures before natural spawn/bonus work. | Yes |
| Trail cadence | Normal point insertion and below-radius no-point behavior are source-tested. | `source_trail_batch.json` with `source-trail-cadence-canary`. | Not a standalone vector comparator gate. Normal point insertion exists inside some supported vector transitions. | Exact threshold, multi-step normal cadence, longer drift. | Add exact-threshold or multi-step fixtures only after tolerance is explicit. | Yes |
| Trail gaps | Four forced trail-gap cases are source-tested: hole-space safe, stored body still kills, print-to-hole boundary kills, hole-to-print boundary kills. One separate natural taped multi-step hole crossing is source-verified. | `source_trail_gap_batch.json`; separate `source_trail_gap_natural_multistep_hole_crossing.json`; default mixed vector comparator includes only the four forced cases. | Vector supports all four forced trail-gap fixtures in the default mixed run, and separately supports the natural multi-step random-tape case in scalar full-trace comparison. Speed defaults still own only the forced cases. | Broader natural gap cadence beyond the one taped crossing, broader row-local RNG, broader emitted-body variants. | Keep the natural case separate until scalar/vector parity stays boring enough to promote into speed defaults deliberately. | Yes |
| Bonuses | Source maps exist for config, spawn, catch order, stack math, timers, expiry, clear, borderless, color, speed/radius/inverse. The promoted slice is narrow active `BonusSelfSmall` catch/no-catch/death-order, one natural one-type `BonusSelfSmall` spawn/type/position RNG proof, one default multi-type weight/type RNG proof, game-world and bonus-world spawn retry proofs, one cap-at-20 skip proof, broad Python source-env stack/effect support for self/enemy/all bonuses, one timed `BonusSelfSmall` expiry/restore proof, one forced `BonusGameClear` immediate clear proof, forced optional-array `BonusGameBorderless` catch plus source/runtime/public seeded duration/expiry, seeded public `BonusSelfSmall`/`BonusGameClear`/`BonusGameBorderless`, partial public natural source-default type selection, same-frame natural bonus plus PrintManager random-order accounting, seeded public bonus replay audit metadata, and a low-level natural spawn helper in `vector_runtime.py` with type/position/retry/cap tests. Runtime bonus support is being consolidated toward an explicit table/spec; natural source-default type selection must not imply unsupported runtime effects. Source CurvyTron remains the oracle, and env runtime work remains the priority. | Source docs: `docs/research/curvytron_source_map/bonuses_config.md`; `source_bonus_self_small_catch_step.json`; `source_bonus_self_small_tangent_no_catch_step.json`; `source_bonus_self_small_wall_death_no_catch_step.json`; `source_bonus_spawn_type_position_rng_step.json`; `source_bonus_default_weights_type_rng_step.json`; `source_bonus_spawn_game_world_retry_step.json`; `source_bonus_spawn_bonus_world_retry_step.json`; `source_bonus_spawn_cap_twenty_step.json`; `source_bonus_self_small_expiry_restore_step.json`; `source_bonus_game_clear_immediate_step.json`; `source_bonus_game_borderless_expiry_restore_step.json`; `tests/test_source_env.py`; `tests/test_vector_runtime.py`; `tests/test_vector_multiplayer_env.py`; `tests/test_multiplayer_replay_contract.py`. | Fast runtime supports optional-array forced `BonusSelfSmall` catch/expiry, `BonusSelfSlow`, `BonusEnemyBig`, forced `BonusGameClear`, forced `BonusGameBorderless` catch and duration/expiry, seeded public `BonusGameBorderless`, metadata-only type/cap helpers, and a low-level natural spawn helper. Public seeded bonus support covers `BonusSelfSmall`/`BonusGameClear`/`BonusGameBorderless`; public natural type selection is partial only. | Broader stack math/expiry ordering, natural spawned `BonusGameClear` catch/clear coupling, remaining runtime/public effects, public natural spawn timer ownership/scheduling/random accounting, full public bonus replay/final state, and broader death interactions. | Keep the current narrow fixtures as the first bonus proofs; add one isolated bonus claim at a time. | Yes |
| Spawn | Map size and max score formulas are source-read. The promoted lifecycle fixtures have direct JS/Python parity for natural spawn position RNG, accepted first-attempt heading RNG, one heading rejection retry, reverse player order, no spawn bodies, 2P/3P/4P first-round spawn streams, focused 3P next-round spawn streams, and focused 4P all-dead plus survivor next-round spawn streams. | The 28 promoted `source_lifecycle_*` fixtures, including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, and `source_lifecycle_multi_round_match_end_3p`; source docs. | Narrow vector spawn support exists in `src/curvyzero/env/vector_spawn.py`: `spawn_round_rows(...)` matches promoted first-round facts for 2P heading retry, 3P reverse spawn order, 3P present/absent, and 4P reverse spawn order. `vector_lifecycle.py` now composes reset/spawn and a strict 1v1/no-bonus warmup/start/runtime-step slice. It still does not support full lifecycle. | Free-position rejection controls, broader heading retry cases, broader lifecycle beyond the pinned continuations, broader present/non-present behavior, reset integration beyond the strict slice, and reset-seed policy. | Extend the strict reset/spawn/warmup/runtime helper toward source-backed warmdown and next-round paths before broadening claims. | Yes |
| Lifecycle | The current pinned lifecycle slice has direct JS/Python parity: 2P warmup, `game:start`, delayed print starts, heading rejection retry, terminal update, `round:end`, warmdown `game:stop`, immediate next `round:new`, next spawn RNG, focused 3P first-round spawn order, focused 3P warmup/PrintManager start, focused 3P present/non-present first-round, survivor-scoring, and warmdown/next-round cases, focused 3P all-dead forced wall-death continuation, focused 3P survivor-scoring `round:end`, focused 3P survivor warmdown/next-round continuation, focused 4P first-round spawn order, focused 4P all-dead and survivor next-round paths, focused 2P/3P match-end cases, focused 3P tie-at-max-score continuation, and `source_lifecycle_multi_round_match_end_3p`. `CurvyTronSourceEnv` separately verifies one active 2P mid-round `removeAvatar`/`player:leave` case. | The 28 promoted `source_lifecycle_*` fixtures including `source_lifecycle_survivor_score_4p_next_round` and `source_lifecycle_present_absent_3p_survivor_score_round_end`; `source_lifecycle_mid_round_remove_avatar_2p.json`; `tests/test_source_lifecycle_runner.py`; `tests/test_lifecycle_oracle.py`; `tests/test_source_env.py`; `tests/test_vector_spawn.py` for the narrow vector spawn helper. | Full lifecycle is not optimized/vector supported. Natural `source_lifecycle_*` fixtures are rejected by the seeder as ordinary seeds with RNG metadata. The strict 1v1/no-bonus vector helper and public env now cover reset/spawn/warmup/start/runtime step, terminal lifecycle flags, vector trainer final-observation/reward handoff, autoreset planning, and reset-to-terminal parity for the named long 1v1 fixture. Direct seeded 3P/4P no-bonus wall-scoring/order canaries, no-bonus N-player reset/warmup helpers, one narrow 3P warmdown/next-round helper, and metadata-only public 2P/3P/4P stepping also exist. They still do not cover broad warmdown/next-round, broad present/non-present lifecycle, learned 3P/4P observations, natural public reset/replay parity, bonuses, or a broad public full-env API. | Round timers beyond the pinned start path, scoring beyond the strict terminal slice, broad warmdown/next-round, natural public reset/replay integration, broad autoreset integration, broader present/non-present lifecycle, broader 4P match lifecycle stress, broader present/alive leave cases, broad bonuses. | Use the pinned source slice, seeder RNG metadata, `vector_spawn.py`, `vector_reset.py`, `vector_lifecycle.py`, `vector_trainer_observation.py`, `vector_trainer_env.py`, and `vector_multiplayer_env.py` to define reset-integrated lifecycle rows; do not claim broad lifecycle until new source fixtures prove it and vector rows implement it. | Yes |
| Wire events | Network source map documents JSON batch events and compression. Common trace is not a wire replay. | `network_render_build.md`; no wire batch command. | Not supported. | Compressed socket payload, `angle`, `borderless`, `clear`, bonus events, spectator catch-up. | Add `wire_event_single_tick` only after state/event semantics are boring. | No |
| Observations | Original source has no learned observation. Toy/debug observation exists. Observation plan exists. | `CurvyTronEnv.observe`; `benchmark_vector_obs_reward_packing.py`; `trainer_observation.py`; `vector_trainer_observation.py`; `observation_fidelity_plan.md`; `trainer_observation_reward_contract_v0_2026-05-09.md`. | Debug packer uses `obs[B,P,9]`; not final. Trainer contract pins `curvyzero_egocentric_rays/v0`; toy/grid and strict vector body-circle helpers now return the pinned `float32[106]` shape, masks, sparse reward metadata, and `float32[B,2,106]` final observations for the strict 1v1/no-bonus proof slice. It is still not broad source-backed observation fidelity. | Trusted source-state observation fixtures, broader perspective/leak checks, broader vector/body-array semantics, LightZero wrapper. | Keep strict vector trainer observations scoped to proof/replay-v0 evidence; harden `VectorMultiplayerEnv` observation contracts separately, and keep visual/pixel observations as a later adapter. | Yes |
| Rewards | Toy sparse reward exists; vector debug reward exists. Source scoring is separate from training reward. | `CurvyTronEnv._terminal_rewards`; `trainer_observation.py`; `vector_trainer_observation.py`; debug packer reward formula. | Sparse round-outcome reward exists for toy/grid and strict vector 1v1/no-bonus terminal handoff. Debug reward still exists separately as `score_delta + round_score_delta - died_this_step`. | Production reward wiring into vector replay, broader terminal/truncation cases, broad multiplayer reward maps. | Wire sparse rewards from vector trainer batches into replay-v0; keep debug rewards labeled debug-only. | Yes |
| Reset/autoreset | Toy env refuses hidden post-terminal stepping and returns final observation. `src/curvyzero/env/vector_reset.py` now provides a production-facing masked `reset_arrays(...)` boundary: validate required row arrays, snapshot selected terminal rows before mutation, copy selected reset-template rows, increment `episode_id`, stamp `reset_seed`/`reset_source`, clear terminal/event/timer-fired fields where present, preserve skipped rows, and return reset metadata plus the terminal snapshot. | `reset_timer_autoreset_plan_2026-05-09.md`; vector reset/autoreset/trainer-observation modules; vector comparator tests; actor bridge debug autoreset test; `tests/test_vector_trainer_env.py`. | Reset boundary, narrow spawn helper, public autoreset planner/apply helper, strict 1v1 final-observation/reward staging, horizon truncation, overflow truncation, terminal metadata, terminal barrier replay policy, live-step replay recording, and `VectorTrainerEnv1v1NoBonus` public reset/step/autoreset now exist as proof/profiling evidence. Still missing: broad lifecycle timers, row-local seed/RNG replay history, and production `reset_many` beyond the strict slice. | Reset-integrated spawn templates beyond the strict slice, seed history, lifecycle timers, reset info arrays, and broader terminal preservation across lifecycle rows. | Use the strict 1v1 proof as a guardrail; add seed/RNG history and broad lifecycle timers to the `VectorMultiplayerEnv` hardening path before widening lifecycle claims. | Yes |
| Replay | Actor bridge stages fixed in-memory chunks with debug metadata. A separate 1v1/no-bonus replay v0 file contract exists, B>1 replay packing exists for narrow chunks, and the live-step recorder builds chunks from strict proof/profiling env batches. | `benchmark_vector_actor_loop_bridge.py`; related tests; `replay_chunk_v0.py`; `tests/test_trainer_replay_v0_builder.py`; `tests/test_vector_env_replay_recorder.py`. | Narrow pack/build support exists for returned public env observations, rewards, actions, terminal final arrays, terminal metadata, and the terminal barrier replay policy. | Seed/RNG replay history, broader event/state refs, manifests, compaction, and broad lifecycle replay rows. | Add seed/RNG history and broader refs once those contracts exist; keep the terminal barrier policy scoped to strict 1v1/no-bonus proof/profiling. | Yes |
| Vector/B>1 | Current vector support is real but narrow; the natural trail-gap scalar comparator remains separate from speed defaults, and `vector_spawn.py` covers only promoted first-round spawn facts. | `compare_vector_arrays_to_fidelity.py`; `benchmark_vector_batch_rows.py`; `benchmark_vector_actor_loop_bridge.py`; `tests/test_vector_spawn.py`; `tests/test_vector_trainer_env.py`. | Real but narrow. Strict public 1v1/no-bonus B-row env exists with public horizon/overflow truncation, terminal metadata, terminal barrier replay policy, and live-step recording; fixed P/K groups, no full lifecycle, and natural trail-gap is not in speed defaults. | Broader fixtures, mixed-P policy, no-event/debug-event production choice, reset-integrated lifecycle rows beyond 1v1, 3P/4P, and bonuses. | Keep correctness-first preflights; widen only from promoted source fixtures. The optimizer may time source-backed `[B,2,106]` plumbing only as plumbing. | Yes |
| Modal/GPU | Modal CPU/vector smoke and Modal/JAX/Mctx boundary runs exist. They prove runtime plumbing, not CurvyTron self-play. | `environment_vector_bench.py`; `mctx_synthetic_benchmark.py`; Modal docs/runbooks. | Not an environment semantics gate. | Remote fixture equivalence job, real observation packer, real rollout loop, transfer/compile/steady metrics. | Keep Modal as coarse artifact jobs; no per-step Modal calls. | No |
| Pixel/browser later | Browser/client source is mapped. Rendering is deferred. | `network_render_build.md`; raw app build notes. | Not supported. | Old app build, browser harness, screenshots, visual trail/render parity. | Do after source state and wire payloads pass. | No |

## How To Test Each Part

Use this order for a new mechanic:

1. Write one source claim in plain language.
2. Read the source files and source-map docs that own that claim.
3. Add one deterministic scenario JSON or JS probe.
4. Run the original JS oracle.
5. Project JS output to common trace.
6. Add or extend the narrow Python source runner.
7. Run `tools/run_fidelity_loop.py` for one scenario.
8. Promote a small batch with `tools/run_fidelity_batch.py`.
9. Add vector seeding and vector comparator support only after source parity.
10. Add optimized parity and timing only after scalar vector comparison matches.
11. Add trainer observation/reward/replay tests only after state/events match.
12. Add browser/pixel checks only after state and wire evidence exists.

Do not use fuzzing as the first proof. Use fuzzing after small goldens exist.

## How We Know This Is Comprehensive

This matrix is comprehensive in the practical sense: every known
source-visible environment surface has a row with evidence, test command,
vector status, missing cases, next action, and training-blocking status.

The claim inventory comes from these sources:

- Source maps over `Game`, `Avatar`, `BaseAvatar`, `World`, `Island`,
  `AvatarBody`, `PrintManager`, `BonusManager`, `BonusStack`, `BaseGame`,
  room config, and network/render code.
- Scenario batch manifests under `scenarios/environment`.
- Python runners in `src/curvyzero/fidelity/source_runners.py`.
- Common trace projection in `src/curvyzero/env/trace_compare.py`.
- Vector comparator and B>1 scripts under `scripts/`.
- Training/debug interfaces in `src/curvyzero/env/core.py` and the vector actor
  bridge scripts.
- Working docs: `coverage_tracker.md`, `source_feature_inventory.md`,
  `vector_state_schema.md`, `reset_timer_autoreset_plan_2026-05-09.md`,
  `observation_fidelity_plan.md`, `modal_vectorization_integration_plan.md`,
  and `spec_backlog_2026-05-09.md`.

The proof is still not "we thought of everything." The proof is weaker and
more useful: every current claim must point to a source-read note, a JS
oracle/probe, Python parity status, optimized parity status if vectorized, and a
clear unsupported list if not.

## Overclaims To Avoid

- "Full fidelity is done." False. Broader lifecycle/spawn, bonuses, training
  contracts, replay, wire, and pixels are still missing.
- "Source-read means verified." False. Source-read is only the first rung.
- "JS-pinned alone means Python is correct." False. Python parity is a separate
  gate. The current 28 pinned lifecycle fixtures including
  `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p` have crossed it when the focused
  tests pass; broader lifecycle/spawn has not.
- "The vector comparator is a production backend." False. It is a narrow
  fixture-seeded array-step proof.
- "B>1 rows/sec is self-play throughput." False. Current speed defaults are
  fixture-cycled CPU slices, not real completed games with policy/search/replay.
- "Debug obs/reward is the training contract." False. It is a temporary packer
  for shape and timing.
- "Modal/JAX/Mctx smoke means real CurvyTron rollout." False. It is boundary
  runtime evidence until it consumes real env rows and returns real replay.
- "Pixels can prove gameplay." False. Pixels can prove render behavior only
  after state and event traces pass.
- "Bonuses can wait but we can still claim full CurvyTron." False. Bonuses are
  source-visible CurvyTron mechanics.
- "Randomness can be faked with constant 0.5." False for spawn, bonuses, and
  broad PrintManager cadence. It is only acceptable in narrow fixtures that say
  so.

## Rabbit Holes

- GPU env stepping before profiling the real actor loop.
- Huge synchronous batches that inflate rows/sec while hurting action latency.
- More planning labels instead of one claim, one fixture, one batch, one
  unsupported list.
- Browser build work before server state and wire stream are trusted.
- Bonus implementation without a row-local RNG policy.
- Production trainer API expansion before reset/autoreset, observations,
  rewards, terminal info, and replay are boring.
- Treating toy-v0/debug paths as source-fidelity, replay, bonus, or product
  runtime evidence. They are quarantined as historical smoke/interface evidence.

## Current Plain Next Actions

1. Expand `natural lifecycle/spawn/RNG` beyond the current 28-fixture pinned
   JS/Python-direct slice only with source-claim-named fixtures.
2. Add row-local seed/RNG replay history for strict public env resets and
   replay chunks.
3. Broaden no-bonus lifecycle only from source-claim-named fixtures.
4. Add bonus fixtures one promoted source claim at a time.
5. Add broader vector support only from promoted source fixtures.
6. Keep existing source and vector checks as regression footers after source
   runner or vector changes.
7. Treat Modal/GPU and pixels as later evidence unless they run the real
    CurvyTron state path.

## Stale Docs Noticed But Not Touched

I did not edit these because this worker owns only this new doc:

- `docs/research/curvytron_source_map/facts_index.md` still lists some
  "probe-needed" items that newer working docs say are now promoted, such as
  varied elapsed movement and delayed PrintManager start.
- Some older timing notes in `docs/working/environment/vector_state_schema.md`
  are explicitly marked as old or pre-fixed-event-row timing and should not be
  quoted as current throughput without rerun.
