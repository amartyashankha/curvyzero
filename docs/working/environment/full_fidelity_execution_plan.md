# Full Fidelity Execution Plan

Status: docs-only execution map
Date: 2026-05-09

This is the orchestration map from the current verified CurvyTron slices to full
environment fidelity. Keep the top short. Put source details, fixture geometry,
and mismatch archaeology in the linked working docs.

One runtime is under hardening: `VectorMultiplayerEnv`. The name is
historical public-env wording; product direction is one fast source-faithful
runtime, not two implementations. `CurvyTronSourceEnv` and the source JS oracle
are proof tools. Strict `VectorTrainerEnv1v1NoBonus` is proof/profiling only,
not the destination.

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Source-Claim Map

Target fidelity is server gameplay state and event fidelity first. Progress is
source claim -> oracle/probe -> Python parity -> optimized parity. The plan is
organized around source claims, because the implementation work should be named
after the behavior it proves.

Current top source claim: `natural lifecycle/spawn/RNG`.

This claim owns natural round creation, player spawn order, spawn position and
heading random calls, print timer scheduling, round termination, warmdown,
next-round scheduling, and focused max-score match ends. The current 20 pinned
lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`,
`source_lifecycle_survivor_score_4p_next_round`,
`source_lifecycle_present_absent_3p_survivor_score_round_end`, and
`source_lifecycle_multi_round_match_end_3p` now have JS oracle
evidence and direct Python parity through `tests/test_source_lifecycle_runner.py`,
which compares events, `randomCalls`, and snapshots against
`tools/reference_oracle/lifecycle_oracle.js`. It covers three 2P lifecycle
fixtures, focused 3P first-round spawn order, focused 3P warmup and delayed
PrintManager start order/random calls, focused 4P first-round spawn order/RNG
labels, focused 4P all-present all-dead warmdown/next-round, focused 4P
survivor warmdown/next-round, focused 3P first-round present/absent, focused 3P
present/absent survivor-scoring, one focused 2P
`max_score: 1` match-end fixture, and one focused 3P `max_score: 2`
match-end fixture, plus one focused 3P all-dead next-round fixture, one
focused 3P survivor-scoring `round:end` fixture, one focused 3P survivor
warmdown into next-round fixture, one focused 3P tie-at-max continuation, and
one focused 3P all-present multi-round match-end. This is not full broad
lifecycle.

Allowed status labels in this plan:

- `verified`: JS oracle and Python/common-trace parity are promoted for the
  named source claim.
- `verified-narrow`: JS oracle and Python direct/common parity are promoted for
  a named narrow slice, with broader cases still open.
- `JS-pinned`: a JS oracle fixture pins the behavior; Python parity is not yet
  promoted.
- `source-read`: original source or source-map docs have been read, but no JS
  oracle fixture pins the behavior yet.
- `needs-spec`: the behavior needs a one-sentence claim and fixture contract
  before JS or Python work.
- `blocked`: waits on an earlier source claim.
- `deferred`: intentionally later than the current source or interface gate.

## Remaining Claims

| Claim | Status | First implementation chunk | Blockers | Acceptance artifact |
| --- | --- | --- | --- | --- |
| `natural lifecycle/spawn/RNG` | `verified-narrow`: Python direct parity for 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p` | Keep the focused 2P/3P/4P lifecycle fixtures as regression guards, including 4P all-dead and survivor next-round, 3P present/absent survivor scoring, 3P match-end/tie/multi-round, and the 2P core paths; expand only from new source claims such as broader 4P match lifecycle or broader present/non-present variants. | Broad lifecycle is still missing: broader 4P match lifecycle, broader present/non-present variants, bonuses, production reset/autoreset/replay/final obs/reward, and optimized/vector full lifecycle. | `tests/test_source_lifecycle_runner.py` compares events, `randomCalls`, and snapshots against `tools/reference_oracle/lifecycle_oracle.js` for all 28 promoted lifecycle source fixtures. |
| `production reset/autoreset` | `partial` | Build natural reset/spawn templates, pre-step timers, terminal preservation, horizon truncation, seed history, and public autoreset on top of `src/curvyzero/env/vector_reset.py`. | Needs lifecycle/spawn/RNG so reset state and seed history are source-shaped. `vector_reset.py` is a reset boundary only; it does not spawn or autoreset. | Reset/autoreset contract tests named by terminal and reset behavior, not by fixture counts. |
| `production replay` | `partial` | Wire the separate 1v1/no-bonus replay v0 contract into the actor bridge, then add final-observation policy, terminal metadata, event/state refs, manifests, and compaction. | Needs reset/autoreset and observation/reward terminal surfaces before production rows can be trusted; current actor-bridge use may still be sample/debug-only. | Replay chunk v0 acceptance commands with mismatch rejection checks and actor-bridge write/read coverage. |
| `trainer observation/reward` | `partial` | Back the pinned observation/reward schema with trusted source-state fixtures and terminal-info behavior. | Can design in parallel; source-backed fixtures need lifecycle and terminal states. | Observation/reward tests named after source state claims. |
| `bonuses` | `verified-narrow`: active `BonusSelfSmall` catch/no-catch/death-order, JS/Python source-env natural spawn/type RNG, default multi-type weight/type RNG, natural game-world and bonus-world retry, cap-at-20 skip, broad Python source-env stack/effect support for self/enemy/all bonuses, runtime optional-array support for `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`, `BonusSelfMaster`, `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`, `BonusEnemyInverse`, `BonusEnemyStraightAngle`, `BonusGameBorderless`, `BonusAllColor`, and `BonusGameClear`, seeded and focused natural public catch/expiry for promoted stack effects including `BonusSelfMaster` and `BonusAllColor`, SelfMaster wall/body parity, AllColor reverse target event order and older-wins overlap behavior, public seeded/natural bonus stack capacity via `SOURCE_MAX_ACTIVE_BONUSES`, partial public natural source-default type selection, same-frame natural bonus/PrintManager random-order protection, replay audit metadata preservation, and a low-level `vector_runtime.py` natural spawn helper with type/position/retry/cap tests | Keep the seeded active `BonusSelfSmall` strict-overlap catch, tangent no-catch, same-tick wall-death no-catch, 7500 ms expiry/restore, `source_bonus_default_weights_type_rng_step.json`, cap-at-20 skip, `BonusGameClear` immediate clear, forced and seeded public `BonusGameBorderless`, source/runtime/public borderless-expiry, partial public natural spawn, `BonusSelfSlow` runtime speed restore, `BonusEnemyBig` enemy-radius restore, `BonusSelfMaster` invincible/printing restore and wall/body parity, `BonusAllColor` color rotate/restore plus overlap order, public seeded/natural stack capacity, and low-level spawn-helper tests as regression guards. Natural source-default type selection must not imply broad natural catch support until the broader natural matrix lands. | Stack math/expiry ordering beyond the promoted restore cases, natural spawned catch/clear coupling for all source-default types, full public natural bonus support, full replay/final state, broader 3P/4P lifecycle/leave, and visual/pixel parity are still open. | Bonus source batches and later vector/public acceptance for promoted effects. |
| `broader vector semantics` | `partial` | Widen only from promoted source fixtures: row-local RNG, mixed player counts, overflow/truncation, timer rows, broader border/body/collision-order, no-event/debug-event policy. | Needs source claims for lifecycle/spawn/RNG, reset/autoreset, and bonuses. | Vector comparator and B>1 commands under the specific source claim they cover. |
| `real LightZero` | `partial` | Local smoke plus thin `curvyzero_v0_lightzero` registration now exist. Next replace local-only confidence with an installed-runtime config/import smoke, env-manager reset/step check, observation dicts, masks, scalar rewards, done/truncated info, replay, and policy/search row mapping. | Needs reset/autoreset, replay, observation/reward, broader vector row semantics, and installed runtime. | Local registered wrapper tests now; installed-runtime no-train config/import smoke next, then later training smoke. |
| `wire and pixels` | `deferred` | Pin one compressed wire event batch; do browser pixels only after state and wire agree. | Needs stable server state/event traces. | Wire fixture first; screenshot/pixel artifacts later. |

## Promotion Gates

Every new behavior moves through these gates.

| Gate | Required output | Who can run in parallel | Stop rule |
| --- | --- | --- | --- |
| Claim gate | One sentence source claim, source files read, scenario id, expected events/counters. | Many package owners. | Unknown source detail is recorded as `needs-spec`, not guessed. |
| JS-pin gate | Scenario or probe run against original JS source with raw artifact and expected trace. | Many owners if they only add unique scenario JSON. | No Python behavior change is promoted before this gate for a new source rule. |
| Python/common-trace gate | Python runner matches JS through common trace or direct oracle output; targeted batch or focused parity test matches. | One owner per shared code file. | First mismatch is explained or the package remains `JS-pinned`. |
| Neighbor regression gate | Adjacent promoted batches still match when shared runner/normalizer code changed. | Batches can run in parallel after edits land. | Regressions are fixed before widening the claim. |
| Observation gate | Observation fixtures match a verified state/event scenario. | Observation owner can design schema while gameplay continues. | No observation claim outranks state/event evidence. |
| Pixel/browser gate | Screenshot or browser artifact is checked only after matching state/wire trace exists. | Browser owner can prepare harness later. | Pixel failure is render/client work unless it exposes a state mismatch. |

## What Must Be JS-Pinned

These rules need JS oracle evidence before Python/common-trace promotion:

- Varied `step_ms` movement is already Python/common-trace verified with
  explicit numeric tolerance.
- Normal wall exact-edge and just-in/out controls.
- Borderless exact-edge and one-axis corner behavior is now promoted with the
  trail/PrintManager wrap fixture and destination-body skip.
- Head-head/order asymmetry is promoted through Python; death-frame point side
  effects and old-body metadata have narrow coverage, while broader
  emitted-trail body collisions still need JS evidence.
- `natural lifecycle/spawn/RNG` beyond the current pinned slice: broader 4P
  lifecycle beyond first-round spawn and broader
  present/non-present variants still need JS oracle evidence before Python
  promotion. The three 2P core fixtures,
  focused 3P spawn-order fixture, focused 3P warmup/PrintManager-start fixture,
  focused 4P spawn-order fixture, focused 3P present/absent fixture, and
  focused 2P max-score match-end fixture, plus the focused 3P all-dead
  next-round fixture, already have JS oracle evidence and direct Python parity.
- Bonus catch order, bonus stack effects, timers/expiry, borderless bonus
  expiry, and weighted type selection. Forced `BonusGameClear`, seeded public
  `BonusGameBorderless`, runtime/public `BonusSelfMaster` and `BonusAllColor`,
  and public seeded/natural stack capacity are promoted narrowly; broader
  natural catch/effect coverage remains open.
- Wire compression and one authoritative event-batch fixture.

## What Must Be Python/Common-Trace Verified

These are not full-fidelity claims until the Python runner matches JS through
common trace and the matching batch is promoted:

- All server-authoritative state transitions: position, angle, alive, present,
  printing, trail/body counters, score, roundScore, round state, borderless,
  bonus state, and world body count.
- Event order when the source order matters: `position`, `angle`, `point`,
  `die`, `property`, `score:round`, `score`, `round:end`, `game:start`,
  `game:stop`, bonus events, `borderless`, and `clear`.
- Shared trace projection for any newly compared field in
  `src/curvyzero/env/trace_compare.py`.
- Batch manifests for each promoted mechanic family.
- Neighbor batches when edits touch shared movement, body/trail, event, runner,
  or normalizer code.

## Observation And Visual Later

These should wait for verified state/event fixtures:

- `observation_rays_spec`: ego-relative rays, scalars, action mask, player order,
  and shape/range checks from known source-state fixtures.
- Local raster observations: only after ray/global observation semantics are
  stable and state traces provide the same body/trail facts.
- CurvyTron visual stacked-frame input from our own renderer is an intended
  LightZero training target. Scalar/ray single-ego rows are the practical bridge
  while replay and lifecycle metadata settle.
- Multiplayer visual replay must carry visual frame provenance together with
  full wrapper action maps, opponent policy ids, player ids, present/alive
  masks, death order, score vectors, reset/RNG metadata, and terminal/final
  observation policy.
- Browser/client trail visuals: after wire/event batches prove the server stream.
- Pixel screenshots: only for visual/client fidelity or human review, never as
  the first proof of gameplay rules.

## Parallel Work Packages

Each package owns a source claim, scenario ids, expected trace table, and
acceptance commands. Shared files are merge locks, not personal workspaces.

| Package | Claim | First work | Blockers | Shared-file risk |
| --- | --- | --- | --- | --- |
| A | `natural lifecycle/spawn/RNG` JS oracle | Done for the 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p`; keep the oracle and expected event/random-call tables attached to the fixture ids. | None for the current pinned slice. | High: `scenario_runner.js`, scenario schema, event capture, timer control. |
| B | `natural lifecycle/spawn/RNG` Python direct parity | Done for the current 28-fixture slice through `tests/test_source_lifecycle_runner.py`; next work is broader 4P match lifecycle or broader present/non-present variants, not redoing the same parity step. | Package A JS artifact for the current slice; new JS artifacts for broader cases. | High: `source_runners.py`, `trace_compare.py`, batch manifest. |
| C | `production reset/autoreset` | Build natural reset/spawn templates, pre-step timer advance, final transition preservation, horizon truncation, seed history, and public autoreset ordering on top of `src/curvyzero/env/vector_reset.py`. | Needs source-shaped reset/spawn/RNG state; the current pinned slice is verified, but natural spawn/reset templates, autoreset, and vector optimized parity are not. | High: vector state arrays, reset helpers, actor bridge, replay staging. |
| D | `production replay` | Promote from sample/debug chunks to writer/reader with episode ids, reset seed/source, terminal/final observation policy, event/state refs, manifest, and compatibility rejection. | Package C and observation/reward schema hashes. | Medium-high: replay helpers, actor bridge, artifact layout. |
| E | `trainer observation/reward` | Keep schema work parallel, but make source-backed observation fixtures come from verified state claims. Wire sparse reward and terminal info into trainer-facing rows. | Source-state fixtures from A/B for natural starts and terminal states. | Medium: observation helpers, reward helpers, trainer wrapper. |
| F | `bonuses` | Continue from the promoted active `BonusSelfSmall` catch/no-catch/death-order, one-type spawn RNG, default multi-type weight/type RNG, game-world retry, expiry/restore, and forced `BonusGameClear` clear slices; split one claim per remaining bonus effect, cap/probability rule, other bonus type, and vector/runtime rule. | Row-local RNG policy from A/B. | High: movement, body/trail, timers, events, random calls. |
| G | `broader vector semantics` | Add row-local RNG arrays, mixed-P policy, timer rows, overflow/truncation, and vector comparators only for promoted source claims. | A/B/C/F depending on mechanic. | High: vector step, event rows, benchmarks. |
| H | `real LightZero` | Local no-train smoke and thin DI-engine registration exist. Next prove the same boundary in the installed LightZero runtime, then keep single-action ego rows mapped explicitly to CurvyZero wrapper action maps, which are converted to source control state. | C/D/E/G and real runtime availability. | Medium: wrapper boundaries, policy/search mapping, replay metadata. |
| I | `wire and pixels` | Pin one compressed wire stream fixture after state events are stable; defer browser pixels until wire/state agree. | Stable source state/event traces. | Medium for wire; high setup for browser. |

## Dependency Shape

Work can run in four parallel streams:

1. Source-claim stream: Package A and B are done for the current 28-fixture
   lifecycle/spawn/RNG slice including `source_lifecycle_spawn_rng_4p_next_round`,
   `source_lifecycle_survivor_score_4p_next_round`,
   `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
   `source_lifecycle_multi_round_match_end_3p`.
   Next source workers should add one broader JS
   artifact at a time, then promote that same named claim through Python.
   Bonus source reading can proceed, but bonus implementation waits for the
   row-local RNG contract.
2. Reset/replay stream: Package C can prepare API shape now, but production
   reset values must wait for natural lifecycle/spawn/RNG. Package D can draft
   replay metadata now and wire it after C.
3. Observation/reward stream: Package E can keep schema and analytic tests
   moving now, then replace toy/debug evidence with trusted source-state
   fixtures from A/B and terminal states from C.
4. Vector/LightZero stream: Package G can widen only from promoted source
   claims. Package H remains a real-runtime boundary task and should not absorb
   source-fidelity language until C/D/E/G are stable.

The practical split is "claim writing and fixture design in parallel, shared
runner edits in sequence." Source reading, expected event tables, scenario
naming, and acceptance command drafts are parallel. Edits to the JS oracle,
Python runner, common trace projection, vector row state, and replay writer need
a queue.

## Merge Locks

Only one package owner should edit each of these at a time:

- `tools/reference_oracle/scenario_runner.js`: JS source setup, fake timers,
  random tape, event capture, and snapshot fields.
- `src/curvyzero/fidelity/source_runners.py`: Python source-fidelity semantics
  and runner dispatch.
- `src/curvyzero/env/trace_compare.py`: common-trace projection and compared
  fields.
- `tools/run_fidelity_loop.py` and `tools/run_fidelity_batch.py`: artifact and
  batch behavior.
- `src/curvyzero/env/scenario_schema.py`: scenario input contract.
- Any one batch manifest under `scenarios/environment/*_batch.json`.
- `docs/working/environment/active_lanes.md` and
  `docs/working/environment/source_feature_inventory.md`: short current-memory
  docs should have one final status writer.

Usually safe in parallel:

- New scenario JSON files with unique ids.
- Docs-only fixture specs under `docs/working/environment/`.
- Source-map notes under `docs/research/curvytron_source_map/` when they do not
  rewrite shared summaries.
- Artifact inspection in `/private/tmp/...`.
- Running already-promoted regression batches after code edits settle.

## Package Output Contract

Each work package should leave the same paper trail:

1. `Claim`: one sentence naming the source rule.
2. `Source`: source files or source-map note read.
3. `Scenario`: one deterministic fixture id and setup table.
4. `JS oracle`: artifact path, event/counter expectation, and status
   `JS-pinned` when Python parity is not done yet.
5. `Python/common trace`: runner id, compared fields, tolerance, and batch
   command.
6. `Promotion`: batch path, command output attached to the claim, unsupported
   list, and neighboring regressions run.
7. `Docs`: update this plan only if dependencies change; update
   `coverage_tracker.md` and `source_feature_inventory.md` when a status changes.

## Milestones

### Milestone A: Natural Lifecycle/Spawn/RNG Is Verified

Exit criteria:

- The 28 promoted lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p`
  have JS oracle evidence and direct
  Python parity: three 2P fixtures, one focused 3P spawn-order first-round
  fixture, one focused 3P warmup/PrintManager-start fixture, one focused 4P
  spawn-order first-round fixture, one focused 3P present/absent first-round
  fixture, one focused 3P present/absent next-round fixture, one focused 2P
  max-score match-end fixture, one focused all-present 3P `max_score: 2`
  match-end fixture, one focused 3P all-dead next-round fixture, one focused
  3P survivor-scoring `round:end` fixture, one focused 3P survivor
  warmdown/next-round fixture, one focused 3P tie-at-max continuation fixture,
  and one focused 3P all-present multi-round match-end fixture.
- The claim records what remains outside it, especially broader 4P match
  lifecycle, broader present/non-present
  variants, vector lifecycle, trainer/replay/final
  observation, reset/autoreset follow-through, and bonus RNG.

### Milestone B: Reset/Autoreset And Replay Are Honest

Exit criteria:

- Production reset builds on the verified lifecycle/spawn/RNG state shape and
  `src/curvyzero/env/vector_reset.py` where they apply, without claiming
  natural spawn, autoreset, or broader lifecycle.
- Public autoreset preserves terminal transition data before row reset.
- Horizon truncation and seed history are explicit.
- Replay writer/reader stores terminal/final observation policy, seed metadata,
  event/state refs, and schema/rules hashes.

### Milestone C: Observation/Reward Are Source-Backed

Exit criteria:

- Observation fixtures are tied to trusted source-state snapshots, not browser
  pixels or toy-only states.
- Reward and terminal info are wired through the trainer-facing surface.
- LightZero-shaped local smoke remains labeled as adapter plumbing unless real
  DI-engine/LightZero registration is available.

### Milestone D: Bonuses Are Boring

Exit criteria:

- Forced catch order is verified.
- One fixture each verifies speed, radius, inverse, public borderless expiry,
  color, and timed expiry behavior. Source/runtime borderless expiry and the
  first clear-trails fixture are already promoted narrowly.
- Bonus random spawn/type selection uses the same random stream policy.

### Milestone E: Vector And Interface Are Honest

Exit criteria:

- Broader vector semantics are widened only from promoted source claims.
- Mixed player-count, overflow/truncation, row-local RNG, timer rows, and
  no-event/debug-event policy have explicit contracts.
- Real LightZero adapter checks pass against the real boundary when the runtime
  exists.

### Milestone F: Wire And Browser/Pixel Review

Exit criteria:

- One compressed event batch is JS-pinned and matched to verified gameplay
  state.
- Browser/client checks are tied to a verified state/wire scenario.
- Screenshots or pixels are used only to assess render/client fidelity.
- Any visual mismatch is triaged as client/render unless it points back to a
  failing state/event trace.

## Completed Parallel Wave

This wave has landed. Keep its outputs as regression evidence and do not list
its agents as active:

| Slot | Package | First deliverable | Shared-file rule |
| --- | --- | --- | --- |
| A | P1 movement clock | Done: `source_kinematics_turn_multistep` is promoted. | Scenario-only first; runner edit after JS pin. |
| B | P2 borderless trails | Done: PrintManager wrap, destination-body skip, and exact-edge/corner-axis controls are promoted. | Shared runner edits should serialize with collision-order work. |
| C | P3 collision order | Done: death-point and head-head/order fixtures are verified via `source_collision_order_batch.json`; the active printing and already-hole PrintManager death-stop fixtures are verified separately in `source_print_manager_batch.json`. | Serialize `source_runners.py` edits with borderless work. |
| D | P7 random stream | Done: random/bonus probe plan exists, and the PrintManager call-order tape is promoted through Python/common trace. | Serialize `scenario_runner.js` edits with timer/lifecycle work. |
| E | P11 observations | Done: observation gates drafted from verified fixtures. | Docs-only done; no env API changes yet. |

This gave the project five useful pieces of evidence without five agents
colliding in the same Python file.

## Previous Active Wave Complete

| Slot | Owner | First deliverable | Shared-file rule |
| --- | --- | --- | --- |
| G | Worker G | Done: promoted the varied elapsed-ms kinematics fixture. | Kinematics batch is the regression guard. |
| H | Worker H | Done: added the first fixture-seeded array comparator. | Use verified fixture state as input; do not start a production backend rewrite. |
| I | Explorer I | Done: pinned head-head/order in the JS path. | Worker K promoted the Python path. |
| J | Worker J | Done: cleaned active docs after the completed wave. | Docs only; no code or scenario edits. |
The old main-thread verification for this context is superseded by the K/L/N
verification below.

## Completed K/L/N Wave

This wave has landed. Keep its outputs as regression and speed-lane evidence:

| Slot | Owner | First deliverable | Shared-file rule |
| --- | --- | --- | --- |
| K | Worker K | Done: promoted head-head/order to Python/common trace. | Collision-order batch is the regression guard. |
| L | Worker L | Done: extended the vector comparator to simple borderless wrap. | Keep unsupported source semantics explicit. |
| N | Worker N | Done: added CPU policy/search batch stand-in and speed docs. | Stand-in is shape/copy evidence, not Mctx/GPU proof. |

The K/L/N verification was superseded by the final-wave verification below.

## Completed P/Q/R/O Wave

| Slot | Owner | First deliverable | Shared-file rule |
| --- | --- | --- | --- |
| P | Worker P | Done: added vector comparator normal-wall death support. | Use verified fixture state; no production backend rewrite. |
| Q | Worker Q | Done: promoted borderless exact-edge/corner-axis fidelity fixture. | Keep the claim narrow and batch-backed. |
| R | Worker R | Done: made the Modal/GPU/JAX-Mctx smoke path concrete. | Worker U later ran it on a Modal L4; label speed evidence separately from fidelity evidence. |
| O | Worker O | Done: kept active docs current for this wave. | Docs only; no code or scenario edits. |

Acceptance output is not a fidelity headline. After source-claim or vector
edits, rerun the relevant acceptance command and record the exact output beside
the claim it protects. Repository-wide test totals and lint stay as local
hygiene notes, not progress claims.

## Completed Speed Wave

See [active_lanes.md](active_lanes.md) for the current work map. The latest
completed wave was speed work with fidelity guardrails:

- Worker A: fixed event arrays for the first 8 supported vector fixtures.
- Worker B: first `B>1` supported batch-row timing.
- Worker C: small Modal/JAX/Mctx sweep.
- Worker D: docs reorientation.
- Worker E: architecture critique.
- Worker F: batched event rows in the `B>1` benchmark path.
- Worker G: first PrintManager no-toggle vector slice.
- Worker H: debug observation/reward packing benchmark.
- Worker I: active PrintManager body-collision source fixture.
- Worker J: exact-zero PrintManager source fixture.
- Worker K: PrintManager hole-to-print vector slice.
- Worker L: batched event writer speedup.
- Worker M: first actor-loop bridge benchmark.
- Worker N: observation-shaped Modal/JAX/Mctx synthetic benchmark.
- Worker P: vector PrintManager print-to-hole and exact-zero support.
- Worker Q: actor bridge rollout blocks and replay byte reporting.
- Worker R: Modal/JAX/Mctx host setup, first device placement, and steady
  search timing split.
- Worker O: delayed `3000 ms` PrintManager source fixture.
- Worker T: vector PrintManager death-stop support.
- Worker U: no-event/debug-event cost split.
- Worker V: fixture-seeded CPU debug-packer output to Modal/JAX/Mctx boundary.
- Latest vector comparator update: delayed-start PrintManager support for the
  narrow fixture comparator. This is not broad reset/timer/autoreset support.
- Latest reset/autoreset slice: `src/curvyzero/env/vector_reset.py` has a
  production-facing masked reset boundary. It snapshots selected terminal rows,
  copies selected reset-template rows, increments `episode_id`, stamps reset
  metadata, clears terminal/event/timer-fired fields where present, preserves
  skipped rows, and returns reset metadata plus terminal snapshot. It does not
  spawn, schedule timers, autoreset, preserve final obs, or write replay.
- Latest lifecycle/spawn/RNG parity: 16 lifecycle fixtures including
  `source_lifecycle_spawn_rng_4p_next_round` and
  `source_lifecycle_multi_round_match_end_3p` now have direct Python parity
  against the JS oracle through
  `tests/test_source_lifecycle_runner.py`. This is not broad lifecycle or vector
  optimized parity.

## Regression Batch Set

Run the relevant targeted batch for each package. When shared runner or
normalizer code changes, also rerun the promoted neighbors:

```bash
uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --fail-on-mismatch --artifact-root /private/tmp/curvy-source-kinematics-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-border-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-regression
uv run python tools/run_fidelity_batch.py scenarios/environment/source_collision_order_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-collision-order-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_batch.json --python-runner source-trail-cadence-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-cadence-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_gap_batch.json --python-runner source-trail-gap-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-gap-regression
```

## Deep Backlog By Layer

### Server Gameplay

| Layer | Status | Next gate |
| --- | --- | --- |
| Source update order | `verified` | Stress with P3 death-frame and head-head fixtures. |
| Movement constants, fixed-60Hz controls, and varied elapsed-ms | `verified` | Bonus-modified movement belongs with P9 bonus core. |
| Normal wall basics | `verified` | Exact-edge and just-in/out controls. |
| Borderless plain wrap, first trail wrap, destination-body skip, and exact-edge/corner-axis | `verified` | Next-frame second-axis wrap only if a later feature needs it. |
| Stored body collision and own latency | `verified` | P5 old age and emitted body variants. |
| Same-frame point materialization | `verified` | P3 death-frame point side effect. |
| PrintManager forced toggles, delayed start, and active death stop | `verified` | Broader timer/lifecycle behavior stays separate. |
| Normal trail cadence and first gap cases | `verified` | Multi-step and emitted-trail gap variants. |
| Frame scoring narrow cases | `verified` | P3/P6 event-order and lifecycle stress. |
| Round lifecycle | `verified-narrow` | 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p` have direct JS/Python parity for the focused 2P/3P/4P slice. Next gates are broader 4P match lifecycle and broader present/non-present variants. |
| Randomness | `partial` | PrintManager call-order tape is promoted; broader random stream, spawn, and bonus behavior remain. |
| Natural spawn | `verified-narrow` | Current 2P spawn order, x/y RNG, accepted first-attempt headings, one heading rejection retry, no spawn bodies, next-round spawn, focused 3P first-round spawn order, one 3P non-present spawn skip, and focused 4P first-round spawn order/RNG labels are verified; broader lifecycle remains. |
| Bonuses | `verified-narrow` | First active `BonusSelfSmall` catch/no-catch/death-order has JS/Python source-env parity. Natural one-type and default multi-type spawn/type RNG, game-world and bonus-world retry, and cap-at-20 skip have narrow parity. Runtime/public support now includes the promoted seeded/focused natural stack effects, including `BonusSelfMaster` and `BonusAllColor`, SelfMaster wall/body parity, AllColor reverse target event order and older-wins overlap behavior, and public seeded/natural stack capacity uses `SOURCE_MAX_ACTIVE_BONUSES`. Broader stack math/expiry ordering, full natural catch/effect coverage, full replay/final state, and broader vector/runtime support remain open. |

### Wire, Browser, And UI

| Layer | Status | Next gate |
| --- | --- | --- |
| Socket event names | `source-read` | P10 one compressed authoritative event batch. |
| Compression | `source-read` | Include in P10; do not mix with gameplay rule changes. |
| Browser client prediction | `deferred` | Reopen after state/wire scenario passes. |
| Canvas rendering and sounds | `deferred` | Screenshot review only after state/wire. |
| Old app build/runtime | `deferred` | Disposable browser host only when render fidelity is requested. |

### Training And Observation

| Layer | Status | Next gate |
| --- | --- | --- |
| Source state for trainer | `deferred` | Wait for Milestone A/B. |
| Observation rays | `source-read` | Later schema/tests from verified scenarios. |
| Raster/pixel observation | `deferred` | After ray/global observation and state evidence. |
| Reward schema | `needs-spec` | Separate source score from trainer reward after lifecycle is pinned. |
| Vector/batch env | `deferred` | After single-env state, observation, and reward contracts. |

## Links

- Live lane map: [active_lanes.md](active_lanes.md)
- Feature inventory: [source_feature_inventory.md](source_feature_inventory.md)
- Coverage tracker: [coverage_tracker.md](coverage_tracker.md)
- Probe backlog: [probe_backlog.md](probe_backlog.md)
- Collision order spec: [collision_order_probe_plan.md](collision_order_probe_plan.md)
- Borderless trail spec: [borderless_trail_probe_plan.md](borderless_trail_probe_plan.md)
- Strategy review: [reconstruction_strategy_review.md](reconstruction_strategy_review.md)
- Workflow contract: [../../design/environment/reconstruction_workflow.md](../../design/environment/reconstruction_workflow.md)
