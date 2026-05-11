# Environment Active Work

Status: live work tracker
Date: 2026-05-11

Use this as the short current-state map. Stable contracts live in
`docs/design/environment/`; detailed evidence lives in the tracker, execution
plan, source inventory, and experiment notes.

Fast recovery packet: [reorientation_packet.md](reorientation_packet.md).
Source claim tracker: [coverage_tracker.md](coverage_tracker.md).
Shared source/vector transition contract:
[EnvironmentTransitionV0](../../design/environment/environment_transition_v0.md).
Full fidelity matrix: [full_fidelity_spec_matrix_2026-05-09.md](full_fidelity_spec_matrix_2026-05-09.md).
Full environment spec: [full_environment_spec_2026-05-09.md](full_environment_spec_2026-05-09.md).
Full execution map: [full_fidelity_execution_plan.md](full_fidelity_execution_plan.md).
Remaining gap catalog: [remaining_gap_catalog_2026-05-10.md](remaining_gap_catalog_2026-05-10.md).
Multiplayer gap targets:
[multiplayer_env_gap_targets_2026-05-10.md](multiplayer_env_gap_targets_2026-05-10.md).
Optimizer handoff:
[optimizer_handoff_2026-05-10.md](optimizer_handoff_2026-05-10.md).
CurvyTron one-shot remaining spec:
[full_curvytron_one_shot_spec_2026-05-10.md](full_curvytron_one_shot_spec_2026-05-10.md).
Optimizer visual tensor handoff:
[optimizer_visual_tensor_handoff_2026-05-10.md](optimizer_visual_tensor_handoff_2026-05-10.md).

## Plain Current State

- We are not at full CurvyTron environment fidelity.
- There is one public runtime name under hardening:
  `VectorMultiplayerEnv`.
- Strict `VectorTrainerEnv1v1NoBonus` is only a proof/profiling boundary. It is
  not the destination.
- `VectorMultiplayerEnv` is the intended public runtime name. The
  old "NoBonus" suffix was historical wording and has now been removed from the public runtime name; do not describe it
  as a separate second product implementation.
- Cleanup decision: product direction is one fast source-faithful runtime, not
  two implementations. Public-env cleanup should consolidate behavior into this
  runtime path and remove confusion from historical naming.
- Treat it as a narrow metadata/public-state surface today, not a trainer-ready
  env and not a full CurvyTron fidelity claim. It has partial public seeded and natural bonus slices only.
- Latest focused environment validation after the bonus stack-capacity,
  SelfMaster wall-parity, AllColor stack-order, generated-RNG extension, and
  natural timer callback-cap removal and natural bonus effect coverage reported `282 passed`; focused source
  bonus validation reported `33 passed`.
  Ruff and the environment doc guard also passed. Keep this as a freshness
  note, not a status scoreboard.
- The largest gaps are full bonus breadth, timer/random ordering for public
  bonus scheduling, broader 3P/4P lifecycle/public parity, full public replay
  and final state, old toy-path quarantine, and final cleanup. Bonus replay is
  still metadata/audit only, not full replay arrays.
- Latest public base hardening: direct public body/trail/collision canary
  tests, the long 2P reset-to-terminal source rollout, warmdown/match-end
  checks, the 2P metadata replay bridge, and the source-state LightZero wrapper
  fixed-opponent sidecar proof.
- Latest validation reported on 2026-05-11: focused public runtime/replay/
  source-state route tests passed on the touched set, the source bonus suite
  reported `33 passed`, focused validation reported `282 passed`, ruff passed,
  and the environment doc guard passed.
- Broad environment-suite status: the stale public metadata/replay
  unsupported-audit claim is retired. Focused checks now require all
  source-default runtime-supported effects, including `BonusSelfMaster`,
  `BonusAllColor`, and `BonusEnemyStraightAngle`, to stay out of unsupported
  audit metadata.
- Halley's capacity audit: seed-generated public random tape auto-extends
  deterministically. Seed-generated natural bonus position retry is no longer
  capped by `natural_bonus_position_attempt_capacity`; that setting is a
  chunk/fixture limit, not a source-fidelity stop for generated rows.
  Fixture/direct finite tapes remain strict and can exhaust by design. The
  public natural bonus timer no longer has an artificial callback cap. The
  remaining capacity question is mostly policy: artificial/manual bonus stack
  overflow is an intentional fixed-array guard for bad or undersized direct
  runtime fixtures, not a public natural/seeded env bug. A fully blocked
  generated map may still need policy if retries never find a position.
  Truncation-by-design includes `max_ticks`, body overflow, and event overflow.
- Archimedes's confirmed `BonusSelfMaster` wall-death bug is fixed in this
  checkout. Source and public runtime regressions now prove that `SelfMaster`
  invincibility blocks body/trail death but not normal-wall death.
- Multi-target `BonusAllColor` event order and overlapping non-additive color
  stack precedence are fixed in this checkout. Source/public tests now prove
  reverse target event order and source older-wins behavior until the older
  stack expires.
- Latest Modal route evidence on 2026-05-11: CPU `dry` plus a tiny CPU command
  for `source_state_fixed_opponent` reached `VectorMultiplayerEnv`,
  logged fixed-opponent action telemetry, and reported non-ALE,
  non-browser-pixel source-state tensors. This is route evidence only. It is
  not the next priority and not an environment-fidelity claim.
- Visual size wording: the source 2P arena is 88 source units from
  `CurvyTronReferenceDefaults.arena_size_for_players(2)`. The 64x64 size is
  our learned raw observation raster from source state, not the original game
  arena size.
- Latest 2P source-state visual check: `scripts/compare_2p_raw_visual_observation.py`
  compares source-shaped state against `VectorMultiplayerEnv` gray64. On
  2026-05-11 the long no-bonus wall fixture matched through terminal: 112
  frames, exact `max_abs_diff=0`, `mismatch_pixels=0`, and original-JS reset
  source-state check pass. This is still not browser/canvas pixel parity.
- Gray64 v0 distinguishes 2P player trails and heads, but every active map
  bonus is value `208`. Keep that caveat visible until a typed bonus visual
  schema exists.
- Cleanup patches landed: stale lifecycle metadata overlay plumbing was removed
  from `vector_multiplayer_env.py`, and the dead legacy scalar replay builder
  was removed from `multiplayer_replay_v0.py`.
- Seeded public bonus slice landed narrowly: public tests now cover default
  bonus support off, seeded `BonusSelfSmall` catch/no-catch/expiry restore,
  seeded `BonusSelfMaster` and `BonusAllColor` catch/expiry restore, and
  seeded `BonusGameClear` immediate clear.
- Latest bonus follow-up: the source oracle has broad Python bonus stack/effect
  support for self/enemy/all bonuses; `vector_runtime` now has table-backed
  optional-array support for `BonusSelfSmall`, `BonusSelfSlow`,
  `BonusSelfFast`, `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`,
  `BonusEnemyInverse`, `BonusEnemyStraightAngle`, `BonusGameBorderless`, and
  `BonusSelfMaster`, `BonusAllColor`, and `BonusGameClear`. Focused natural
  bonus tests now pass in `VectorMultiplayerEnv` for every source-default
  effect family: self small/slow/fast/master, enemy slow/fast/big/inverse/
  straight-angle, game borderless, all color, and game clear.
  public seeded `BonusSelfSmall`/`BonusGameClear`/`BonusGameBorderless`
  includes public borderless expiry, and public seeded
  `BonusEnemyStraightAngle` now carries angular-velocity stack state through
  catch/expiry. Public seeded/natural `BonusSelfMaster` carries
  invincible/printing stack state through catch/expiry, and public
  seeded/natural `BonusAllColor` rotates/restores avatar color state through
  catch/expiry. Public natural
  source-default type selection and same-frame natural bonus plus PrintManager
  random-order accounting are protected; do not narrow source defaults. Metadata
  replay preserves bonus metadata and audit fields only. This is focused public
  natural catch/effect coverage, not broad long-run bonus replay/final-state or
  stack/death stress coverage. Public seeded and focused natural bonus stack
  capacity now uses `SOURCE_MAX_ACTIVE_BONUSES`. Capacity
  policy still needs promotion for manual/direct stack guard wording and maybe
  a fully blocked generated map if retries never find a position.
  Remaining gaps include stack/death interactions beyond the fixed
  `BonusSelfMaster` wall/body guard, full replay arrays/final state, broader
  3P/4P lifecycle/leave, visual pixel parity, and final cleanup.
- Docs worker status: this is the only active docs lane now. Keep updates
  concise, and avoid broad refactors unless they directly remove confusion or
  unblock parity tests.

Why this is taking long: we are no longer debating versions or searching for a
different runtime. We are hardening one public runtime, and the remaining work
is specific feature work plus source-backed proof: bonuses, lifecycle, replay
and final observations, public metadata, and later browser pixel parity.

## 2P Status

Full 2P fidelity is not done. The public base core is much stronger after
direct public body/trail/collision canaries, the long reset-to-terminal public
check `test_2p_public_reset_to_terminal_matches_source_long_wall_fixture`,
focused warmdown/match-end checks, and the 2P metadata replay bridge. This is
not full gameplay, trainer replay, trainer observations, real two-seat training,
bonus behavior, full replay/final observations, or visual/browser pixel parity.
The source-state route proof is fixed-opponent plumbing only, not an
environment-fidelity claim. Do not describe the game as 64x64: for 2P the
source arena is 88 units, while 64x64 is only the learned raw observation
raster size.

## Current Direction

Main priority: harden one fast, source-faithful CurvyTron runtime:
`VectorMultiplayerEnv`. It is the
current intended public runtime name, not a separate second product path. The
working proof path is:

```text
source claim -> JS oracle/probe -> CurvyTronSourceEnv parity -> optimized parity
```

Native CurvyTron should be described as held control state advanced through
elapsed-ms frames. `step` and `joint_action` are wrapper/API terms; do not let
them replace the source mental model.

Strict `VectorTrainerEnv1v1NoBonus` is a fixed decision wrapper over source
control state. It accepts trainer action ids, maps them to native source moves,
and holds that control state for `decision_ms`; it is not native discrete
simultaneous actions and not the product runtime.
Its restrictions are temporary explicit non-fidelity profile choices, not the
reconstruction path. Reconstruct source-default CurvyTron behavior in
`VectorMultiplayerEnv`.

Wrapper-facing action rows can stay as sidecar metadata while environment rules
are reconstructed. They do not replace native source semantics and they are not
the current priority.

`CurvyTronSourceEnv` and the JS oracle are proof/oracle tools while source
rules move into `VectorMultiplayerEnv`. They are not alternate product
environments. Speed/vector work must directly promote or measure verified source
behavior on the intended runtime path.

Do not let fixture counts, pass counts, Modal runs, or rows/sec become the goal.
The 1v1/no-bonus slice is a useful trainer-boundary proof only; multiplayer
2P/3P/4P behavior, scoring, presence/alive edges, match lifecycle, bonuses,
full env reset/autoreset semantics, replay, and policy-row mapping still need
source-backed production treatment.

Current execution wave:

- Environment owns source truth and reconstruction: public multiplayer
  lifecycle, reset/autoreset/final-observation policy, row-local RNG provenance,
  replay metadata, masks/rewards, and bonus promotion.
- Optimizer owns visual smoke/profiling and LightZero adapter plumbing.
  Environment owns only whether visual tensors are source-faithful.
- Latest landed pieces: multiplayer final-row metadata, explicit public
  `autoreset_done_rows`, metadata-only multiplayer replay record/chunk
  packaging plus recorder, optional-array fast-runtime `BonusSelfSmall`
  catch/expiry and forced `BonusGameClear` clear, seeded public bonus fixture
  support for `BonusSelfSmall`, `BonusGameClear`, and `BonusGameBorderless`,
  visual truth/schema metadata, local/
  installed no-train scalar LightZero smoke boundaries, and the source-state
  LightZero wrapper fixed-opponent sidecar proof.
- Active Environment implementation targets are deeper bonus stack/death
  interactions; fuller bonus metadata/replay audit without claiming full replay
  arrays; full public replay/final observations; broader lifecycle/multiplayer
  parity including 3P/4P leave; browser pixel parity later; toy-path
  quarantine; and final cleanup.
- The fixed-opponent source-state route smoke is landed and has a tiny CPU Modal
  proof. Treat it as route evidence only. The low-level natural bonus spawn
  helper is landed, and focused public natural source-default catch/effect paths
  now have `VectorMultiplayerEnv` tests. This is still not broad long-run bonus
  replay/final-state support.
- Bonus guardrail: current runtime bonus work stays optional-array only and is
  now table-backed for the landed runtime effects: self small/slow/fast/master,
  enemy slow/fast/big/inverse/straight-angle, all color, game borderless, and
  game clear. Public natural spawn can select and catch source-default bonus
  types in focused tests, but this must not imply broad replay/final-state or
  stack/death stress coverage. Keep
  natural bonus and replay claims separate until the corresponding contracts
  exist.
- Current scalar/ray 1v1 checks are a guardrail, not the destination.
- Old toy paths are quarantined as historical smoke/interface evidence only.
  They must not be used as environment-fidelity, bonus, replay, or public
  runtime evidence.

Boundary with Optimizer: optimizer owns timing, bottleneck reads, CPU/GPU and
Modal/process decisions. Environment owns whether the measured env path is
faithful enough to time. The current optimizer-safe path is strict
`VectorTrainerEnv1v1NoBonus` plus the source-backed
`CurvyTronSourceEnv -> source_snapshot_to_vector_trainer_state(...)` timing
surface when the report says so explicitly. These are proof/profiling surfaces
for the intended `VectorMultiplayerEnv` runtime, not full environment
fidelity and not separate product envs.

Visual tensor split with Optimizer: Environment owns source-fidelity/pixel
truth, visual schema meaning, source/browser comparison, metadata, final
observation policy, and promotion gates. Optimizer owns debug visual smoke,
profilers, LightZero adapter plumbing, batching, Modal/GPU/CPU alternatives,
and bottleneck reads after the tensor contract is named. Existing JS reference
tooling under `tools/reference_oracle` and `tools/js_reuse_probe` can produce
golden source-state snapshots. There is not yet a finished browser/canvas pixel
golden-frame harness.

Route-smoke boundary: the current Modal/source-state command
`source_state_fixed_opponent` goes through a source-state wrapper backed by
`VectorMultiplayerEnv`. It is fixed-opponent, not ALE, not browser
pixels, and not environment fidelity. The scalar/ray single-ego contract is a
diagnostic sidecar. The tiny CPU Modal smoke proves the route can reach the
right env identity and telemetry; it does not change the next environment
priority.

Current fast-path snapshot:

- `src/curvyzero/env/vector_runtime.py::step_many` is the supported
  fixture-backed, source-ordered CPU transition kernel.
- Fast bonus runtime slices landed narrowly: optional arrays can drive
  table-backed catches and expiry/restore for `BonusSelfSmall`,
  `BonusSelfSlow`, `BonusSelfFast`, `BonusEnemySlow`, `BonusEnemyFast`,
  `BonusEnemyBig`, `BonusEnemyInverse`, `BonusEnemyStraightAngle`,
  `BonusSelfMaster`, `BonusAllColor`, `BonusGameBorderless`, plus immediate
  `BonusGameClear` main-world clear.
  `vector_runtime.py` also has a low-level
  natural bonus spawn helper with type/position/retry/cap tests. Public seeded
  and focused natural bonus stack capacity uses `SOURCE_MAX_ACTIVE_BONUSES`.
  No optional arrays means the old no-bonus path. Focused public natural
  support now covers source-default catch/effect families, but it is not broad
  replay, final-state support, stack/death stress coverage, or a broad bonus-system
  claim.
- `scripts/benchmark_vector_batch_rows.py` sends normal benchmark calls through
  that public runtime. Its private `_step_many_kernel(..., phase_timers=...)`
  path is only for benchmark diagnostics, and the dead duplicate old benchmark
  body has been removed.
- The vector runtime now marks optional `done`, `terminated`, `reset_pending`,
  `terminal_reason`, `draw`, and `winner` row arrays after survivor or draw
  terminal events when those arrays exist.
- `vector_lifecycle.run_warmup_start_step_1v1_no_bonus_rows` composes the
  strict 1v1/no-bonus reset/spawn/warmup/timer/runtime step slice. A focused
  test proves wall-death terminal state plus the real vector trainer
  final-observation/reward handoff into autoreset planning.
- `src/curvyzero/env/vector_trainer_observation.py` now builds the narrow
  1v1/no-bonus vector trainer observation surface: `float32[106]` ego rays from
  vector body circles, `float32[B,2,106]` final observations, and sparse final
  reward maps from `terminal_reason`/`winner`/`draw`.
- `src/curvyzero/env/vector_runtime.py` now has `print_manager_mode =
  "natural_toggle"` for public base stepping: live rows use natural
  print/hole toggling, and wall/body deaths also run PrintManager death
  cleanup.
- `src/curvyzero/env/vector_autoreset.py::apply_autoreset_rows(...)` now stages
  final observation/reward through `plan_autoreset_rows(...)`, then mutates
  selected rows via `vector_reset.reset_arrays(...)` while preserving terminal
  snapshot-before-reset ordering.
- `src/curvyzero/env/vector_trainer_env.py` now exposes the narrow public
  `VectorTrainerEnv1v1NoBonus` API. It owns B rows of vector state, returns real
  `float32[B,2,106]` trainer observations, maps trainer actions to source moves,
  stages terminal final observation/reward before autoreset, and resets/spawns/
  warms only selected done rows. Reset and step info include
  `native_control_model_id`, `trainer_control_wrapper_id`, and `decision_ms`.
- The public env now applies narrow horizon truncation in the live step path:
  active rows increment `episode_step`, rows at `max_ticks` end with
  `terminated=false`, `truncated=true`, `done=true`,
  `terminal_reason=timeout_truncated`, zero reward, and the same
  final-array/autoreset handoff used by source terminal rows.
- `src/curvyzero/training/vector_env_replay_recorder.py` now records live
  `VectorTrainerEnv1v1NoBonus.step(...)` batches into replay-v0 chunks with the
  returned trainer observations, rewards, actions, policy/search side inputs,
  terminal final arrays, and replay metadata defaults from returned env info for
  the strict 1v1/no-bonus slice.
- The strict public env now applies overflow truncation and terminal metadata in
  the same narrow public handoff: rows that exceed supported capacity terminate
  as truncations with explicit terminal reasons, truncation reason labels, final
  arrays, and replay-visible metadata rather than silent partial writes.
- The live-step recorder uses the terminal barrier replay policy for this slice:
  once any row in the returned batch is final, the chunk closes at that barrier
  so terminal final arrays and post-reset observations are not mixed as one
  continuous live segment.
- The long source-fidelity bridge now has two public-vector checks.
  `test_public_vector_env_matches_source_long_1v1_wall_round_done_terminal_step`
  seeds from the source penultimate frame and checks the terminal step.
  `test_public_vector_env_reset_to_terminal_matches_source_long_1v1_fixture`
  uses the source fixture random tape plus exact warmup policy and compares
  public reset/spawn/warmup through terminal for the same long wall-round-done
  fixture.
- Planck's earlier source-fidelity bridge test
  `test_public_vector_env_matches_source_long_1v1_wall_round_done_terminal_step`
  remains useful as a narrow terminal-step regression.
- Safe timer diagnostic cleanup has landed: benchmark-only timer instrumentation
  stays behind private diagnostic paths and normal `vector_runtime.step_many`
  callers use the public runtime surface.
- The source-pinned bridge-test wave has landed: same-frame wall draw replay,
  borderless wrap destination-body/next-frame kill, borderless PrintManager
  wrap toggle, collision-order batch support, direct public body/trail/collision
  canaries, seed/reset metadata across autoreset into recorder chunks, an
  optional strict replay/profile manifest, 3P/4P runtime wall-scoring canaries,
  one narrow 3P warmdown/next-round helper, a metadata-only
  `VectorMultiplayerEnv` public surface, the 2P metadata replay bridge,
  and both long 1v1 wall-round-done public-vector bridges: terminal-step and
  reset-to-terminal. Keep this framed as source-pinned bridge evidence only,
  not broad lifecycle, bonuses, learned 3P/4P observations, visual LightZero,
  or full CurvyTron.
- Cleanup note: lifecycle metadata now comes from the public env state path
  without the stale overlay plumbing, and old scalar replay builder code has
  been removed. This is maintenance, not a new training claim.
- This is not full environment fidelity. First gaps remain: broader natural
  bonus catch/effect matrix, fuller bonus metadata/replay audit, full public
  replay/final observations, broader lifecycle/multiplayer parity, browser
  pixel parity later, and final cleanup. Current replay preservation is
  metadata/audit only, not full replay arrays.

## Current Environment Gap Queue

Plain remaining issues before stronger environment claims:

1. Broader natural bonus catch/effect matrix beyond the promoted focused
   public seeded/natural slices.
2. Halley capacity-audit follow-up: generated public random tape auto-extends,
   generated natural bonus position retry is not stopped by
   `natural_bonus_position_attempt_capacity`, and fixture/direct finite tapes
   remain strict. Public natural bonus timer advancement no longer has an
   artificial callback cap. Artificial/manual bonus stack overflow is a
   fixed-array guard for bad direct runtime fixtures, not a public env bug; a
   fully blocked generated map may still need policy. Keep `max_ticks`, body
   overflow, and event overflow classified as truncation-by-design.
3. Keep the fixed `BonusSelfMaster` wall-death parity guard green: normal-wall
   death still kills a SelfMaster/invincible avatar, while body collision
   remains suppressed.
4. Keep the fixed multi-target `BonusAllColor` event-order and older-wins stack
   precedence guards green while broader stack/death cases expand.
5. Timer/random ordering for public bonus scheduling: source/runtime slices are
   pinned, but public ownership of timers, RNG cursor/draw counts, and natural
   bonus scheduling is not broad.
5. Borderless stack behavior and wrap/collision side effects beyond the seeded
   public `BonusGameBorderless` expiry slice. Source/runtime/public
   duration/expiry is now covered by focused tests and a source fixture.
6. Fuller bonus public metadata and replay audit: spawned bonus identity,
   catch/expiry/clear events, random cursor/draw counts, active stack facts,
   and source refs. Current replay preservation is metadata/audit only, not
   full replay arrays.
7. Broader bonus effects and interactions beyond the current promoted
   runtime/public slices.
8. Full public replay and final observations for multiplayer rows, including
   reset/RNG provenance, terminal facts, reward/mask maps, and final-row policy.
9. Broader lifecycle and multiplayer parity: natural reset/warmup, warmdown
   frame movement, next-round/match-end policy, present/absent and leave edges,
   masks, and rewards.
10. Browser/source pixel parity later, after source state, public replay, and
   lifecycle rows are stable. Current raw 64x64 source-state raster parity is
   model-observation evidence only, not original browser/canvas pixel evidence.
11. Old toy path quarantine: toy-v0/debug routes are historical smoke evidence
   only, not a product runtime or fidelity proof.
12. Modal, speed, and fixed-opponent route smokes stay labeled as route evidence
   or measurement evidence. They are not the next environment priority.

## Current Source State

- Lifecycle/spawn/RNG now has pinned lifecycle fixtures, including
  `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p`, with JS oracle and Python parity
  coverage through the focused lifecycle runner.
- That lifecycle slice covers focused 2P warmup/start, terminal-to-next-round,
  heading retry, 3P and 4P first-round spawn/RNG, 4P all-present all-dead
  warmdown/next-round, 4P survivor next-round, 3P warmup plus delayed
  PrintManager start, focused present/non-present cases including survivor
  scoring, 2P and 3P match-end
  cases, 3P all-dead continuation, survivor scoring, tie-at-max continuation,
  and the focused all-present 3P multi-round match end.
- `BonusGameClear` immediate clear is promoted narrowly as a forced source-env
  parity claim. It is not a broad bonus-system claim.
- The narrow active bonus slice also includes `BonusSelfSmall` catch/no-catch,
  same-tick death-order no-catch, one natural one-type spawn/type/position RNG
  path, one default multi-type weight/type RNG path, one game-world retry path,
  and one expiry/restore path.
- Movement should stay framed as source elapsed-ms kinematics with held control
  state. Wrapper APIs may expose `step` or `joint_action`, but source behavior
  is not defined by those wrapper names.
- Existing promoted mechanics include movement, normal wall and borderless
  behavior, body collision/order canaries, PrintManager toggles/start/death
  stops, normal trail cadence, forced trail gaps, one natural trail-gap source
  case, old-body metadata, and focused scalar source-env scoring/timer guards.

## Vector And Speed Stance

- Current vector support remains fixture-backed, but `vector_runtime.step_many`
  is now the public supported CPU step boundary for those promoted
  source-ordered rows. The mixed comparator covers the promoted body,
  borderless, normal-wall, PrintManager, and forced trail-gap transition set.
  Natural trail-gap and lifecycle work stay separate unless a source-backed
  claim is explicitly promoted.
- Lifecycle fixtures still have honest unsupported seeder reports with RNG
  metadata. First-round spawn facts and the strict 1v1/no-bonus
  warmup/start/runtime-step slice now have narrow vector helpers, including a
  wall-death terminal vector trainer handoff into autoreset planning and the
  strict public 1v1 vector trainer env. That is not broad vector lifecycle,
  replay semantics beyond the strict live-step recorder, broad/full env API
  beyond strict 1v1, visual rendering, performance integration, or trainer
  readiness.
- Existing CPU/Modal/JAX/Mctx numbers are runtime evidence only. Use them to
  measure the path toward one faithful fast environment, not to declare
  production self-play. Current optimizer timings are fenced to strict wrapper
  `VectorTrainerEnv1v1NoBonus` `[B,2,106]` plus replay-v0 plumbing only; do not
  generalize them to bonuses, broad lifecycle, 3P/4P, visual LightZero, or full
  CurvyTron.
- Optimizer critique: native vector timing is useful only when the included
  components are explicit. If ray/observation work is the bound, that is an
  actionable optimization target. A large CPU batch regression must be broken
  down by env step, observation, replay/reset, and policy/search before making
  rewrite claims.

## Next Tasks

1. Widen natural bonus support beyond the promoted focused public
   seeded/natural slices.
2. Promote Halley's capacity-audit policy: generated public random tape
   auto-extends, generated natural bonus position retry is not capped by
   `natural_bonus_position_attempt_capacity`, fixture/direct finite tapes stay
   strict, and public natural bonus timer advancement has no artificial
   callback cap. Artificial/manual bonus stack overflow is an intentional
   fixed-array guard; a fully blocked generated map may still need policy.
   Designed truncations are `max_ticks`, body overflow, and event
   overflow.
3. Keep the fixed `BonusSelfMaster` wall/body parity checks in the focused
   bonus suite.
4. Finish timer/random ordering for public bonus scheduling.
5. Finish borderless stack/wrap/collision semantics beyond the seeded public
   expiry slice. Source/runtime/public duration/expiry has focused coverage; do
   not list that proof as missing.
6. Widen bonus public metadata/replay audit for spawn, catch, expiry, clear,
   active stack, and RNG facts while keeping the claim metadata-only until full
   replay arrays exist.
7. Add broader bonus effects and interactions one source claim at a time.
8. Fill full public replay and final-observation rows for multiplayer
   lifecycle and bonus states.
9. Broaden lifecycle/multiplayer parity after the bonus and replay facts are
   named: natural reset/warmup, warmdown movement, next-round/match-end,
   present/absent, leave, masks, and rewards.
10. Keep browser/source pixel parity later. It follows stable source state,
   public replay, and lifecycle rows. Source-vs-vector raw 64x64 parity proves
   the learned source-state observation raster only, not browser canvas pixels.
11. Keep old toy/debug paths quarantined as historical smoke evidence only.
12. Keep Modal, speed, and fixed-opponent route smokes labeled as route evidence
   only.

## Boundaries

- Source fidelity is the semantic guardrail.
- Training should use source-like randomness; later domain randomization is a
  separate robustness layer. Keep RNG provenance for fidelity/replay separate
  from extra training noise, and keep source-fidelity checks controlled and
  replayable.
- Keep front-door docs concise and link to evidence instead of copying logs.
- Do not treat `step_many` as a trainer-facing environment API. It is the
  supported fixture-backed runtime kernel. The strict vector trainer handoff is
  real, and the strict public 1v1 vector trainer env now covers public-step
  horizon truncation, overflow truncation, terminal metadata,
  reset/step control-wrapper info, terminal barrier replay policy, and live-step
  replay recording. Final `reset_many` beyond that strict slice, source fixture
  random-tape reset parity, seed/RNG history, visual rendering, and broad batch
  semantics remain unpinned.
- Do not treat fixture-cycled rollouts, debug observations/rewards, synthetic
  Mctx, or Modal boundary samples as real CurvyTron training.
