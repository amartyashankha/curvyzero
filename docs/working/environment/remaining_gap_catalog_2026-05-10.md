# Remaining Environment Gap Catalog - 2026-05-10

Status: current reproduction gap audit
Goal: one fast shared, source-faithful CurvyTron runtime under hardening:
`VectorMultiplayerEnv`. Old no-bonus public-env naming was stale, not a separate second product implementation. Fidelity first; speed
later. Strict
`VectorTrainerEnv1v1NoBonus` is only the older proven 1v1 proof/profiling
boundary. JS source checks and `CurvyTronSourceEnv` are truth/proof tools, not
alternate product environments. Proven source behavior should be absorbed into
`VectorMultiplayerEnv`. Route smokes and speed numbers only count after
the environment claim is source-backed.
Cleanup decision: do not split this into two implementations. Consolidate the
historical public env path into one fast runtime direction.
Restricted wrappers are temporary explicit profile configs; they are not the
reconstruction path and must not replace source-default CurvyTron behavior.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).
Current 2P public-runtime base guards now include direct public body/trail/
collision canaries, terminal wall step, autoreset final metadata, active leave
immediate scoring, the long natural 1v1 wall rollout from reset to terminal,
draw warmdown into the next round, unique-leader max-score match end, the 2P
metadata replay bridge, and the source-state LightZero wrapper fixed-opponent
sidecar proof. These are still focused source-backed slices, not a broad
full-fidelity claim. Latest validation reported on 2026-05-11: focused
environment validation reported `282 passed`, source bonus validation reported
`33 passed`, ruff passed, and the environment doc guard passed.

Read this as a punch list, not as a progress scorecard. The audit evidence is
narrow: focused vector-runtime, multiplayer-env, multiplayer-replay,
source-env bonus, env-scenario bonus, and CurvyZero LightZero smoke checks
covered specific boundaries. Rerun the relevant commands before treating any of
that as current status. It is useful evidence. It is not full CurvyTron.

## Green Boundary

What is real:

- Strict `VectorTrainerEnv1v1NoBonus` reset/step/autoreset/final-observation/
  reward/replay behavior is source-backed for the narrow 1v1 no-bonus long wall
  fixture.
- `vector_runtime.step_many` is a fixture-backed helper for promoted no-bonus
  transition slices. It is not a separate product env.
- `VectorMultiplayerEnv` exists for 2P/3P/4P as a metadata/public-state surface with base and partial seeded/natural bonus support. It has explicit done-row autoreset, final-row metadata,
  reset random tape source/length/RNG implementation metadata, metadata-only
  replay packaging, a metadata replay recorder, and public lifecycle ids.
  Public reset starts `round_id=1`, next-round warmdown increments it, and
  match-end rows keep the final round id. Focused 2P public-runtime guards now
  cover direct body/trail/collision canaries, the long no-bonus reset-to-terminal
  wall rollout, draw warmdown into next round, unique-leader max-score match
  end, and the 2P metadata replay bridge.
  Public info/replay carry
  `lifecycle_policy_id`, `reset_episode_id_policy`, and
  `source_round_id_policy`.
- Generated public reset rows use `seed_generated_source_random_history` /
  `curvyzero_seeded_source_math_random_history/v0`. The only natural reset
  claim is `seeded_source_history_reset_spawn_warmup_call_order/v0`, tested for
  2P/3P/4P reset spawn plus warmup call-order parity. It is not V8
  `Math.random` bit parity and not broad generated reset, lifecycle, replay,
  trainer, visual, or bonus parity.
- Public lifecycle metadata fields are now real `VectorMultiplayerEnv`
  state arrays from reset: `round_done`, `warmdown_pending`, `match_done`,
  `round_winner`, and `match_winner`. This removes the old stitched-array
  smell, but does not make natural reset or full lifecycle source-faithful.
  The stale lifecycle metadata overlay plumbing has been removed from
  `vector_multiplayer_env.py`.
- A narrow explicit public warmdown frame bridge exists:
  `VectorMultiplayerEnv.advance_warmdown_frame(..., elapsed_ms=...)`.
  It is guarded for one 3P match-mode survivor movement/death/no-rescore case.
  Ordinary `step()` still stays blocked during warmdown.
- A narrow active-round public leave bridge exists:
  `VectorMultiplayerEnv.remove_player(...)` is guarded for the narrow
  3P and 4P continuation fixture paths plus a narrow immediate round-end path.
  The immediate path is source-proven for 2P leave and has a 4P source-rule
  canary for survivor scoring after already-dead players. It is metadata-only, uses
  zero-based public ids with source ids equal to public id plus one, marks the
  leaver present/alive false, and keeps the leaver out of `death_player`. It
  rejects warmdown, terminal, absent-player, dead-player, and bad-shape calls.
- `src/curvyzero/env/vector_multiplayer_observation.py` adds
  `curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0`: a pure
  projection over `VectorMultiplayerEnv.state`, not a second env. It
  emits `float32[R,27]` rows for present+alive ego players only.
- `build_multiplayer_scalar_observation_replay_artifact_v0(...)` packages the
  3P/4P scalar observation rows, masks, row ids, ego ids, source shape, and
  nested public metadata records into a replay-shaped artifact. It is useful
  trace packaging, not trainer replay.
  The dead legacy scalar replay builder has been removed from
  `multiplayer_replay_v0.py`.
- Optional-array bonus support in the fast runtime is now table-backed for
  `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`, `BonusEnemySlow`,
  `BonusEnemyFast`, `BonusEnemyBig`, `BonusEnemyInverse`,
  `BonusGameBorderless`, and `BonusGameClear`.
- `vector_runtime.bonus_type_selection_metadata` exists as metadata-only
  weighted type-selection support for caller-supplied row-local `bonus.type`
  draws, including reduced/full `BonusGameClear` probability edges.
- Source-env bonus scouts cover minimal natural spawn/type RNG, default
  multi-type selection including natural `BonusGameClear` selection, one
  game-world spawn retry, one bonus-world spawn retry, and cap-at-20
  schedule-and-skip behavior. The bonus-world retry proof is
  `source_bonus_spawn_bonus_world_retry_step.json`, guarded by
  `tests/test_env_scenarios.py` and `tests/test_source_env.py`.
- `source_lifecycle_tie_at_max_score_4p.json` now has JS oracle, Python source
  runner, and focused public metadata parity for tied 4P leaders continuing to
  the next round. `source_lifecycle_match_end_at_max_score_4p.json` now has JS
  oracle and Python source-runner parity under `source-lifecycle-v25` for the
  unique-leader match end, and a focused public metadata proof now protects the
  same public path. `source_lifecycle_multi_round_match_end_4p.json` also now
  has JS oracle and Python source-runner parity for one all-present 4P
  continue-then-end match, plus focused public metadata parity. These do not
  prove broader public 4P multi-round variants or 4P present/absent match
  variants.
- `source_lifecycle_remove_avatar_to_single_present_3p.json` is promoted under
  `source-lifecycle-v25`: avatar 3 dies first, removing live avatar 2 leaves
  only avatar 1 alive, avatar 1 gets `roundScore=2` from total avatar count,
  warmdown does not emit `end` because avatar 3 is still present, and the next
  round uses two-present-player size with avatar 2 in deaths. Focused public
  metadata parity for this leave edge is green.
- The fixture bridge preserves per-step `advance_timers_ms` as
  `timer_advance_ms` for the promoted bonus expiry path.
- Route smoke evidence is plumbing only. Local fallback and Modal-installed
  smokes can prove commands reach the intended env identity, but they are route
  evidence only and not the next environment priority.

What is still not real:

- No full environment fidelity claim.
- Strict 1v1/no-bonus remains a proof boundary only. Do not promote it to a
  broader multiplayer or bonus claim.
- No broad natural public multiplayer reset parity claim. The generated reset
  claim is only the tested 2P/3P/4P
  `seeded_source_history_reset_spawn_warmup_call_order/v0` scope.
- No full public 3P/4P observation or final-observation claim. The scalar
  projection is useful, but it is not full environment fidelity.
- No production 3P/4P replay writer/reader/shard/manifest claim. The scalar
  replay-shaped artifact is not full public replay.
- No source-pixel renderer or browser/source pixel parity claim.
- No full public bonus env, full bonus replay arrays, broad natural bonus
  effects claim, or visual pixel parity claim.
- Narrow seeded/natural public bonus fixture support is landed for promoted
  effects, including `BonusSelfMaster` and `BonusAllColor`. Focused public
  natural tests now pass in `VectorMultiplayerEnv`, including source-default
  type selection and same-frame bonus/PrintManager random-order accounting.
  Seeded public bonus replay metadata preserves bonus metadata/audit fields
  only, not full replay arrays. Runtime optional-array effects are table-backed
  for promoted runtime effects including `BonusSelfMaster`, `BonusAllColor`,
  and `BonusEnemyStraightAngle`; the source oracle now has broader stack/effect
  support for self/enemy/all bonuses. `BonusSelfMaster` wall/body parity and
  `BonusAllColor` reverse target event order plus older-wins overlap behavior
  are fixed. Focused public natural source-default catch/effect coverage is now
  in place for self, enemy, game, and all-target source-default effects.
  Remaining gaps are manual/direct stack guard documentation, possible fully-blocked
  generated-map policy, full public bonus replay/final observations, broader
  multiplayer parity, visual pixel parity, toy-path quarantine, and final
  cleanup.
- The fixed-opponent source-state Modal smoke is landed. It is route evidence
  only and not the next priority. The natural spawn helper is landed, and
  focused public natural spawn tests now pass; that is still not broad runtime
  effect support.
- No broad public leave/disconnect claim, broad warmdown leave, replay/trainer/
  visual leave support, or bonus leave support. Immediate round-end public
  leave is narrow: 2P fixture-backed plus one 4P source-rule canary. The
  single-present 3P leave-edge fixture has focused public metadata parity, but
  broader leave variants remain open.
- No source-fidelity completion claim from the multiplayer scalar projection.
- No whole-loop self-play performance claim tied to full source-faithful env
  behavior.

## Stale Or Dangerous Claims To Delete

- Replace "public 3P/4P env is missing" with "metadata-only
  `VectorMultiplayerEnv` exists; broad natural reset, lifecycle parity,
  trainer observations, and production replay are missing."
- Replace "public env has no leave/disconnect action" with "public env has only
  narrow metadata-only active-round leave through
  `VectorMultiplayerEnv.remove_player(...)`; 3P/4P continuation and
  2P immediate round-end are covered, focused 3P staged match-mode warmdown
  leave metadata is covered, and the single-present 3P leave edge has focused
  public metadata parity. Broad warmdown leave, replay, trainer, visual, and
  bonus support are missing."
- Replace "multiplayer replay is missing" with "metadata-only multiplayer
  replay packaging, a metadata replay recorder, and a scalar replay-shaped
  observation artifact exist; trainer-ready 3P/4P replay writer/reader/shards,
  visual replay, and policy/search/value target replay are missing."
- Replace "3P/4P observation is missing" with
  "`curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0` exists as a
  pure `float32[R,27]` state projection over `VectorMultiplayerEnv`;
  `build_multiplayer_scalar_observation_replay_artifact_v0(...)` can package
  those rows with public metadata; trainer-ready env, visual/pixel,
  source-fidelity completion, and policy/search/value target claims are still
  false."
- Replace "bonus vector support is missing" with "optional-array fast-runtime
  table-backed support exists for promoted runtime effects including
  `BonusSelfMaster`, `BonusAllColor`, and `BonusEnemyStraightAngle`;
  source-default type-selection probability exists; and `vector_runtime.py` also
  has a low-level natural spawn helper with type/position/retry/cap tests.
  Public natural spawn timer ownership/scheduling/random accounting, full bonus
  replay, and manual/direct stack guard documentation remain missing.
  Generated tape/position retries are
  separate from strict fixture/direct oracle tapes, and public natural bonus
  timer advancement no longer has an artificial callback cap."
- Replace "route is ready" with "local fallback and Modal smokes are route
  evidence only; the fast source-faithful CurvyTron env still owns the next
  priority."
- Replace "visual observation exists" with "debug occupancy exists; source
  visual truth and pixel parity do not."
- Do not use pass counts, fixture counts, Modal runs, or rows/sec as
  reproduction claims.
- Do not call strict 1v1/no-bonus the destination. It is the first proof
  boundary.
- Quarantine toy-v0/debug paths as historical smoke/interface evidence only.
  They are not product runtime, replay, bonus, or fidelity evidence.

## Prioritized Punch List

| Priority | Gap | Implementation files | Tests to add or widen | Acceptance |
| --- | --- | --- | --- | --- |
| 1 | Broader natural bonus types/effects beyond promoted focused public seeded/natural slices. | `src/curvyzero/env/source_env.py`, `src/curvyzero/env/vector_runtime.py`, `src/curvyzero/env/vector_multiplayer_env.py`, `src/curvyzero/env/vector_lifecycle.py`, `src/curvyzero/env/vector_reset.py`, `scenarios/environment/source_bonus_*.json` | `tests/test_source_env.py`, `tests/test_env_scenarios.py`, `tests/test_vector_runtime.py`, `tests/test_vector_multiplayer_env.py` | Focused public natural `BonusSelfMaster` and `BonusAllColor` support is covered narrowly. Broader natural types/effects need source-backed public runtime claims. |
| 2 | Capacity-audit policy. | `src/curvyzero/env/vector_runtime.py`, `src/curvyzero/env/vector_multiplayer_env.py`, `src/curvyzero/env/source_env.py` | `tests/test_vector_runtime.py`, `tests/test_vector_multiplayer_env.py`, `tests/test_source_env.py` | Seed-generated random tape auto-extends, generated position retry is not capped by `natural_bonus_position_attempt_capacity`, public natural bonus timer advancement has no artificial callback cap, fixture/direct finite tapes plus `vector_runtime` finite helpers remain strict, and artificial/manual stack overflow is documented as a fixed-array guard. |
| 3 | Timer/random ordering for public bonus scheduling. | `src/curvyzero/env/vector_multiplayer_env.py`, `src/curvyzero/env/vector_runtime.py`, `src/curvyzero/env/vector_reset.py` | `tests/test_vector_multiplayer_env.py`, `tests/test_multiplayer_replay_contract.py` | Public rows own bonus timers, RNG cursor/draw counts, source refs, and scheduling facts without broadening beyond the proven source claims. |
| 4 | Borderless stack ordering, wrap/collision side effects beyond the seeded public expiry slice, and replay facts. | `src/curvyzero/env/source_env.py`, `src/curvyzero/env/vector_runtime.py`, `src/curvyzero/env/vector_multiplayer_env.py`, `third_party/curvytron-reference/src/server/model/Bonus/*.js` | `tests/test_source_env.py`, `tests/test_env_scenarios.py`, `tests/test_vector_runtime.py`, `tests/test_vector_multiplayer_env.py` | Public seeded `BonusGameBorderless` catch/expiry support has landed, and source/runtime duration/expiry has a runtime test plus source fixture. Remaining borderless work is stack/wrap/collision and replay facts. |
| 5 | Fuller bonus public metadata/replay audit without claiming full replay arrays. | `src/curvyzero/env/vector_multiplayer_env.py`, `src/curvyzero/env/vector_runtime.py` | `tests/test_vector_multiplayer_env.py`, `tests/test_multiplayer_replay_contract.py` | Spawned bonus identity, catch/expiry/clear events, active stack facts, random cursor/draw counts, and source refs survive public info and replay metadata. Full replay arrays remain separate. |
| 6 | Broader bonus effects. | `src/curvyzero/env/source_env.py`, `src/curvyzero/env/vector_runtime.py`, `src/curvyzero/env/vector_multiplayer_env.py`, `scenarios/environment/source_bonus_*.json` | `tests/test_source_env.py`, `tests/test_env_scenarios.py`, `tests/test_vector_runtime.py`, `tests/test_vector_multiplayer_env.py` | Add speed, slow, inverse, straight-angle, color, clear coupling, and death interactions one source claim at a time. |
| 7 | Full public replay and final observations. | `src/curvyzero/env/vector_multiplayer_env.py`, `src/curvyzero/env/vector_multiplayer_observation.py` | `tests/test_vector_multiplayer_env.py`, `tests/test_multiplayer_replay_contract.py`, future observation tests | Public rows carry reset/RNG provenance, terminal facts, reward/mask maps, final-observation policy, and source refs across lifecycle and bonus states. |
| 8 | Broader lifecycle/multiplayer parity. | `src/curvyzero/env/vector_multiplayer_env.py`, `src/curvyzero/env/vector_reset.py`, `src/curvyzero/env/vector_spawn.py`, `src/curvyzero/env/vector_lifecycle.py`, `scenarios/environment/source_lifecycle_*.json` | `tests/test_vector_multiplayer_env.py`, `tests/test_vector_reset.py`, `tests/test_vector_spawn.py`, `tests/test_vector_lifecycle.py`, `tests/test_source_lifecycle_runner.py` | Natural reset/warmup, warmdown movement, next-round/match-end policy, present/absent, leave, masks, and rewards are source-backed in the public runtime. |
| 9 | Old toy-path quarantine. | Docs and route selection only. | Doc guard plus route-smoke labels. | Toy-v0/debug routes stay historical smoke/interface evidence and are not cited as product runtime, replay, bonus, or fidelity proof. |
| 10 | Browser/source pixel parity later. | Future environment renderer/browser harness files only after state and replay settle. | Future source/browser artifact validators | Pixel parity follows stable source state, public replay, and lifecycle rows. It is not the current priority. |

## Bonus Gap Detail

Current optional-array bonus work is real but narrow. It is not a public bonus
environment or replay claim. Keep the next bonus patches small:

The bonus-world retry path is now proven narrowly by
`source_bonus_spawn_bonus_world_retry_step.json` plus JS/source-env tests. It
only proves a first candidate rejected by the bonus world, one retry x/y pair,
and the accepted `bonus:pop`. It does not promote public natural spawn timer
ownership/scheduling/random accounting, a public bonus env, bonus replay, or
broad bonus effects.

1. Public seeded bonus support is narrow: `BonusSelfSmall`, `BonusGameClear`,
   and `BonusGameBorderless` fixture checks only.
2. Public natural `BonusSelfSmall` spawn is covered narrowly in
   `VectorMultiplayerEnv`; broaden natural spawn only one source-backed
   type/effect at a time.
3. Natural spawned `BonusGameClear` catch/clear coupling.
4. Speed and slow turn-rate effects.
5. Inverse while turning and inverse double-cancel.
6. Straight-angle while already turning.
7. Borderless stack ordering plus wrap/collision side effects. Source/runtime/
   public seeded duration/expiry is already covered by focused tests and a
   source fixture.
8. `BonusAllColor` snapshot/restore.
9. Multiple active stacks and same-timestamp expiry ordering.
10. Death interactions for non-`BonusSelfSmall` effects.

Cap behavior itself is now only a narrow source-env claim:
`source_bonus_spawn_cap_twenty_step.json` proves `popBonus()` draws the next
delay before the cap check, then consumes no type/position RNG and emits no new
`bonus:pop` while 20 map bonuses remain active.

## Visual Stop Sign

CurvyTron visual work is not ALE and not Atari. Browser/source pixel parity is
later. The current environment priority is source-state reconstruction:
broader natural bonus types/effects, remaining borderless stack/wrap/collision
semantics, bonus audit/replay facts, public replay/final observations, broader
lifecycle parity, and final cleanup. Debug visual smoke cannot be promoted as
pixel truth.
The current Modal/source-state route command is fixed-opponent route evidence
only and does not move the environment priority.

## Promotion Rule

Every implementation item needs:

1. Source claim id.
2. JS oracle or source fixture when the behavior is source-native.
3. Python/source-env parity for exactly that claim.
4. Fast runtime or public env parity only after the source claim is named.
5. Tests that state the unsupported boundary.
6. Replay/metadata update when reset, terminal, RNG, observation, reward, or
   action meaning changes.
