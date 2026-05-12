# Remaining Reconstruction Gap Catalog - 2026-05-11

Status: practical Environment Reconstruction execution catalog.
Scope: one fast CurvyTron runtime direction, `VectorMultiplayerEnv`.

This is not a second taxonomy. It is the remaining work needed before we can
honestly say the fast runtime reconstructs source-default CurvyTron behavior.
The JavaScript oracle and `CurvyTronSourceEnv` are source-truth tools.
`VectorTrainerEnv1v1NoBonus` is only a strict proof/profile surface, not the
product env. Optimizer and Coach should consume explicit runtime surfaces; they
should not patch environment truth.

Observation-path guardrail: the active 2P product visual path is source-state
browser-like 704x704 RGB raw frame -> deterministic gray64 -> frame stack. Bonus64 and
rich tensors are diagnostics only. Browser/canvas pixels are not P0, and trainer
wrapper/replay propagation remains open unless an exact proof is named.
Renderer direction for that path is now explicit: `browser_lines` is the default
browser-style source-state renderer, and `body_circles_fast` remains an
explicitly selected circle-per-body approximation. This is still
native/source-state rendering, not a browser pixel parity claim.

## Plain Answer

Halley's capacity audit changes the current risk map. Seed-generated public
random tape auto-extends deterministically. Seed-generated natural bonus
position retry is no longer capped by
`natural_bonus_position_attempt_capacity`; that setting is a chunk/fixture
limit, not a source-fidelity stop for generated rows. Fixture/direct finite
tapes remain strict and can exhaust by design. `vector_runtime` finite helpers
also remain strict and raise when callers do not provide enough position draws
or random tape. Public natural bonus timer advancement no longer has an
artificial callback cap. Artificial/manual bonus stack overflow is an
intentional fixed-array guard for bad or undersized direct runtime fixtures,
not a public natural/seeded env bug. A fully blocked generated map may still
need policy if retries never find a position. Truncation-by-design includes
`max_ticks`, body overflow, and event overflow.

Archimedes's confirmed `BonusSelfMaster` wall-death bug is fixed in this
checkout. Source and public runtime regressions now prove the intended rule:
`BonusSelfMaster` invincibility blocks body/trail death, but normal-wall death
still kills the avatar.

Multi-target `BonusAllColor` event order and overlapping non-additive color
stack precedence are also fixed in this checkout. Source and public runtime
regressions now prove reverse target event order and source older-wins behavior
until the older stack expires.

The previous blocker was source-default natural bonus catch/effect support.
That blocker is now fixed for focused public paths: public natural tests cover
self small/slow/fast/master, enemy slow/fast/big/inverse/straight-angle, game
borderless, all color, and game clear. These bonuses spawn, catch, apply their
state changes, and expire or clear where the source effect has an expiry.
Speed-changing bonuses now also update turn rate with the source formula and
restore it on expiry; this was a concrete 2P fidelity hole found during the
visual/coverage audit.

The source-state visual profile path uses `VectorMultiplayerEnv` with
`natural_bonus_spawn=True`. That means the runtime can naturally spawn any
source-default bonus. The old optimizer profile failed when type code `11`
(`BonusAllColor`) spawned and was caught before table support had landed. That
table/effect gap is now closed. Public seeded and natural bonus stack capacity
also uses `SOURCE_MAX_ACTIVE_BONUSES`.

Do not fix this by narrowing `natural_bonus_type_codes`. That would create a
partial configured surface, not source-default reconstruction. Source defaults
include both effects:

- `BonusSelfMaster` is type code `4`.
- `BonusAllColor` is type code `11`.

`CurvyTronSourceEnv` already models these source effects. The fast runtime now
has matching focused state semantics for source-default natural catch/effect
families. Broad bonus replay/final-state and stack/death stress parity are
still not finished.

Before the single fast runtime can claim source-default reconstruction, these
must be true in `VectorMultiplayerEnv`:

- Source-default natural bonuses have focused public spawn/catch/effect/expiry
  or clear coverage.
- `BonusAllColor` rotates alive avatar colors from a snapshot and restores
  source player colors on expiry.
- `BonusSelfMaster` applies self invincibility plus source printing behavior
  and restores cleanly on expiry.
- Bonus timer order, stack order, death interactions, replay metadata, final
  state, and public info stay source-backed when bonuses are active.
- Existing no-bonus body/trail/collision/lifecycle proofs remain green; they
  are not the main blocker now.

## Blockers Now

| Priority | Gap | Why it blocks | Done when |
| --- | --- | --- | --- |
| B0 | Optimizer rerun of the failed profile | The live profile is the real route that exposed the original bonus gap. | Optimizer reruns the same source-state visual no-death profile and reaches the intended profiler stages without bonus-type or bonus-stack-capacity crash. |
| B1 | Capacity-audit policy | Halley's audit found long-survival traps and fixed-array guards that must be named honestly. | Generated tape/position retries extend deterministically; public natural bonus timer advancement has no artificial callback cap; fixture/direct finite tapes and `vector_runtime` finite helpers remain strict by design; artificial/manual stack overflow is documented as a fixed-array guard; a fully blocked generated map has an explicit policy if needed. |
| B1 | Bonus replay/final-state facts | Metadata currently preserves audit fields, not full bonus replay/final surfaces. | Public info/replay/final rows expose spawned bonus identity, catch, clear, expiry, stack, random cursor, and source refs without pretending to be browser replay. |
| B2 | Stack/timer/death interactions | AllColor overlap and SelfMaster wall/body parity are now guarded, but this does not prove every same-timestamp expiry, overlapping stack family, or death while boosted. | Focused source fixtures and public/runtime tests cover overlap, expiry ordering, wall/body deaths, and clear/borderless interactions. |
| B3 | Trainer propagation, replay/final observations, browser pixels | Not today's crash, but needed before broad env claims. The source-state renderer has default `browser_lines` plus explicit approximate `body_circles_fast`, but that is not browser canvas parity. | Public replay/final observations and later browser/source pixel comparison have separate source-backed gates. Raw 64x64 source-state raster parity is not browser/canvas pixel parity and does not prove trainer wrapper/replay propagation. |

## Done Enough

These should stay as regression guards, not as distractions:

- Runtime direction is settled: `VectorMultiplayerEnv` is the public fast
  runtime under hardening.
- `VectorTrainerEnv1v1NoBonus` remains useful for strict 1v1/no-bonus proof
  and profiling only.
- Focused no-bonus body, trail, collision, scoring, warmup, warmdown, match,
  leave, and reset-to-terminal slices already have source-backed tests.
- Source bonus spawn/type RNG has narrow JS/Python coverage, including default
  type selection that can select `BonusAllColor`, dynamic `BonusGameClear`
  probability edges, game-world retry, bonus-world retry, and cap-at-20 skip.
- Runtime/public seeded bonus work is real for the currently table-backed
  effects: `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`,
  `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`,
  `BonusEnemyInverse`, `BonusEnemyStraightAngle`, `BonusGameBorderless`,
  `BonusSelfMaster`, `BonusAllColor`, and `BonusGameClear`.
- Public seeded and focused natural stack capacity now uses
  `SOURCE_MAX_ACTIVE_BONUSES`.
- Source-state gray64 frames are Environment-owned source-state geometry rasters
  for the active image path: 704x704 RGB raw frame -> gray64 -> stack. The source 2P
  arena is 88 units from
  `CurvyTronReferenceDefaults.arena_size_for_players(2)`; 64x64 is only the
  learned gray64 tensor size. Bonus64/rich tensors are diagnostic only.
  These tensors are not ALE, not browser/canvas pixel truth, not trainer
  propagation proof, and not Coach learning evidence. The trail renderer default
  for this path is `browser_lines`; `body_circles_fast` remains an explicit
  approximation.
- Latest 2P source-state visual gate:
  `scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain`
  passes 35 full source-vs-vector gray64 scenarios with exact
  `max_abs_diff=0` and `mismatch_pixels=0`. The covered set includes long wall
  terminal, movement traces, normal wall/draw cases, collision-order cases,
  borderless cases, `BonusSelfSmall` catch/no-catch/expiry/wall-death cases,
  `BonusGameClear`, `BonusGameBorderless`, the four natural bonus
  spawn/retry/cap fixtures, programmatic source-snapshot visual stress cases,
  typed bonus diagnostics for all 12 source-default bonus types, and
  final-observation checks. Two intentional mismatch canaries prove the harness
  fails when a visible world body or visible map bonus is missing. This is
  native source-state render evidence only: full-size RGB raw frame -> gray64.
- Renderer-status caveat: this latest gate is under the native source-state
  renderer. It is not evidence that real browser canvas pixels are matched.
- Use `scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain`
  for the combined current visual proof. Latest result:
  `PASS full_2p_source_state_visual_gate canvas_gray64=35/35 typed_bonus=12/12 final_obs=pass canaries=2/2 mismatch_pixels=0 max_abs_diff=0.0 expected_canary_mismatch_pixels=26`.
  This still does not claim browser pixels, trainer wrapper/replay propagation,
  or final training readiness.
- Gray64 v0 keeps 2P player trails and heads distinct, but it draws every active
  map bonus with the same value (`208`). The separate bonus64 v1 typed/status
  gate now checks source-default active bonus type planes and post-catch
  self/other/game status planes. It still does not encode `BonusAllColor`
  color rotation or a typed post-catch `BonusGameClear` status plane.
- Focused 2P survivor movement during warmdown is now source/public guarded:
  `source_lifecycle_survivor_score_2p_next_round.json` pins original/source
  behavior, and public `VectorMultiplayerEnv.advance_warmdown_frame(...)`
  proves no rescore, no second `round:end`, correct death order, and correct
  next-round RNG cursor for the continuing-round case.
- Optimizer route/profile evidence is useful only after the exact environment
  surface is named.

## Evidence And Missing Tests

| Claim | Current proof | Missing tests before promotion |
| --- | --- | --- |
| Source defaults include `BonusSelfMaster` and `BonusAllColor`. | `tests/test_env_reference_defaults.py`; `src/curvyzero/env/config.py`; `vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES`. | None for the default list unless defaults change. |
| Public natural bonus metadata keeps the full source-default list and marks no implemented default effect as unsupported. | `tests/test_vector_multiplayer_env.py -k public_natural_bonus_metadata_keeps_source_defaults`; `tests/test_multiplayer_replay_contract.py -k bonus_audit`. | Replay/final-state and long-run stress checks remain. |
| The source-state visual profile path uses `VectorMultiplayerEnv(..., natural_bonus_spawn=True)`. | `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`; `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -k natural_bonus`. | Optimizer profile rerun remains separate route evidence. Trainer/replay propagation and browser/canvas pixel parity still wait. |
| `BonusAllColor` source behavior is known and runtime/public supported. | `tests/test_source_env.py -k bonus_all_color`; default-weight source scenarios in `tests/test_env_scenarios.py -k bonus`; focused public seeded/natural bonus checks. | Replay/audit, final-state, and broader stack stress checks remain. |
| `BonusSelfMaster` source behavior is available in `CurvyTronSourceEnv` and runtime/public supported. | `src/curvyzero/env/source_env.py` models invincible plus printing stack behavior; focused public seeded/natural bonus checks. | Replay/audit, final-state, and broader death/stack stress checks remain. |
| Runtime supports source-default bonus effects. | `tests/test_vector_runtime.py -k bonus` covers forced catches/expiry for small, slow/fast velocity, master, enemy big, inverse, straight-angle, all color, clear, and borderless slices. | Overlap/death/expiry-order stress where source rules differ. |
| Public natural source-default catch/effect support exists for focused slices. | `tests/test_vector_multiplayer_env.py -k "natural_bonus_self_effects or natural_bonus_enemy_effects or natural_bonus_game_borderless or natural_bonus_game_clear or natural_bonus_self_master or natural_bonus_all_color"` plus replay metadata checks. Slow/fast bonuses now assert source-style turn-rate changes as well as speed changes. | Replay/final-state, longer natural traces, and stack/death stress remain. |
| Natural bonus spawn/type/retry/cap is source-backed narrowly. | `tests/test_source_env.py -k bonus`; `tests/test_env_scenarios.py -k bonus`; `tests/test_vector_runtime.py -k "bonus_spawn or bonus_type_selection or bonus_spawn_cap"`. | Timer ownership across longer traces and replay/final-state tests. |
| Halley capacity audit separates failures from designed truncations. | Public guards `test_public_seed_generated_random_tape_extends_deterministically_on_demand`, `test_seed_generated_natural_bonus_position_retries_past_attempt_capacity`, `test_seed_generated_natural_bonus_timer_handles_many_due_callbacks`, `test_public_source_fixture_random_tape_stays_strict_on_exhaustion`, and `test_fixture_natural_bonus_position_does_not_autoextend_on_tape_exhaustion` prove generated tape/retries extend deterministically, fixture/direct tapes stay strict, and natural bonus timers have no callback cap; `vector_runtime` finite helpers remain strict. Existing strict public slices already treat `max_ticks`, body overflow, and event overflow as truncation metadata. | Artificial/manual stack overflow is an intentional fixed-array guard; add a fully blocked generated-map policy only if it becomes real. Do not reclassify generated tape extension, generated position retries, strict fixture tapes, strict runtime helpers, `max_ticks`, body overflow, or event overflow as bugs. |
| `BonusSelfMaster` wall-death parity is fixed. | Source update checks wall death before body collision invincibility; runtime now keeps wall death unmasked by `invincible`. `tests/test_source_env.py -k "self_master and wall"` and `tests/test_vector_multiplayer_env.py -k "self_master and wall"` pass. | Keep this as a regression guard while stack/death interactions broaden. |
| `BonusAllColor` multi-target order and overlap precedence are fixed. | Source/public tests prove reverse target event order and older-wins color stack behavior until the older stack expires. | Extend the same source-order/precedence treatment to other multi-target and non-additive stacks where needed. |
| No-bonus core is strong enough to protect while bonus work proceeds. | `tests/test_vector_runtime.py`; `tests/test_vector_multiplayer_env.py`; `tests/test_source_lifecycle_runner.py`; `tests/test_lifecycle_oracle.py`; focused body/trail/collision/source-env tests. | Broader lifecycle/replay/observation gaps remain, but they should not block fixing source-default bonus crashes. |
| Source-state visual route is a valid plumbing surface. | `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`; `tests/test_vector_visual_observation.py`; `scripts/compare_2p_raw_visual_observation.py`; current raw uint8 704x704 source-state access and source-vs-vector gray64 parity work. The `full2p` gate covers 35 source-vs-vector gray64 scenarios with exact `max_abs_diff=0`, `mismatch_pixels=0`, including natural bonus spawn/retry/cap fixtures and programmatic source-snapshot stress cases. It also has intentional mismatch canaries for missing visible body/bonus geometry. The bonus64 v1 gate covers typed active bonus identity and core status planes for all 12 source-default bonus types. | This proves the model-observation raster from source state for the covered fixtures. It does not prove trainer wrapper/replay propagation. Browser/canvas pixel parity is optional later human/debug evidence, not the current blocker. Existing JS reference tooling under `tools/reference_oracle` and `tools/js_reuse_probe` can produce golden source-state snapshots. `source_print_manager_random_call_order_step` remains intentionally outside gray64 because it verifies RNG/event order, not a distinct rendered state. Bonus64 v1 still does not encode `BonusAllColor` color rotation or a post-catch `BonusGameClear` status plane. |

## Execution Checklist

1. Lock the source claim.
   - Keep source-default bonus list unchanged.
   - Confirm type codes: `BonusSelfMaster=4`, `BonusAllColor=11`.
   - Name the claim ids for source-default natural catch/effect support.

2. Add focused source regressions only where they are missing.
   - `BonusAllColor`: source-env rotate and restore already has a direct test;
     add a natural-catch fixture only if needed for event/timer ordering.
   - `BonusSelfMaster`: add a focused source-env or JS fixture if the fast
     implementation needs exact event order for invincible/printing changes.

3. Implement fast-runtime effect support. Done for the first forced/public
   seeded and focused natural promotion.
   - Keep `BonusAllColor` in the runtime effect path with all-alive-avatar
     targeting, color snapshot, property events, stack storage, expiry, and
     restore.
   - Keep `BonusSelfMaster` with source invincible and printing semantics,
     stack storage, property events, expiry, and restore.
   - Preserve existing effect counters and unsupported-type errors for truly
     unknown types.

4. Promote through `VectorMultiplayerEnv`.
   - Forced seeded catch/expiry tests are landed.
   - Focused natural `BonusSelfMaster` and `BonusAllColor` support is landed.
   - Broader natural source-default spawn/catch tests are next.
   - Keep capacity outcomes explicit: seed-generated public random tape extends,
     generated natural bonus position retry is not capped by
     `natural_bonus_position_attempt_capacity`, fixture/direct finite tapes and
     `vector_runtime` finite helpers are strict, public natural bonus timers no
     longer have an artificial callback cap, artificial/manual stack overflow is
     an intentional fixed-array guard, and `max_ticks`, body overflow, and event
     overflow are truncation-by-design.
   - `BonusSelfMaster` wall parity is fixed: normal walls still kill
     invincible avatars, while body collisions remain suppressed.
   - `BonusAllColor` multi-target order and overlapping color stack precedence
     are fixed for the promoted public/source slices.
   - Metadata/audit updates only after runtime effects work.

5. Preserve replay/final-state truth.
   - Public info must say which bonus spawned, which player caught it, which
     stack changed, which timer expired, and which RNG calls were consumed.
   - Replay metadata should stop marking implemented source-default effects as
     unsupported.
   - Full replay arrays can remain a separate later gate if the doc says so.

6. Hand back to Optimizer.
   - Optimizer reruns the same source-state visual profile now that the
     focused bonus support is fixed.
   - Environment reviews only semantic claims from that profile.
   - Timing, GPU utilization, search cost, and frame-stack cost stay Optimizer
     owned.

7. Hand back to Coach.
   - Coach evaluates learning and policy quality only after the runtime surface
     is explicit.
   - Coach does not turn optimizer profile success into environment truth.

## Parallel And Sequenced Work

Can run in parallel:

- `BonusAllColor` source/test specification and `BonusSelfMaster`
  broader source/test specification, as long as one worker owns shared runtime
  stack schema edits.
- Replay/audit expectation updates can be prepared in parallel, but should
  merge after runtime support is real.
- Broader no-bonus lifecycle/replay cleanup can continue separately if it does
  not touch bonus arrays, timers, or public info contracts.
- Optimizer can keep profiling debug-only or no-death partial surfaces, but
  those runs must stay labeled partial.

Must be sequenced:

- Source claim before runtime implementation.
- Runtime effect before public natural catch claim.
- Public info/replay audit update after runtime effect.
- Optimizer rerun after the focused bonus support is fixed.
- Coach learning/eval claims after Environment and Optimizer publish the exact
  runtime/profile surface used.

## Ownership Split

Environment Reconstruction owns:

- source truth, source fixtures, and `CurvyTronSourceEnv` parity;
- `vector_runtime` and `VectorMultiplayerEnv` gameplay semantics;
- source-default bonus behavior, timers, random call order, stack state,
  public info, replay metadata, final observations, and contract wording;
- deciding whether a surface is source-default, partial, debug-only, or
  browser-pixel faithful.

Optimizer owns:

- timing/profile runs, Modal commands, GPU/CPU breakdowns, LightZero adapter
  plumbing, visual stack plumbing, and profiler artifacts;
- rerunning the visual profile on the named focused bonus-support surface;
- reporting partial surfaces as partial.

Coach owns:

- learning, evaluation, policy quality, gates, and experiment interpretation;
- deciding whether a trained agent improved, once the runtime/profile surface
  is named.

## Acceptance Commands

Existing focused guards to rerun around this catalog:

```bash
uv run pytest tests/test_env_reference_defaults.py -q
uv run pytest tests/test_source_env.py tests/test_env_scenarios.py -q -k bonus
uv run pytest tests/test_vector_runtime.py -q -k bonus
uv run pytest tests/test_vector_multiplayer_env.py tests/test_multiplayer_replay_contract.py -q -k bonus
uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -q -k natural_bonus
uv run python scripts/check_environment_doc_status.py docs/working/environment
```

New blocker guards:

- forced runtime `BonusAllColor` catch/expiry restore: landed;
- public seeded `BonusAllColor` catch/expiry restore: landed;
- public natural `BonusAllColor` spawn/catch/expiry: landed for the focused
  support slice;
- forced runtime `BonusSelfMaster` catch/expiry restore: landed;
- public seeded `BonusSelfMaster` catch/expiry restore: landed;
- public natural `BonusSelfMaster` spawn/catch/expiry: landed for the focused
  support slice;
- replay/audit check that neither implemented effect remains in unsupported
  natural bonus metadata: landed for metadata/audit;
- seeded/natural bonus stack capacity uses `SOURCE_MAX_ACTIVE_BONUSES`: landed;
- `BonusAllColor` reverse target event order and older-wins overlap behavior:
  landed;
- focused validation: `282 passed`; source bonus validation: `33 passed`; ruff
  and the environment doc guard passed.
