# Environment Active Work

Status: live work tracker
Date: 2026-05-13

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
Observation-path purge ledger:
[observation_path_purge_2026-05-11.md](observation_path_purge_2026-05-11.md).
Full-game fidelity reorientation:
[full_game_fidelity_reorientation_2026-05-13.md](full_game_fidelity_reorientation_2026-05-13.md).
Fidelity testing strategy:
[fidelity_testing_strategy_2026-05-13.md](fidelity_testing_strategy_2026-05-13.md).
Operating patterns:
[operating_patterns_2026-05-13.md](operating_patterns_2026-05-13.md).
Orchestration patterns:
[orchestration_patterns_2026-05-13.md](orchestration_patterns_2026-05-13.md).
Current queue:
[current_queue_2026-05-13.md](current_queue_2026-05-13.md).
Multiplayer training readiness:
[multiplayer_training_readiness_2026-05-13.md](multiplayer_training_readiness_2026-05-13.md).
Source-state multiplayer trainer surface spec:
[source_state_multiplayer_trainer_surface_spec_2026-05-13.md](source_state_multiplayer_trainer_surface_spec_2026-05-13.md).
Source-state multiplayer target-row adapter spec:
[source_state_multiplayer_lightzero_adapter_spec_2026-05-13.md](source_state_multiplayer_lightzero_adapter_spec_2026-05-13.md).
Optimized path reorientation:
[optimized_path_reorientation_2026-05-13.md](optimized_path_reorientation_2026-05-13.md).
Full fidelity parallel spec:
[full_fidelity_parallel_spec_2026-05-13.md](full_fidelity_parallel_spec_2026-05-13.md).
Parallel execution todo:
[parallel_execution_todo_2026-05-13.md](parallel_execution_todo_2026-05-13.md).

## Plain Current State

- We are not at full CurvyTron environment fidelity.
- Environment Reconstruction should not own public tournament leaderboard
  publishing or training-run ranking analysis. Those are Coach/Training
  consumer tasks unless they reveal that the environment contract is wrong.
  If they need context from this thread, write a short handoff and return to
  source-fidelity work.
- Current top priority is Environment Reconstruction: faithful multiplayer
  CurvyTron behavior first, then speed/training integration. Keep
  source-state visual/replay/target-row/sample-batch paths honest, but treat
  them as guarded downstream interfaces rather than the main focus.
- Main active threads are multiplayer fidelity gaps, controls fidelity,
  renderer/fast-path boundary, and docs/orchestration rhythm.
- Latest parallel fidelity wave promoted source-backed lifecycle/leave, bonus,
  and collision fixtures through public env and trainer/replay proof. This
  narrows the gap list but is not a full CurvyTron completion claim.
- Working-memory snapshot: trainer surface done, replay arrays done,
  repo-owned target-row adapter done, deterministic sample batches done, and
  fake/injected native `GameSegment` mapping from target rows done.
  Hume's opt-in real-LightZero construction helper is also done, but it is
  construction-smoke only. Real LightZero buffer sampled-target parity remains
  unproven.
- There is one public runtime name under hardening:
  `VectorMultiplayerEnv`.
- There is one active 2P visual product path:
  source-state browser-like 704x704 RGB raw frame -> 11x11 area-downsampled
  gray64 -> frame stack. Bonus64/rich tensors are diagnostics only.
- The two-player trainer stack uses the optimized
  `render_source_state_canvas_gray64_player_perspectives(...)` helper with
  dirty render caching when safe. This is an equivalence optimization under the
  declared render mode, not a second product observation path.
- Visual terminology guardrail: `browser_lines` is now the default
  browser-style source-state renderer for this product image path, and
  `body_circles_fast` is an explicit approximation for speed/profiling. This is
  still native/source-state rendering, not browser canvas pixel parity. It
  prefers `visual_trail_*` points when present and falls back to sparse bodies
  when visual trail points are absent.
- Bonus sprites in the current source-state renderer come from the one 300x400
  `web/images/bonus.png` atlas: 3x4 tiles, 12 source-default bonus types.
- Browser/canvas pixels are not P0, and the current source-state visual gate is
  not browser pixel parity.
- Training/profile additions such as `profile_no_death` are intentional
  project features, not original CurvyTron behavior. Preserve them, but keep
  their metadata explicit and never cite those runs as source-fidelity proof.
- Trainer surface, in-memory replay arrays, repo-owned target rows, and
  deterministic sample batches now have explicit proof for the source-state
  multiplayer surface. Fake/injected native `GameSegment` mapping is also done,
  but it is injection-only and does not import LightZero. The opt-in
  real-LightZero construction helper is done, still construction-smoke only,
  and still not real `MuZeroGameBuffer` sampled-target parity. Real native
  LightZero buffer integration, durable artifacts, learner updates, and eval
  quality remain open; do not infer them from gray64, bonus64, replay-array,
  target-row, sample-batch, injected-bridge, or construction-helper passes.
- Current trainer-surface state: `SourceStateMultiplayerTrainerSurface` now
  wraps `VectorMultiplayerEnv` and emits per-seat source-state visual stacks,
  live-seat policy-row mappings, masks, survival-plus-bonus rewards, terminal
  visual final observations, and honest render/source metadata. It rejects
  `fast_gray64_direct`, defaults to `browser_lines`, and labels
  `body_circles_fast` approximate. `SourceStateMultiplayerTrainerReplayRecorder`
  now stores copied in-memory trainer arrays over time, including terminal
  visual final observations and variable live-policy rows.
  `SourceStateMultiplayerTargetRowsV0` now consumes those replay arrays plus
  explicit policy-row records, validates one-step transition alignment,
  preserves project-only helper metadata, and keeps native LightZero claims
  false. `SourceStateMultiplayerSampleBatchV0` now builds deterministic sample
  batches on top of those rows. The injection-only native bridge now maps those
  rows to injected `GameSegment`-like objects while preserving project-only
  metadata and keeping native/LightZero/training/buffer/learner claims false;
  Hume's separate opt-in real-LightZero construction helper is done but remains
  construction-smoke only, while real buffer sampled-target parity and durable
  artifact plumbing are later separate work.
- Confirmed cadence/collision rule: source CurvyTron frames are about 16.67 ms
  from `BaseGame.framerate = (1 / 60) * 1000`. Collision truth is stored body
  circle endpoint overlap plus death/collision metadata, not visual trail
  crossing alone.
- Trainer wrappers now use `decision_source_frames` as the real knob, derive
  `decision_ms`, hold controls for that window, simulate source-sized internal
  frames, and stop early on death. The old pattern of treating one 300 ms
  decision as one physics step can tunnel through stored bodies.
  Controls-specific evidence and next tests:
  [controls_fidelity_audit_2026-05-13.md](controls_fidelity_audit_2026-05-13.md).
- Latest controls proof covers original JS keyboard reduction/server move
  delivery, 2P/3P/4P public source-frame control mapping, held-control parity,
  release-to-straight, invalid/live action rejection, inactive noops, one
  direct terminal early-stop trace, and one LightZero-facing wrapper terminal
  trace. Terminal-padding noop behavior is proven through a controlled padded
  row fixture, not relaxed post-terminal public stepping. It still does not
  prove touch/gamepad input, real transport/browser integration, or full replay
  arrays.
- Latest direct 2P product-route proof covers `VectorMultiplayerEnv` raw RGB ->
  gray64, seeded `BonusGameClear`, stale trail/body clear, terminal wall death,
  rewards, final observation masks, and metadata replay. Latest
  LightZero-facing wrapper proof covers wrapper-side scalar joint-action
  decoding, raw RGB -> gray64 stack, held source frames, terminal final
  observation, rewards, masks, and native sidecars.
- Latest bonus probability fix: original JS effectively gives every
  non-`BonusGameClear` source-default bonus probability `1`; subclass prototype
  fields `0.8/0.6/0.8` are not effective type-selection weights.
- Latest public natural bonus proof pins corrected `BonusGameClear` boundary
  draws, natural bonus RNG labels/cursors, next-delay scheduling, spawned
  position, and the full 12-item source-default bonus set.
- Latest hit-owner stress proof covers 4P newest-owner overlap, 4P source-style
  corner island order, 3P own-body latency, and 4P two-victim metadata
  alignment. Focused propagation now reaches public env, trainer surface,
  replay records, and debug die events for one 3P terminal body-hit case and
  one 4P nonterminal two-victim case. Raw JS oracle fixtures now pin those
  exact 3P/4P stress shapes, and public `VectorMultiplayerEnv` mirrors them
  from fixture-seeded state. Broader collision edges remain open.
- Fresh 2026-05-13 local validation: source env, vector runtime, controls,
  collision, and direct 2P product-route focused tests reported `155 passed`.
  New multiplayer trainer-surface and replay focused tests reported `14
  passed`, and the combined surface/replay/product/LightZero-wrapper/render
  sweep reported `41 passed, 1 skipped`.
  The full 2P source-state visual gate still reports
  `canvas_gray64=35/35`, `typed_bonus=12/12`, `final_obs=pass`,
  `mismatch_pixels=0`, and `max_abs_diff=0.0`. Keep this as a freshness note,
  not a full-fidelity claim.
- Strict `VectorTrainerEnv1v1NoBonus` is only a proof/profiling boundary. It is
  not the destination.
- `VectorMultiplayerEnv` is the intended public runtime name. The
  old "NoBonus" suffix was historical wording and has now been removed from the public runtime name; do not describe it
  as a separate second product implementation.
- Cleanup decision: product direction is one fast source-faithful runtime, not
  two implementations. Public-env cleanup should consolidate behavior into this
  runtime path and remove confusion from historical naming.
- Treat it as a narrow public-state surface today, not a full CurvyTron fidelity
  claim. It has promoted seeded and focused natural bonus effect slices, but
  trainer observation/replay/final-state coverage is still not a blanket claim.
- Latest focused environment validation after the bonus stack-capacity,
  SelfMaster wall-parity, AllColor stack-order, generated-RNG extension, and
  natural timer callback-cap removal and natural bonus effect coverage reported `282 passed`; focused source
  bonus validation reported `33 passed`.
  Ruff and the environment doc guard also passed. Keep this as a freshness
  note, not a status scoreboard.
- Latest 2P bonus fidelity fix: slow/fast speed bonuses now also change turn
  rate with the source formula, and expiry restores it. This was a real gap
  caught by the current coverage audit.
- Public 3P/4P lifecycle now has a focused mixed-row match-mode proof:
  reset/warmup, round win, warmdown, one row starting the next round, one row
  ending the match, masks, rewards, and public final rows.
- Public env plus trainer/replay now also have a focused P=3/P=4
  presence/leave proof for one active row and one staged-warmdown row in the
  same batch. It covers present/alive masks, absent action slots, warmdown
  next-round carryover, trainer live-policy rows, and replay storage.
- Focused source-backed public leave proofs are also green for
  `source_lifecycle_remove_avatar_to_single_present_3p.json` and
  `source_lifecycle_remove_avatar_during_warmdown_3p.json`. The largest
  current environment gaps are broader leave variants, renderer/fast-path
  boundary guards, JS/original stress fixtures for the remaining
  bonus/collision cases, bonus metadata/replay beyond audit rows, old toy-path
  quarantine, and final cleanup. Real LightZero `MuZeroGameBuffer`
  sampled-target parity remains downstream interface work, not the main
  reconstruction priority.
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
  `CurvyTronReferenceDefaults.arena_size_for_players(2)`. The 704x704 RGB frame
  is our source-state raw visual product surface, and 64x64 is the downsampled
  learned gray64 tensor size, not the original game arena size.
- Latest 2P source-state visual check:
  `scripts/compare_2p_raw_visual_observation.py --suite core2p --format plain`
  compares source-shaped state against `VectorMultiplayerEnv` through the
  active product image path: source-state browser-like 704x704 RGB raw frame ->
  11x11 area-downsampled gray64. The 2026-05-11 source-state/native check passed
  35 scenarios exactly. Keep it as source-vs-vector state/raster regression
  evidence, not browser canvas pixel evidence. `browser_lines` is the default
  source-state mode over sparse body points; `body_circles_fast` remains
  selectable and clearly labeled as approximate.
- The fuller one-line visual command is now
  `scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain`.
  Recorded result:
  `PASS full_2p_source_state_visual_gate canvas_gray64=35/35 typed_bonus=12/12 final_obs=pass canaries=2/2 mismatch_pixels=0 max_abs_diff=0.0 expected_canary_mismatch_pixels=26`.
  This means only that source and vector agreed through the source-state/native
  render path and that the separate diagnostic bonus64 gate passed. By itself
  it does not prove real browser canvas pixels, target rows, native LightZero
  integration, or full training readiness.
- Gray64 v0 distinguishes 2P player trails and heads, but every active map
  bonus is value `208`. The separate bonus64 v1 typed/status gate now checks
  active map bonus mask/type planes for all 12 source-default bonus types and
  post-catch self/other/game status planes. It still does not encode
  `BonusAllColor` post-catch color rotation or a typed post-catch
  `BonusGameClear` status plane. Treat bonus64/rich tensors as proof and
  diagnostics only, never as the product trainer default.
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
  Latest 2P stack/death proof adds a JS fixture and source/vector/public tests
  for three stacked `BonusSelfFast` catches followed by normal-wall death:
  death clears active stack state, does not restore the dead avatar's boosted
  velocity, and late timeout callbacks stay inert. Remaining gaps include other
  stack/death combinations. Latest same-step timer proof adds
  `source_bonus_self_fast_expiry_then_wall_death_same_tick_step.json` and
  source/public tests for one `BonusSelfFast` expiry draining before a
  wall-death update: velocity restores to `16`, p0 moves to `x=-1.4`, dies,
  p1 scores, and the stack remains empty. Trainer/replay now has the matching
  focused terminal packaging proof for both `BonusSelfFast` cases: final visual
  observation rows, final reward maps, step counters, death facts, winner/loser
  facts, and compact bonus audit metadata are preserved in replay records.
  New 4P target-filter proof covers enemy/all/game bonus targeting with one
  alive target, one dead seat, and one absent seat: source/public enemy bonuses
  hit only other alive avatars, all-avatar bonuses hit only alive avatars, and
  game bonuses update global state. New focused 4P terminal proof covers
  `BonusEnemySlow` stack/death packaging with a JS oracle fixture, public vector
  mirror, and trainer/replay check: p0 applies slow stacks to p1/p2/p3, those
  targets wall-die before expiry, death clears the stack rows without restoring
  dead-player speed, p0 wins the round, and replay preserves final visual/reward
  rows, death order, winner/loser facts, step counters, and compact bonus
  metadata. This is a narrow source-runner, public-vector, and trainer/replay
  proof, not browser event-loop or render proof. Remaining gaps include other
  stack/death combinations, broader 3P/4P lifecycle/leave, visual pixel parity,
  and final cleanup.
- Docs worker status: this is the only active docs lane now. Keep updates
  concise, and avoid broad refactors unless they directly remove confusion or
  unblock parity tests.

Why this is taking long: we are no longer debating versions or searching for a
different runtime. We are hardening one public runtime, and the remaining work
is specific feature work plus source-backed proof: bonuses, lifecycle, replay
and final observations, public metadata, and later browser pixel parity.

## 2P Status

Full 2P fidelity is not done. The current source-state visual gate is internal;
the latest recorded `full2p` run matched `canvas_gray64=35/35`,
`typed_bonus=12/12`, and `final_obs=pass` through the native source-state render
path. This is not browser canvas pixel parity. The new 2P collision/body-depth
canaries pin the important source rule: stored body circle endpoint overlap and
death metadata are collision truth, while a rendered line crossing alone is not.
Large one-step decisions can tunnel; source-frame substeps catch the death.
Deeper evidence and gap detail live in
[two_player_fidelity_gap_catalog_2026-05-11.md](two_player_fidelity_gap_catalog_2026-05-11.md)
and
[remaining_reconstruction_gap_catalog_2026-05-11.md](remaining_reconstruction_gap_catalog_2026-05-11.md).

Next 2P environment-fidelity checklist:

1. Keep the candidate trainer observation on the clean source-state image path:
   704x704 RGB raw frame -> 11x11 area-downsampled gray64 -> stack. Keep
   bonus64/rich tensors as diagnostic proof for hidden bonus facts.
2. Keep continuous trail topology guarded: prefer `visual_trail_*` when present,
   connect sparse persisted bodies otherwise, and split only on explicit break
   state.
3. Keep typed map bonus sprites visible through the source atlas path, while
   remembering gray64 is still not a full hidden bonus-state tensor.
4. Expand the 2P wall/borderless/collision canaries around wrap destination
   bodies, wall priority, trail latency, and same-frame death ordering.
5. Add JS/original fixture parity for bonus stack/death stress cases that are
   currently source-env programmatic probes.
6. Promote final/replay bonus state beyond metadata-only audit rows.
7. Add a real browser/canvas pixel harness and golden reference frames before
   making browser pixel parity claims. The existing `full2p` command is a
   source-state/native visual consistency gate.
8. Keep row-local RNG/replay history and fully blocked generated bonus-position
   policy explicit.

Closed on 2026-05-11: focused 2P survivor movement during warmdown now has
source/original proof and public `VectorMultiplayerEnv.advance_warmdown_frame`
coverage. It proves no second `round:end`, no rescore, death order preservation,
and correct next-round RNG cursor for the continuing-round case.

Do not describe the game as 64x64: for 2P the source arena is 88 units, while
704x704 is the raw source-state visual frame and 64x64 is only the learned
downsampled gray64 tensor size. The source-state route proof is fixed-opponent
plumbing only, not an environment-fidelity claim.

## Current Direction

Main priority: harden one fast, source-faithful CurvyTron runtime:
`VectorMultiplayerEnv`. It is the
current intended public runtime name, not a separate second product path. The
working proof path is:

```text
source claim -> JS oracle/probe -> CurvyTronSourceEnv parity -> optimized parity
```

Native CurvyTron should be described as held control state advanced through
elapsed-ms frames. Source frames are about 16.67 ms from
`BaseGame.framerate = (1 / 60) * 1000`. `step` and `joint_action` are
wrapper/API terms; do not let them replace the source mental model.

Strict `VectorTrainerEnv1v1NoBonus` is a fixed decision wrapper over source
control state. It accepts trainer action ids, maps them to native source moves,
and holds that control state for `decision_source_frames` source frames while
internally advancing source-sized frames and stopping early on death. Current
wrappers derive `decision_ms` from that frame count. One 300 ms decision must
not be collapsed into one physics step; that can tunnel through stored bodies.
It is not native discrete simultaneous actions and not the product runtime.
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
- Optimizer owns visual smoke/profiling and later real native LightZero buffer
  plumbing. Environment owns whether visual tensors, replay arrays,
  repo-owned target rows, and injected bridge rows are source-faithful and
  honestly claimed.
- Coach consumes named runtime/training surfaces for learning, evaluation,
  policy quality, and experiment interpretation. Coach results do not promote
  route, profile, or learning evidence into Environment source-fidelity claims.
- Latest landed pieces: multiplayer final-row metadata, explicit public
  `autoreset_done_rows`, metadata-only multiplayer replay record/chunk
  packaging plus recorder, optional-array fast-runtime `BonusSelfSmall`
  catch/expiry and forced `BonusGameClear` clear, seeded public bonus fixture
  support for `BonusSelfSmall`, `BonusGameClear`, and `BonusGameBorderless`,
  visual truth/schema metadata, local/
  installed no-train scalar LightZero smoke boundaries, and the source-state
  LightZero wrapper fixed-opponent sidecar proof.
- Active Environment implementation targets are multiplayer fidelity gaps,
  controls fidelity, renderer/fast-path boundaries, and docs/orchestration
  rhythm. Deeper bonus stack/death interactions, fuller bonus metadata/replay
  audit, broader lifecycle/multiplayer parity including 3P/4P leave, browser
  pixel parity later, toy-path quarantine, and final cleanup stay in that
  reconstruction queue. Hume's real-LightZero construction helper is done but
  remains downstream construction-smoke only.
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
- This is not full environment fidelity. First gaps now center on Environment
  Reconstruction: multiplayer lifecycle and 3P/4P breadth, controls fidelity,
  renderer/fast-path boundaries, continuous-trail and bonus-sprite regressions,
  wall/borderless/collision stress, fuller bonus metadata/replay audit,
  browser pixel parity later, and final cleanup. Source-backed
  visual/replay/target rows/sample batches, injected bridge rows, and Hume's
  construction helper are not real native LightZero-ready until sampled-target
  buffer parity is proven.

## Current Environment Gap Queue

Plain remaining issues before stronger environment claims:

1. Convert the completed full-game multiplayer audit into focused tests/fixes:
   lifecycle, presence/leave, scoring, match-end, replay/final observations,
   bonus stack/death stress, and 3P/4P breadth.
2. Keep controls fidelity source-backed: source-frame control delivery, held
   and released inputs, terminal-padding boundaries, touch/gamepad input, and
   browser/transport semantics.
3. Keep renderer/fast-path boundaries explicit: optimized and approximate
   render paths must stay labeled and separate from engine rule fidelity.
4. Keep continuous trail rendering source-faithful enough for training:
   `visual_trail_*` first, sparse body connection second, explicit breaks for
   clears/gaps/wraps/resets, and `body_circles_fast` labeled approximate.
5. Keep typed bonus visibility honest: source atlas sprites for active map
   bonuses in the RGB path, downsampled visibility in gray64, bonus64/rich
   tensors only as diagnostics, and no claim that gray64 exposes all hidden
   bonus stack facts.
6. Finish 2P wall/borderless/collision depth: normal-wall priority, borderless
   wrap destination-body behavior, stored-body circle overlap, own-trail latency,
   head-head/death ordering, and source-frame substep decisions that avoid
   tunneling.
7. Keep the new JS/original 2P `BonusSelfFast` stack/death fixture green, and
   add parity for the remaining 2P bonus stack/death stress cases.
8. Promote final/replay bonus state beyond metadata-only audit rows.
9. Halley capacity-audit follow-up: generated public random tape auto-extends,
   generated natural bonus position retry is not stopped by
   `natural_bonus_position_attempt_capacity`, and fixture/direct finite tapes
   remain strict. Public natural bonus timer advancement no longer has an
   artificial callback cap. Artificial/manual bonus stack overflow is a
   fixed-array guard for bad direct runtime fixtures, not a public env bug; a
   fully blocked generated map may still need policy. Keep `max_ticks`, body
   overflow, and event overflow classified as truncation-by-design.
10. Keep the fixed `BonusSelfMaster` wall-death parity guard green: normal-wall
   death still kills a SelfMaster/invincible avatar, while body collision
   remains suppressed.
11. Keep the fixed multi-target `BonusAllColor` event-order and older-wins stack
   precedence guards green while broader stack/death cases expand.
12. Timer/random ordering for public bonus scheduling: source/runtime slices are
   pinned, but public ownership of timers, RNG cursor/draw counts, and natural
   bonus scheduling is not broad.
13. Borderless stack behavior and wrap/collision side effects beyond the seeded
   public `BonusGameBorderless` expiry slice. Source/runtime/public
   duration/expiry is now covered by focused tests and a source fixture.
14. Fuller bonus public metadata and replay audit: spawned bonus identity,
   catch/expiry/clear events, random cursor/draw counts, active stack facts,
   and source refs. Current replay preservation is metadata/audit only, not
   full replay arrays.
15. Broader bonus stack/death interactions beyond the promoted
   `BonusSelfFast` wall-death and current catch/expiry/effect slices.
16. Full public replay and final observations for multiplayer rows, including
   reset/RNG provenance, terminal facts, reward/mask maps, and final-row policy.
17. Broader lifecycle and multiplayer parity: natural reset/warmup, warmdown
   frame movement, next-round/match-end policy, present/absent and leave edges,
   masks, and rewards.
18. Browser/source pixel parity later, after source state, public replay, and
   lifecycle rows are stable. Current 704x704 raw -> gray64 source-state raster
   parity is model-observation evidence only, not original browser/canvas pixel
   evidence.
19. Old toy path quarantine: toy-v0/debug routes are historical smoke evidence
   only, not a product runtime or fidelity proof.
20. Modal, speed, and fixed-opponent route smokes stay labeled as route evidence
   or measurement evidence. They are not the next environment priority.
21. Real LightZero `MuZeroGameBuffer` sampled-target parity remains unproven
   downstream work after environment reconstruction priorities.

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
  state and about 16.67 ms source frames. Wrapper APIs may expose `step` or
  `joint_action`, but source behavior is not defined by those wrapper names.
- Collision should stay framed as stored body circle endpoint overlap plus
  death/collision metadata. Visual trail crossing alone is not source truth.
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

1. Convert the completed multiplayer fidelity gap audit into focused
   `VectorMultiplayerEnv` tests/fixes for lifecycle, presence/leave, scoring,
   match-end, replay/final observations, bonus stack/death stress, and 3P/4P
   breadth.
2. Continue the controls fidelity audit for source-frame delivery, held/released
   inputs, terminal-padding boundaries, touch/gamepad input, and
   browser/transport semantics.
3. Keep renderer/fast-path boundaries explicit; optimized/approximate paths
   stay labeled and separate from engine rule fidelity.
4. Keep docs/orchestration rhythm current: docs as working memory, main thread
   plans/delegates/orchestrates, subagents handle bounded audits/tests/docs.
5. Keep continuous trails and segment breaks regression-tested across source,
   vector, wrapper, and GIF/raw-frame inspection paths.
6. Keep bonus sprites/type visibility on the default source-state path; use
   bonus64/rich tensors only for diagnostic proof of hidden bonus facts.
7. Expand 2P wall/borderless/collision-body-depth parity around wrap,
   destination-body collisions, wall priority, own-trail latency, head-head/death
   ordering, and source-frame substep decisions.
8. Keep the new JS/original 2P `BonusSelfFast` stack/death fixture green, and
   add parity for remaining 2P bonus stack/death stress cases.
9. Widen bonus public metadata/replay audit for spawn, catch, expiry, clear,
   active stack, and RNG facts while keeping the claim metadata-only until full
   replay arrays exist.
10. Promote Halley's capacity-audit policy: generated public random tape
   auto-extends, generated natural bonus position retry is not capped by
   `natural_bonus_position_attempt_capacity`, fixture/direct finite tapes stay
   strict, and public natural bonus timer advancement has no artificial
   callback cap. Artificial/manual bonus stack overflow is an intentional
   fixed-array guard; a fully blocked generated map may still need policy.
   Designed truncations are `max_ticks`, body overflow, and event
   overflow.
11. Keep the fixed `BonusSelfMaster` wall/body parity checks in the focused
   bonus suite.
12. Finish timer/random ordering for public bonus scheduling.
13. Finish borderless stack/wrap/collision semantics beyond the seeded public
   expiry slice. Source/runtime/public duration/expiry has focused coverage; do
   not list that proof as missing.
14. Add broader bonus stack/death interactions one source claim at a time,
    starting from the promoted `BonusSelfFast` wall-death proof.
15. Fill full public replay and final-observation rows for multiplayer
   lifecycle and bonus states.
16. Broaden lifecycle/multiplayer parity after the bonus and replay facts are
   named: natural reset/warmup, warmdown movement, next-round/match-end,
   present/absent, leave, masks, and rewards.
17. Keep browser/source pixel parity later. It follows stable source state,
   public replay, and lifecycle rows. Source-vs-vector 704x704 raw ->
   downsampled gray64 parity proves the source-state observation raster only,
   not browser canvas pixels.
18. Keep old toy/debug paths quarantined as historical smoke evidence only.
19. Keep Modal, speed, and fixed-opponent route smokes labeled as route evidence
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
