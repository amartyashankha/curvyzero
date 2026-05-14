# Environment Current Queue - 2026-05-13

Status: working queue, not a completion claim.
Owner surface: Environment docs/process.

Use this as the short operating queue for the current environment
reconstruction push. Deeper evidence stays in focused notes; this page only
records how the work is sequenced and what proof is needed next.

North star: faithful multiplayer CurvyTron environment first, then
speed/training integration. LightZero/training plumbing is only a guarded
downstream interface over the reconstructed environment, not the center of the
work. Do not divert into speed rabbit holes unless the measurement directly
protects source-fidelity reconstruction. Environment owns fidelity; Optimizer
and Coach consume explicit runtime, visual, replay, target-row, profiling, and
learning surfaces without turning those consumer results into source claims.

Guardrail: project-only helpers such as `profile_no_death`,
`death_immunity`/opponent-immortal modes, optimizer modes, and training-helper
modes are valid project features. Preserve them through trainer/replay/target
surfaces with explicit metadata, but do not cite those rows as original
CurvyTron/source-fidelity behavior.
Source CurvyTron control remains held real-time control state advanced through
elapsed-ms source frames; `step` and `joint_action` are wrapper/API terms.

## Latest Reorientation

Do not let tournament, leaderboard, Coach, or Optimizer work become the center
of this thread. Those surfaces are consumers of the reconstructed environment.
Environment Reconstruction should only touch them long enough to state a clear
handoff or to fix a bug that proves the environment surface is wrong.

Current active Environment work is:

1. Close the remaining source-fidelity proof queue for multiplayer
   `VectorMultiplayerEnv`.
2. Harden the remaining behavior proofs for movement-affecting bonuses:
   borderless catch-to-wrap and AllColor visual/observation effects.
   Velocity-while-turning, inverse-while-turning, and radius collision/render
   lifecycle now have focused runtime proof.
3. Preserve the no-death/profile/training helper modes while keeping them
   explicitly separate from original CurvyTron claims.
4. Keep default visual training observations on the source-state
   `browser_lines` raw frame -> gray64 stack path; keep approximate render
   modes labeled.
5. Keep docs as working memory, not as a pass-count dashboard.

## Working Rhythm

- Main thread: plan, delegate, orchestrate, synthesize, and decide. Keep the
  route to faithful multiplayer `VectorMultiplayerEnv` behavior clear and
  reorder the queue when evidence changes risk.
- Subagents: bounded source reads, focused tests, narrow docs updates, and
  small experiments. Each handoff should name the question, files or commands,
  finding, evidence, remaining risk, and next step.
- Docs: working memory, current queue, evidence, conclusions, and gaps. Do not
  turn docs into broad taxonomies or pass-count dashboards.

Active reorientation threads:

- Completed parallel fidelity wave 1: lifecycle/leave, bonus breadth, and
  collision/borderless fixture families were promoted through public env and
  trainer/replay proof where they belong. This was mostly proof hardening, not
  a large engine rewrite.
- Completed parallel fidelity wave 2: lifecycle present/absent and match-end,
  natural bonus spawn/RNG/retry, borderless/death-point edges, and optimized
  render-cache guards were promoted through focused tests. This added one real
  source-fidelity knob, `natural_bonus_rate`, and one trainer helper,
  `advance_warmup(...)`.
- Completed full-game multiplayer gap audit: lifecycle, presence/leave, scoring,
  match-end, replay/final observations, bonus stack/death stress, and 3P/4P
  breadth.
- Completed controls fidelity audit note: source-frame control delivery,
  held/released inputs, and terminal-padding behavior now have focused proof;
  real browser DOM/gamepad package/transport evidence remains product/eval-only.
- Completed renderer/fast-path boundary audit: keep source-state/native
  fidelity claims separate from optimized or approximate render paths.
- Added focused optimized-path reorientation:
  [optimized_path_reorientation_2026-05-13.md](optimized_path_reorientation_2026-05-13.md).
- Added full parallel environment spec and todo:
  [full_fidelity_parallel_spec_2026-05-13.md](full_fidelity_parallel_spec_2026-05-13.md)
  and [parallel_execution_todo_2026-05-13.md](parallel_execution_todo_2026-05-13.md).
- Docs/orchestration rhythm update: docs stay working memory; main thread
  plans/delegates/orchestrates; subagents handle bounded audits, tests, and
  docs.
- Completed right-angle movement fix: `BonusEnemyStraightAngle` was a real
  missed behavior gap, not only bonus metadata. Native/vector movement now
  carries source-like `current_angular_velocity` and `direction_in_loop`, and
  internal source-frame loops arm source moves only once per outer decision.
- Completed default bonus probability fix: scalar and vector paths now use the
  source default map, not a flat fallback. Defaults are inverse `0.8`,
  straight-angle `0.6`, borderless `0.8`, dynamic clear probability, and
  `1.0` for the other source-default bonuses.
- Completed `BonusSelfMaster` print-manager side-effect fix: vector runtime
  now applies the source-like invincible/printing contract and public env proof
  exists. Remaining audit queue starts after SelfMaster.
- Completed focused active-turn proof for speed and inverse bonuses:
  `tests/test_vector_runtime.py` now verifies that a speed bonus refreshes the
  turn rate for an already held turn, and that inverse preserves the current
  turn sign on catch/expiry until the next source input event re-arms it.
- Completed focused radius lifecycle proof: `tests/test_vector_runtime.py`
  now verifies that `BonusSelfSmall` radius changes affect normal wall checks,
  body collision checks, the raw browser-like RGB frame, and the downsampled
  gray64 observation.

## Current Snapshot

Done:

- `SourceStateMultiplayerTrainerSurface` emits per-seat source-state visual
  stacks, live-seat policy rows, masks, survival-plus-bonus rewards, terminal
  visual final observations, and honest render/source metadata.
- `SourceStateMultiplayerTrainerReplayRecorder` stores copied in-memory replay
  arrays over time, including terminal visual final observations and variable
  live-policy rows.
- `SourceStateMultiplayerTargetRowsV0` now builds repo-owned target rows from
  trainer replay arrays plus policy-row records. Focused tests cover
  reset-to-step alignment, terminal final observations/rewards, P=4 row
  mapping, `to_play=-1`, copied arrays, invalid policy/action rejection, and
  no-death/death-immunity metadata preservation.
- `SourceStateMultiplayerSampleBatchV0` now builds deterministic sample
  batches on top of target rows through
  `build_source_state_multiplayer_sample_batch_v0`. Focused target-row/sample
  batch tests reported `12 passed` locally per worker.
- Fake/injected native `GameSegment` mapping from
  `SourceStateMultiplayerTargetRowsV0` is implemented in
  `src/curvyzero/training/multiplayer_source_state_native_bridge.py`, with
  focused tests in `tests/test_multiplayer_source_state_native_bridge.py`. It
  is injection-only, does not import LightZero, preserves project-only mode
  metadata, and keeps native/LightZero/training/buffer/learner claims false.
- The separate opt-in real-LightZero construction helper exists as
  construction smoke only. It does not prove `MuZeroGameBuffer` sampled-target
  parity, learner updates, evaluation quality, or true multiplayer self-play.
- Focused controls, product-route, bonus-default, hit-owner, old-body, and
  LightZero boundary proofs are current enough to support the training surface,
  with gaps below.

Current focus:

- Environment Reconstruction is on the next multiplayer fidelity pass:
  broader lifecycle families, bonus RNG/stack stress, broader collision edges,
  renderer/fast-path boundaries, and docs/orchestration rhythm. Treat the
  completed audits and wave-1 through wave-5 tests as inputs, not broad
  completion claims.
- The optimized two-player trainer image path is an equivalence optimization
  inside `SourceStateGray64Stack4`, not a second environment. It reads
  `VectorMultiplayerEnv.state`, emits the same declared source-state gray64
  stack, exposes dirty-render stats through trainer-surface metadata, and now
  has focused guards for stale bonus sprites, trail/body clears, wrap breaks,
  reset/new-round state, and terminal final-frame equality against a direct
  source-state render. Dirty render stats aggregation now sums numeric
  top-level fields and merges nested dict counters instead of overwriting
  rows.

## Wave 3 Integrated

Wave 3 is integrated and broad-validated. It was fixture promotion and
stale-claim cleanup, with engine fidelity kept separate from
render/trainer-observation caching.

- 4P lifecycle/present-absent/match-end target fixtures:
  `source_lifecycle_present_absent_4p_round_new.json`,
  `source_lifecycle_present_absent_4p_survivor_score_round_end.json`,
  `source_lifecycle_present_absent_4p_next_round.json`,
  `source_lifecycle_present_absent_4p_tie_at_max_score.json`,
  `source_lifecycle_match_end_at_max_score_4p.json`, and
  `source_lifecycle_multi_round_match_end_4p.json`. These now have focused
  source/public/trainer-replay promotion in
  `tests/test_multiplayer_lifecycle_source_fixture_promotion.py`.
- Natural bonus cap/default-weight target fixtures:
  `source_bonus_spawn_cap_twenty_step.json`,
  `source_bonus_default_weights_type_rng_step.json`,
  `source_bonus_default_weights_select_game_clear_step.json`, and
  `source_bonus_default_weights_game_clear_full_probability_step.json`.
  The cap-twenty and default-weights type-RNG fixtures now have public-env
  promotion in `tests/test_multiplayer_bonus_spawn_rng_breadth.py`. Wave 4
  later promoted the two GameClear probability fixtures.
- Wave-3 collision/trail/body proof targets:
  `source_trail_gap_hole_space_safe_step.json`,
  `source_trail_gap_stored_body_still_kills_step.json`,
  `source_trail_gap_print_to_hole_boundary_kills_step.json`,
  `source_trail_gap_hole_to_print_boundary_kills_step.json`,
  `source_body_opponent_tangent_safe_step.json`,
  `source_body_opponent_overlap_kills_step.json`,
  `source_body_old_opponent_overlap_kills_step.json`,
  `source_body_same_frame_point_kills_step.json`, and
  `source_body_same_frame_point_control_safe_step.json`. Treat existing
  source/runtime canaries as inputs. The four `source_trail_gap_*` fixtures now
  have public/trainer/replay propagation in
  `tests/test_multiplayer_collision_breadth_fidelity.py`; Wave 4 later promoted
  the listed body same-frame/opponent canaries where useful.
  `source_collision_order_batch.json` is covered by its promoted underlying
  fixtures,
  `source_collision_death_point_kills_later_player_step.json` and
  `source_collision_head_head_reverse_order_single_death_step.json`.

## Wave 4 Integrated

Wave 4 is integrated and broad-validated. It stayed a narrow
environment-fidelity wave; render/trainer-observation caching remains a
separate optimization surface and is not engine-fidelity proof.

- Natural GameClear bonus fixture targets:
  `source_bonus_default_weights_select_game_clear_step.json` and
  `source_bonus_default_weights_game_clear_full_probability_step.json`.
  Both now have public-env spawn/probability promotion in
  `tests/test_multiplayer_bonus_spawn_rng_breadth.py`.
- Lifecycle leave target:
  `source_lifecycle_mid_round_remove_avatar_2p.json`. This now has public env
  plus trainer/replay promotion in
  `tests/test_multiplayer_presence_leave_fidelity.py`.
- Lower-priority body/collision canaries:
  `source_body_opponent_tangent_safe_step.json`,
  `source_body_opponent_overlap_kills_step.json`,
  `source_body_old_opponent_overlap_kills_step.json`,
  `source_body_same_frame_point_kills_step.json`, and
  `source_body_same_frame_point_control_safe_step.json`. The opponent
  tangent/overlap and same-frame point fixtures now have public env plus
  trainer/replay propagation in
  `tests/test_multiplayer_collision_breadth_fidelity.py`. The old-body
  `old:true` case is now promoted too: public info, trainer surface, and
  trainer replay carry `death_hit_old`. `source_collision_order_batch.json` is
  already covered by its promoted death-point and head-head underlying fixtures.

## Wave 5 Integrated

Wave 5 is integrated for environment-owned work. Keep the downstream claims
narrow.

- Old-body `old:true` metadata is now explicit. Runtime stores `body_birth_ms`,
  source-order body hits retain the exact hit body slot, public info exposes
  `death_hit_old`, debug die events use the same old flag, and trainer replay
  copies the field.
- Controls/browser tail proof is split cleanly. Trainer-relevant controls are
  proven through original JS input reduction, public runtime action mapping,
  held/released source frames, and 3P/4P control cases. Real browser DOM,
  actual gamepad package, and Socket.IO transport remain browser-play/product
  evidence, not a current source-state trainer blocker.
- The LightZero boundary now has a real source-state `[4,64,64]` fake
  GameSegment preservation test. It proves our export shape and metadata, not
  full LightZero sampled-target parity.
- Real LightZero sampled-target parity stays downstream of reconstructed
  environment and repo-owned target rows. A `MuZeroGameBuffer` sampled reward,
  value, policy, action, mask, observation, and `to_play` audit is useful
  downstream, but it is not primary environment-fidelity evidence.
- Validation after Wave 5: focused collision/old-body plus seed checks
  reported `16 passed`; controls reported `12 passed`; LightZero boundary
  reported `16 passed, 1 skipped`; vector runtime/lifecycle reported
  `96 passed`; broader environment validation reported `571 passed, 2 skipped`.

## Latest Completed Fixes

These fixes are integrated after Wave 5. Keep the claims scoped to the tested
surfaces named here.

- `BonusEnemyStraightAngle` movement fidelity is fixed for native/vector
  movement. The runtime now tracks source-like `current_angular_velocity` and
  `direction_in_loop`; source-frame loops arm the source move once per outer
  decision instead of once per internal frame. Proof includes a low-level snap
  test and a public seeded catch test across `decision_source_frames=4`.
- Default bonus probabilities were a real source-default bug. Scalar and vector
  paths now use inverse `0.8`, straight-angle `0.6`, borderless `0.8`, dynamic
  clear probability, and `1.0` for the other source-default bonuses. The
  probability and spawn tests were updated.
- Dirty render stats aggregation now sums numeric top-level fields and merges
  nested dict counters.
- `BonusSelfMaster` print-manager side effects are implemented in vector
  runtime. Contract: catch gives `invincible=true` and `printing=-1` for
  `7500` ms; body/trail death is blocked while active; normal wall death still
  kills unless project-only `profile_no_death`, `death_immunity`, or
  opponent-immortal modes are enabled; expiry restores `invincible=false` and
  restarts printing/PrintManager through source-like side effects; death before
  expiry clears the stack and must not restart printing or leave invincible
  true.
- Validation after the right-angle/default/dirty-render fixes: the focused
  controls/bonus/runtime/
  render-surface suite
  `tests/test_source_env.py tests/test_vector_runtime.py tests/test_bonus_spawn_rng_public_fidelity.py tests/test_multiplayer_bonus_spawn_rng_breadth.py tests/test_vector_multiplayer_env.py -q`
  reported `260 passed`; the broad environment sweep reported
  `578 passed, 2 skipped`; `ruff`, the environment doc guard, and
  `git diff --check` passed.
- Focused SelfMaster validation reported runtime `self_master or print_manager`
  `11 passed`, public `self_master` `5 passed`, and the broader focused
  environment suite `321 passed`; `ruff` and diff checks passed. The full
  environment sweep after the SelfMaster fix reported `591 passed, 2 skipped`.

Next:

- Convert the completed full-game multiplayer, controls, and renderer boundary
  audits into focused tests and fixes against `VectorMultiplayerEnv` before
  treating any speed/training task as primary.
- Treat Hume's opt-in real-LightZero construction helper as done but
  construction-smoke only. Real `MuZeroGameBuffer` sampled reward, value,
  policy, action, mask, observation, and `to_play` parity remains unproven.
  Keep native integration, learner updates, eval quality, and true multiplayer
  self-play false-claimed until those surfaces are actually tested.

## Remaining Proof Queue

1. Target-row adapter, deterministic sample batches, and fake/injected native
   `GameSegment` mapping for source-state multiplayer training. Done for the
   repo-owned v0 row, sample-batch, and injection-only bridge contracts.
   Remaining downstream follow-up is real LightZero buffer sampled parity, not
   the current primary Environment lane.
2. Controls source-frame fidelity.
   Latest proof covers original JS keyboard reduction/server move delivery, 2P
   public action-to-native-control mapping, held controls across
   `decision_source_frames`, release-to-straight, invalid/live action errors,
   inactive noops, terminal-padding noops under `decision_source_frames`, 3P/4P
   public one-frame trajectory parity, 4P held-control parity, and terminal
   early-stop through both direct runtime and the LightZero-facing wrapper.
   Real browser DOM, gamepad package, and Socket.IO proof are product/eval
   evidence only, not trainer blockers.
3. End-to-end 2P product route.
   Latest direct `VectorMultiplayerEnv` proof covers raw RGB -> gray64, seeded
   `BonusGameClear`, stale trail/body clear, live ticks, terminal wall death,
   rewards, final observation masks, and metadata replay. Latest
   LightZero-facing wrapper proof covers wrapper-side scalar joint-action
   decoding, raw RGB -> gray64 stack, held source frames, terminal final
   observation, rewards, masks, and native sidecars. New
   `SourceStateMultiplayerTrainerSurface`
   proof covers per-seat source-state visual stacks, live-seat policy-row
   mapping, survival-plus-bonus rewards, render-mode guards, and terminal
   visual final observations over `VectorMultiplayerEnv`. New in-memory replay
   proof stores those trainer arrays over time. New target-row and sample-batch
   proof validates replay -> target transition rows -> deterministic sample
   batches. New fake/injected native bridge proof maps those rows to injected
   `GameSegment`-like objects without importing LightZero or changing native
   claims. New public `VectorMultiplayerEnv` lifecycle proof covers P=3/P=4
   match-mode rows where one row starts the next round and one row ends the
   match after warmdown. New trainer/replay lifecycle proof carries that shape
   through `SourceStateMultiplayerTrainerSurface.advance_warmdown(...)` and
   `SourceStateMultiplayerTrainerReplayRecorder`: final visual rows, final
   rewards, masks, copied arrays, and variable live-policy rows are preserved.
   Remaining work is durable artifact plumbing and downstream real LightZero
   buffer sampled parity.
4. Bonus probability and source defaults.
   Latest source/default fix uses the source default map in both scalar and
   vector paths: inverse `0.8`, straight-angle `0.6`, borderless `0.8`,
   dynamic clear probability, and `1.0` for the other source-default bonuses.
   Latest public/runtime proof pins corrected default boundary draws, RNG
   labels/cursors, next-delay scheduling, spawned position, and the full
   source-default set. New focused 2P `BonusSelfFast`
   stack/death proof catches three speed bonuses, kills the boosted player on
   the normal wall, and proves death clears the active stack while later source
   timeout callbacks only emit inert stack-removal events. New same-step timer
   proof pins the narrower ordering where one `BonusSelfFast` expiry drains
   before the wall-death update: speed is restored to `16`, movement uses that
   speed, p0 dies on the normal wall, p1 scores, and the stack stays empty.
   New trainer/replay proof carries both terminal cases through
   `SourceStateMultiplayerTrainerSurface` and
   `SourceStateMultiplayerTrainerReplayRecorder`: terminal visual final
   observation rows, final reward maps, death metadata, winner/loser facts,
   step counters, and compact bonus audit metadata survive into replay
   records. This is source-runner/public-vector plus trainer/replay
   preservation, not browser event-loop or pixel proof. New 4P target-filter
   proof pins the source/public rule that enemy bonuses affect only other alive
   avatars, all-avatar bonuses affect only alive avatars, absent seats are not
   targeted because they are not alive, and game bonuses still apply to global
   game state. New source-backed 4P terminal proof covers `BonusEnemySlow`: a
   JS oracle fixture and public vector mirror now pin p0 catching the enemy
   bonus, p1/p2/p3 receiving slowed stack entries, those targets wall-dying
   before expiry, death clearing their stack rows without restoring dead-player
   speed, and p0 winning the round. The matching trainer/replay proof preserves
   final visual rows, final reward rows, death order, winner/loser facts, step
   counters, and compact bonus metadata.
   Fresh bonus-breadth proof in
   `tests/test_multiplayer_bonus_breadth_fidelity.py` promotes
   `source_bonus_game_borderless_expiry_restore_step.json`,
   `source_bonus_self_small_expiry_restore_step.json`, and
   `source_bonus_self_small_wall_death_no_catch_step.json` through
   `SourceStateMultiplayerTrainerSurface` and
   `SourceStateMultiplayerTrainerReplayRecorder`. Replay audit metadata now
   preserves `borderless` so the borderless catch/expiry fact survives into
   records.
   Fresh natural-spawn proof in
   `tests/test_multiplayer_bonus_spawn_rng_breadth.py` promotes
   `source_bonus_spawn_type_position_rng_step.json`,
   `source_bonus_spawn_game_world_retry_step.json`, and
   `source_bonus_spawn_bonus_world_retry_step.json` through public
   `VectorMultiplayerEnv`. The public env now supports `natural_bonus_rate`
   so source fixture `bonus_rate=1` produces the same shortened spawn delay.
   Wave-3 natural-spawn proof also promotes
   `source_bonus_spawn_cap_twenty_step.json` and
   `source_bonus_default_weights_type_rng_step.json` through public env.
   Wave 4 promotes the two GameClear probability fixtures. `BonusSelfMaster`
   printing side effects are now covered in vector runtime and public env
   proof: active SelfMaster blocks body/trail death, normal wall death still
   kills unless project-only no-death/`death_immunity`/opponent-immortal
   helpers are enabled, expiry restarts printing, and death before expiry
   clears the stack without restarting printing. Remaining bonus-spawn work is
   broader retry/RNG stress and the still-open queue items below without
   narrowing the default bonus set.
5. Hit-owner ordering.
   Latest runtime fix scans source-compatible body-hit corner islands and newest
   bodies first. Latest stress tests cover 4P newest-owner overlap, 4P corner
   island order, 3P own-body latency, and 4P two-victim hit-owner metadata.
   New focused propagation proof carries a 3P terminal body-hit case and a 4P
   nonterminal two-victim case through public env, trainer surface, replay
   records, and debug die events. New raw JS oracle fixtures now pin the exact
   3P terminal and 4P nonterminal stress shapes, and public
   `VectorMultiplayerEnv` mirrors them from fixture-seeded state. Remaining
   work is broader collision edges beyond those two promoted shapes.
   Fresh collision-breadth proof in
   `tests/test_multiplayer_collision_breadth_fidelity.py` promotes
   head-head reverse-order single death, own-trail latency delta 3 safe versus
   delta 4 kill, borderless destination-body skip followed by next-frame kill,
   and 4P ordered wall deaths/survivor scoring through public env plus
   trainer/replay records. Fresh borderless/collision proof in
   `tests/test_multiplayer_borderless_collision_breadth.py` promotes
   death-point ordering, plain borderless wrap, and exact-edge/corner-axis wrap
   through public env plus trainer/replay where player count permits; the 1P
   PrintManager wrap-toggle fixture is source-runner proof only. Remaining
   work is broader collision stress and wider visual pixel proof.
6. Wider multiplayer.
   Public 3P/4P lifecycle now has a focused mixed-row match-mode proof for
   reset/warmup, round win, warmdown, next-round, match-end, masks, rewards,
   and public final rows. Trainer/replay now has the matching focused proof
   for the same shape, including legal/live policy rows and terminal visual
   final rows. Public env plus trainer/replay now also have a focused P=3/P=4
   presence/leave proof for mixed active-row and staged-warmdown removal:
   present/alive masks, absent action slots, warmdown next-round carryover,
   trainer live-policy rows, and replay array storage. Focused source-backed
   public proofs also exist for `source_lifecycle_remove_avatar_to_single_present_3p.json`
   and `source_lifecycle_remove_avatar_during_warmdown_3p.json`. Remaining
   work is broader leave variants, more 3P/4P bonus stack/death combinations
   beyond the focused 4P `BonusEnemySlow` terminal replay proof, and later
   browser/canvas pixel parity.
   Fresh lifecycle/leave proof in
   `tests/test_multiplayer_presence_leave_fidelity.py` promotes
   `source_lifecycle_remove_avatar_to_single_present_3p.json`,
   `source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json`,
   `source_lifecycle_mid_round_remove_avatar_4p_continue_round_end.json`, and
   `source_lifecycle_remove_avatar_during_warmdown_3p.json` through public env
   and/or trainer/replay surfaces. Wave 4 also promotes the dedicated 2P
   mid-round leave fixture through public env plus trainer/replay.
   Fresh lifecycle-source promotion in
   `tests/test_multiplayer_lifecycle_source_fixture_promotion.py` covers 3P
   present/absent round-new, survivor-score round-end, next-round, 2P/3P
   max-score match-end, and 3P multi-round match-end fixtures. Trainer
   surface now has `advance_warmup(...)` so warmup rows can be recorded.
   Wave-3 lifecycle-source promotion now covers the named 4P present/absent,
   tie-at-max-score, max-score match-end, and multi-round match-end fixtures
   through public env plus trainer/replay assertions. Remaining lifecycle work
   is broader leave variants and any new source fixture families we choose to
   promote, not the earlier 4P starter list.
7. Native LightZero bridge after environment reconstruction and target
   rows/sample batches.
   Fake/injected native `GameSegment` mapping from
   `SourceStateMultiplayerTargetRowsV0` is done and injection-only. Hume's
   separate opt-in real-LightZero construction helper is also done, but it is
   still construction-smoke only. Real `MuZeroGameBuffer` sampled reward, value,
   policy, action, mask, observation, and `to_play` parity remains unproven.
   This remains downstream interface work, not the main reconstruction job.

Validation history through latest fixes:

- Parallel fidelity wave 1 combined suite:
  `tests/test_multiplayer_presence_leave_fidelity.py`,
  `tests/test_multiplayer_bonus_breadth_fidelity.py`,
  `tests/test_multiplayer_collision_breadth_fidelity.py`,
  trainer replay bonus/lifecycle tests, hit-owner tests, and 2P collision
  tests reported `58 passed`.
- Parallel fidelity wave 2 focused suite:
  lifecycle-source promotion, bonus spawn/RNG breadth, borderless collision
  breadth, prior wave-1 tests, render/trainer visual tests, and related
  replay tests reported `157 passed, 1 skipped`.
- Parallel fidelity wave 3 focused suite:
  4P lifecycle-source promotion, bonus cap/default-weight type-RNG promotion,
  trail-gap/body persistence propagation, and related public/trainer/replay
  tests reported `37 passed`.
- Parallel fidelity wave 4 focused suite:
  GameClear probability fixtures, 2P mid-round leave, body/same-frame
  collision canaries, and related public/trainer/replay tests reported
  `40 passed`.
- Focused render/trainer test sweep:
  `tests/test_curvytron_two_seat_render_mode.py`,
  `tests/test_multiplayer_source_state_trainer_surface.py`, and
  `tests/test_vector_visual_observation.py` reported `78 passed, 1 skipped`.
- Wave-5 broader environment sweep over source, runtime, multiplayer, controls,
  trainer/replay/target/native bridge, render, and visual observation tests
  reported `571 passed, 2 skipped`.
- Latest focused controls/bonus/runtime/render-surface suite:
  `tests/test_source_env.py tests/test_vector_runtime.py tests/test_bonus_spawn_rng_public_fidelity.py tests/test_multiplayer_bonus_spawn_rng_breadth.py tests/test_vector_multiplayer_env.py -q`
  reported `260 passed`.
- Pre-SelfMaster broad environment sweep reported `578 passed, 2 skipped`.
- Focused SelfMaster validation reported runtime `self_master or print_manager`
  `11 passed`, public `self_master` `5 passed`, and broader focused suite
  `321 passed`.
- Full environment sweep after the SelfMaster fix reported
  `591 passed, 2 skipped`; `ruff`, the environment doc guard, and
  `git diff --check` passed.

## Queue Discipline

- Keep at most one main active implementation lane and one docs cleanup lane.
- Call a gap closed only for the surface actually tested: source truth,
  product runtime, trainer/replay, or renderer.
- Record blocked items with the missing proof, not just the symptom.
- Keep old toy/debug paths as historical smoke evidence unless a focused note
  promotes one for a narrow proof surface.
- Preserve no-death/profile/`death_immunity`/opponent-immortal helpers as
  project features, but do not cite them as source-fidelity proof.
- Keep optimized and approximate render paths explicitly labeled and separate
  from engine rule fidelity.
