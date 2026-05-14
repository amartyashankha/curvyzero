# Parallel Execution Todo - 2026-05-13

Status: concrete checklist, not a completion claim.
Owner surface: Environment Reconstruction.
Depends on:
[full_fidelity_parallel_spec_2026-05-13.md](full_fidelity_parallel_spec_2026-05-13.md).

Use this as the working checklist for parallel waves. Keep engine/runtime
fidelity separate from render/trainer-observation optimization and downstream
LightZero work.

## Wave 0 - Coordination

- [x] Keep `VectorMultiplayerEnv` as the product runtime target.
- [x] Keep source fixture names in each worker handoff.
- [x] For each worker, name the proof surface: source, runtime, public env,
  trainer/replay, render/trainer observation, or downstream LightZero.
- [x] Keep optimization language narrow: current optimization work is mostly
  render/trainer-observation caching and metadata.
- [x] Keep main fidelity language narrow: engine/runtime/lifecycle/bonus/
  collision/replay/final-state work is the priority.

Wave 1 partial result, 2026-05-13:

- Lifecycle worker promoted four source lifecycle/leave fixtures through
  public and trainer/replay proof. Wave 4 later closed the 2P mid-round leave
  trainer/replay gap.
- Bonus worker promoted three source bonus fixtures through trainer/replay and
  added `borderless` to source-state trainer replay audit records.
- Collision worker promoted five collision/borderless/death-order fixtures
  through public plus trainer/replay proof.
- Combined worker validation reported `58 passed`; render/trainer validation
  reported `78 passed, 1 skipped`; broader environment validation reported
  `526 passed, 2 skipped`.

Wave 2 partial result, 2026-05-13:

- Lifecycle worker promoted 3P present/absent and 2P/3P match-end fixtures,
  and added trainer `advance_warmup(...)` so warmup rows can be recorded.
- Bonus worker promoted three natural bonus spawn/RNG/retry fixtures and added
  `natural_bonus_rate`, matching source delay math for fixture `bonus_rate=1`.
- Collision worker promoted death-point ordering, plain borderless wrap,
  exact-edge/corner-axis wrap, and source-only 1P PrintManager wrap-toggle
  proof.
- Render worker added stronger 2P optimized gray64 equivalence guards for
  bonus type changes, trail/body clears, wrap breaks, reset/new-round state,
  and terminal final frames.
- Merged wave-2 validation reported `157 passed, 1 skipped`; broader
  environment validation reported `547 passed, 2 skipped`.

Wave 3 integrated, 2026-05-13:

Main integration broad-validated this wave after worker results landed.

- [x] 4P lifecycle/present-absent/match-end fixture closure. Exact targets:
  `source_lifecycle_present_absent_4p_round_new.json`,
  `source_lifecycle_present_absent_4p_survivor_score_round_end.json`,
  `source_lifecycle_present_absent_4p_next_round.json`,
  `source_lifecycle_present_absent_4p_tie_at_max_score.json`,
  `source_lifecycle_match_end_at_max_score_4p.json`, and
  `source_lifecycle_multi_round_match_end_4p.json`.
- [x] Natural bonus cap/default-weight fixture promotion. Exact targets:
  `source_bonus_spawn_cap_twenty_step.json`,
  `source_bonus_default_weights_type_rng_step.json`,
  `source_bonus_default_weights_select_game_clear_step.json`, and
  `source_bonus_default_weights_game_clear_full_probability_step.json`.
  The first two landed in Wave 3; the two GameClear probability fixtures landed
  in Wave 4.
- [x] Remaining collision/trail/body proof for the named `source_trail_gap_*`
  fixtures. Exact targets handled:
  `source_trail_gap_hole_space_safe_step.json`,
  `source_trail_gap_stored_body_still_kills_step.json`,
  `source_trail_gap_print_to_hole_boundary_kills_step.json`, and
  `source_trail_gap_hole_to_print_boundary_kills_step.json`.
- [x] Lower-priority body/collision canaries promoted where useful. Exact
  targets:
  `source_body_opponent_tangent_safe_step.json`,
  `source_body_opponent_overlap_kills_step.json`,
  `source_body_old_opponent_overlap_kills_step.json`,
  `source_body_same_frame_point_kills_step.json`, and
  `source_body_same_frame_point_control_safe_step.json`.
  `source_collision_order_batch.json` is covered by its promoted underlying
  death-point and head-head fixtures.
- [x] Keep optimization language simple: optimization work is
  render/trainer-observation caching; engine fidelity remains lifecycle,
  bonus, collision, replay, and final-state behavior.
- Focused wave-3 validation reported `37 passed`; broader environment
  validation reported `559 passed, 2 skipped`.

Wave 4 integrated, 2026-05-13:

- [x] Natural GameClear bonus targets:
  `source_bonus_default_weights_select_game_clear_step.json` and
  `source_bonus_default_weights_game_clear_full_probability_step.json`.
- [x] 2P mid-round leave target:
  `source_lifecycle_mid_round_remove_avatar_2p.json`.
- [x] Lower-priority body/collision canaries promoted where useful:
  `source_body_opponent_tangent_safe_step.json`,
  `source_body_opponent_overlap_kills_step.json`,
  `source_body_same_frame_point_kills_step.json`, and
  `source_body_same_frame_point_control_safe_step.json`.
- [x] Old-body body/collision canary promoted:
  `source_body_old_opponent_overlap_kills_step.json`. Public info, trainer
  surface, and trainer replay now carry `death_hit_old`.
- [x] Collision-order batch covered by promoted underlying fixtures:
  `source_collision_order_batch.json` contains the promoted death-point and
  head-head fixtures that cover the important death-order behavior.
- [x] Keep optimization language narrow: render/trainer-observation caching is
  separate from engine fidelity.
- Focused wave-4 validation reported `40 passed`; broader environment
  validation reported `566 passed, 2 skipped`.

Wave 5 integrated, 2026-05-13:

- [x] Old-body `old:true` metadata promoted through public env, trainer
  surface, and trainer replay as `death_hit_old`.
- [x] Controls/browser tail proof split: source-state trainer controls are
  covered; real browser DOM/gamepad package/Socket.IO remains product/eval
  evidence only.
- [x] LightZero boundary test now uses real source-state `[4,64,64]` rows with
  the fake GameSegment bridge.
- [x] Validation after Wave 5: collision/seed `16 passed`, controls
  `12 passed`, LightZero boundary `16 passed, 1 skipped`, vector
  runtime/lifecycle `96 passed`, broader environment sweep
  `571 passed, 2 skipped`.
- [ ] Keep real LightZero sampled-target parity audit downstream. A
  `MuZeroGameBuffer` sampled reward, value, policy, action, mask, observation,
  and `to_play` parity probe is not primary environment-fidelity proof and
  should consume the reconstructed environment/target-row contract after that
  contract is stable.
- [ ] Keep optimization language narrow: only render/trainer-observation
  caching is in scope here, and it must stay separate from engine fidelity.

Latest completed fixes, 2026-05-13:

- [x] Fix `BonusEnemyStraightAngle` as native/vector movement behavior:
  source-like `current_angular_velocity`, `direction_in_loop`, and internal
  source-frame loops that arm source moves only once per outer decision.
- [x] Add straight-angle proof: low-level snap test plus public seeded catch
  test across `decision_source_frames=4`.
- [x] Fix source-default bonus probabilities in scalar and vector paths:
  inverse `0.8`, straight-angle `0.6`, borderless `0.8`, dynamic clear
  probability, and `1.0` for other source-default bonuses.
- [x] Update probability and spawn tests for the default map.
- [x] Fix dirty render stats aggregation by summing numeric top-level fields
  and merging nested dict counters.
- [x] Implement `BonusSelfMaster` print-manager side effects in vector runtime:
  `invincible=true` and `printing=-1` for `7500` ms, body/trail death blocked
  while active, normal wall death still lethal unless project-only no-death or
  `death_immunity`/opponent-immortal modes are enabled, expiry restarts
  printing/PrintManager, and death before expiry clears the stack without
  restarting printing.
- [x] Validation after the right-angle/default/dirty-render fixes: focused
  controls/bonus/runtime/
  render-surface suite
  `tests/test_source_env.py tests/test_vector_runtime.py tests/test_bonus_spawn_rng_public_fidelity.py tests/test_multiplayer_bonus_spawn_rng_breadth.py tests/test_vector_multiplayer_env.py -q`
  reported `260 passed`; broad environment sweep reported
  `578 passed, 2 skipped`; `ruff`, environment doc guard, and
  `git diff --check` passed.
- [x] Focused SelfMaster validation reported runtime filter
  `self_master or print_manager` `11 passed`, public `self_master`
  `5 passed`, and broader focused environment suite `321 passed`; `ruff` and
  diff checks passed.
- [x] Full environment sweep after the SelfMaster fix reported
  `591 passed, 2 skipped`; the environment doc guard and `git diff --check`
  passed.

Post-right-angle audit queue, 2026-05-13:

Use this as a ranked active queue from audits. It is not a claim that all items
are broken; it is the order for checking source behavior and adding the missing
proof or fix.

Completed:

- [x] `BonusEnemyStraightAngle` native/vector movement semantics and public
  seeded catch proof across `decision_source_frames=4`.
- [x] Default bonus probabilities: inverse `0.8`, straight-angle `0.6`,
  borderless `0.8`, dynamic clear probability, and `1.0` for the other
  source-default bonuses.
- [x] `BonusSelfMaster` printing side effects: active invincibility plus
  `printing=-1`, body/trail death immunity only while active, normal wall death
  still lethal unless project-only no-death/`death_immunity`/opponent-immortal
  helpers are enabled, expiry restart, and death-before-expiry cleanup.
- [x] Velocity and inverse active-turn behavior: focused runtime proof now
  checks that speed bonuses refresh the current held-turn rate, and inverse
  preserves the current turn sign on catch/expiry until the next source input
  event applies the new direction.
- [x] Radius collision/render lifecycle: focused runtime proof now checks that
  `BonusSelfSmall` radius changes affect wall checks, body collision checks,
  raw browser-like RGB rendering, and the downsampled gray64 observation.
- [x] Direct borderless catch-to-wrap: focused runtime proof catches
  `BonusGameBorderless`, crosses an arena edge while active, wraps, and avoids
  normal wall death.
- [x] AllColor visual/observation: focused public-env proof catches
  `BonusAllColor`, verifies rotated `avatar_color` reaches browser-like RGB
  and gray64, then verifies expiry restores the baseline frame.

Remaining:

- No open items from this post-right-angle proof queue.

Wrapper/source input semantics note:

- Trainer `joint_action`/held action is a wrapper decision surface and can
  re-emit controls per decision.
- Source-native input is input-event/current-turn state advanced through
  elapsed-ms frames.
- Proofs should label which surface they exercise before comparing behavior.

## Wave 1 - Immediate No-Refactor Lanes

These can start in parallel now if workers stay on their target files.

### Lane L1 - Mid-Round Remove

- [x] Promote `source_lifecycle_mid_round_remove_avatar_2p.json` through
  `VectorMultiplayerEnv`.
- [x] Promote `source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json`
  through `VectorMultiplayerEnv`.
- [x] Promote `source_lifecycle_mid_round_remove_avatar_4p_continue_round_end.json`
  through `VectorMultiplayerEnv`.
- [x] Add public proof in `tests/test_multiplayer_presence_leave_fidelity.py`
  or `tests/test_multiplayer_lifecycle_fidelity.py`.
- [x] Add trainer/replay preservation only after the public env facts are
  stable.
- [ ] Proof needed: present/alive masks, absent action slots, death order,
  round-end or continue-round facts, final rows, and replay metadata.
- [ ] Refactor needed now: no.

### Lane L2 - Present/Absent And Match End

- [x] Mirror the 3P present/absent fixtures through the public env.
- [x] Mirror the 4P present/absent fixtures through the public env.
- [x] Mirror `source_lifecycle_match_end_at_max_score_2p.json`,
  `source_lifecycle_match_end_at_max_score_3p.json`, and
  `source_lifecycle_match_end_at_max_score_4p.json`.
- [x] Mirror `source_lifecycle_multi_round_match_end_3p.json` and
  `source_lifecycle_multi_round_match_end_4p.json`.
- [ ] Target tests: `tests/test_source_lifecycle_runner.py`,
  `tests/test_lifecycle_oracle.py`, `tests/test_vector_multiplayer_env.py`,
  `tests/test_multiplayer_lifecycle_fidelity.py`, and
  `tests/test_multiplayer_source_state_lifecycle_replay.py`.
- [ ] Proof needed: source events, public state, winner/match facts, timers,
  masks, reset/autoreset separation, and final observations agree.
- [ ] Refactor needed now: no, unless lifecycle summary assembly starts
  duplicating across public info, replay, and trainer records.

### Lane B1 - Bonus Spawn And RNG

- [x] Reuse `source_bonus_spawn_type_position_rng_step.json`.
- [x] Reuse `source_bonus_spawn_game_world_retry_step.json`.
- [x] Reuse `source_bonus_spawn_bonus_world_retry_step.json`.
- [x] Reuse `source_bonus_spawn_cap_twenty_step.json`.
- [x] Reuse the default-weight fixtures, including
  `source_bonus_default_weights_type_rng_step.json`.
- [ ] Target tests: `tests/test_env_scenarios.py`, `tests/test_source_env.py`,
  `tests/test_vector_runtime.py`,
  `tests/test_bonus_spawn_rng_public_fidelity.py`, and
  `tests/test_vector_multiplayer_env.py`.
- [ ] Proof needed: RNG labels/cursors, next-delay scheduling, retries, cap
  behavior, spawned identity, public info, and replay metadata agree.
- [ ] Refactor needed now: no.

### Lane B2 - Bonus Stack, Timer, And Death

- [ ] Reuse `source_bonus_self_fast_stack_death_late_expiry_step.json`.
- [ ] Reuse `source_bonus_self_fast_expiry_then_wall_death_same_tick_step.json`.
- [ ] Reuse `source_bonus_enemy_slow_4p_stack_wall_death_terminal_step.json`.
- [ ] Reuse `source_bonus_game_borderless_catch_step.json`.
- [x] Reuse `source_bonus_game_borderless_expiry_restore_step.json`.
- [x] Reuse `source_bonus_self_small_expiry_restore_step.json`.
- [x] Reuse `source_bonus_self_small_wall_death_no_catch_step.json`.
- [x] Add missing public/trainer/replay mirrors one fixture family at a time
  for the three fixtures above.
- [ ] Target tests: `tests/test_vector_runtime.py`,
  `tests/test_vector_multiplayer_env.py`,
  `tests/test_multiplayer_source_state_bonus_terminal_replay.py`,
  `tests/test_multiplayer_replay_contract.py`, and
  `tests/test_multiplayer_source_state_trainer_replay.py`.
- [ ] Proof needed: catch, stack append/remove, expiry order, death cleanup,
  winner/loser facts, final reward map, and final visual rows agree.
- [ ] Refactor needed now: no for fixture promotion; yes if another bonus
  family repeats stack mutation and audit-event code.

### Lane C1 - Collision And Head-Head

- [x] Reuse `source_collision_order_batch.json` through the promoted
  underlying death-point and head-head fixtures.
- [x] Reuse `source_collision_death_point_kills_later_player_step.json`.
- [x] Reuse `source_collision_head_head_reverse_order_single_death_step.json`.
- [x] Reuse `source_body_own_delta3_safe_step.json`.
- [x] Reuse `source_body_own_delta4_kills_step.json`.
- [x] Reuse `source_normal_wall_4p_ordered_deaths_survivor_score.json`.
- [ ] Target tests: `tests/test_env_scenarios.py`, `tests/test_source_env.py`,
  `tests/test_vector_runtime.py`, `tests/test_2p_collision_fidelity.py`,
  `tests/test_hit_owner_multiplayer_fidelity.py`, and
  `tests/test_multiplayer_hit_owner_replay_fidelity.py`.
- [ ] Proof needed: source death order, public death cause/owner arrays,
  terminal or nonterminal outcome, debug die events, and replay metadata agree.
- [ ] Refactor needed now: no, unless death event packaging diverges.

### Lane C2 - Borderless And Trail/Body Edges

- [x] Reuse `source_borderless_wrap_step.json`.
- [x] Reuse `source_borderless_print_manager_wrap_toggle_step.json`.
- [x] Reuse `source_borderless_wrap_skips_destination_body_then_next_frame_kills.json`.
- [x] Reuse `source_borderless_exact_edge_corner_axis_step.json`.
- [ ] Include trail-gap/body persistence tests if the borderless row touches
  visual trail versus collision body semantics.
- [ ] Target tests: `tests/test_env_scenarios.py`, `tests/test_source_env.py`,
  `tests/test_vector_runtime.py`, `tests/test_env_wall_collision.py`, and
  `tests/test_2p_trail_gap_source_public_parity.py`.
- [ ] Proof needed: wrap position, destination-body skip, next-frame death,
  exact-edge/corner-axis behavior, PrintManager state, and replay facts agree.
- [ ] Refactor needed now: no.

### Lane R1 - Render/Trainer-Observation Cache Guards

- [x] Keep this lane scoped to observation equivalence and metadata.
- [x] Add stale-cache guards for bonus sprite changes, trail clears, trail
  gaps, borderless wraps, resets, and terminal final frames as needed.
- [x] Aggregate dirty render stats by summing numeric top-level fields and
  merging nested dict counters.
- [x] Keep `browser_lines` as the default source-state render mode.
- [x] Keep `body_circles_fast` explicitly approximate.
- [x] Keep `fast_gray64_direct` out of the default trainer surface.
- [ ] Target files: `src/curvyzero/env/vector_visual_observation.py`,
  `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`, and
  `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`.
- [ ] Target tests: `tests/test_curvytron_two_seat_render_mode.py`,
  `tests/test_vector_visual_observation.py`,
  `tests/test_multiplayer_source_state_trainer_surface.py`, and
  `tests/test_compare_2p_raw_visual_observation.py`.
- [ ] Proof needed: optimized gray64 equals direct source-state render for the
  same env state, and metadata names default versus approximate modes.
- [ ] Refactor needed now: no for 2P guards; yes before broad P3/P4 dirty
  cache work if invalidation state would spread across modules.

### Lane K1 - Controls Tail

- [ ] Add touch/gamepad input proof only if a route depends on it.
- [ ] Add real browser or Socket.IO proof only if browser evaluation depends on
  it. Keep this product/eval-only, not a trainer blocker.
- [ ] Add wider trainer/replay propagation for controls only after the target
  replay surface is named.
- [ ] Target tests: `tests/test_controls_source_input_fidelity.py`,
  `tests/test_controls_vector_fidelity.py`,
  `tests/test_controls_multiplayer_vector_fidelity.py`, and
  `tests/test_lightzero_source_state_wrapper_product_fidelity.py`.
- [ ] Proof needed: native source moves, held source frames, wrapper action
  ids, terminal handling, sidecars, and replay rows agree.
- [ ] Refactor needed now: no.

## Wave 2 - Integration And Replay

- [ ] For each lifecycle lane that reaches public env, add trainer/replay
  preservation through `SourceStateMultiplayerTrainerSurface` and
  `SourceStateMultiplayerTrainerReplayRecorder`.
- [ ] For each bonus lane that reaches public env, add final visual rows,
  final reward maps, death metadata, bonus audit metadata, and policy-row
  mapping through trainer/replay.
- [ ] For each collision lane that reaches public env, add death cause/owner,
  debug die events, final rows, and replay metadata.
- [ ] Add target-row coverage only after the replay record is a real action
  transition, not a setup, leave, warmdown, or reset event.
- [ ] Keep reset observations and final observations separate.
- [ ] Refactor trigger: if public replay and trainer replay need the same event
  facts, create a shared event/final-row schema or adapter.

## Wave 3 - Refactors Only If Triggered During Fixture Work

- [ ] Lifecycle transition summary helper:
  only if public info, replay, and trainer records build the same lifecycle
  facts in separate code.
- [ ] Bonus effect/event table:
  only if new bonus families repeat target selection, stack mutation, expiry,
  death cleanup, and audit-event logic.
- [ ] Death event packet writer:
  only if wall, body, head-head, borderless, debug event, and replay paths
  drift.
- [ ] Renderer cache object:
  only if P3/P4 dirty caching needs shared invalidation state.
- [ ] Replay/final-row schema:
  only if public and trainer replay schemas cannot share event facts cleanly.
- [ ] LightZero bridge split:
  only if real native imports or episode segmentation blur the existing
  repo-owned-row and injection-only boundaries.

## Wave 4 - Downstream LightZero

- [ ] Keep this downstream of the environment/trainer surfaces being tested.
- [ ] Build a real `MuZeroGameBuffer` sampled-target parity probe from
  `SourceStateMultiplayerTargetRowsV0`.
- [ ] Compare sampled reward, value, policy, action, mask, observation, and
  `to_play` against repo-owned expected rows.
- [ ] Keep learner updates, eval quality, and true multiplayer self-play as
  separate Coach/LightZero gates.
- [ ] Do not mutate the injection-only bridge into the real-LightZero import
  path.

## Validation Commands

Run the narrow commands for the slice changed, then the doc guard:

```bash
uv run pytest tests/test_source_lifecycle_runner.py tests/test_lifecycle_oracle.py -q
uv run pytest tests/test_env_scenarios.py tests/test_source_env.py -q -k "bonus or collision or borderless"
uv run pytest tests/test_vector_runtime.py tests/test_vector_multiplayer_env.py -q
uv run pytest tests/test_multiplayer_source_state_trainer_surface.py tests/test_multiplayer_source_state_trainer_replay.py -q
uv run pytest tests/test_multiplayer_source_state_target_rows.py tests/test_multiplayer_source_state_native_bridge.py -q
python3 scripts/check_environment_doc_status.py docs/working/environment
git diff --check
```
