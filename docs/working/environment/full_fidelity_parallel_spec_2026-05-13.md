# Full Fidelity Parallel Spec - 2026-05-13

Status: working spec, not a completion claim.
Owner surface: Environment Reconstruction.

This is the split for the remaining CurvyTron environment work. The product
runtime is `VectorMultiplayerEnv`. The main fidelity work is still engine
rules, lifecycle, bonuses, collisions, replay facts, and final observations.
Most optimization work right now is render/trainer-observation caching around
`SourceStateGray64Stack4`; it must preserve the declared visual contract rather
than define game rules. LightZero work is downstream of those surfaces.

## Simple Boundary

- Engine rule fidelity:
  source/original behavior, `CurvyTronSourceEnv`, `vector_runtime`, and
  `VectorMultiplayerEnv`. This owns movement, controls, lifecycle, spawning,
  scoring, bonuses, timers, random cursors, collisions, presence/leave,
  replay facts, and final observations.
- Render and trainer-observation optimization:
  `SourceStateGray64Stack4`, `vector_visual_observation.py`,
  `render_source_state_canvas_gray64_player_perspectives(...)`, dirty render
  caching, render-mode metadata, source-state RGB -> gray64 stacks, and
  trainer wrapper packaging. This work optimizes or verifies observations from
  source state. It does not close engine rule gaps by itself.
- Downstream LightZero:
  repo-owned target rows, sample batches, injected/native-shaped bridges,
  real `MuZeroGameBuffer` sampled parity, learner updates, and eval quality.
  This work consumes the environment and trainer surfaces. It must not turn
  route, buffer, or learning evidence into source-fidelity evidence.

## Proof Rule

Close each gap only for the surface that was tested:

1. JS/original fixture or source-map evidence.
2. `CurvyTronSourceEnv` or source runner parity.
3. `vector_runtime` low-level parity when the gap lives in batched kernels.
4. `VectorMultiplayerEnv` product-route parity.
5. Trainer/replay/final-observation preservation when a wrapper consumes it.
6. Render/trainer-observation equivalence when the gap is visual.
7. LightZero sampled-target parity only after the repo-owned target rows are
   treated as expected data.

## Fixture Leverage

Use these existing source fixtures before inventing new ones.

Fresh wave-1 result, 2026-05-13: the first parallel pass promoted focused
lifecycle/leave, bonus, and collision fixtures through public and
trainer/replay proof. This does not mean full CurvyTron fidelity is complete;
it means the remaining list is now narrower and better separated. The code
change in this wave was small: source-state trainer replay audit metadata now
keeps `borderless`. Most other work was proof expansion.

Wave 3 running, not complete: keep the next target list explicit. For
lifecycle, the exact 4P fixture targets are
`source_lifecycle_present_absent_4p_round_new.json`,
`source_lifecycle_present_absent_4p_survivor_score_round_end.json`,
`source_lifecycle_present_absent_4p_next_round.json`,
`source_lifecycle_present_absent_4p_tie_at_max_score.json`,
`source_lifecycle_match_end_at_max_score_4p.json`, and
`source_lifecycle_multi_round_match_end_4p.json`. For natural bonuses, the
exact cap/default-weight targets are
`source_bonus_spawn_cap_twenty_step.json`,
`source_bonus_default_weights_type_rng_step.json`,
`source_bonus_default_weights_select_game_clear_step.json`, and
`source_bonus_default_weights_game_clear_full_probability_step.json`. For
collision/trail/body proof, promote only the named source/runtime/public or
trainer/replay surfaces that the worker handoff needs. Optimization work stays
limited to render/trainer-observation caching; it does not close engine
fidelity gaps.

Lifecycle and presence:

- `source_lifecycle_mid_round_remove_avatar_2p.json`
- `source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json`
- `source_lifecycle_mid_round_remove_avatar_4p_continue_round_end.json`
- `source_lifecycle_present_absent_3p_round_new.json`
- `source_lifecycle_present_absent_3p_survivor_score_round_end.json`
- `source_lifecycle_present_absent_3p_next_round.json`
- `source_lifecycle_present_absent_3p_tie_at_max_score.json`
- `source_lifecycle_present_absent_4p_round_new.json`
- `source_lifecycle_present_absent_4p_survivor_score_round_end.json`
- `source_lifecycle_present_absent_4p_next_round.json`
- `source_lifecycle_present_absent_4p_tie_at_max_score.json`
- `source_lifecycle_survivor_score_2p_next_round.json`
- `source_lifecycle_survivor_score_3p_round_end.json`
- `source_lifecycle_survivor_score_3p_next_round.json`
- `source_lifecycle_survivor_score_4p_next_round.json`
- `source_lifecycle_match_end_at_max_score_2p.json`
- `source_lifecycle_match_end_at_max_score_3p.json`
- `source_lifecycle_match_end_at_max_score_4p.json`
- `source_lifecycle_multi_round_match_end_3p.json`
- `source_lifecycle_multi_round_match_end_4p.json`

Bonus and random:

- `source_bonus_self_small_catch_step.json`
- `source_bonus_self_small_tangent_no_catch_step.json`
- `source_bonus_self_small_wall_death_no_catch_step.json`
- `source_bonus_self_small_expiry_restore_step.json`
- `source_bonus_self_fast_stack_death_late_expiry_step.json`
- `source_bonus_self_fast_expiry_then_wall_death_same_tick_step.json`
- `source_bonus_enemy_slow_4p_stack_wall_death_terminal_step.json`
- `source_bonus_game_clear_immediate_step.json`
- `source_bonus_game_borderless_catch_step.json`
- `source_bonus_game_borderless_expiry_restore_step.json`
- `source_bonus_spawn_type_position_rng_step.json`
- `source_bonus_spawn_game_world_retry_step.json`
- `source_bonus_spawn_bonus_world_retry_step.json`
- `source_bonus_spawn_cap_twenty_step.json`
- `source_bonus_default_weights_type_rng_step.json`
- `source_bonus_default_weights_select_game_clear_step.json`
- `source_bonus_default_weights_game_clear_full_probability_step.json`

Collision, head-head, trail, and borderless:

- `source_collision_order_batch.json`
- `source_collision_death_point_kills_later_player_step.json`
- `source_collision_head_head_reverse_order_single_death_step.json`
- `source_borderless_wrap_step.json`
- `source_borderless_print_manager_wrap_toggle_step.json`
- `source_borderless_wrap_skips_destination_body_then_next_frame_kills.json`
- `source_borderless_exact_edge_corner_axis_step.json`

Spawn and warmup RNG fixtures are also useful when lifecycle rows depend on
spawn labels, heading retries, or PrintManager timing:

- `source_lifecycle_spawn_rng_2p_next_round.json`
- `source_lifecycle_spawn_rng_3p_next_round.json`
- `source_lifecycle_spawn_rng_4p_next_round.json`
- `source_lifecycle_spawn_rng_order_3p.json`
- `source_lifecycle_spawn_rng_order_4p.json`
- `source_lifecycle_spawn_rng_warmup_print_start_2p.json`
- `source_lifecycle_spawn_rng_warmup_print_start_3p.json`
- `source_lifecycle_spawn_heading_rejection_retry_2p.json`

## Remaining Slices

### E1. Lifecycle, Presence, And Match End

Priority: P0.

Remaining gap:
source fixtures exist for mid-round remove, present/absent rows, match-end
rows, and multi-round match-end rows. Some public and trainer/replay slices are
focused. The gap is broad product-route coverage through
`VectorMultiplayerEnv`, public final rows, warmdown/next-round behavior,
presence masks, score/winner facts, reset/autoreset separation, and
trainer/replay preservation for the same shapes.

Target files:

- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_lifecycle.py`
- `src/curvyzero/env/vector_reset.py`
- `src/curvyzero/env/vector_autoreset.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_replay.py`

Target tests and proof:

- Keep source proof in `tests/test_source_lifecycle_runner.py` and
  `tests/test_lifecycle_oracle.py`.
- Add or widen product-route mirrors in
  `tests/test_multiplayer_lifecycle_fidelity.py`,
  `tests/test_multiplayer_presence_leave_fidelity.py`,
  `tests/test_vector_multiplayer_env.py`, and
  `tests/test_vector_autoreset.py`.
- Add trainer/replay preservation in
  `tests/test_multiplayer_source_state_lifecycle_replay.py` and
  `tests/test_multiplayer_source_state_trainer_replay.py`.
- Proof needed: for each promoted fixture, source runner and public runtime
  agree on present/alive masks, deaths, score deltas, winner/match facts,
  warmdown timers, next-round or match-end outcome, final rows, and replay
  metadata.

Parallel/refactor call:
can be attacked immediately in parallel without refactoring by assigning
separate workers to mid-round remove, present/absent, and match-end fixtures.
A refactor is justified if the same lifecycle summary is rebuilt separately in
public info, replay records, and trainer records. In that case, add one small
transition-summary helper rather than spreading more branch logic.

### E2. Bonus RNG, Timers, Stack, Death, And Replay Facts

Priority: P0 for source-default runtime behavior and replay/final facts; P1
for wider long-run stress.

Remaining gap:
default weights and focused source-default effects are covered in narrow rows,
but broader retry/RNG stress, timer order, same-tick expiry, overlapping stack
families, death while boosted, clear/borderless interactions, and full replay
facts remain partial. The public runtime should preserve spawned identity,
catch, expiry, clear, stack state, RNG labels/cursors, death effects, and final
reward/observation facts when bonuses are active.

Target files:

- `src/curvyzero/env/source_env.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_source_random.py`
- `src/curvyzero/training/multiplayer_replay_contract.py`
- `src/curvyzero/training/vector_env_replay_recorder.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_replay.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`

Target tests and proof:

- Keep source proof in `tests/test_env_scenarios.py` and
  `tests/test_source_env.py`.
- Keep low-level kernel proof in `tests/test_vector_runtime.py`.
- Add or widen product-route proof in
  `tests/test_bonus_spawn_rng_public_fidelity.py` and
  `tests/test_vector_multiplayer_env.py`.
- Add replay/trainer proof in `tests/test_multiplayer_replay_contract.py`,
  `tests/test_multiplayer_source_state_bonus_terminal_replay.py`, and
  `tests/test_multiplayer_source_state_trainer_replay.py`.
- Proof needed: source fixture, runtime state transition, public info/replay,
  final observation/reward, and compact audit metadata all describe the same
  bonus event sequence.

Parallel/refactor call:
can be attacked immediately in parallel by fixture family: spawn/RNG,
timer/expiry, stack/death, borderless/clear, and replay propagation. A refactor
is justified if adding another family requires copy-pasting stack mutation,
target selection, expiry, or audit-event code. The likely refactor is a small
bonus-effect event table plus a shared stack/audit packet.

### E3. Collision, Hit Owner, Head-Head, Trail, And Borderless

Priority: P0 for engine death rules; P1 for visual/replay propagation.

Remaining gap:
the promoted hit-owner cases cover important owner-order shapes, but broader
collision edges remain: head-head reverse order, death-point ordering,
borderless destination-body skip, exact-edge/corner-axis wrap, PrintManager
wrap toggles, trail-gap body persistence, and propagation through public info,
trainer/replay records, and debug die events.

Target files:

- `src/curvyzero/env/source_env.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_replay.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`

Target tests and proof:

- Keep source proof in `tests/test_env_scenarios.py` and
  `tests/test_source_env.py`.
- Keep/runtime widen product proof in `tests/test_vector_runtime.py`,
  `tests/test_env_wall_collision.py`, `tests/test_2p_collision_fidelity.py`,
  `tests/test_hit_owner_multiplayer_fidelity.py`,
  `tests/test_multiplayer_hit_owner_replay_fidelity.py`, and
  `tests/test_2p_trail_gap_source_public_parity.py`.
- Add trainer/replay propagation where needed in
  `tests/test_multiplayer_source_state_trainer_replay.py` and
  `tests/test_multiplayer_source_state_trainer_surface.py`.
- Proof needed: source death order, public death cause/owner arrays, winner or
  nonterminal facts, replay/debug die events, and final observations agree.

Parallel/refactor call:
can be attacked immediately in parallel by fixture type: head-head, death
point, borderless wrap, and trail-gap/body persistence. A refactor is justified
if death cause, hit owner, and debug-event packaging diverge between wall,
body, head-head, borderless, and replay paths. The likely refactor is one death
event packet writer used by public info and replay sidecars.

### E4. Controls Tail

Priority: P1.

Remaining gap:
keyboard/server move delivery, source-frame held controls, release to
straight, inactive noops, and terminal-padding fixture behavior have focused
proof. Touch/gamepad input, real browser or Socket.IO transport, and wider
trainer/replay propagation remain open only if those routes are required.

Target files:

- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/trainer_contract.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`

Target tests and proof:

- `tests/test_controls_source_input_fidelity.py`
- `tests/test_controls_vector_fidelity.py`
- `tests/test_controls_multiplayer_vector_fidelity.py`
- `tests/test_lightzero_source_state_wrapper_product_fidelity.py`
- `tests/test_multiplayer_source_state_trainer_surface.py`
- Proof needed: browser or shimmed input reduces to native source moves, public
  action ids hold those moves over source frames, terminal handling stops at the
  same point, and wrapper/replay sidecars preserve the same control facts.

Parallel/refactor call:
can be attacked without refactoring if the task is a new input shim or wrapper
propagation test. A refactor is justified only if a real browser/transport
harness becomes a repeated dependency for several tests.

### E5. Replay, Final Observation, And Durable Artifacts

Priority: P0 for public/trainer final-row truth; P1 for durable artifacts.

Remaining gap:
in-memory trainer replay arrays, target rows, and sample batches exist, but
durable artifacts and broad public replay/final rows still need stronger
coverage for lifecycle, bonuses, collision, RNG, terminal state, reset versus
final observation, and event metadata.

Target files:

- `src/curvyzero/training/vector_env_replay_recorder.py`
- `src/curvyzero/training/multiplayer_replay_contract.py`
- `src/curvyzero/training/multiplayer_replay_v0.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_replay.py`
- `src/curvyzero/training/multiplayer_source_state_target_rows.py`
- `src/curvyzero/env/vector_multiplayer_env.py`

Target tests and proof:

- `tests/test_vector_env_replay_recorder.py`
- `tests/test_multiplayer_replay_contract.py`
- `tests/test_multiplayer_source_state_trainer_replay.py`
- `tests/test_multiplayer_source_state_target_rows.py`
- `tests/test_multiplayer_source_state_lifecycle_replay.py`
- `tests/test_multiplayer_source_state_bonus_terminal_replay.py`
- Proof needed: replay arrays and metadata can reconstruct the transition
  without consulting live env state, and final observations are captured before
  any reset/autoreset mutation.

Parallel/refactor call:
some tests can be added now without refactoring by copying existing replay
fixtures into lifecycle/bonus/collision cases. A refactor is justified if
public replay and trainer replay keep separate metadata schemas for the same
event facts. The likely refactor is a shared event/final-row schema or adapter.

### E6. Render And Trainer-Observation Optimization

Priority: P1. This is important, but it is mostly optimization and observation
guarding right now, not the main engine-fidelity lane.

Remaining gap:
the optimized 2P helper and dirty render cache should stay equivalent to a
direct source-state render. P3/P4 still fall back to per-player renders.
Browser/canvas pixel parity is later. More guards are needed for stale bonus
sprites, trail clears/gaps/wraps/resets, terminal final frames, approximation
metadata, and accidental use of direct-gray/profile paths in trainer runs.

Target files:

- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`
- `scripts/compare_2p_raw_visual_observation.py`
- `scripts/profile_curvytron_render_trajectory_lengths.py`

Target tests and proof:

- `tests/test_curvytron_two_seat_render_mode.py`
- `tests/test_vector_visual_observation.py`
- `tests/test_multiplayer_source_state_trainer_surface.py`
- `tests/test_compare_2p_raw_visual_observation.py`
- `tests/test_benchmark_render_lane_microbench.py`
- Proof needed: optimized output equals direct source-state renderer output for
  the same `VectorMultiplayerEnv.state`; render metadata names default versus
  approximate modes; trainer final observations use the final source state.

Parallel/refactor call:
render guard tests can run in parallel with engine work because they should
not change rules. A refactor is justified before P3/P4 cache work if the cache
state would otherwise spread across the trainer stack, renderer, and profiling
scripts. The likely refactor is a small renderer-cache object with explicit
invalidations.

### E7. Downstream LightZero

Priority: P2 until the environment/trainer surfaces above are stable enough
for the sampled-target check being attempted.

Remaining gap:
repo-owned target rows, deterministic sample batches, fake/injected native
mapping, and opt-in real `GameSegment` construction smoke exist. Real
`MuZeroGameBuffer` sampled reward, value, policy, action, mask, observation,
and `to_play` parity remains open. Learner updates, eval quality, and true
multiplayer self-play are separate downstream gates.

Target files:

- `src/curvyzero/training/multiplayer_source_state_target_rows.py`
- `src/curvyzero/training/multiplayer_source_state_native_bridge.py`
- `src/curvyzero/training/multiplayer_source_state_lightzero_native_bridge.py`
- `src/curvyzero/training/two_seat_native_replay_bridge.py`
- `src/curvyzero/training/multiplayer_ego_lightzero_coach_smoke.py`

Target tests and proof:

- `tests/test_multiplayer_source_state_target_rows.py`
- `tests/test_multiplayer_source_state_native_bridge.py`
- `tests/test_multiplayer_source_state_lightzero_native_bridge.py`
- `tests/test_curvytron_two_seat_native_replay_bridge.py`
- `tests/test_multiplayer_ego_lightzero_coach_smoke.py`
- Proof needed: sampled targets from the real buffer match repo-owned target
  rows for reward, value, policy, action, mask, observation, and `to_play`.

Parallel/refactor call:
can be investigated in parallel, but should not block P0 engine work. A
refactor is justified if real LightZero imports start leaking into the
injection-only bridge or if target rows need episode segmentation that the
current row schema cannot express cleanly.

## Immediate Parallel Work Without Refactoring

- Lifecycle fixture mirrors: one worker on mid-round remove, one on
  present/absent, one on match-end and multi-round rows.
- Bonus fixture mirrors: one worker on spawn/RNG, one on stack/death, one on
  borderless/clear, one on replay/final facts.
- Collision fixture mirrors: one worker on head-head/death-point order, one on
  borderless wrap/destination-body skip, one on trail-gap/body persistence.
- Controls tail: one worker can add touch/gamepad or transport proof only if a
  route needs it.
- Render guard work: one worker can add stale-cache/final-frame/metadata guards
  while engine workers continue, as long as the renderer does not redefine
  rules.
- Downstream LightZero: one worker can write a sampled-target parity probe
  against the existing repo-owned rows, but it should report as downstream.

## Refactor Triggers

Refactor only when it removes real duplication or prevents claim drift:

- Lifecycle refactor trigger:
  the same present/death/score/winner/warmdown facts are assembled in more than
  one public, replay, or trainer path.
- Bonus refactor trigger:
  a new bonus family requires copy-pasting target selection, stack mutation,
  expiry, RNG/audit, or death cleanup logic.
- Collision refactor trigger:
  death cause, hit owner, winner facts, and debug events diverge between
  engine, public info, and replay records.
- Replay refactor trigger:
  public replay and trainer replay encode the same final-row/event facts with
  incompatible schemas.
- Render refactor trigger:
  P3/P4 dirty caching needs shared invalidation state outside a single helper.
- LightZero refactor trigger:
  real native imports or episode segmentation would blur the current boundary
  between repo-owned rows, injection-only mapping, and native buffer smoke.
