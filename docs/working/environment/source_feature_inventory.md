# Source Feature Inventory

Status: working rebuild map
Date: 2026-05-09

Use this file as the concrete rebuild map for CurvyTron source fidelity. It says
what to copy, what is already proved, what to defer, and what fixture should come
next.

Environment Reconstruction has one intended runtime under hardening:
`VectorMultiplayerEnv`. Strict `VectorTrainerEnv1v1NoBonus` is only the
older proven 1v1 proof/profiling boundary. JS source probes and
`CurvyTronSourceEnv` are proof/oracle tools; they are not alternate product
environments.
Restricted wrappers are temporary explicit profile configs. This inventory is
about reconstructing source-default CurvyTron behavior.

For package ownership, gates, and parallel execution order, use
[full_fidelity_execution_plan.md](full_fidelity_execution_plan.md).

Allowed item labels:

- `source-read`: source code/docs have been read, but no dedicated JS oracle
  fixture is pinned.
- `JS-pinned`: a JS oracle/raw source fixture pins the behavior, but Python parity
  is not promoted.
- `Python-verified`: JS oracle plus Python/common-trace parity or direct
  Python/oracle parity exists for the named slice.
- `deferred`: intentionally later.
- `unknown`: source behavior or the test contract is not clear enough yet.

## Top Map

Gameplay rebuild order:

1. Keep source update order central: move changes, reverse avatar loop, movement,
   wall/body checks, PrintManager, bonus catch, then round-end check.
2. Finish the remaining core physics gaps: the next PrintManager death edge if
   needed, any follow-up borderless corner controls, and broader emitted-trail
   collisions.
3. Define randomness before natural spawns, long print holes, or bonuses.
4. Add bonuses after core body/trail state is boring.
5. Add wire/browser checks and trainer observations after gameplay state parity.

Current strongest evidence is in these batches and pinned fixtures:

- `source_kinematics_batch.json`, including the varied elapsed-ms fixture
- `source_border_batch.json`
- `source_normal_wall_multiplayer_batch.json`
- `source_body_canary_batch.json`
- `source_body_old_metadata_batch.json`
- `source_collision_order_batch.json`
- `source_print_manager_batch.json`
- `source_trail_batch.json`
- `source_trail_gap_batch.json`
- `source_lifecycle_spawn_rng_warmup_print_start_2p.json`,
  `source_lifecycle_spawn_rng_2p_next_round.json`, and
  `source_lifecycle_spawn_heading_rejection_retry_2p.json`, checked by
  `tests/test_source_lifecycle_runner.py` against
  `tools/reference_oracle/lifecycle_oracle.js`
- `source_lifecycle_spawn_rng_order_3p.json`, checked by the same lifecycle
  runner as a focused first-round spawn-order/RNG-label fixture only
- `source_lifecycle_present_absent_3p_round_new.json`, checked by the same
  lifecycle runner as a focused first-round present/non-present fixture only
- `source_lifecycle_match_end_at_max_score_2p.json`, checked by the same
  lifecycle runner as a focused 2P max-score match-end fixture only
- `source_lifecycle_match_end_at_max_score_3p.json`, checked by the same
  lifecycle runner as a focused 3P max-score match-end fixture only
- `source_lifecycle_match_end_at_max_score_4p.json`, checked by the same
  lifecycle runner as a focused 4P unique-leader source-only match-end fixture
  under `source-lifecycle-v25`
- `source_lifecycle_multi_round_match_end_4p.json`, checked by the same
  lifecycle runner as a focused 4P all-present multi-round match-end fixture
  under `source-lifecycle-v25`, with focused public metadata parity
- `source_lifecycle_tie_at_max_score_3p.json`, checked by the same lifecycle
  runner as a focused 3P tie-at-max-score continuation fixture only
- `source_lifecycle_present_absent_3p_tie_at_max_score.json`, checked by the
  same lifecycle runner as a focused 3P present/absent tie-at-max source
  fixture under `source-lifecycle-v25`
- `source_lifecycle_present_absent_4p_tie_at_max_score.json`, checked by the
  same lifecycle runner as a focused 4P present/absent tie-at-max source
  fixture under `source-lifecycle-v25`
- `source_lifecycle_remove_avatar_to_single_present_3p.json`, checked by the
  same lifecycle runner as a focused active 3P single-present leave-edge source
  fixture under `source-lifecycle-v25`
- `source_lifecycle_spawn_rng_3p_next_round.json`, checked by the same lifecycle
  runner as a focused 3P all-dead forced wall-death warmdown/next-round fixture
  only
- `source_lifecycle_survivor_score_3p_round_end.json`, checked by the same
  lifecycle runner as a focused 3P survivor-scoring `round:end` fixture only
- `source_lifecycle_survivor_score_3p_next_round.json`, checked by the same
  lifecycle runner as a focused 3P survivor warmdown/next-round fixture only
- `source_lifecycle_mid_round_remove_avatar_2p.json`, checked in
  `tests/test_source_env.py` against `tools/reference_oracle/lifecycle_oracle.js`
  for one active 2P removeAvatar/player-leave case
- `source_bonus_self_small_catch_step.json`,
  `source_bonus_self_small_tangent_no_catch_step.json`,
  `source_bonus_self_small_wall_death_no_catch_step.json`,
  `source_bonus_spawn_type_position_rng_step.json`,
  `source_bonus_spawn_game_world_retry_step.json`,
  `source_bonus_spawn_bonus_world_retry_step.json`,
  `source_bonus_spawn_cap_twenty_step.json`,
  `source_bonus_self_small_expiry_restore_step.json`, and
  `source_bonus_game_clear_immediate_step.json`, checked in
  `tests/test_source_env.py` against the JS scenario oracle for narrow active
  `BonusSelfSmall` catch/no-catch/death-order, one-type natural spawn RNG, and
  one timed expiry/restore behavior, plus natural spawn retries against the
  main game world and bonus world, one cap-at-20 skip, and one forced immediate
  `BonusGameClear` clear behavior

Toy-v0 API status, which is CurvyZero interface work rather than source
gameplay: `observe(ego_player)` and `legal_action_mask(ego_player)` are real,
missing live-player actions are errors, `StepResult.infos` carries step/schema
metadata, and `reset(seed)` populates `last_reset_info` while still returning
only the observation dict.

## Parallelizable Next Work

These can be handled independently right now as fixture/spec lanes:

- `normal_wall_edges`: next normal-wall border controls. The borderless
  PrintManager wrap, destination-body skip, and exact-edge/corner-axis fixtures
  are now promoted in `source_border_batch.json`.
- `print_manager_edges`: delayed `3000 ms` start is now promoted after the
  active wall/body death-stop and exact-zero toggle fixtures; keep future
  PrintManager work to natural/multi-step edges only when needed.
- `random_stream_policy`: scenario/oracle contract for listed random values and
  call logs.
- `bonus_forced_catch_order`: source-read fixture spec for catch timing, before
  full random bonus spawn.
- `wire_event_single_tick`: compressed socket payload spec, separate from state
  parity.
- `observation_rays_spec`: trainer-facing observation fixture design, separate
  from gameplay runners.

Avoid merge conflicts by assigning only one owner at a time to each of:

- `src/curvyzero/fidelity/source_runners.py`
- `src/curvyzero/env/trace_compare.py`
- `tools/reference_oracle/scenario_runner.js`

## Gameplay Mechanics

| Item | Label | Rebuild rule | Evidence now | Next fixture/spec |
| --- | --- | --- | --- | --- |
| Map size formula | source-read | Arena is square. Size is `round(sqrt(80^2 + (players - 1) * 80^2 / 5))`: 1P `80`, 2P `88`, 3P `95`, 4P `101`. | Source maps and reference defaults. 3P/4P scenarios use source sizes. | Add `random_stream_spawn_canary` that asserts map size with natural spawn setup. |
| Player count and avatar order | Python-verified | Room insertion order becomes avatar order. `Game.update` loops last avatar to first. First-round natural spawn also uses reverse order for present avatars. | Same-frame point canary, multiplayer wall-score canaries, the promoted head-head fixture, and focused 3P lifecycle fixtures depend on reverse order. | Add broader player-count lifecycle fixtures only when they isolate a new rule. |
| Present/alive and player leave | Python-verified | `present` and `alive` are separate. Non-present avatars are added to `deaths` on new round. Mid-round leave destroys avatar but does not add it to current `deaths`; in the verified 2P active case it emits `player:leave` and immediately ends the round when one avatar remains alive. | `source_lifecycle_present_absent_3p_round_new.json` covers first-round non-present setup, `source_lifecycle_present_absent_3p_next_round.json` covers one focused continuation, and `source_lifecycle_mid_round_remove_avatar_2p.json` is Python/oracle verified in `tests/test_source_env.py`. | Broader present/non-present variants and player-count variants only when they isolate a new rule. |
| Max score formula | Python-verified | Default max score is `max(1, (room.players.count() - 1) * 10)`: 1P `1`, 2P `10`, 3P `20`, 4P `30`. A tied top score at/above max is not a match winner; source starts another round. | Source maps, reference defaults, focused max-score match-end fixtures including `source_lifecycle_match_end_at_max_score_4p.json`, plus all-present and present/absent tie-at-max fixtures for 3P/4P. | Broader present/absent match variants only when they isolate a new rule. |
| Live loop timing | Python-verified | Live source targets `1000/60` ms with `setTimeout`, but each frame uses measured elapsed `step_ms`. | `source_kinematics_varied_elapsed_multistep.json` is promoted in `source_kinematics_batch.json` and verifies 10/20/15/21.666667 ms steps with the same total elapsed time as the fixed four-step controls. | Keep the kinematics batch green after runner or trace projection edits. |
| Fixed-step movement fixtures | Python-verified | Apply source control values, turn first, then move with elapsed-ms kinematics; source update order is reverse avatar order for movement events. | `source_kinematics_batch.json`, including `source_kinematics_straight_multistep.json` and `source_kinematics_turn_multistep.json`. | Keep fixed elapsed-ms controls green when movement code changes. |
| Input values | Python-verified | Left is `-1`, right is `1`, both/neither becomes `0` on the wire. | Movement fixtures and source map. | None unless browser input fidelity is reopened. |
| Base movement constants | Python-verified | Base speed `16` units/s. Base turn rate `2.8/1000` rad/ms. | Normal source movement batch plus straight, turn, and varied elapsed-ms multi-step fixtures with explicit `0.000001` tolerance. | Bonus-modified movement belongs with bonus mechanics. |
| Bonus-modified movement | source-read/vector-focused | Speed changes alter turn rate. Inverse flips turn. Straight-angle bonus sets `directionInLoop=false`, uses `angularVelocityBase=Math.PI/2`, applies the signed current turn once, then clears it. Vector runtime now carries source-like current turn velocity so public source-frame decisions arm straight-angle once per outer decision, not once per internal frame. | Source maps plus focused vector/runtime tests. | `inverse_while_turning`; `speed_bonus_turn_rate`; broader `straight_angle_existing_turn` source fixture. |
| Per-frame update order | Python-verified | One avatar fully updates and checks collision before the next lower-index avatar. A higher-index avatar can create a point body before lower-index collision. | Same-frame point materialization, normal-wall multiplayer batches, and the promoted head-head fixture. | Add emitted-trail order stress only when it isolates a new rule. |
| Normal wall | Python-verified | Normal border check uses avatar radius as margin. Crossing kills. Border check runs before body check. | `source_border_batch.json`. | `normal_wall_edges`: exact edge, just inside, just outside for all four walls. |
| Borderless wrap | Python-verified | Borderless uses margin `0`; strict edge equality is safe; center crossing teleports to the opposite edge; if x and y are both out of bounds, the source resolves the first x-axis hit only. Normal point insertion can happen before wrap, and `PrintManager.test()` runs after the survivor wraps. A body at the wrap destination is skipped on that wrap frame, but can kill on the next frame if the head stays on it. It is not torus collision lookup. | `source_borderless_wrap_step.json`, `source_borderless_print_manager_wrap_toggle_step.json`, `source_borderless_wrap_skips_destination_body_then_next_frame_kills.json`, and `source_borderless_exact_edge_corner_axis_step.json` in `source_border_batch.json`. | Next-frame second-axis wrap only if a later feature depends on it. |
| Body circle overlap | Python-verified | Stored bodies are circles. Collision is strict `distance < radiusA + radiusB`; exact tangent is safe. | Opponent tangent-safe and overlap-kill body canaries. | Epsilon below/above tangent, plus island-boundary body lookup. |
| Own body latency | Python-verified | Own stored body matches only when `currentBody.num - storedBody.num > 3`. | Own delta `3` safe and delta `4` kill body canaries. | Natural emitted own-body loop, not only seeded body. |
| Opponent body latency | Python-verified | Opponent bodies match immediately. | Opponent overlap-kill body canary. | Opponent body emitted by natural trail cadence. |
| Same-frame point insertion | Python-verified | Point events insert `AvatarBody` into world immediately while game/world are active. | Same-frame point kill and control safe body canaries. | Death-frame point side-effect fixture. |
| Head-head/reverse-order backup | Python-verified | Live heads are not collision bodies by themselves. In same-endpoint movement, the higher-index avatar's emitted normal point can kill the lower-index avatar later in the same reverse-order update, while the higher-index avatar survives. | `source_collision_head_head_reverse_order_single_death_step.json` is promoted in `source_collision_order_batch.json`; it pins `p1` survivor, `p0` killed by `p1`, `score=[0,1]`, deaths `[p0]`, and `worldBodyCount=3`. | Add 3P order stress only when it isolates a new rule. |
| Death point side effects | Python-verified | `BaseAvatar.die()` clears bonuses, sets alive false, and emits a point at death position. If an active PrintManager stops, printing-state side effects happen before the `die` event. If the avatar is already in a hole, stop emits `property printing=false` without an important stop point. Body-collision deaths carry the body owner as killer and `old:false` for young seeded bodies. | `source_collision_death_point_kills_later_player_step.json` is promoted in `source_collision_order_batch.json`; `source_print_manager_active_stop_on_death_step.json`, `source_print_manager_active_hole_stop_on_death_step.json`, and `source_print_manager_body_collision_stop_on_death_step.json` are promoted in `source_print_manager_batch.json`. | Natural emitted-body death variant only if needed. |
| Death event killer fields | Python/public/trainer-verified | Wall killer is null. Body killer is body owner. `old` comes from body age. | Body canaries verify body killer and `old:false`; `source_body_old_metadata_batch.json` verifies one seeded old-body `old:true` event; wall fixtures verify wall death. Public info, trainer surface, and trainer replay now carry `death_hit_old`. | Wall/body/self event comparison. |
| `old` flag | Python/public/trainer-verified | `AvatarBody.oldAge = 2000`; it affects only emitted death event, not collision. | `source_body_old_opponent_overlap_kills_step.json` verifies `age_ms=2000` emits `old:true`; `tests/test_multiplayer_collision_breadth_fidelity.py` promotes it through public env and trainer replay as `death_hit_old`. Current coverage is one old seeded opponent-body kill. | Add a young/old boundary pair only if broader metadata coverage is needed. |
| Frame scoring | Python-verified | `Game.update` captures the source death collection count once at frame start. All deaths in that frame get that same round score; absent avatars already in `game.deaths` count. | 3P two-die-one-survivor, 4P ordered score, 4P terminal draw, direct source-env absent-player scoring corner, `source_lifecycle_present_absent_3p_survivor_score_round_end.json`, `source_lifecycle_present_absent_3p_next_round.json`, `source_lifecycle_survivor_score_3p_round_end.json`, `source_lifecycle_tie_at_max_score_3p.json`, and the 3P/4P present/absent tie-at-max fixtures. | Broader present/non-present variants. |
| Round end scoring | Python-verified | Sole survivor gets `max(total avatars - 1, 1)`, then roundScore resolves into score. Everyone dead means no survivor bonus unless the game has exactly one avatar: source `resolveScores` then treats that sole avatar as winner even if it just died. A tied max-score leader set emits `round:end` winner null and continues. The focused 3P single-present leave-edge fixture proves the survivor score still uses total avatar count, not present count. | Normal-wall multiplayer batch, direct source-env 1P wall-death scoring, `source_lifecycle_present_absent_3p_survivor_score_round_end.json`, `source_lifecycle_present_absent_3p_next_round.json`, `source_lifecycle_survivor_score_3p_round_end.json` for full 3P `round:end` score event order, one focused all-present 3P `max_score: 2` match-end path, all-present plus present/absent tie-at-max fixtures, and `source_lifecycle_remove_avatar_to_single_present_3p.json`. | Broader present/non-present variants and multi-round match cases remain separate. |
| 2P lifecycle/spawn/RNG fixture slice | Python-verified | `newRound` sets started/inRound and schedules start after warmup. `onStart` emits `game:start`, schedules print start, activates world, and starts loop. The next-round fixture pins terminal death/score order, `round:end`, warmdown `game:stop`, and synchronous next `round:new`. The heading-retry fixture pins one rejected heading attempt followed by an accepted retry. | `tests/test_source_lifecycle_runner.py` compares Python events, `randomCalls`, and snapshots against `tools/reference_oracle/lifecycle_oracle.js` for `source_lifecycle_spawn_rng_warmup_print_start_2p.json`, `source_lifecycle_spawn_rng_2p_next_round.json`, and `source_lifecycle_spawn_heading_rejection_retry_2p.json`. | Keep this as the 2P portion of the pinned lifecycle slice; broader lifecycle fixtures stay separate. |
| 3P first-round natural spawn order | Python-verified | The focused 3P fixture proves only first-round natural spawn order and RNG labels at 0 ms: avatars spawn in reverse order 3, 2, 1, and each consumes `position_x`, `position_y`, then `angle_attempt_0`. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_spawn_rng_order_3p.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim 3P warmdown, next-round, present/non-present, match end, bonuses, optimized/vector, or trainer/replay final-observation support. A separate fixture covers focused 3P warmup/PrintManager start. |
| 3P warmup and delayed PrintManager start | Python-verified | The focused 3P fixture proves first-round spawn RNG/order, `game:start`, and delayed `print_manager:start` order/random calls at 3000 ms. The source order is avatar 3, then 2, then 1. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_spawn_rng_warmup_print_start_3p.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim 3P warmdown, next-round, match end, bonuses, optimized/vector, or trainer/replay final-observation support. |
| 4P first-round natural spawn order | Python-verified | The focused 4P fixture proves only first-round natural spawn order and RNG labels at 0 ms: avatars spawn in reverse order 4, 3, 2, 1, and each consumes `position_x`, `position_y`, then `angle_attempt_0`. Separate focused fixtures cover 4P all-dead and survivor next-round paths. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_spawn_rng_order_4p.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim broader 4P match lifecycle, bonuses, optimized/vector, or trainer/replay final-observation support. |
| 3P first-round present/non-present | Python-verified | The focused 3P fixture proves only first-round `Game.onRoundNew()` with avatar 2 non-present. Source skips avatar 2 for natural spawn RNG, spawns avatar 3 then avatar 1, and adds avatar 2 to `game.deaths`. The snapshot pins avatar 2 as `alive=false`, `present=false`, at `(0.6, 0.6)`, with `deathCount=1` and `deaths=[2]`. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_3p_round_new.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim 3P warmdown, next-round, match end, bonuses, optimized/vector, or trainer/replay final-observation support. |
| Non-present delayed PrintManager start | Python-verified | Source delayed `print_manager:start` runs for every avatar in reverse avatar order, including non-present avatars. The absent avatar still emits start point/property/random events. | `tests/test_source_env.py` compares a focused 2P present/non-present scenario against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim broader mid-round leave or next-round present/non-present behavior. |
| 3P present/non-present next round | Python-verified | The focused 3P fixture proves only a present/non-present continuation: avatar 2 is non-present, the two present avatars die in one elapsed-ms source update, source emits `round:end` winner null, `game:stop` at 8000 ms resizes the arena from size 95 to present-player size 88, and the next `round:new` re-adds avatar 2 to `deaths` while spawning only avatars 3 and 1. The absent avatar's delayed PrintManager state remains active across the next `onRoundNew()` because source clears only present avatars. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_3p_next_round.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim broader present/non-present lifecycle, broader 4P match lifecycle, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. |
| 3P present/non-present survivor-scoring round end | Python-verified | The focused 3P fixture proves only present/non-present survivor scoring at `round:end`: avatar 2 is non-present and already in source `deaths` without a `die` event, avatar 3 then dies on a wall and gets `roundScore=1`, and avatar 1 survives with `roundScore=2` and `round:end` winner 1. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_3p_survivor_score_round_end.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim broader present/non-present lifecycle, next-round continuation, broader 4P match lifecycle, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. |
| 4P first-round present/non-present | Python/public-metadata verified | The focused 4P fixture proves only first-round `Game.onRoundNew()` with avatar 2 non-present. Source skips avatar 2 for natural spawn RNG, spawns avatars 4, 3, then 1, and adds avatar 2 to `game.deaths`. The snapshot pins size 101, `deathCount=1`, and `deaths=[2]`. Focused public metadata parity exists for reset/spawn metadata. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_4p_round_new.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broad public/vector parity, match end, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. |
| 4P present/non-present survivor-scoring round end | Python/public-metadata verified | The focused 4P fixture proves only present/non-present survivor scoring at `round:end`: avatar 2 is non-present and already in source `deaths` without a `die` event, avatars 4 and 3 die on walls, avatar 1 survives with `roundScore=3`, and `round:end` winner 1. Focused public metadata parity exists for this survivor-scoring shape. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_4p_survivor_score_round_end.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broad public/vector parity, broad next-round public parity, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. |
| 4P present/non-present next round | Python/public-metadata verified | The focused 4P fixture proves only a present/non-present continuation: avatar 2 is non-present, present avatars 4, 3, and 1 die, source emits `round:end` winner null, `game:stop` at 8000 ms resizes the arena from size 101 to present-player size 95, and the next `round:new` re-adds avatar 2 to `deaths` while spawning only avatars 4, 3, and 1. Focused public metadata parity exists for the next-round bridge. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_4p_next_round.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broad public/vector parity, broader present/non-present lifecycle, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. |
| 3P present/non-present tie-at-max continuation | Python/public-metadata verified | The focused 3P fixture proves only present/non-present tied leaders at max score: source emits `round:end` winner null and continues to the next round instead of ending the match. Focused public metadata parity exists for this tied continuation shape. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_3p_tie_at_max_score.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broader present/non-present lifecycle, bonuses, reset/autoreset, replay, trainer observations, or visual support. |
| 4P present/non-present tie-at-max continuation | Python/public-metadata verified | The focused 4P fixture proves only present/non-present tied leaders at max score: source emits `round:end` winner null and continues to the next round instead of ending the match. Focused public metadata parity exists for this tied continuation shape. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_present_absent_4p_tie_at_max_score.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broader present/non-present lifecycle, bonuses, reset/autoreset, replay, trainer observations, or visual support. |
| 3P removeAvatar during warmdown | Python/public-metadata verified | The focused 3P fixture proves only `removeAvatar` after `round:end` and before `game:stop`: source emits the leaver's stop/die/`player:leave` side effects, does not re-score, does not emit a second `round:end`, leaves current `deaths` as `[3, 2]`, then starts the next round at the two-present-player map size with leaver avatar 1 in `deaths`. Focused 3P staged match-mode warmdown leave metadata parity is green. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_remove_avatar_during_warmdown_3p.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broad public warmdown leave, 4P warmdown leave, one/zero-present leave edges, replay, trainer observations, or broad public leave support. |
| 3P removeAvatar to single present | Python/public-metadata verified | The focused active 3P fixture proves only this leave edge: avatar 3 dies first and enters `deaths=[3]`; removing live avatar 2 emits `die` then `player:leave`, sets avatar 2 `present=false/alive=false`, does not add avatar 2 to current deaths, immediately emits `round:end` because only avatar 1 remains alive, gives avatar 1 `roundScore=2` using total avatar count, does not emit `end` at warmdown because avatar 3 is still present, and starts the next round at the two-present-player size with avatar 2 in next-round deaths. Focused public metadata parity exists for the same leave-edge shape. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_remove_avatar_to_single_present_3p.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broad public leave, warmdown leave, replay, trainer observations, visual support, or broader leave edge variants. |
| 2P max-score match end | Python-verified | The focused match-end fixture proves only `max_score: 1`: after avatar 2 dies and avatar 1 reaches score 1, source emits `round:end` winner 1 at 3000 ms, then `game:stop` and `end` at 8000 ms, with no immediate next `round:new`. The final snapshot has `started=false`, `inRound=false`, cleared world fields, and no avatars. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_match_end_at_max_score_2p.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim multi-round match, 3P/4P lifecycle, bonuses, reset/autoreset, vector, or replay support. |
| 3P max-score match end | Python-verified | The focused match-end fixture proves only `max_score: 2`: after delayed PrintManager start, avatars 3 and 2 die on walls in one `Game.update(100)`, avatar 1 reaches score 2 from the survivor bonus, source emits `round:end` winner 1 at 3000 ms, then `game:stop` and `end` at 8000 ms, with no immediate next `round:new`. The final snapshot has `started=false`, `inRound=false`, cleared world fields, and no avatars. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_match_end_at_max_score_3p.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim multi-round match, broader present/non-present lifecycle, broader 4P match lifecycle, bonuses, reset/autoreset, vector, or replay support. |
| 4P max-score match end | Python/public-metadata verified | The focused all-present fixture proves only the source unique-leader match end in `source-lifecycle-v25`: one 4P leader reaches max score and source emits match end instead of a next `round:new`. A focused public metadata test now protects the same unique-leader match-end shape through `VectorMultiplayerEnv`. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_match_end_at_max_score_4p.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broader 4P public match parity, broader present/non-present lifecycle, bonuses, reset/autoreset, replay, trainer observations, or visual support. |
| 3P tie-at-max-score continuation | Python-verified | The focused all-present fixture proves only `max_score: 1` tied leaders: avatar 3 dies first, then avatars 2 and 1 die together, avatars 2 and 1 both resolve to score 1, source emits `round:end` winner null at 3000 ms, then `game:stop` and a new `round:new` at 8000 ms, with no `end`. Scores `[1, 1, 0]` carry into the next round. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_tie_at_max_score_3p.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim broad multi-round behavior, broader 4P match lifecycle, broader present/non-present lifecycle, bonuses, reset/autoreset, vector, or trainer/replay/final-observation support. |
| 3P all-present multi-round match end | Python-verified | The focused all-present fixture proves only one `max_score: 3` path: avatar 1 carries score 2 through `game:stop` and `round:new` at 8000 ms, later reaches score 4, then source emits `game:stop` and `end` at 19000 ms with no later `round:new`. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_multi_round_match_end_3p.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim broader 4P match lifecycle, broader present/non-present lifecycle, bonuses, reset/autoreset, vector, or trainer/replay/final-observation support. |
| 4P all-present multi-round match end | Python/public-metadata verified | The focused all-present fixture proves one `max_score: 4` path: avatar 1 carries score 3 through `game:stop` and `round:new` at 8000 ms, later reaches score 6, then source emits `game:stop` and `end` at 19000 ms with no later `round:new`. Focused public metadata parity exists for this all-present multi-round path. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_multi_round_match_end_4p.json` against `tools/reference_oracle/lifecycle_oracle.js`; `tests/test_vector_multiplayer_env.py` covers focused public metadata. | Do not use this to claim broader 4P multi-round variants, broader present/non-present lifecycle, bonuses, reset/autoreset, vector, replay, trainer observations, or visual support. |
| 3P all-dead next round | Python-verified | The focused 3P continuation fixture proves only all-three-dead forced wall deaths after delayed PrintManager start: source emits `round:end` winner null at 3000 ms, then `game:stop` and next `round:new` at 8000 ms, then consumes next natural 3P spawn RNG/order. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_spawn_rng_3p_next_round.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this fixture alone to claim 3P survivor scoring, the separate focused present/non-present next-round behavior, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. A separate fixture covers focused 3P survivor scoring only through `round:end`. |
| 3P survivor-scoring round end | Python-verified | The focused 3P fixture proves only terminal survivor scoring after delayed PrintManager start: avatars 3 and 2 die on walls in one `Game.update(100)`, avatar 1 survives, source emits winner score:round `roundScore=2`, resolves score events in reverse avatar order, and emits `round:end` winner 1 at 3000 ms with deaths `[3, 2]`. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_survivor_score_3p_round_end.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this fixture alone to claim survivor warmdown/next-round continuation, focused present/non-present next-round behavior, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. |
| 3P survivor next round | Python-verified | The focused 3P fixture extends the survivor-scoring setup through warmdown only: after `round:end` winner 1 at 3000 ms, the original JS frame loop keeps moving live avatar 1, avatar 1 hits the wall and emits `point`, `property printing=false`, `random`, `die`, and `score:round` at 4150 ms, then source emits `game:stop` and synchronous next `round:new` at 8000 ms with next natural spawn RNG/order 3, 2, 1. | `tests/test_source_lifecycle_runner.py` compares `source_lifecycle_survivor_score_3p_next_round.json` against `tools/reference_oracle/lifecycle_oracle.js`. | Do not use this to claim focused present/non-present next-round behavior, bonuses, optimized/vector lifecycle, or trainer/replay final-observation support. |
| Broader natural lifecycle | source-read | Source has 3P/4P rounds, broader present/non-present behavior, broader multi-round behavior, and more timer combinations beyond the current pinned slice. | Source maps plus the current 2P lifecycle parity fixtures, focused 3P fixtures including all-dead next-round, survivor-scoring round-end, survivor warmdown/next-round, present/non-present survivor, next-round, and tie-at-max cases, max-score match-end, all-present tie-at-max continuation, all-present multi-round match-end, warmdown leave, and single-present active leave-edge cases; focused 4P first-round, all-dead next-round, survivor next-round, all-present tie-at-max continuation, unique-leader match-end, all-present multi-round match-end, and present/non-present reset/survivor/next-round/tie-at-max fixtures; and the focused 2P match-end fixture. | Add one fixture only when it isolates a missing source rule, such as broader present/non-present handling beyond the exact pinned continuation/tie cases, broader warmdown leave, broader leave edges beyond the promoted single-present case, or a broader multi-round variant. |
| Natural spawn positions | Python-verified | New round clears state, then each present avatar gets random free position using spawn margin. The current 2P fixtures verify reverse avatar order, x/y RNG labels, no immediate world bodies after spawn, and next-round spawn order. The focused 3P and 4P fixtures verify first-round reverse spawn order and labels. | Lifecycle parity fixtures above. Fixtures elsewhere mostly force state. | Broaden spawn rejection or reset integration only if it isolates a new rule; vector row-local spawn RNG is still open. |
| 2P natural spawn headings and retry | Python-verified | Heading is random and may be rejected if it points too close to a border by angle margin. The current 2P parity fixtures verify accepted first-attempt angles and one rejected attempt followed by an accepted retry. | Lifecycle parity fixtures above. | Add broader heading-retry controls only if they isolate a new source rule. |
| PrintManager forced toggles/death stop | Python-verified | Active manager subtracts traveled distance after collision and toggles at `distance <= 0`, including exact zero. Overshoot is not carried. On active printing death, stop emits the important stop point/property before `die` and clears manager state. On active already-hole death, stop emits only the property event before `die` and clears manager state. The same stop ordering is now pinned for seeded body-collision death before `PrintManager.test()`. | `source_print_manager_batch.json` includes print-to-hole, hole-to-print, exact-zero, no-toggle, active wall stop-on-death, active hole wall stop-on-death, body-collision stop-on-death, and delayed start. | Natural emitted-body death variant only if needed. |
| PrintManager delayed start | Python-verified | `Game.onStart()` schedules each avatar's `printManager.start` after 3000 ms. At 2999 ms the manager is still inactive. At 3000 ms, `start()` sets `active=true`, copies current x/y into `lastX/lastY`, calls `setPrinting(true)`, emits the important point and `property printing=true`, and sets deterministic distance `39` under `Math.random()=0.5`. | `source_print_manager_delayed_start_timer_step.json` is promoted in `source_print_manager_batch.json`. | Full production reset/timer/autoreset remains separate. |
| Normal trail cadence | Python-verified | While printing, first point draws if no last point; later points draw only when distance is strictly greater than avatar radius. | `source_trail_batch.json`. | Exact threshold with float plan; multi-step cadence. |
| Trail gap interior | Python-verified | While printing is false, no normal per-radius point body appears in the hole. | `source_trail_gap_hole_space_safe_step`. | Keep as regression. |
| Old bodies in visual gap | Python-verified | `Trail.clear()` clears visual/cursor state only. Existing world bodies remain and can kill. | `source_trail_gap_stored_body_still_kills_step`. | Keep as regression. |
| Print-to-hole boundary body | Python-verified | `setPrinting(false)` emits an important point before clearing visual trail; that body remains in the world and can kill a later player in the same source update. | `source_trail_gap_print_to_hole_boundary_kills_step` is promoted in `source_trail_gap_batch.json`. | Keep as regression; add multi-step/emitted-trail variants later. |
| Hole-to-print boundary body | Python-verified | `setPrinting(true)` emits an important point, keeps it visible in trail state, inserts its collision body immediately, and that body can kill a later player in the same source update. | `source_trail_gap_hole_to_print_boundary_kills_step` is promoted in `source_trail_gap_batch.json`. | Keep as regression; add natural multi-step gap variants later. |
| World island grid | Python-verified | Bodies are inserted/queried by four bounding-box corners into island buckets; duplicate bodies are not repeated inside one island; out-of-world corners are tolerated. | `tests/test_source_env.py` compares `SourceWorldState` with a vendored JS source probe. | Keep as regression; broader full-game body scans stay with emitted-body fixtures. |
| Randomness contract | partial | Source uses `Math.random` for spawns, headings, print/hole distances, bonus timing/type/position, and room passwords. The three pinned 2P lifecycle parity fixtures verify labeled spawn, one heading retry, and PrintManager start/stop calls; the focused 3P fixtures verify first-round spawn RNG labels, warmup/PrintManager start RNG labels, one non-present avatar skip, one present/non-present survivor-scoring stop-distance slice, one present/non-present next-round spawn stream, one present/non-present tie-at-max plus next-round spawn stream, one all-dead next-round spawn stream, one all-present survivor-scoring stop-distance slice, one survivor warmdown stop-distance plus next-round spawn stream, one max-score match-end stop-distance slice, one all-present tie-at-max-score plus next-round spawn stream, and one all-present multi-round match-end stream. The focused 4P fixtures verify first-round spawn RNG labels, all-dead and survivor next-round spawn streams, one present/non-present spawn skip, one present/non-present survivor-scoring stop-distance slice, one present/non-present next-round spawn stream, one present/non-present tie-at-max plus next-round spawn stream, one all-present tie-at-max-score plus next-round spawn stream, and one all-present multi-round match-end stream. The focused 2P max-score fixture also pins the stop-distance RNG call on the match-end path. `source_bonus_spawn_type_position_rng_step.json` adds a JS/Python source-env five-draw bonus proof for start delay, next delay, one-type selection, and x/y position; `source_bonus_default_weights_type_rng_step.json` adds the default multi-type weight/type proof; `source_bonus_spawn_game_world_retry_step.json` and `source_bonus_spawn_bonus_world_retry_step.json` each add one retry pair. Broader row-local RNG policy is not done. | Source maps plus forced deterministic fixtures, the pinned lifecycle parity fixtures, and the JS/Python source-env bonus spawn/type/retry fixtures. | Define row-local tape/state/cursor behavior for spawn, PrintManager, and bonuses. |
| Vector lifecycle seeder guard | deferred | Natural `source_lifecycle_*` `Game.newRound()` fixtures are deliberately rejected by `scripts/seed_vector_state_from_fixtures.py` as unsupported ordinary initial-state seeds. The report records RNG contract metadata: call index, site, avatar, value, at-ms, expected call count, and capacity pressure. | Seeder rejection/metadata output covers the 28 promoted lifecycle fixtures, including focused 4P all-dead and survivor next-round, focused 3P/4P present/non-present survivor scoring and tie-at-max, focused 3P warmdown leave, focused 3P single-present leave edge, and focused 3P/4P all-present multi-round match-end. | Use the metadata to design production reset/spawn later; do not call this optimized/vector lifecycle. |
| Production-facing masked reset | deferred | `src/curvyzero/env/vector_reset.py` now has `reset_arrays(target, reset_template, reset_mask, *, reset_seed, reset_source, snapshot_array_names=None)`. It validates required row arrays, snapshots selected terminal rows, copies selected reset-template rows, increments `episode_id`, stamps reset metadata, clears terminal/event/timer-fired fields where present, and preserves skipped rows. | Reset module and reset plan. | This is a reset boundary only; natural spawn, seed generation/history, lifecycle timers, autoreset, final obs, replay, and trainer API remain open. |
| Bonus config | source-read/vector-focused | Default bonus set is enabled. `bonusRate` changes pop-time base. Source-default probabilities are inverse `0.8`, straight-angle `0.6`, borderless `0.8`, dynamic clear probability, and `1.0` for the other source-default bonuses. Hidden `BonusSelfGodzilla` exists but is not normally selectable. | Bonus source map plus updated scalar/vector probability and spawn tests. | `bonus_config_defaults` source/JS fixture only if a new source-default selection rule is suspected. |
| Bonus spawn | JS/Python source-env narrow | Bonus cap `20`. Spawn delay is base pop time times `1 + random`. Spawn position must avoid game world and bonus world. Type choice is weighted with the source default probability map. `source_bonus_spawn_type_position_rng_step.json` proves the minimal one-type path only: `BonusManager.start()` draws the first delay, `popBonus()` draws the next delay before type/position, `BonusSelfSmall` is selected, position draws land at `(23.94, 64.06)`, and `bonus:pop` precedes the zero-elapsed source update's position events. `source_bonus_default_weights_type_rng_step.json` proves one default multi-type weight/type path selecting `BonusAllColor`. `source_bonus_spawn_game_world_retry_step.json` adds one rejected main-world candidate and one accepted retry at `(68.072, 19.928)`. `source_bonus_spawn_bonus_world_retry_step.json` adds the matching rejected bonus-world candidate and accepted retry. `source_bonus_spawn_cap_twenty_step.json` proves the source draws the next delay before the cap check, then skips type/position RNG and emits no new `bonus:pop`. | Bonus source map, JS oracle fixtures `source_bonus_spawn_type_position_rng_step.json`, `source_bonus_default_weights_type_rng_step.json`, `source_bonus_spawn_game_world_retry_step.json`, `source_bonus_spawn_bonus_world_retry_step.json`, and `source_bonus_spawn_cap_twenty_step.json`, plus JS scenario checks in `tests/test_env_scenarios.py`, matching `CurvyTronSourceEnv` checks in `tests/test_source_env.py`, low-level `vector_runtime.py` helper tests for type/position/retry/cap, and public spawn/probability tests. | Public natural spawn coverage is stronger for the promoted probability/retry/cap cases, but do not treat these proofs as broad natural catch support, full bonus replay, or broad bonus effects. |
| Bonus catch order | Python-verified | Catch runs after movement, wall/body collision, and PrintManager for alive avatars. It uses strict circle overlap in bonus world. The current proof is only seeded active `BonusSelfSmall`: one catch after movement, one strict-overlap no-catch/tangent case, and one same-tick wall-death no-catch case. The death case proves wall/scoring wins before catch; Python currently filters JS's non-important death point from the surfaced event comparison. | JS oracle fixtures `source_bonus_self_small_catch_step.json`, `source_bonus_self_small_tangent_no_catch_step.json`, and `source_bonus_self_small_wall_death_no_catch_step.json`; matching `CurvyTronSourceEnv` checks in `tests/test_source_env.py`. | Add other catch/death interactions and other bonus types as separate fixtures. |
| Bonus stack effects | JS/Python source-env narrow | Stack resets defaults, applies active effects, then calls setters. `source_bonus_self_small_expiry_restore_step.json` pins one radius restore path: a caught `BonusSelfSmall` expires after `7500` ms, restores radius from `0.3` to `0.6`, emits `bonus:stack remove`, and does not emit another `bonus:clear`. `source_bonus_game_clear_immediate_step.json` separately pins forced `BonusGameClear`: `bonus:clear` then `clear`, `worldActive=true`, `worldBodyCount=0`, no stack/property changes. `source_bonus_game_borderless_expiry_restore_step.json` plus runtime/public coverage now pins one seeded `BonusGameBorderless` duration/expiry path. Runtime bonus support is being consolidated toward an explicit table/spec, and natural source-default type selection must not imply unsupported runtime effects. Speed/radius/inverse/invincible/printing/color/borderless all have special behavior beyond these proofs. | Bonus source map, JS oracle fixtures `source_bonus_self_small_expiry_restore_step.json`, `source_bonus_game_clear_immediate_step.json`, and `source_bonus_game_borderless_expiry_restore_step.json`, plus matching `CurvyTronSourceEnv`/runtime/public checks. | One fixture each for speed, radius collision beyond restore, inverse double-cancel, color, and broader stack math. |
| BonusSelfMaster | source-read/vector-focused | Gives `invincible=true` and `printing=-1` for `7500` ms. While active, body/trail death is blocked, but normal wall death still kills unless project-only `profile_no_death`, `death_immunity`, or opponent-immortal modes are enabled. Expiry restores `invincible=false` and restarts printing/PrintManager through source-like side effects. Death before expiry clears the stack and must not restart printing or leave invincible true. Project-only no-death/`death_immunity`/opponent-immortal modes remain valid training/profile helpers, not source-fidelity claims. | Vector runtime print-manager/SelfMaster tests and public env SelfMaster proofs. Focused validation reported runtime `self_master or print_manager` `11 passed` and public `self_master` `4 passed`. | Broader environment validation after the SelfMaster fix remains pending. |
| Bonus timers/expiry | JS/Python source-env narrow | Timed bonuses use `setTimeout`. The current proof pins one `BonusSelfSmall` expiry after catch, with expiry events before the following zero-elapsed movement events. Source/runtime/public seeded `BonusGameBorderless` duration/expiry is also pinned narrowly. Same-frame expiry with other timers and broader stack ordering are not pinned. | `source_bonus_self_small_expiry_restore_step.json`, `source_bonus_game_borderless_expiry_restore_step.json`, and matching `CurvyTronSourceEnv`/runtime/public checks. | Add public timer/order, same-frame timer/order, and multi-stack expiry fixtures only when they isolate a new rule. |

## UI, Browser, Rendering, And Wire

These are not gameplay-source blockers unless the task is browser demo or wire
replay parity.

| Item | Label | Rebuild rule or deferral | Evidence now | Next fixture/spec |
| --- | --- | --- | --- | --- |
| Socket protocol | source-read | Messages are JSON batches of array events. Callback id is optional. | Network source map. | `wire_event_single_tick`. |
| Wire compression | source-read | Position and angle are compressed to integer hundredths with `(0.5 + value * 100) | 0`, then decompressed by `/100`. | Network source map. | Include compression in `wire_event_single_tick`. |
| Gameplay wire events | source-read | Events include `position`, `angle`, important `point`, `die`, `property`, `score`, `score:round`, bonus events, `round:new`, `round:end`, `game:start`, `game:stop`, `clear`, `borderless`, and `end`. | Network source map. | One wire death+score fixture; spectator catch-up fixture later. |
| Browser client authority | deferred | Browser mirrors server state and predicts visuals locally. It is not gameplay authority. | Network/render source map. | Browser check only after state/wire parity. |
| Client trail visuals | source-read | Client has visual-only trail gap handling and only receives important point events over wire. Server world bodies are still source of gameplay truth. | Trail and network source maps. | Client visual gap screenshot if demo parity needs it. |
| Canvas rendering | deferred | Stacked canvases, interpolation, bonus sprites, explosions, resize behavior, sounds, and CSS are render/UI work. | Network/render source map. | Fixed viewport screenshot after a matching state trace. |
| Raw old-app build | deferred | Gulp 3/Bower/generated `bin` and `web/js` files are needed only for browser hosting. | Network/build source map and raw-run probe. | Disposable old-node build only if browser demo is requested. |
| App shell UI | deferred | Angular routing, room list UI, chat, profile, sounds, Inspector/Influx, and trackers are not environment rules. | Source map. | None for environment rebuild. |
| Source wire replay | unknown | No separate gameplay-authoritative replay format is pinned. If replay means replaying socket batches, use the wire event stream. | Network source map. | Decide replay meaning; then record/replay one compressed event batch. |

## Trainer-Facing Interfaces

These are CurvyZero contracts. They should not change source gameplay rules.

| Item | Label | Rebuild rule or deferral | Evidence now | Next fixture/spec |
| --- | --- | --- | --- | --- |
| Learned observations | deferred | Original CurvyTron source does not define learned observations. CurvyZero now has strict 1v1 rays and a narrow 3P/4P scalar projection over `VectorMultiplayerEnv.state`, but these are trainer contracts over simulator state, not source gameplay rules or browser pixels. | `observation_fidelity.md`; `src/curvyzero/env/vector_multiplayer_observation.py`; `tests/test_vector_multiplayer_observation.py`. | Keep the 3P/4P scalar projection narrow: no trainer-ready env, replay writer, visual/pixel, source-fidelity completion, or LightZero training claim. |
| Debug/global observation | deferred | Keep privileged global state as debug/oracle only. | Observation docs and toy-v0 now cover `observe(ego_player)`, `legal_action_mask(ego_player)`, `last_reset_info`, step infos/schema metadata, and missing live-player action errors. | Keep schema id/hash attached to replay and trainer samples. |
| Ray observation | deferred | Ego-relative rays/scalars/action mask are trainer schema work after state parity. | Observation plan. | Shape/range/permutation fixtures. |
| Local raster observation | deferred | Local raster is later CNN/MuZero work, not source gameplay. | Observation plan. | Raster fixtures only when needed. |
| Pixel observation | deferred | Browser pixels are expensive render fidelity, not first training observation. | Observation plan. | Use only after state trace passes for same scenario. |
| Reward/training score | deferred | Source has roundScore/match score. Trainer rewards can be a separate schema, but must record rules/reward ids. | Rounds source map and training docs. | Reward schema note after source scoring fixtures broaden. |
| Training replay rows | deferred | Training replay is not CurvyTron source. Rows need rules hash, observation schema, action schema, reward schema, ego id, wrapper action map / `joint_action` sidecar, and terminal observation policy. A separate 1v1/no-bonus replay v0 contract now exists, but production actor-bridge integration may still be sample/debug-only depending on the main-thread fixes. | Observation/training docs and `src/curvyzero/training/replay_chunk_v0.py`. | Wire replay v0 through the actor bridge only after reset, terminal, observation, and reward contracts are stable. |
| Vector/batch env API | deferred | Do not add `reset_many`/`step_many` or vector backends before single-env source semantics are stable. | Active lanes and performance docs. | Interface freeze after gameplay state parity. |

## Messy Gameplay Details

Source files to keep open while implementing:

- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/model/Avatar.js`
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/core/Island.js`
- `third_party/curvytron-reference/src/server/core/AvatarBody.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `third_party/curvytron-reference/src/server/manager/BonusManager.js`
- `third_party/curvytron-reference/src/server/model/BonusStack.js`
- `third_party/curvytron-reference/src/server/model/GameBonusStack.js`
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`
- `third_party/curvytron-reference/src/shared/model/BaseRoomConfig.js`

Important source facts that are easy to lose:

- `Game.update` captures frame-start death score once. Do not update all players
  first and then resolve collisions.
- Normal trail points happen inside `Avatar.update`, before body collision.
- Live heads are not collision bodies unless they emit point bodies; do not add
  an all-heads collision phase.
- PrintManager toggles happen after body collision, so a same-frame hole toggle
  cannot save the avatar from a body already hit.
- `setPrinting(false)` emits a point, then clears visual trail state. World
  bodies remain.
- Borderless wrap uses one-axis teleport through `getOposite`; it does not check
  body collision on the wrap branch of that frame.
- Exact tangent body distance is safe because collision is strict `<`.
- Exact wall equality is safe because border checks use strict `<` and `>`.
- `oldAge` is event metadata only.
- Bonus catch runs after PrintManager. Bonus effects can change movement, radius,
  invincibility, printing, color, borderless, and clear-trails behavior.
- `BonusSelfMaster` active invincibility blocks body/trail death, not normal
  wall death. `profile_no_death`, `death_immunity`, and opponent-immortal modes
  are project-only helpers, not source-fidelity claims.

## Regression Commands

Run these when touching source-fidelity behavior:

```bash
uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --fail-on-mismatch --artifact-root /private/tmp/curvy-source-kinematics-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-border-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_old_metadata_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-old-metadata-regression
uv run python tools/run_fidelity_batch.py scenarios/environment/source_collision_order_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-collision-order-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_batch.json --python-runner source-trail-cadence-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-cadence-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_gap_batch.json --python-runner source-trail-gap-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-gap-regression
```

## Source Docs

- [facts_index.md](../../research/curvytron_source_map/facts_index.md)
- [movement_controls.md](../../research/curvytron_source_map/movement_controls.md)
- [collisions_trails_world.md](../../research/curvytron_source_map/collisions_trails_world.md)
- [rounds_scoring_multiplayer.md](../../research/curvytron_source_map/rounds_scoring_multiplayer.md)
- [bonuses_config.md](../../research/curvytron_source_map/bonuses_config.md)
- [network_render_build.md](../../research/curvytron_source_map/network_render_build.md)
- [coverage_tracker.md](coverage_tracker.md)
- [full_fidelity_execution_plan.md](full_fidelity_execution_plan.md)
