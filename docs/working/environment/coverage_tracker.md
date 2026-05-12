# Environment Source Claim Tracker

Status: working source-claim checklist
Date: 2026-05-10

Use this as the concise checklist for what is covered, what is still open, and
how to test the next claims. Detailed evidence stays in the source maps,
scenario fixtures, batch summaries, and investigation notes.

This tracker records source claims, not a test-count scoreboard. Progress means
source claim -> oracle/probe -> Python parity -> optimized parity. The current
source-backed core covers movement, selected border rules, body and collision
order canaries, PrintManager behavior, normal trail cadence, forced trail gaps,
one separate natural trail-gap source case, old-body metadata, the promoted
live movement event trace
`source_live_movement_event_trace_2p_no_bonus_multistep`, and 28 pinned
lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`,
`source_lifecycle_survivor_score_4p_next_round`,
`source_lifecycle_present_absent_3p_survivor_score_round_end`, and
`source_lifecycle_present_absent_3p_tie_at_max_score`,
`source_lifecycle_present_absent_4p_tie_at_max_score`,
`source_lifecycle_remove_avatar_during_warmdown_3p`, and
`source_lifecycle_remove_avatar_to_single_present_3p`.
The natural
trail-gap case has a separate
scalar vector comparator path through row-local Math.random tape arrays, but it
is still not part of the four-case trail-gap batch or vector speed defaults.
`CurvyTronSourceEnv` also now carries source-shaped body/island/world storage,
source point body insertion, world-backed `worldBodyCount`, source-shaped
border detection, and reset/source game-state `borderless`; its focused test
file covers direct source-env cases for body, insertion, active
already-hole PrintManager stop, normal trail exact-threshold and epsilon
behavior, borderless branch checks, direct borderless PrintManager wrap/toggle
behavior, 3P ordered death scoring, 3P same-frame wall-death scoring, a 3P
absent-player scoring corner, tie-at-max next-round behavior,
timer-drain large advance across `game:stop -> round:new -> game:start`, a
zero-delay timer-loop guard, non-present delayed PrintManager starts, 1P
wall-death scoring, source-verified world island corner lookup, mid-round 2P
`removeAvatar`/`player:leave`, 3P and 4P source-verified mid-round leave
continuation fixtures, and narrow active `BonusSelfSmall`
catch/no-catch plus default multi-type bonus weight/type RNG, the first natural
bonus spawn retry proofs against both the main game world and bonus world,
natural `BonusGameClear` type selection, and one immediate `BonusGameClear`
clear-trails proof.

Acceptance commands live in `scripts/run_environment_fidelity_matrix.py` and the
focused source/vector test files. The matrix expected/description fields should
name the behavior protected and the boundary still unsupported, not restate pass
totals. Rerun the relevant command after changing a mechanic, then update the
claim it protects rather than copying status totals into this front-door tracker.
Docs and code now use the 28-fixture lifecycle claim. `SOURCE_LIFECYCLE_RULES_HASH`
is `source-lifecycle-v25`.

Current direction: this is not full fidelity, and there is one public runtime
name under hardening: `VectorMultiplayerEnv`. The name is
old no-bonus public-env naming, not a separate second product
implementation. Strict
`VectorTrainerEnv1v1NoBonus` is only the older proven 1v1 proof/profiling
boundary. `CurvyTronSourceEnv` and the JS oracle are truth/proof tools while
source rules move into `VectorMultiplayerEnv`; they are not alternate
product environments. Keep source batches as claim guardrails while batched
event rows, reset/timer/autoreset, observation/reward packing, and
Modal/JAX/Mctx speed work prove the intended runtime against named source
contracts.
Cleanup decision: keep one fast source-faithful runtime path. The historical
public env name should not be used to imply a second implementation.

Latest fixed-state validation reported `282 passed` for the focused
environment subset, `33 passed` for the source bonus suite, ruff passed, and
the environment doc guard passed. Treat that as a recency marker only. Natural
public bonus support is partial, not broad support. `BonusSelfMaster` wall/body
parity and `BonusAllColor` reverse target event order plus older-wins overlap
behavior are fixed.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).
Current 2P public-runtime base guards cover direct public body/trail/
collision canaries, terminal wall step, autoreset final metadata, active leave
immediate scoring, the long natural 1v1 wall rollout from reset to terminal,
draw warmdown into the next round, unique-leader max-score match end, and the
2P metadata replay bridge. Public hardening also includes the source-state
LightZero wrapper fixed-opponent sidecar proof. These are focused proof slices,
not a full 2P claim. Also keep visual dimensions plain: the source 2P arena is
88 units from `CurvyTronReferenceDefaults.arena_size_for_players(2)`. The raw
product visual frame is the full-size canvas-like RGB render, currently 704x704
pixels for 2P. The 64x64 value is only the derived gray64 model tensor, not the
original game size.
Latest validation reported on 2026-05-11: focused environment validation
reported `282 passed`, source bonus validation reported `33 passed`, ruff
passed, and the environment doc guard passed.
`tests/test_vector_runtime.py` covers the focused vector runtime boundary for
validation, counters, PrintManager row-local random distances, final
state-derived counter normalization, direct 3P/4P wall fixtures, and no-bonus
N-player warmup timer advancement. This is useful production-runtime movement,
but it is not public 3P/4P env parity or full optimized lifecycle evidence yet.

The strict public `VectorTrainerEnv1v1NoBonus` path now includes public-step
horizon truncation, overflow truncation, terminal metadata, terminal barrier
replay policy, replay metadata defaults from env info, truncation reason
labeling, and live-step replay recording into replay-v0 chunks. Safe timer
diagnostic cleanup keeps benchmark-only timer instrumentation out of normal
runtime calls.

The source-pinned bridge wave has landed. It includes same-frame wall draw
replay packing, borderless destination-body skip then next-frame kill full-trace
compare, borderless PrintManager wrap toggle compare, collision-order batch
comparator support, direct public body/trail/collision canaries, seed/reset
metadata across autoreset into recorder chunks, an optional strict
replay/profile manifest, two long 1v1 wall-round-done public-vector source
bridges, direct seeded 3P/4P no-bonus fast-runtime wall-scoring/order canaries,
no-bonus N-player reset/warmup helpers, one
narrow no-bonus 3P warmdown/next-round helper proof against
`source_lifecycle_spawn_rng_3p_next_round`, one narrow no-bonus 4P
warmdown/next-round helper proof against `source_lifecycle_spawn_rng_4p_next_round`,
one no-bonus 3P survivor warmdown-death/next-round proof against
`source_lifecycle_survivor_score_3p_next_round`, one no-bonus 4P survivor-score/
next-round proof against `source_lifecycle_survivor_score_4p_next_round`, one
no-bonus 3P present/absent draw warmdown/next-round proof against
`source_lifecycle_present_absent_3p_next_round`, and a metadata-only public
`VectorMultiplayerEnv` surface for 2P/3P/4P.
That public surface now has 2P terminal/autoreset/leave guards, one long 2P
source reset-to-terminal wall rollout, focused 2P draw-next-round and max-score
match-end warmdown guards, a 2P metadata replay bridge, a source-state
LightZero wrapper fixed-opponent sidecar proof, seeded 3P/4P
wall-canary coverage, 3P/4P
fixture-tape present/absent reset proofs, 3P/4P present/absent
survivor-scoring proofs, and 3P/4P present/absent metadata-only
warmdown/next-round proofs through
`VectorMultiplayerEnv.advance_warmdown(...)`, plus one focused 3P
match-mode survivor warmdown movement/death/no-rescore proof through explicit
`VectorMultiplayerEnv.advance_warmdown_frame(...)` using
`source_lifecycle_survivor_score_3p_next_round`. Ordinary public `step()` still
blocks while `warmdown_pending=true`; the explicit metadata bridge advances the
source-pinned 1150 ms warmdown frame before `game:stop`, then
`advance_warmdown(3850.0)` reaches the next round. It exposes debug transition
metadata only; it does not claim learned/trainer observations, full natural
public reset parity, replay parity, trainer-ready natural lifecycle, hidden autoreset,
or broad public lifecycle parity. Focused lifecycle/runtime/multiplayer guard,
ruff, and doc guard passed on the touched set. Treat this as narrow bridge
hygiene, not as a full-fidelity claim.

The public lifecycle metadata-array smell is fixed for the current public env:
`round_done`, `warmdown_pending`, `match_done`, `round_winner`, and
`match_winner` are real `VectorMultiplayerEnv` state arrays from reset,
guarded by `test_public_lifecycle_metadata_arrays_exist_from_reset` plus the
focused round/match warmdown tests. This does not claim full natural public reset,
trainer-ready lifecycle, or full public lifecycle parity.
The stale lifecycle metadata overlay plumbing has been removed from
`vector_multiplayer_env.py`; the current claim is still only the guarded public
state-array path above.

The public seed-generated reset path now uses
`seed_generated_source_random_history` with
`curvyzero_seeded_source_math_random_history/v0`. It produces deterministic
row-local `float64` histories in `[0,1)` from `reset_seed` and labels that
history separately from fixture tapes. The exact source claim is narrow:
`test_public_seed_generated_reset_history_matches_source_env_call_order` feeds
the generated public row history into `CurvyTronSourceEnv` and proves spawn plus
warmup `print_manager.start_distance` random-call order for 2P/3P/4P. Generated
reset rows can therefore set
`natural_multiplayer_reset_claim=true` for
`seeded_source_history_reset_spawn_warmup_call_order/v0`; fixture-tape and
direct-state resets still keep the claim false. This is not V8
`Math.random` bit parity, broad warmup frame movement parity, replay parity, or
trainer-ready lifecycle.

Multiplayer replay now has three narrow checks, not trainer replay:
metadata-only public-env packaging, a metadata sequence recorder, and
`build_multiplayer_scalar_observation_replay_artifact_v0(...)` for scalar
observation rows plus nested public metadata. These reject missing 3P/4P
metadata, reject 3P/4P claims against strict replay-v0 shape, and block 3P/4P
from using the strict 1v1 ray observation schema. A production trainer replay
writer/shard/manifest plus policy/search/value targets remains open.
The dead legacy scalar replay builder has been removed from
`multiplayer_replay_v0.py`; that cleanup does not promote the scalar artifact
into trainer replay.

Helper-level and public metadata-only 3P match/tie/multi-round lifecycle checks
are also green. They protect unique-leader match end, tied max-score
continuation, and one multi-round continue-then-end path. Public metadata also
has a focused 4P tied max-score continuation proof for
`source_lifecycle_tie_at_max_score_4p.json`: two leaders tie at max score, no
match winner is emitted, and warmdown starts the next round. The 4P
unique-leader fixture `source_lifecycle_match_end_at_max_score_4p.json` is
source-promoted under `source-lifecycle-v25`: JS oracle plus Python
source-runner. Public metadata now has a focused 4P unique-leader match-end
proof through `VectorMultiplayerEnv`. The 4P all-present multi-round
fixture `source_lifecycle_multi_round_match_end_4p.json` is also source-promoted
under `source-lifecycle-v25`; focused public metadata parity for that
all-present 4P multi-round path is green. The three 4P present/absent
reset/survivor/next-round source fixtures are also promoted under the same hash
and have focused public metadata parity. The 3P and 4P present/absent
tie-at-max source fixtures are promoted under `source-lifecycle-v25`; public
parity for those tie fixtures is still separate. Replay, trainer-ready
observations, and broader lifecycle claims are still open.

Leave behavior is split deliberately. Source fixtures now cover 3P and 4P
mid-round `removeAvatar` continuation through later round end with JS oracle
and `CurvyTronSourceEnv` checks. The 2P source fixture also proves immediate
round end when a leave leaves one live present player. The new 3P warmdown
leave source fixture proves `removeAvatar` after `round:end` does not re-score
or emit a second `round:end`, and the next round treats the leaver as
non-present. The 3P single-present leave-edge source fixture proves avatar 3
dies first and enters `deaths=[3]`; removing live avatar 2 emits `die` then
`player:leave`, sets avatar 2 `present=false/alive=false`, does not add avatar
2 to current deaths, immediately round-ends because only avatar 1 remains
alive, gives avatar 1 `roundScore=2` using total avatar count, does not emit
`end` at warmdown because avatar 3 is still present, and starts the next round
at the two-present-player size with avatar 2 in next-round deaths. Public
parity for this new leave-edge fixture is not done. Public metadata has narrow
`VectorMultiplayerEnv.remove_player(...)`
support for active-round leave: 3P/4P continuation, 2P immediate round end,
and a 4P source-rule canary for survivor scoring after already-dead players.
It also has one focused 3P staged match-mode warmdown leave metadata proof.
Public ids are zero-based, source ids equal public id plus one, the leaver
becomes present/alive false, and the leaver is not added to `death_player`.
Broad public leave, broad public warmdown leave, broader leave edge variants,
replay, trainer, visual, and bonus support remain open.

Long 1v1 wall-round-done bridge coverage now exists in both the older strict
proof wrapper and the intended runtime. The strict tests
`test_public_vector_env_matches_source_long_1v1_wall_round_done_terminal_step`
and `test_public_vector_env_reset_to_terminal_matches_source_long_1v1_fixture`
remain proof-wrapper guards. `test_2p_public_reset_to_terminal_matches_source_long_wall_fixture`
runs the same long natural source rollout through `VectorMultiplayerEnv`
from reset to terminal and compares per-tick source state. This is still only a
focused 2P/no-bonus wall rollout, not broad 2P fidelity.

Amdahl guardrail: optimize env-step only with measurements against the whole
self-play loop, including MCTS/search/model cost, observation packing, reset,
and replay.

Optimizer critique: native vector timing is useful only when the included
components are explicit. Ray/observation-bound timing is actionable. Large CPU
batch regressions need a breakdown by env step, observation, replay/reset, and
policy/search before any rewrite claim.

Remaining gap checklist: seeded public `BonusSelfSmall` and `BonusGameClear`
fixture support has landed with public tests; broad natural reset/warmup remains
beyond the
seed-generated source-history call-order proof; native lifecycle state has
reset-owned public arrays for the core fields but still needs broader
public-env ownership beyond the narrow bridges; broad public warmdown frame-loop
movement/death is still missing beyond the one explicit 3P metadata bridge;
broad public leave, broad public warmdown leave beyond the focused 3P staged
metadata proof, and broader leave edge cases are still open; the new
single-present 3P leave edge has focused public metadata parity; immediate
round-end public leave is only narrow 2P fixture-backed plus one 4P canary;
masks and
rewards still need full coverage across live, dead, absent, terminal, warmdown,
timeout, overflow, draw, survivor, and match-end states; replay compaction and
broader source refs are not done;
trainer-ready public 3P/4P observation/env support is still missing even though
the scalar projection and scalar replay-shaped artifact exist; CurvyTron visual
stacked-frame input from our own renderer, raw uint8 source-state observation
access, source-vs-vector full-size RGB render parity plus gray64 parity, broad
public reset/warmdown/replay parity, broader world/island boundaries, and moving
more source semantics into the fast runtime remain open. The source-state visual
work proves native-render model-observation parity from source state; it does
not prove browser/canvas pixels.
Multiplayer makes visual replay stricter: visual stacks need frame provenance
plus player ids, present/alive masks, death order, score vectors, opponent
policy ids, full wrapper action logs, and reset/RNG metadata.

Bonus replay metadata, forced optional-array `BonusGameBorderless` runtime
proof, and the low-level natural bonus spawn helper in `vector_runtime.py` have
landed narrowly. `vector_runtime` now has table-backed optional-array support
for the promoted runtime effects, including `BonusSelfMaster`, `BonusAllColor`,
and `BonusEnemyStraightAngle`. Focused seeded/natural public bonus tests now
cover promoted slices including `BonusSelfMaster` catch/expiry and wall/body
parity, `BonusAllColor` catch/expiry, reverse target event order, and
older-wins overlap behavior. Focused validation reported `282 passed`, source
bonus validation reported `33 passed`, ruff passed, and the environment doc
guard passed. Replay currently preserves bonus metadata/audit only, not full
replay arrays. Focused public natural source-default catch/effect coverage now
covers self small/slow/fast/master, enemy slow/fast/big/inverse/
straight-angle, game borderless, all color, and game clear. Remaining gaps
include fuller replay/final observations, Halley's remaining capacity-policy cleanup
(manual/direct stack guard wording and possible fully blocked generated-map
policy), broader 3P/4P lifecycle/leave parity, visual pixel parity, toy-path
quarantine, and final cleanup.

Next gap queue: keep the landed bridge wave green, then stay on environment
reconstruction. First: widen natural bonus support beyond the promoted focused
seeded/natural slices; keep generated tape/position retries distinct from strict
fixture/direct tapes and strict `vector_runtime` finite helpers; document the
manual/direct stack guard and add fully-blocked generated-map policy only if it
becomes real; then widen bonus metadata/replay audit while keeping replay
arrays out of the claim; full public replay/final observations; broader 3P/4P
lifecycle/leave parity; toy-path quarantine; visual pixel parity later; and
final cleanup.
Modal and fixed-opponent route smokes are route evidence only, not the next
priority.

## Source Pointers

- Live work map: [active_lanes.md](active_lanes.md)
- Full fidelity matrix:
  [full_fidelity_spec_matrix_2026-05-09.md](full_fidelity_spec_matrix_2026-05-09.md)
- Multiplayer gap targets:
  [multiplayer_env_gap_targets_2026-05-10.md](multiplayer_env_gap_targets_2026-05-10.md)
- Optimizer handoff:
  [optimizer_handoff_2026-05-10.md](optimizer_handoff_2026-05-10.md)
- Measurement critique:
  [measurement_critique_2026-05-09.md](measurement_critique_2026-05-09.md)
- Body/trail detail: [body_trail_investigation.md](body_trail_investigation.md)
- Loop contract: [trace_loop_contract.md](../../design/environment/trace_loop_contract.md)
- Source facts: [facts_index.md](../../research/curvytron_source_map/facts_index.md)
- Collision/trail source map:
  [collisions_trails_world.md](../../research/curvytron_source_map/collisions_trails_world.md)
- Next mismatch plan:
  [next_mismatch_plan.md](../../research/environment/next_mismatch_plan.md)

## Verified Mechanics

| Mechanic | Current evidence | Keep it honest with |
| --- | --- | --- |
| Local trace loop | Common-trace diff is the normal comparison path; sidecars include `js.common_trace.json`, `python.common_trace.json`, and compact timelines. | Run one scenario before promotion; inspect `diff_status`, first mismatch, and timelines. |
| Source movement | `source_kinematics_batch.json` protects the one-step movement fixtures, the forced two-player turn case, straight and turn four-step fixed-60Hz movement, and varied 10/20/15/21.666667 ms movement with the same total elapsed time as the fixed four-step control. `source_live_movement_event_trace_2p_no_bonus_multistep` adds the promoted live movement event-order trace. | Rerun the promoted movement batch and live event trace after runner or trace projection edits. Bonus-modified movement belongs with bonus mechanics. |
| Normal-wall and borderless basics | `source_border_batch.json` protects normal-wall cases, one plain source borderless wrap case, `source_borderless_print_manager_wrap_toggle_step`, `source_borderless_wrap_skips_destination_body_then_next_frame_kills`, and `source_borderless_exact_edge_corner_axis_step`. | Rerun the batch; add next-frame second-axis wrap only if broader corner behavior becomes important. |
| 3P/4P normal-wall scoring/order | `source_normal_wall_multiplayer_batch.json` protects the narrow death-order, survivor score, and terminal-draw canaries. Comparator/direct-runtime fixtures are green. | Rerun the batch; add head-head and round-lifecycle fixtures before general multiplayer claims. |
| No-bonus N-player reset/warmup helpers | Focused vector lifecycle/runtime checks protect dynamic 2P/3P/4P reset/spawn/warmup and PrintManager start ordering. | Rerun focused vector lifecycle/runtime checks after reset, spawn RNG, timer, or PrintManager start edits. Keep public 3P/4P env parity separate. |
| No-bonus 3P warmdown/next-round helpers | Focused fast helpers protect the all-dead 3P path, the survivor warmdown-death/no-rescore path, and one present/absent draw continuation. The present/absent proof pins first-round map size 95, next-round present-count map size 88, death list `[1,2,0]` before warmdown, absent death list `[1,-1,-1]` after spawn, spawn RNG indices `11..16`, and absent PrintManager state preservation. The public metadata env now has one focused match-mode survivor warmdown bridge for `source_lifecycle_survivor_score_3p_next_round`: normal `step()` stays blocked, explicit `advance_warmdown_frame(..., elapsed_ms=1150.0)` moves/kills the survivor without terminal re-score, and `advance_warmdown(3850.0)` reaches next round. | Rerun focused vector runtime/lifecycle and public multiplayer warmdown-frame tests after warmdown, spawn RNG, timer, present/absent, or death-list edits. This is not replay, observation parity, hidden autoreset, or broader present/non-present coverage. |
| Stored body overlap and own latency | `source_body_canary_batch.json` protects opponent tangent safe, opponent overlap kill, own delta `3` safe, own delta `4` kill, and same-frame point cases. `tests/test_source_env.py` also checks the scalar source env directly for opponent strict overlap versus tangent safety, own delta `3` safe and delta `4` kill, old metadata at 2000 ms, and wall priority over body collision. `tests/test_vector_multiplayer_env.py` now has direct public body canary fixture-step coverage through `VectorMultiplayerEnv`. | Rerun the body batch, focused source-env tests, and public multiplayer body canaries; add gap/epsilon and emitted-trail variants before broad body claims. |
| Same-frame point materialization | The same body batch covers same-frame point kill and same-frame control safe. `CurvyTronSourceEnv` now inserts bodies when source point emission occurs while the game has started and the world is active, and its focused tests cover same-frame reverse-order point insertion killing a lower-index avatar. | Rerun the body batch and focused source-env tests; add broader event-order stress cases before treating all same-frame collisions as covered. |
| Collision order | `source_collision_order_batch.json` protects death-point-kills-later-player and the head-head-looking same-endpoint single-death fixture through common trace via `source-body-canary`. `tests/test_vector_multiplayer_env.py` now carries direct public collision canary fixture-step coverage. | Rerun the collision batch and public multiplayer collision canaries; keep PrintManager death-stop variants in the separate print-manager batch. |
| Deterministic print manager | `source_print_manager_batch.json` protects print-to-hole, hole-to-print, exact-zero hole-to-print toggle, active no-toggle control, delayed 3000 ms start, active printing wall stop-on-death, active already-hole wall stop-on-death, and active printing body-collision stop-on-death through `source-print-manager-canary`. `source_print_manager_random_batch.json` separately protects same-frame random tape call order and one straight-line multi-step cadence from real taped print/hole distances. | Rerun both print-manager batches; keep timer/lifecycle-driven cadence separate before broad trail claims. |
| Normal trail cadence threshold | `source_trail_batch.json` protects normal point insertion and below-radius no-point behavior through `source-trail-cadence-canary`. | Rerun the trail batch; keep visual trail points separate from the hidden draw cursor. Add exact-threshold and multi-step cadence only after gap checks. |
| Trail gap body absence, stored-body danger, and boundary transition | `source_trail_gap_batch.json` protects forced hole-space safety, stored-body-in-visual-hole kill, print-to-hole boundary kill, and hole-to-print same-update emitted-body kill through `source-trail-gap-canary`. `source_trail_gap_natural_multistep_hole_crossing.json` separately matches JS/Python common trace, proves one natural taped PrintManager hole crossing where p0 remains alive, and now has scalar vector parity through row-local random tape arrays. `tests/test_vector_multiplayer_env.py` now has direct public trail-gap canary fixture-step coverage. | Rerun the four-case batch, the separate natural source loop, the separate natural vector comparator, and public multiplayer trail-gap canaries after gap edits. Do not put the natural case into vector speed defaults until promotion is deliberate. |
| Source-env scalar scoring and timer guards | `tests/test_source_env.py` now adds focused scalar checks for 3P ordered death scoring, 3P same-frame wall-death scoring, a 3P absent-player scoring corner, timer-drain large advance across `game:stop -> round:new -> game:start`, and a zero-delay timer-loop guard. `advance_timers` drains every due timer up to the target time, including newly scheduled due timers. | Rerun focused source-env tests after scalar scoring or timer changes; keep this separate from broad lifecycle, production vector timers, and full reset/autoreset claims. |
| Source lifecycle spawn, timers, and next round | `tests/test_lifecycle_oracle.py` and `tests/test_source_lifecycle_runner.py` protect 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_tie_at_max_score_4p`, `source_lifecycle_match_end_at_max_score_4p`, `source_lifecycle_multi_round_match_end_4p`, the 3P/4P present/absent tie-at-max fixtures, the three 4P present/absent reset/survivor/next-round fixtures, `source_lifecycle_remove_avatar_during_warmdown_3p`, and `source_lifecycle_remove_avatar_to_single_present_3p` when focused tests pass: three 2P core fixtures, focused 3P spawn order and warmup/PrintManager start, focused 4P first-round spawn, focused 4P all-dead next round, focused 4P survivor next round, focused 4P tie-at-max continuation, focused 4P unique-leader match end, focused 4P all-present multi-round match end, focused 3P/4P present/absent first-round, survivor-scoring, next-round, and tie-at-max cases, focused 2P and 3P match ends, focused 3P tie-at-max continuation, focused 3P all-present multi-round match end, focused 3P warmdown leave, focused 3P single-present leave edge, focused 3P all-dead next round, and focused 3P survivor round-end plus next-round cases. | Rerun the `source-lifecycle` matrix check or the focused pytest command after lifecycle, spawn RNG, timer, or print-start edits. The matrix command is `uv run python scripts/run_environment_fidelity_matrix.py --run source-lifecycle`. Keep broader present/non-present variants, bonuses, production reset/autoreset/replay/final obs/reward, optimized/vector full lifecycle, and server-message lifecycle separate. |
| Fast helper/public match/tie/present-absent lifecycle | `tests/test_vector_lifecycle.py` protects helper-level 3P unique max-score match end, tied max-score continuation, and one multi-round continue-then-end path. `tests/test_vector_multiplayer_env.py` now protects the long 2P source reset-to-terminal wall rollout, focused 2P draw warmdown into next round, focused 2P unique-leader max-score match end, the same focused 3P paths, focused 4P tie-at-max continuation, 4P unique-leader match end, 4P all-present multi-round match end, and 4P present/absent reset, survivor scoring, and next-round metadata through metadata-only public warmdown/warmup bridges. | Rerun focused lifecycle and public multiplayer tests after match scoring, warmdown, `match_done`, `match_winner`, max-score, present masks, or public lifecycle metadata edits. This is not replay, trainer-observation, visual stacked-frame, broad match-mode episode support, or broad public warmdown leave. |
| RemoveAvatar/leave | `source_lifecycle_mid_round_remove_avatar_2p.json` is Python/oracle verified in `tests/test_source_env.py`. It proves active 2P `Game.removeAvatar()` destroys the leaving avatar to `present=false/alive=false`, emits `player:leave`, does not add that avatar to current-round `deaths`, and immediately ends the round when one avatar remains alive. The 3P and 4P continuation fixtures are also JS oracle plus `CurvyTronSourceEnv` checked. `source_lifecycle_remove_avatar_to_single_present_3p.json` is source-promoted under `source-lifecycle-v25` and proves the active 3P single-present edge, including avatar 1 getting `roundScore=2` from total avatar count and the next round using two-present-player size. Public metadata has narrow active-round support through `VectorMultiplayerEnv.remove_player(...)`: 3P/4P continuation, 2P immediate round end, the 3P single-present leave edge, and a 4P source-rule canary for survivor scoring after already-dead players. The focused 3P staged match-mode warmdown leave metadata proof is also green. In all covered public cases the leaver becomes present/alive false and is omitted from `death_player`. | Rerun `uv run pytest tests/test_lifecycle_oracle.py tests/test_source_env.py -q -k mid_round_remove_avatar`, `uv run pytest tests/test_lifecycle_oracle.py tests/test_source_lifecycle_runner.py -q -k remove_avatar_to_single_present`, and `uv run pytest tests/test_vector_multiplayer_env.py -q -k 'remove_player or active_round_leave'` after leave/present/death-list changes. Keep broad warmdown leave, replay, trainer, visual, bonus, and broader present/non-present variants separate. |
| Narrow active BonusSelfSmall catch/no-catch/death-order | JS oracle fixtures `source_bonus_self_small_catch_step.json`, `source_bonus_self_small_tangent_no_catch_step.json`, and `source_bonus_self_small_wall_death_no_catch_step.json` now have matching `CurvyTronSourceEnv` checks in `tests/test_source_env.py`. They prove seeded active `BonusSelfSmall` catch after movement, strict-overlap tangent no-catch, and same-tick wall-death no-catch. Python filters JS's non-important death point in the wall-death event comparison. | Rerun `uv run pytest tests/test_source_env.py -q` after bonus catch/stack/world changes. This row does not prove expiry; the separate expiry/restore row owns exactly one timed restore case. Other bonus effects, broader death interactions, and broader vector/runtime bonus support remain open. |
| BonusSelfSmall expiry/restore | `source_bonus_self_small_expiry_restore_step.json` has JS/Python source-env parity for one caught `BonusSelfSmall` expiring after `7500` ms. It proves the timer fires before the zero-elapsed update, emits `property radius=0.6`, emits `bonus:stack remove`, leaves `bonusCount=0` and `bonusWorldBodyCount=1`, and does not emit a second `bonus:clear`. `tests/test_vector_runtime.py` now also protects the optional-array fast-runtime expiry/restore slice and the fixture bridge now preserves per-step `advance_timers_ms` as `timer_advance_ms`. | Rerun `uv run pytest tests/test_source_env.py -q -k bonus`, `uv run pytest tests/test_env_scenarios.py -q -k bonus`, and `uv run pytest tests/test_vector_runtime.py -q` after source-env, JS runner, bonus fixture, fixture-bridge, or vector-runtime bonus edits. This does not prove broader stack math, same-frame expiry ordering with other timers, other effects, other bonus types, or public-env/replay bonus support. |
| Minimal natural bonus spawn/type RNG | `source_bonus_spawn_type_position_rng_step.json` has JS/Python source-env parity for one enabled type. It enables just `BonusSelfSmall`, advances source elapsed time to the 1500 ms first pop before PrintManager starts, and proves the labeled random order `bonus.start_delay`, `bonus.next_delay_after_pop`, `bonus.type.BonusSelfSmall`, `bonus.position.x`, `bonus.position.y`; the frame has `bonusCount=1`, `bonusWorldBodyCount=1`, `bonus:pop` before the zero-elapsed position events, type `BonusSelfSmall`, and position `(23.94, 64.06)`. `source_bonus_default_weights_type_rng_step.json` and `source_bonus_default_weights_select_game_clear_step.json` add the default multi-type source order with two of four present avatars already dead, proving the reduced `BonusGameClear` dynamic probability window: type draw `0.945` selects `BonusAllColor`, while type draw `0.965` selects `BonusGameClear`, both at position `(27.255, 73.745)`. `tests/test_vector_runtime.py` pins the same reduced edge plus the one-dead full-probability `0.93` edge through `bonus_type_selection_metadata`, and also protects the low-level natural spawn helper type/position path. | Rerun `uv run pytest tests/test_source_env.py -q -k bonus`, `uv run pytest tests/test_env_scenarios.py -q -k bonus`, and `uv run pytest tests/test_vector_runtime.py -q -k bonus` after source-env, JS runner, bonus fixture, or vector-runtime metadata edits. This does not prove catch/effects for newly selectable spawned types, natural spawned `BonusGameClear` catch/clear coupling, public spawn timer ownership/scheduling/random accounting, public bonus env, or replay bonus support. |
| Natural bonus spawn retry against game world | `source_bonus_spawn_game_world_retry_step.json` has JS/Python source-env parity for one rejected natural `BonusSelfSmall` position against a seeded main game-world body, followed by one accepted position. It pins seven draws: start delay, next delay, one-type selection, first `x/y`, retry `x/y`; exactly one retry pair; `worldBodyCount=1`, `bonusCount=1`, `bonusWorldBodyCount=1`; `bonus:pop` before zero-elapsed position events; final bonus type `BonusSelfSmall` at `(68.072, 19.928)`. `tests/test_vector_runtime.py` now protects the matching low-level helper retry path. | Rerun `uv run pytest tests/test_source_env.py -q -k bonus`, `uv run pytest tests/test_env_scenarios.py -q -k bonus`, and `uv run pytest tests/test_vector_runtime.py -q -k bonus` after source-env, JS runner, bonus fixture, or vector-runtime helper edits. Broader stack math/expiry ordering, borderless/speed/radius/inverse/color effects after catch, public timer ownership/scheduling/random accounting, public bonus env, and bonus replay remain open. |
| Natural bonus spawn retry against bonus world | `source_bonus_spawn_bonus_world_retry_step.json` has JS/Python source-env parity for one rejected natural `BonusSelfSmall` position against an already active bonus-world body, followed by one accepted position. It pins seven draws: start delay, next delay, one-type selection, first `x/y`, retry `x/y`; exactly one retry pair; `worldBodyCount=0`, `bonusCount=2`, `bonusWorldBodyCount=2`; `bonus:pop` before zero-elapsed position events; final new bonus type `BonusSelfSmall` at `(68.072, 19.928)`. `tests/test_vector_runtime.py` now protects the matching low-level helper retry path. | Rerun `uv run pytest tests/test_source_env.py -q -k bonus`, `uv run pytest tests/test_env_scenarios.py -q -k bonus`, and `uv run pytest tests/test_vector_runtime.py -q -k bonus` after source-env, JS runner, bonus fixture, or vector-runtime helper edits. This is not public bonus env support, bonus replay, or broad bonus effects. |
| Natural bonus cap at 20 | `source_bonus_spawn_cap_twenty_step.json` has JS/Python source-env parity for the source cap branch. It seeds 20 active map bonuses after `BonusManager.start()`, advances to the first natural pop, proves only `bonus.start_delay` and `bonus.next_delay_after_pop` are consumed, emits no new `bonus:pop`, and leaves `bonusCount=20` and `bonusWorldBodyCount=20`. `tests/test_vector_runtime.py` now protects the metadata-only `bonus_spawn_cap_metadata` slice and the low-level natural spawn helper cap path: for caller-known eligible pop rows, `bonus_count >= 20` marks a row capped and excludes it from subsequent type/position work. | Rerun `uv run pytest tests/test_source_env.py -q -k bonus`, `uv run pytest tests/test_env_scenarios.py -q -k bonus`, and `uv run pytest tests/test_vector_runtime.py -q -k bonus` after source-env, JS runner, bonus fixture, or vector-runtime bonus metadata edits. This is not public bonus env, replay, public timer scheduling/random accounting, bonus mutation, or overflow policy beyond source parity. |
| BonusGameClear immediate clear | `source_bonus_game_clear_immediate_step.json` has JS/Python source-env parity for one forced active `BonusGameClear` catch after safe movement. It starts with one seeded main-world trail/body far from p0, catches the seeded map bonus with no random calls, emits `bonus:clear` then `clear`, keeps p0 alive, leaves both avatars with radius `0.6` and no active avatar bonuses, emits no `bonus:stack` or avatar `property`, and proves the main world is cleared then active with `worldActive=true` and `worldBodyCount=0` while `bonusCount=0` and `bonusWorldBodyCount=1`. `tests/test_vector_runtime.py` now protects the matching optional-array fast-runtime clear slice. | Rerun `uv run pytest tests/test_source_env.py -q -k bonus`, `uv run pytest tests/test_env_scenarios.py -q -k bonus`, and `uv run pytest tests/test_vector_runtime.py -q` after source-env, JS runner, bonus fixture, or vector-runtime bonus edits. This does not prove natural spawned `BonusGameClear` catch/clear coupling, borderless, speed, inverse, color, broader stack timing, or public-env/replay bonus support. |

## Open Mechanics And Tests

| Hole | How to test it |
| --- | --- |
| Print-manager cadence | `source_print_manager_random_call_order_step` is promoted through Python/common trace and pins same-frame PrintManager random call order. `source_print_manager_random_cadence_multistep` now adds one four-tick print-to-hole-to-print cadence case that starts from a real taped print distance, spends it over multiple ticks, then spends a real taped hole distance. Keep this separate from normal trail cadence, delayed start, and broader round lifecycle. |
| Natural trail-gap speed promotion | Source has one separate natural multi-step taped gap crossing, and scalar vector comparison now owns that exact full trace. It is not in speed defaults because B>1 promotion still needs an intentional batch-row path and broader reset/RNG policy. |
| Visual gaps versus collision bodies | First prove server state with counters and events. The current product visual path is full-size source-state RGB raw frame -> deterministic gray64 -> frame stack. `scripts/compare_2p_raw_visual_observation.py --suite full2p --format plain` now passes 35 full 2P source-vs-vector gray64 scenarios with `max_abs_diff=0` and `mismatch_pixels=0`, plus typed bonus diagnostics for all 12 source-default bonus types, final-observation checks, and two intentional mismatch canaries. This includes the long wall rollout through terminal plus movement, normal-wall/draw, collision-order, borderless, narrow bonus frames, the four natural bonus spawn/retry/cap fixtures, and programmatic source-snapshot stress cases. `source_print_manager_random_call_order_step` stays outside gray64 because it proves RNG/event order, not a distinct rendered state. The renderer skips shapes fully outside the source arena, matching source-world visibility for wall-death points. Current gray64 distinguishes 2P player trails/heads and uses browser-like trail lines by default; bonus identity proof lives in the separate bonus64 diagnostic gate, not the product tensor. Browser/canvas pixel checks are optional later human/debug evidence, not P0. Use exact short source-state checks and long-rollout divergence reporting. Do not claim frame-for-frame visual parity after trajectories diverge; record `first_divergent_tick` and compare from source-state resync checkpoints when needed. Future metrics: `max_abs_diff`, `mismatch_pixels`, tolerant pixel threshold, centroid/body/state drift, resync cadence, and saved source/frame/diff/metrics artifacts. |
| Broader opponent/self trails | Source-env point-time body insertion now exists for focused tests. Add broader fixtures where bodies come from emitted points across longer traces, gaps, and more players. Check body owner, point number, killer, and state counters. |
| Death-frame point side effects | Focused same-frame reverse-order insertion is covered in the source env. Add broader event-order cases before making a general claim. Compare event order, alive state, killer, score, and body counters. |
| Head-head and reverse-order collisions | The 2P death-point and head-head-looking same-endpoint fixtures are promoted. Add 3P scripted setups only when they isolate a new source rule. |
| Longer traces and numeric tolerance | Straight, turn, and varied per-step elapsed-ms multi-step movement are Python/common-trace verified. The varied fixture keeps explicit `0.000001` position/angle/velocity tolerance. |
| Borderless follow-up corner cases | The first exact-edge/corner control is promoted. Add next-frame second-axis wrap only if a later feature depends on it. |
| World island edges | Source-verified in `tests/test_source_env.py`: corner insertion across bucket boundaries, duplicate prevention, missing-island tolerance near arena edge, and lookup by query-body corners. |
| Bonuses | The first `BonusSelfSmall` catch/no-catch/death-order, one-type spawn RNG, default multi-type weight/type RNG including natural `BonusGameClear` selection, one game-world spawn retry, one bonus-world retry, one cap-at-20 skip fixture, one stack-expiry fixture, one forced `BonusGameClear` immediate clear fixture, and forced `BonusGameBorderless` catch plus duration/expiry fixture are promoted. The source oracle has broad Python stack/effect support for self/enemy/all bonuses. Fast runtime now covers table-backed optional-array support for promoted runtime effects including `BonusSelfMaster`, `BonusAllColor`, and `BonusEnemyStraightAngle`, plus source-default type selection/probability, a cap gate for caller-known eligible pop rows, and a low-level natural spawn helper with type/position/retry/cap tests. `VectorMultiplayerEnv` has focused seeded/natural coverage for source-default catch/effect families: `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`, `BonusSelfMaster`, `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`, `BonusEnemyInverse`, `BonusEnemyStraightAngle`, `BonusGameBorderless`, `BonusAllColor`, and `BonusGameClear`; SelfMaster wall/body parity and AllColor reverse target event order plus older-wins overlap behavior are fixed. Seeded public bonus replay metadata preserves bonus metadata/audit fields only, not full replay arrays. Seed-generated public random tape auto-extends deterministically, generated natural bonus position retry is no longer capped by `natural_bonus_position_attempt_capacity`, and public natural bonus timer advancement no longer has an artificial callback cap; fixture/direct finite tapes and `vector_runtime` finite helpers stay strict. Current gaps still include manual/direct stack guard documentation, a possible fully-blocked generated-map policy, full public replay/final observations, broader stack/death stress, broader 3P/4P lifecycle/leave parity, visual pixel parity, toy-path quarantine, and final cleanup. Keep the next bonus proof equally small. |
| Broader lifecycle and replay/server messages | The 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_tie_at_max_score_4p`, `source_lifecycle_match_end_at_max_score_4p`, `source_lifecycle_multi_round_match_end_4p`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, `source_lifecycle_present_absent_3p_tie_at_max_score`, `source_lifecycle_present_absent_4p_tie_at_max_score`, `source_lifecycle_present_absent_4p_next_round`, `source_lifecycle_remove_avatar_during_warmdown_3p`, `source_lifecycle_remove_avatar_to_single_present_3p`, and `source_lifecycle_multi_round_match_end_3p` cover the current focused 2P/3P/4P lifecycle slice. No-bonus N-player reset/warmup helpers cover part of the optimized path, focused helper-level 3P/4P continuation proofs are green, and metadata-only public 2P/3P/4P stepping plus focused 2P draw/match-end and 3P/4P present/absent public warmdown bridges exist. Focused public metadata proofs also exist for 4P all-present multi-round match end, 3P staged match-mode warmdown leave, and the 3P single-present active leave edge. Natural public reset/warmdown/replay/trainer-observation parity, broader public present/non-present variants, production autoreset/replay/final obs/reward, broad public warmdown leave, and optimized/vector full lifecycle remain outside this slice. Add new fixtures only when they isolate a broader lifecycle rule. Keep wire compression and browser messages out of the core proof loop for now. |
| Multiplayer replay contract | `src/curvyzero/training/multiplayer_replay_contract.py` and `tests/test_multiplayer_replay_contract.py` define and guard required future 3P/4P metadata fields. | Keep strict 1v1 replay-v0 separate. Add a real 3P/4P replay writer only after public metadata and final observation policy settle. |
| Observation and reward surface | Defer until source state/events are pinned. Use source-state observation checks first; browser/canvas pixel checks are a separate later gate. |

## Current Priority

1. Keep the goal plain: one fast, source-faithful environment runtime.
2. Widen natural bonus support beyond the promoted focused seeded/natural
   slices.
3. Keep Halley's capacity policy explicit: generated tape/position retry
   extension is separate from fixture/direct finite tapes and strict
   `vector_runtime` finite helpers; artificial/manual stack overflow is a
   fixed-array guard; add fully-blocked generated-map policy only if needed.
4. Pin timer/random ordering for public bonus scheduling.
5. Finish borderless stack/wrap/collision semantics beyond the seeded public
   expiry slice. Source/runtime/public duration/expiry has focused coverage; do
   not list that proof as missing.
6. Widen bonus public metadata/replay audit for spawned bonus identity,
   catch/expiry/clear events, active stacks, RNG cursor/draw counts, and source
   refs while keeping the claim metadata-only until full replay arrays exist.
7. Add broader bonus effects only from source-backed claims.
8. Fill full public replay/final observations, then broaden lifecycle and
   multiplayer parity for reset/warmup, warmdown movement, next-round/match-end,
   present/absent, leave, masks, and rewards.
9. Quarantine toy/debug paths as historical smoke evidence only.
10. Browser/source pixel parity is later, after source state and replay rows are
   stable.
11. Finish final cleanup after the remaining source-backed claims settle.
12. Modal, speed, and fixed-opponent route smokes stay labeled as evidence only.

## Regression Batches

Run these when a source-fidelity slice changes:

```bash
uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --fail-on-mismatch --artifact-root /private/tmp/curvy-source-kinematics-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-border-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-regression
uv run python tools/run_fidelity_batch.py scenarios/environment/source_collision_order_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-collision-order-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_batch.json --python-runner source-trail-cadence-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-cadence-regression
uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_gap_batch.json --python-runner source-trail-gap-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-gap-regression
uv run pytest tests/test_source_lifecycle_runner.py tests/test_lifecycle_oracle.py -q
uv run pytest tests/test_source_env.py -q
uv run pytest tests/test_source_env.py tests/test_vector_runtime.py tests/test_source_lifecycle_runner.py tests/test_lifecycle_oracle.py tests/test_env_reference_defaults.py -q
uv run ruff check src/curvyzero/env/source_env.py tests/test_source_env.py
```

Repository-wide tests and lint are hygiene footnotes after code changes, not
source-claim acceptance evidence. The combined focused command above is the
current integrated status command, not a scoreboard.

## Ways To Poke Holes

- Source-read the relevant server source before naming the rule.
- Add a JS oracle fixture that isolates one behavior.
- Compare through common trace diff before reading raw runner noise.
- Check event order when a mechanic depends on frame/update order.
- Add state counters when parity could hide missing bodies, points, scores, or
  round state.
- Add longer traces after the one-step fixture matches.
- Use fuzz/scripted probes after source-state goldens exist, not as the first
  proof. Existing JS reference tooling under `tools/reference_oracle` and
  `tools/js_reuse_probe` can produce golden source-state snapshots, but there
  is no finished browser/canvas pixel golden-frame harness yet.
- Add observation checks after state/events settle.
- Add browser/canvas pixel checks later for rendering and human review only.
