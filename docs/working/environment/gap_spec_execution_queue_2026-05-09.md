# Environment Gap Spec Execution Queue

Status: working execution queue
Date: 2026-05-09
Owner: working memory

## Current Truth

We are not at full CurvyTron environment fidelity.

The proven lifecycle/spawn/RNG source claim is direct JS/Python parity for
28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`,
`source_lifecycle_survivor_score_4p_next_round`,
`source_lifecycle_present_absent_3p_survivor_score_round_end`, and
`source_lifecycle_multi_round_match_end_3p`: three 2P core
lifecycle/spawn/RNG fixtures, focused 3P first-round spawn-order, focused 3P
warmup and delayed PrintManager start, focused 4P first-round spawn-order,
focused 4P all-present all-dead warmdown/next-round, focused 4P survivor
scoring and next-round continuation, focused 3P first-round present/absent,
focused 3P present/absent survivor scoring, focused 3P present/absent
warmdown/next-round, focused 2P max-score match-end, focused 3P max-score
match-end, focused 3P all-dead warmdown/next-round, focused 3P survivor-scoring
`round:end`, focused 3P survivor warmdown/next-round, focused 3P
tie-at-max-score continuation, and focused 3P all-present multi-round match-end,
checked by `tests/test_source_lifecycle_runner.py` against
`tools/reference_oracle/lifecycle_oracle.js`:

- `scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_2p.json`
- `scenarios/environment/source_lifecycle_spawn_rng_2p_next_round.json`
- `scenarios/environment/source_lifecycle_spawn_heading_rejection_retry_2p.json`
- `scenarios/environment/source_lifecycle_spawn_rng_order_3p.json`
- `scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_3p.json`
- `scenarios/environment/source_lifecycle_spawn_rng_3p_next_round.json`
- `scenarios/environment/source_lifecycle_spawn_rng_order_4p.json`
- `scenarios/environment/source_lifecycle_spawn_rng_4p_next_round.json`
- `scenarios/environment/source_lifecycle_survivor_score_4p_next_round.json`
- `scenarios/environment/source_lifecycle_present_absent_3p_round_new.json`
- `scenarios/environment/source_lifecycle_present_absent_3p_survivor_score_round_end.json`
- `scenarios/environment/source_lifecycle_present_absent_3p_next_round.json`
- `scenarios/environment/source_lifecycle_match_end_at_max_score_2p.json`
- `scenarios/environment/source_lifecycle_match_end_at_max_score_3p.json`
- `scenarios/environment/source_lifecycle_tie_at_max_score_3p.json`
- `scenarios/environment/source_lifecycle_survivor_score_3p_round_end.json`
- `scenarios/environment/source_lifecycle_survivor_score_3p_next_round.json`
- `scenarios/environment/source_lifecycle_multi_round_match_end_3p.json`

That claim includes one 2P heading rejection retry and one pinned 3P
first-round natural spawn order/RNG-label check: avatars spawn in reverse order
3, 2, 1 with `position_x`, `position_y`, and `angle_attempt_0` calls at 0 ms.
It includes one focused 3P warmup and delayed PrintManager start check for
order and random calls. It also includes one focused 4P first-round natural
spawn order/RNG-label check: avatars spawn in reverse order 4, 3, 2, 1.
It also includes one focused 4P all-present all-dead continuation: after
delayed PrintManager starts at 3000 ms, same-frame wall deaths process in
reverse avatar order 4, 3, 2, 1, source emits `round:end` winner null at
3000 ms, then `game:stop` and the next `round:new` at 8000 ms with the next
natural 4P spawn RNG/order.
It also includes one focused 4P survivor continuation: avatars 4, 3, and 2 die
across separate forced wall-death updates, receive round scores 0, 1, and 2,
avatar 1 receives the survivor bonus, then `game:stop` and the next
`round:new` emit at 8000 ms with reverse 4P spawn RNG/order.
It also includes one pinned 3P `Game.onRoundNew()` present/non-present case:
avatar 2 is non-present, consumes no spawn RNG, is added to `game.deaths`, and
is snapshotted as `present=false`, `alive=false`, at `(0.6, 0.6)`, with
`deathCount=1` and `deaths=[2]`. It also includes one focused 3P
present/non-present survivor-scoring case: avatar 2 is already absent and in
source `deaths` without a `die` event, avatar 3 dies and gets `roundScore=1`,
avatar 1 survives with `roundScore=2`, and `round:end` winner is avatar 1. It
also includes one focused 3P present/non-present continuation: after the two
present avatars die, `game:stop`
resizes the arena to present-player size 88 and the next `round:new` re-adds
avatar 2 to `deaths` while spawning only avatars 3 and 1. It also includes one
focused 2P max-score
match-end case: with `max_score: 1`, avatar 2 dies, avatar 1 reaches score 1,
source emits `round:end` with winner 1 at 3000 ms, then `game:stop` and `end`
at 8000 ms, and does not immediately emit another `round:new`; the final
snapshot has `started=false`, `inRound=false`, cleared world fields, and no
avatars. The 3P match-end fixture proves only the focused all-present
`max_score: 2` path: avatars 3 and 2 die in one elapsed-ms source update,
avatar 1 reaches score 2, source emits `round:end` with winner 1 at 3000 ms,
then `game:stop` and `end` at 8000 ms, and does not immediately emit another
`round:new`; the final snapshot has `started=false`, `inRound=false`, cleared
world fields, and no avatars. The 3P all-dead fixture proves only forced
same-frame wall deaths:
`round:end` winner null at 3000 ms, `game:stop` then `round:new` at 8000 ms,
and next natural 3P spawn RNG/order. The 3P survivor fixture proves only
focused score resolution through `round:end`: avatar 1 survivor `roundScore=2`,
winner 1, and deaths `[3, 2]`. The 3P survivor warmdown fixture proves only
that avatar 1 continues moving after `round:end`, dies at 4150 ms, then
`game:stop` and next `round:new` emit at 8000 ms with next natural 3P spawn
RNG/order. The focused 3P tie-at-max fixture proves tied leaders continue to
next round, not broad multi-round: avatar 3 dies first, avatars 2 and 1 then
die together, avatars 1 and 2 both resolve to score 1 at `max_score: 1`, source
emits `round:end` winner null at 3000 ms, then `game:stop` and `round:new` at
8000 ms with no `end`. The focused 3P multi-round match-end fixture proves
only one all-present path: avatar 1 carries score 2 through `game:stop` and
the next `round:new` at 8000 ms, then reaches score 4 and source emits
`game:stop` and `end` at 19000 ms with no later `round:new`. The promoted
slice does not include broader 4P match lifecycle beyond the focused all-dead
and survivor-next-round cases, broader present/non-present lifecycle beyond the
focused first-round, survivor-scoring, and next-round cases, bonuses,
production reset/autoreset, production replay/final observations/rewards,
optimized/vector full lifecycle, or trainer/replay final observation.

`CurvyTronSourceEnv` separately has Python/oracle verification for
`source_lifecycle_mid_round_remove_avatar_2p.json`: active 2P
`removeAvatar` destroys the leaving avatar, emits `player:leave`, does not add
that avatar to current-round `deaths`, and immediately ends the round when one
avatar remains alive.

The first bonus source-env slices are narrow but real:
`source_bonus_self_small_catch_step.json`,
`source_bonus_self_small_tangent_no_catch_step.json`,
`source_bonus_self_small_wall_death_no_catch_step.json`,
`source_bonus_spawn_type_position_rng_step.json`, and
`source_bonus_spawn_game_world_retry_step.json`, and
`source_bonus_self_small_expiry_restore_step.json` are JS oracle fixtures with
matching `tests/test_source_env.py` checks for active seeded `BonusSelfSmall`
catch/no-catch/death-order, minimal one-type natural spawn/type/position RNG,
one game-world rejected-position retry, and one timed expiry/restore case.
`source_bonus_default_weights_type_rng_step.json` adds one default multi-type
selection proof: with two of four present avatars dead, source reduces
`BonusGameClear`'s dynamic probability, selects `BonusAllColor`, and emits
`bonus:pop` before zero-elapsed position events at `(27.255, 73.745)`.
`source_bonus_game_clear_immediate_step.json` adds one forced `BonusGameClear`
immediate clear proof: seeded catch after safe movement, `bonus:clear` then
`clear`, `worldActive=true`, `worldBodyCount=0`, no `bonus:stack`, no avatar
property change, and no active avatar bonuses. This does not prove cap behavior,
catch/effects for the newly selectable spawned types, broader stack/effect behavior, death interactions,
natural `BonusGameClear` probability/selection, or vector/runtime bonus support.

The optimized/vector path does not yet support full natural lifecycle or reset
RNG. It now has a narrow vector spawn helper:
`src/curvyzero/env/vector_spawn.py::spawn_round_rows(state, row_mask, *,
player_count)`. Tests in `tests/test_vector_spawn.py` match promoted
first-round spawn facts for 2P heading retry, 3P reverse spawn order, 3P
present/absent, and 4P reverse spawn order. The helper uses row-local random tape/cursor/draw-count,
per-row map size, reverse player order, present masks, heading retries, and
absent-player death-list arrays when present. It does not insert or mutate
world bodies. `src/curvyzero/env/vector_lifecycle.py` now composes reset plus
spawn for selected rows and can optionally stamp 1v1 delayed-start timer
metadata when the timer arrays exist. The stricter
`reset_spawn_warmup_1v1_no_bonus_rows(...)` helper clears selected round-local
arrays and schedules only the first `GAME_START` timer. It still reports
`full_lifecycle=false` and remains a reset/spawn/warmup boundary, not full
lifecycle. `src/curvyzero/env/vector_runtime.py` owns the first runtime shell:
input validation, counters, row-local PrintManager random distances, movement,
and borderless wrap. Normal wall handling, body collision, body append, events,
terminal scoring, and timers still need extraction behind `step_many`.

`scripts/seed_vector_state_from_fixtures.py` still rejects natural
`source_lifecycle_*` `Game.newRound()` fixtures as ordinary initial-state
seeds and records RNG contract metadata: call index, site, avatar, value,
at-ms, expected call count, and capacity pressure. That guard plus the narrow
spawn helper are useful, but they are not optimized/vector lifecycle, round
timers, `game:start`, warmup/warmdown, PrintManager start, next-round, scoring,
terminal handling, autoreset integration, final obs, or replay.

LightZero work is plumbing only. Local and installed no-train smokes prove
adapter/config/import/reset/step boundaries where explicitly named; they are
not training runs and not stronger environment-fidelity claims.

Multiplayer replay now has a widened metadata-only guard in
`src/curvyzero/training/multiplayer_replay_contract.py` and a recorder/row
builder in `src/curvyzero/training/multiplayer_replay_v0.py`. It requires
public env/rules ids, rules hash, native/trainer control ids, `decision_ms`,
episode/round/step/tick/elapsed fields, present/alive masks, legal masks, full
wrapper action sidecar, reward/action/observation schema ids and hashes,
terminal/truncation/reset metadata, needs-reset, round/match winner ids, death
order, final-observation policy, and explicit false
`trainer_observation_claim`, `trainer_replay_claim`, and
`learned_observation_claim`. It accepts current public multiplayer step batches
and can carry optional `random_tape_source`, `random_tape_length`,
`rng_impl_id`, and `source_fixture_ref` if emitted. This remains metadata-only
replay, not a production trainer replay writer, learned 3P/4P observation
tensor, replay sampler, or LightZero trainer-ready claim.

Source state and source event order are the gameplay authority. Browser pixels
come later. Commands are named below as acceptance gates; pass totals are not
proof by themselves.

Training block labels:

- `yes`: blocks source-faithful training.
- `indirect`: can be deferred for a restricted/no-feature subset, but blocks a
  broader CurvyTron claim.
- `no`: useful for replay, demos, or render proof, but not a trainer blocker.

## Parallel Feature-Set Queue

Each lane can move in parallel as long as it keeps its claim narrow and ties any
new rule to a named source fixture or production transition contract.

| Lane | Feature set | Acceptance target | Likely touch points |
| --- | --- | --- | --- |
| A | Fast 1v1 no-bonus lifecycle end-to-end | One row can reset, spawn, warm up, fire `game:start`/PrintManager timers, step to normal-wall `round:end`, preserve terminal obs/reward/info, and stage replay-v0 metadata without fixture cycling. | `vector_lifecycle.py`, `vector_runtime.py`, `vector_autoreset.py`, trainer/replay tests. |
| B | 3P survivor/match-end lifecycle | Source proof is pinned for the focused survivor warmdown/next-round path and one focused 3P match-end path; keep broader present/absent variants separate. | `source_lifecycle_survivor_score_3p_next_round.json`, `source_lifecycle_match_end_at_max_score_3p.json`, plus lifecycle oracle/runner tests. |
| C | Row-local RNG/reset history | Reset rows carry episode id, reset seed/source, RNG state/cursor/draw counts, exhaustion, and enough labels/history for replay and B>1 independence. | `vector_reset.py`, `vector_lifecycle.py`, `vector_runtime.py`, replay metadata tests. |
| D | First `BonusSelfSmall` catch | Source-backed forced catch proves ordering after movement/collision/PrintManager, no catch after same-tick death, and the first self-small effect/stack reset fact. | First bonus scenario, source bonus runner, reference default canary. |
| E | Trainer/replay terminal surface | Terminal transition exposes done/terminated/truncated, final observation, final reward map, terminal info, reset observation ordering, hashes, and compatibility rejection. | `trainer_observation.py`, `trainer_replay_v0_builder.py`, `replay_chunk_v0.py`, autoreset/replay tests. |
| F | Runtime extraction chunks | Move long-lived runtime pieces behind `vector_runtime.step_many` in small chunks: normal wall mask, body collision scan, body append, event rows, terminal scoring, and timers. | `vector_runtime.py`, batch-row bridge, focused vector runtime tests. |

## 1. Source Lifecycle, Spawn, And RNG

### 1.1 Promoted Lifecycle Claim Boundary

- Current proof: the three promoted 2P fixtures prove reverse-player spawn
  order, x/y spawn RNG, accepted first-attempt heading RNG, one rejected
  heading attempt followed by an accepted retry, delayed print-start RNG,
  active stop RNG, warmdown, and next-round spawn RNG. The focused 3P fixture
  proves only first-round natural spawn order/RNG labels at 0 ms: avatars 3, 2,
  1, each with `position_x`, `position_y`, and `angle_attempt_0`.
  `source_lifecycle_spawn_rng_warmup_print_start_3p.json` proves focused 3P
  warmup and delayed PrintManager start order/random calls. The focused 4P
  fixture proves only first-round natural spawn order/RNG labels at 0 ms:
  avatars 4, 3, 2, 1.
  `source_lifecycle_present_absent_3p_round_new.json` proves only first-round
  3P `Game.onRoundNew()` with avatar 2 non-present: source skips avatar 2
  spawn RNG, spawns avatar 3 then avatar 1, and adds avatar 2 to `game.deaths`.
  `source_lifecycle_match_end_at_max_score_2p.json` proves only the focused 2P
  max-score match-end path: `round:end` winner 1 at 3000 ms, `game:stop` and
  `end` at 8000 ms, no immediate next `round:new`, and final stopped/cleared
  snapshot. `source_lifecycle_match_end_at_max_score_3p.json` proves only one
  focused 3P max-score match-end path: two same-frame wall deaths, avatar 1
  score 2 at `max_score: 2`, `round:end` winner 1 at 3000 ms, then
  `game:stop` and `end` at 8000 ms with no immediate next `round:new`.
  `source_lifecycle_survivor_score_3p_round_end.json` proves only one
  focused 3P survivor-scoring `round:end`: two same-frame wall deaths, avatar 1
  survivor `roundScore=2`, reverse score resolution, winner 1, and deaths
  `[3, 2]`. `source_lifecycle_survivor_score_3p_next_round.json` proves only
  the focused survivor warmdown/next-round continuation: live avatar 1 keeps
  moving after `round:end`, dies at 4150 ms, then `game:stop` and next
  `round:new` emit at 8000 ms with next natural 3P spawn RNG/order.
- Missing proof outside this claim: broader 4P match lifecycle beyond the
  focused current cases
  spawn, broader present/non-present variants, broader heading retry controls,
  bonuses, production
  reset/autoreset/replay/final observations/rewards, and optimized/vector full
  lifecycle rows.
- Next file/test work: do not add another 2P heading-retry fixture unless it
  isolates a different source rule. Do not extend the 3P claim beyond spawn
  order until a separate fixture proves that next rule.
- Blocks training: `yes` for broad lifecycle training claims; `no` for the
  already pinned source claim.

### 1.2 Broader 3P/4P Natural Spawn/Lifecycle

- Current proof: 2P natural spawn order, 2P next-round spawn order, and pinned
  3P first-round natural spawn order have direct parity. A focused 3P warmup
  and delayed PrintManager start fixture has direct parity. A focused 4P
  first-round natural spawn order/RNG-label fixture also has direct parity. A
  focused 3P survivor warmdown/next-round fixture has direct parity, as does one
  focused 3P present/absent warmdown/next-round fixture.
- Missing proof: broader present/non-present variants, broader 4P
  lifecycle beyond first-round spawn, and broader spawn rejection controls.
- Next file/test work: add one broader present/absent or broader 4P match lifecycle
  fixture only if it catches a different rule. Widen the
  lifecycle oracle, runner, and lifecycle tests only for that named claim.
- Blocks training: `yes`.

### 1.3 Present And Non-Present Players

- Current proof: one focused 3P first-round `Game.onRoundNew()` fixture and one
  focused 3P continuation fixture are promoted. In
  `source_lifecycle_present_absent_3p_round_new.json`, avatar 2 is
  non-present, source skips avatar 2 for natural spawn RNG, spawns avatar 3 then
  avatar 1, and adds avatar 2 to `game.deaths`. The snapshot pins avatar 2 as
  `alive=false`, `present=false`, at `(0.6, 0.6)`, with `deathCount=1` and
  `deaths=[2]`. In `source_lifecycle_present_absent_3p_next_round.json`, the
  two present avatars die, `game:stop` resizes to present-player size 88, and
  the next `round:new` re-adds avatar 2 to deaths while spawning only avatars 3
  and 1.
- Missing proof: broader present/non-present match end, scoring variants, timer
  variants, and other player-count shapes.
- Next file/test work: add another present/non-present fixture only if it
  isolates one of those broader rules; compare JS oracle output and Python
  lifecycle output in `tests/test_source_lifecycle_runner.py`.
- Blocks training: `indirect` for fixed all-present 2P training; `yes` for a
  broader player-count target.

### 1.4 Match Win And End

- Current proof: the 2P next-round fixture proves terminal round end, warmdown
  `game:stop`, and synchronous next `round:new` when the match is not won. The
  focused match-end fixture proves `max_score: 1`: after avatar 2 dies and
  avatar 1 reaches score 1, source emits `round:end` with winner 1 at 3000 ms,
  then `game:stop` and `end` at 8000 ms, with no immediate next `round:new`.
  The final snapshot has `started=false`, `inRound=false`, cleared world fields,
  and no avatars. The 3P match-end fixture proves `max_score: 2`: after avatars
  3 and 2 die and avatar 1 reaches score 2, source emits `round:end` with
  winner 1 at 3000 ms, then `game:stop` and `end` at 8000 ms, with no
  immediate next `round:new`. The focused 3P tie-at-max fixture proves tied
  leaders continue to next round. The focused 3P all-present multi-round fixture
  proves score carryover into a second round and match `end` only for
  `source_lifecycle_multi_round_match_end_3p`.
- Missing proof: broader 4P match-end behavior and replay/trainer policy for
  match-end episodes.
- Next file/test work: add another match lifecycle fixture only if it isolates
  one of those broader rules.
- Blocks training: `indirect` if training episodes are single-round; `yes` for
  source-faithful match episodes.

## 2. Bonuses

### 2.1 Forced Bonus Catch Order

- Current proof: JS oracle fixtures plus `CurvyTronSourceEnv` checks now verify
  a narrow active seeded `BonusSelfSmall` catch/no-catch/death-order slice:
  `source_bonus_self_small_catch_step.json` catches after movement and applies
  radius `0.3`; `source_bonus_self_small_tangent_no_catch_step.json` keeps the
  active bonus uncaught at strict-overlap tangent distance; and
  `source_bonus_self_small_wall_death_no_catch_step.json` keeps the bonus
  active when the overlapping avatar dies on the same tick.
- Missing proof: borderless, speed/radius/inverse/color effects,
  caps, broader stack math/expiry ordering, other
  bonus types, vector/runtime support, and broader death interactions.
- Next file/test work: keep the current three active fixtures as the first
  Python/source-env bonus proof, plus keep
  `source_bonus_spawn_type_position_rng_step.json` as the first natural
  JS/Python source-env spawn/type RNG proof, plus keep
  `source_bonus_self_small_expiry_restore_step.json` as the first timed
  expiry/restore proof, and `source_bonus_game_clear_immediate_step.json` as
  the first game-level clear proof, and
  `source_bonus_default_weights_type_rng_step.json` as the default multi-type
  weight/type proof. Add the next isolated source claim for caps, probability,
  or another effect.
- Blocks training: `indirect` if bonuses are disabled by rule; `yes` if bonuses
  are in the training target.

### 2.2 Bonus Spawn And RNG Labels

- Current proof: source maps list bonus timeout, type, and position RNG sites.
  `source_bonus_spawn_type_position_rng_step.json` now adds a JS/Python
  source-env minimal natural proof for one enabled type: `bonus.start_delay`,
  `bonus.next_delay_after_pop`, `bonus.type.BonusSelfSmall`,
  `bonus.position.x`, and `bonus.position.y`, producing one `BonusSelfSmall`
  at `(23.94, 64.06)` with `bonusCount=1` and `bonus:pop` before the
  zero-elapsed source position events. `source_bonus_spawn_game_world_retry_step`
  adds one rejected game-world candidate followed by one accepted retry.
- Missing proof: cap behavior, vector/runtime support, and row-local RNG state.
- Next file/test work: add a cap JS fixture, but keep it
  separate from catch/effect/expiry behavior.
- Blocks training: `indirect` for no-bonus training; `yes` for bonus training.

### 2.3 Bonus Stack Effects And Expiry

- Current proof: `source_bonus_self_small_expiry_restore_step.json` pins one
  timed stack expiry path for `BonusSelfSmall`: after catch, the `7500` ms
  timeout restores radius to `0.6`, emits `bonus:stack remove`, and does not
  emit a second `bonus:clear`. Source maps also describe speed, radius,
  inverse, invincible, printing, borderless, color, and broader expiry.
- Missing proof: one source-backed fixture per remaining effect, plus broader
  stack reset/math and expiry ordering.
- Next file/test work: create a small bonus batch such as
  `scenarios/environment/source_bonus_effects_batch.json` with first cases for
  speed turn-rate, radius collision, inverse turn, borderless expiry, and
  color/property events; cover it in `tests/test_source_bonus_runner.py`.
- Blocks training: `indirect` unless bonuses are enabled.

## 3. Reset And Autoreset

### 3.1 Production Masked Reset

- Current proof: `src/curvyzero/env/vector_reset.py` now provides a
  production-facing `reset_arrays(target, reset_template, reset_mask, *,
  reset_seed, reset_source, snapshot_array_names=None)` boundary. It validates
  required row arrays, snapshots selected terminal rows before mutation, copies
  selected rows from the reset template, increments selected `episode_id` from
  the pre-reset target, stamps `reset_seed`/`reset_source`, clears terminal
  flags/reasons/`tick`/`elapsed_ms`/event rows/`timer_fired_count` where
  present, leaves skipped rows unchanged, and returns reset metadata plus the
  terminal snapshot.
- Missing proof: natural spawn/reset-template creation, seed generation and
  row-local RNG history, lifecycle timer scheduling, autoreset loop, final
  observation policy, replay policy, and trainer API integration.
- Next file/test work: use `src/curvyzero/env/vector_reset.py` as the reset
  boundary; add natural reset/spawn templates, timer scheduling, autoreset,
  final-observation, and replay tests in the appropriate production test files.
- Blocks training: `yes`.

### 3.2 Row-Local Seed History

- Current proof: debug replay now explicitly says seed and episode fields are
  absent; the production-facing reset boundary can stamp caller-provided
  `reset_seed`, `reset_source`, and incremented `episode_id`.
- Missing proof: every row reset records seed, episode id, reset source, and RNG
  cursor/state so replay can reproduce the row.
- Next file/test work: add seed-history arrays to the production reset module;
  thread them into replay metadata tests in
  `tests/test_debug_actor_loop_replay.py` or the production replay tests once
  they exist.
- Blocks training: `yes`.

### 3.3 Public Autoreset Boundary

- Current proof: the actor bridge has debug-only internal autoreset after
  replay staging; comparator helpers can snapshot terminal rows before reset.
- Missing proof: final observation, reward, events, refs, and info are returned
  before row reset; next state has a new episode id and `reset_source=1`.
- Next file/test work: add public autoreset tests in `tests/test_vector_reset.py`
  and integration coverage in `tests/test_benchmark_vector_actor_loop_bridge.py`.
- Blocks training: `yes`.

### 3.4 Horizon And Overflow Truncation

- Current proof: terminal reason constants and row-lifecycle helper logic exist
  locally.
- Missing proof: production rows map horizon, event overflow, body overflow, and
  timer overflow to `truncated=true`, `done=true`, correct terminal reason, and
  zero pure-truncation reward.
- Next file/test work: add truncation cases to `tests/test_vector_reset.py`,
  `tests/test_compare_vector_arrays_to_fidelity.py`, and
  `tests/test_trainer_contract.py` for reward/info surfaces.
- Blocks training: `yes`.

## 4. Vector And Optimized Semantics

### 4.1 Row-Local RNG Arrays

- Current proof: one natural trail-gap scalar vector path uses row-local random
  tape arrays. The narrow `spawn_round_rows(...)` helper in
  `src/curvyzero/env/vector_spawn.py` uses row-local random tape/cursor/draw
  count for promoted first-round spawn facts, including the 2P heading retry.
  Lifecycle source fixtures still provide the source call labels.
- Missing proof: optimized/vector state has production RNG seed, cursor,
  exhaustion, call count, call-site labels, and per-row independence across
  reset-integrated spawn, PrintManager, next-round, and bonuses.
- Next file/test work: add RNG arrays to
  `scripts/compare_vector_arrays_to_fidelity.py` first; add focused tests in
  `tests/test_compare_vector_arrays_to_fidelity.py`; then move the shape into
  the production vector/reset module.
- Blocks training: `yes`.

### 4.2 Vector Lifecycle/Spawn From Source Fixtures

- Current proof: vector comparators cover a narrower fixture-seeded slice:
  movement, body, wall/border, selected PrintManager, forced trail gaps, and one
  separate natural trail-gap scalar path. The seeder now deliberately rejects
  natural `source_lifecycle_*` `Game.newRound()` fixtures as unsupported
  ordinary initial-state seeds and reports RNG contract metadata: call index,
  site, avatar, value, at-ms, expected call count, and capacity pressure.
  `src/curvyzero/env/vector_spawn.py::spawn_round_rows(...)` now matches the
  promoted first-round spawn facts for 2P heading retry, 3P reverse spawn
  order, 3P present/absent, and 4P reverse spawn order. It uses per-row map size, reverse player order,
  present masks, heading retry loops, and absent-player death-list arrays when
  present, and it does not insert or mutate world bodies.
- Missing proof: vector rows reproduce full lifecycle from the promoted
  fixtures. Optional 1v1 delayed-start metadata exists, but timer advancement,
  `game:start`, warmup/warmdown, PrintManager start, next-round, scoring,
  terminal handling, reset/autoreset integration, final obs, replay, and
  broader 3P/4P lifecycle fixtures are still absent.
- Next file/test work: integrate `spawn_round_rows(...)` with reset templates
  and lifecycle timer rows; keep lifecycle fixtures in the seeder
  rejection/metadata guard as ordinary seeds until real lifecycle rows exist;
  add lifecycle comparator cases to `tests/test_compare_vector_arrays_to_fidelity.py`.
- Blocks training: `yes`.

### 4.3 Event Rows And Overflow Policy

- Current proof: fixed event rows cover current supported event types and have
  overflow counters.
- Missing proof: lifecycle, bonus, clear, borderless, angle, and timer events
  have stable rows; overflow becomes a truncation surface instead of silent
  loss.
- Next file/test work: add event type rows only when the source fixture needs
  them; add an event-overflow forcing test in
  `tests/test_compare_vector_arrays_to_fidelity.py` and connect it to
  row-lifecycle truncation tests.
- Blocks training: `yes`.

### 4.4 B>1 Lifecycle And Mixed Player Counts

- Current proof: B>1 speed defaults cover only named supported fixture slices;
  delayed-start B>1 timer support is focused helper evidence.
- Missing proof: B>1 lifecycle rows, row-local timer/RNG independence, and a
  policy for mixed P batching or separate fixed-P workers.
- Next file/test work: add B>1 lifecycle cases to
  `tests/test_benchmark_vector_batch_rows.py`; document and test either padded
  mixed-P batches or fixed-P worker grouping before speed claims.
- Blocks training: `yes`.

## 5. Observation And Reward

### 5.1 Source-Backed Observation Fixtures

- Current proof: `curvyzero_egocentric_rays/v0` exists with analytic empty-arena
  and one distilled source movement canary in `tests/test_trainer_contract.py`.
- Missing proof: observations from trusted source states for trail gaps,
  borderless wrap, same-frame death, normal-wall terminal, and lifecycle spawn.
- Next file/test work: add manifests under
  `scenarios/environment/observation/`, starting with
  `obs_trail_gap_hole_safe_v0.json` or `obs_normal_wall_terminal_v0.json`;
  extend `tests/test_trainer_contract.py`.
- Blocks training: `yes`.

### 5.2 Validate Ray Internals Against Simulator State

- Current proof: `src/curvyzero/env/trainer_observation.py` reads current toy
  grid occupancy and player positions; tests cover shape, purity, range, masks,
  and simple perspective.
- Missing proof: ray channels match source-backed body/trail/radius semantics,
  not just toy grid assumptions.
- Next file/test work: extend `src/curvyzero/env/trainer_observation.py` only
  after trusted state fixtures exist; add tests for own trail, opponent trail,
  opponent head, wall, and borderless no-wall channels.
- Blocks training: `yes`.

### 5.3 Sparse Reward And Terminal Info Wiring

- Current proof: `curvyzero_sparse_round_outcome/v0` reward helper covers
  survivor, loser, draw, pure truncation, and terminal-plus-truncated precedence
  in focused tests.
- Missing proof: production step/replay rows emit the same reward, final
  observation, legal masks, terminal reason, winners/losers, refs, and
  `final_reward_map`.
- Next file/test work: wire the reward/info contract through the production
  transition builder and replay writer; extend `tests/test_trainer_contract.py`
  and replay tests.
- Blocks training: `yes`.

### 5.4 Legal Mask Batch Boundary

- Current proof: env mask `bool[3]` and LightZero mask `int8[3]` are pinned for
  live and terminal/dead rows in local helper tests.
- Missing proof: vector batch rows and the real wrapper preserve masks for live,
  dead, terminal, truncated, reset-pending, and padded policy rows.
- Next file/test work: add batch mask tests around policy row mapping and the
  actor bridge; reuse `src/curvyzero/training/policy_row_mapping.py`.
- Blocks training: `yes`.

## 6. Replay

### 6.1 Production Replay Writer/Reader

- Current proof: `src/curvyzero/training/debug_actor_loop_replay.py` writes and
  reads one debug `.npz` chunk, and it rejects missing debug absent policies.
- Missing proof: production shards store observation/reward/action rows with
  episode id, reset seed, reset source, done/terminated/truncated, schema/rules
  hashes, and compatibility rejection.
- Next file/test work: create a production replay module such as
  `src/curvyzero/training/replay_writer.py`; add
  `tests/test_replay_writer.py`; keep debug replay tests explicitly debug-only.
- Blocks training: `yes`.

### 6.2 Final Observation Versus Reset Observation

- Current proof: toy-v0 returns `final_observation`; actor bridge debug replay
  records no production final-observation policy.
- Missing proof: terminal transition rows keep final obs/reward/info while the
  next state may already be reset by autoreset.
- Next file/test work: add final/reset observation ordering tests in the
  production replay tests and actor bridge integration tests.
- Blocks training: `yes`.

### 6.3 Event, State, And Trace References

- Current proof: contracts list refs, but debug chunks do not store production
  refs or event ranges.
- Missing proof: replay rows carry event ranges and optional state/trace refs
  without placing raw source traces in policy observations.
- Next file/test work: add `event_ref`, `event_range`, `state_ref`,
  `trace_ref`, and `trace_hash` metadata to production replay chunks and reject
  mismatched rules/schema hashes in tests.
- Blocks training: `yes`.

### 6.4 Replay Manifest And Compaction

- Current proof: Modal/vector docs propose artifact layouts; replay writer does
  not own a finalized layout.
- Missing proof: shard manifests, complete markers, compaction rules, and
  compatibility checks across multiple chunks.
- Next file/test work: define local and Modal replay artifact paths in the
  production replay module; add manifest finalization tests.
- Blocks training: `yes`.

## 7. LightZero

### 7.1 Real CurvyZero LightZero Env

- Current proof: `src/curvyzero/training/curvyzero_lightzero_smoke.py` is a
  local no-train LightZero-shaped smoke around toy `CurvyTronEnv`.
  `src/curvyzero/training/curvyzero_lightzero_env.py` now adds the thin
  `curvyzero_v0_lightzero` DI-engine registration/timestep boundary while
  reusing the smoke semantics. `tests/test_curvyzero_lightzero_env.py` verifies
  that the registered wrapper matches the smoke wrapper for reset/step facts.
- Missing proof: installed LightZero/DI-engine config/import smoke, env-manager
  reset behavior, real package info retention, and any trainer/search/replay
  integration at that actual runtime boundary.
- Next file/test work: add a Modal or installed-runtime no-train config/import
  smoke pointed at `curvyzero_v0_lightzero`; keep
  `tests/test_curvyzero_lightzero_smoke.py` and
  `tests/test_curvyzero_lightzero_env.py` as local plumbing coverage.
- Blocks training: `yes` for LightZero training, after reset/replay/observation
  contracts exist.

### 7.2 Single-Action Policy Rows To Joint Actions

- Current proof: `src/curvyzero/training/policy_row_mapping.py` maps compact or
  padded ego rows to joint actions, and the actor bridge uses it around a fake
  policy/search stand-in.
- Missing proof: the LightZero adapter and real search boundary preserve env row
  ids, player ids, legal masks, padded rows, dead rows, and no-op actions.
- Next file/test work: reuse policy row mapping inside the real LightZero
  adapter; add adapter tests for 2P live rows, terminal rows, and padded rows.
- Blocks training: `yes`.

### 7.3 Real Search Boundary

- Current proof: Modal/JAX/Mctx and non-CurvyTron toy-game work are runtime or
  unrelated training plumbing; current CurvyZero actor bridge search is
  synthetic. Current scalar/ray single-ego rows are a practical bridge, while
  the intended CurvyTron training end state includes visual stacked-frame input
  from our own renderer shaped for LightZero.
- Missing proof: real observation rows, legal masks, root values, action
  weights, policy ids, visual frame provenance, and replay metadata pass
  through the search boundary. For multiplayer, replay must also retain the
  full wrapper action map, opponent policy ids, player ids, present/alive masks,
  death order, score vectors, and reset/RNG metadata alongside visual stacks.
- Next file/test work: replace the synthetic actor bridge stand-in only after
  production reset, observation/reward, and replay rows exist; test the local
  CPU loop before Modal/GPU claims.
- Blocks training: `yes`.

## 8. Wire And Pixels

### 8.1 Wire Event And Compression Fixture

- Current proof: socket protocol and compression are source-read; common trace
  is not wire replay.
- Missing proof: compressed position/angle payloads and one death+score payload
  match source socket output.
- Next file/test work: add
  `scenarios/environment/wire_event_single_tick.json`; extend
  `tools/reference_oracle/scenario_runner.js` or add a narrow wire oracle; add
  a focused wire test file.
- Blocks training: `no`.

### 8.2 Wire Replay Meaning

- Current proof: no source-authoritative wire replay format is pinned.
- Missing proof: whether replay means training replay rows or socket batch
  replay is explicitly separated.
- Next file/test work: write one small socket-batch record/replay spec after
  `wire_event_single_tick` lands; keep it separate from production training
  replay.
- Blocks training: `no`.

### 8.3 Browser Pixel Checks

- Current proof: browser/client rendering is deferred and is not gameplay
  authority.
- Missing proof: fixed viewport screenshots match after server state and wire
  traces pass.
- Next file/test work: add browser screenshot checks only after state/event and
  wire fixtures are stable; do not use pixels to prove gameplay rules.
- Blocks training: `no`.

## Near-Term Order

1. Keep the strict 1v1/no-bonus terminal trainer/replay transition as a
   guardrail, while moving source-backed multiplayer behavior into
   `VectorMultiplayerEnv`.
2. Keep the 3P survivor warmdown/next-round source fixture narrow; do not fold in match-end or broader present/absent variants.
3. Promote row-local RNG/reset history before bonus RNG or broad B>1 claims.
4. Add the first `BonusSelfSmall` catch only as a thin source-backed slice.
5. Extract runtime chunks behind `vector_runtime.step_many` one at a time:
   normal wall mask, body collision scan, body append, events, terminal scoring,
   then timers.
7. Wire final observation/reward/info into reset/autoreset and replay.
8. Only then promote LightZero plumbing from no-train smokes to a real training
   adapter claim.
