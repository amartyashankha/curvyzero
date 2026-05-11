# Reset, Timer, And Autoreset Implementation Plan

Status: implementation checklist for the next worker.
Date: 2026-05-09
Owner: RESET/VECTOR

This document turns the current reset/timer/autoreset gap into code work. Keep
the language plain and the claims narrow.

Short status:

- `src/curvyzero/env/vector_runtime.py::step_many` is the supported
  fixture-backed source-ordered CPU transition kernel.
- `scripts/benchmark_vector_batch_rows.py` routes normal calls through that
  public runtime. `_step_many_kernel(..., phase_timers=...)` is private
  benchmark diagnostics only, and the dead duplicate old benchmark body is
  gone.
- Terminal survivor/draw rows now mark optional `done`, `terminated`,
  `reset_pending`, `terminal_reason`, `draw`, and `winner` arrays when present.
- `vector_lifecycle.run_warmup_start_step_1v1_no_bonus_rows` composes strict
  1v1/no-bonus reset/spawn/warmup/timer/runtime stepping, and a focused test
  proves wall-death terminal state plus real vector trainer final-observation
  and reward handoff into autoreset planning.
- This is not trainer-ready. Urgent gaps remain: replay writer integration from
  vector batches, a public full env API, broad warmdown/next-round/3P/4P/bonus
  lifecycle coverage, the visual renderer, and performance integration.

Current truth:

- A production-facing masked reset boundary now exists in
  `src/curvyzero/env/vector_reset.py`. It is not a natural spawn implementation,
  not autoreset, and not a trainer API.
- A narrow vector spawn helper now exists in `src/curvyzero/env/vector_spawn.py`:
  `spawn_round_rows(state, row_mask, *, player_count)`. It matches promoted
  first-round spawn facts for 2P heading retry, 3P reverse spawn order, 3P
  present/absent, and 4P reverse spawn order. It is not full lifecycle and is
  not wired into autoreset.
- Narrow autoreset mutation now exists for the strict handoff:
  `apply_autoreset_rows(...)` stages final arrays through
  `plan_autoreset_rows(...)`, then resets selected rows through
  `vector_reset.reset_arrays(...)`. A public full env API is still not ready.
- The lifecycle/spawn/RNG claim has now moved past JS-only proof for the
  current pinned slice. `source-lifecycle` matches the original JS lifecycle
  oracle for 28 pinned lifecycle fixtures including
  `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p`: the three 2P core fixtures, focused 3P
  first-round spawn-order, focused 3P warmup/PrintManager start, focused 4P
  first-round spawn-order, focused 4P all-present all-dead warmdown/next-round,
  focused 4P survivor warmdown/next-round, focused 3P present/non-present
  first-round, focused 3P present/non-present survivor scoring,
  focused 3P present/non-present warmdown/next-round, focused 2P max-score
  match-end, one focused all-present 3P `max_score: 2` match-end path, focused
  3P tie-at-max continuation, focused 3P all-present multi-round match-end,
  focused 3P all-dead warmdown/next-round, focused 3P survivor-scoring
  `round:end`, and focused 3P survivor warmdown/next-round.
- That promoted slice is enough to specify reset seeds, row-local RNG cursors,
  warmup/warmdown timers, and print-start timer order for the first production
  reset shape.
- It is not enough to claim broad lifecycle support. Still missing:
  broader 4P match lifecycle, broader present/non-present variants, bonuses,
  production autoreset,
  replay final-observation policy, and optimized/vector parity for natural
  lifecycle rows.
- The vector seeder now rejects natural `source_lifecycle_*` `Game.newRound()`
  fixtures as unsupported ordinary initial-state seeds and reports RNG contract
  metadata: call index, site, avatar, value, at-ms, expected call count, and
  capacity pressure. This is useful reset/spawn input evidence, not production
  reset/spawn or optimized/vector lifecycle.
- Still open after the production-facing reset boundary, narrow spawn helper,
  and strict vector trainer handoff: reset-integrated spawn templates, seed
  generation and row-local RNG history, broad lifecycle timers, public env
  autoreset loop, replay writer integration, and visual/performance integration.
- Regression command output belongs next to the commands that produced it, not
  in this status block.

## Lifecycle Facts Now Available For Reset

These source facts can drive the next reset/RNG implementation:

- A new round clears the world, sets `started=true`, sets `in_round=true`, and
  keeps `world_active=false` until `game:start`.
- Natural spawn consumes one row-local random stream in reverse player order:
  `spawn.position_x`, `spawn.position_y`, then `spawn.angle_attempt_n`. The
  current promoted fixtures include accepted first attempts and one rejected
  attempt followed by an accepted retry. The focused 3P fixture proves only
  first-round avatar order 3, 2, 1 with `position_x`, `position_y`, and
  `angle_attempt_0` at 0 ms.
- The focused 3P present/non-present fixture proves only first-round
  `Game.onRoundNew()`: avatar 2 is non-present, consumes no spawn RNG, is added
  to deaths, and is snapshotted as `present=false`, `alive=false`, at
  `(0.6, 0.6)`, with `deathCount=1` and `deaths=[2]`.
- `spawn_round_rows(...)` now carries those first-round spawn facts into a
  narrow vector helper: row-local random tape/cursor/draw-count, per-row map
  size, reverse player order, present mask, heading retry loop, optional absent
  death-list arrays, and no world-body insertion.
- Spawn does not insert collision bodies. `world_body_count` stays `0` until
  the delayed PrintManager start emits important points while the world is
  active.
- `game:start` activates the world and schedules PrintManager start timers for
  alive avatars in reverse player order.
- The delayed PrintManager starts fire at `3000 ms`, emit important
  point/property events, insert bodies, and consume
  `print_manager.start_distance` RNG in reverse player order.
- A terminal round schedules warmdown. In the promoted 2P double-death case,
  `round:end` happens at `3000 ms`, `game:stop` happens at `8000 ms`, and the
  next `round:new` happens immediately at `8000 ms`.
- The focused 2P max-score match-end fixture proves only `max_score: 1`:
  avatar 2 dies, avatar 1 reaches score 1, source emits `round:end` winner 1
  at `3000 ms`, then `game:stop` and `end` at `8000 ms`, with no immediate
  next `round:new`. Its final snapshot is stopped and cleared with no avatars.
- The next round clears world bodies again and consumes the next row-local
  spawn RNG calls.
- The focused 3P all-dead continuation proves only forced same-frame wall
  deaths: `round:end` winner null at `3000 ms`, then `game:stop` and next
  `round:new` at `8000 ms`, followed by the next natural 3P spawn RNG/order.

## Already Implemented

These are real now:

- Toy-v0 refuses hidden post-terminal stepping until reset and includes
  `final_observation` in terminal info.
- `scripts/compare_vector_arrays_to_fidelity.py` has narrow timer arrays:
  - `timer_active[B,T] bool`
  - `timer_remaining_ms[B,T] float64`
  - `timer_kind[B,T] int16`
  - `timer_player[B,T] int16`
  - `timer_seq[B,T] int32`
  - `timer_overflow[B] bool`
- The comparator uses `T=4` through `DEFAULT_TIMER_CAPACITY`.
- Timer kind codes exist:
  - `TIMER_KIND_NONE = 0`
  - `TIMER_KIND_PRINT_MANAGER_START = 1`
- The delayed-start fixture schedules PrintManager start timers during
  `array_state_from_seed()`, through `_seed_print_manager_start_timers()`.
- `prepare_fixture_array_step()` carries a scalar `timer_advance_ms` for the
  fixture step.
- `_advance_pre_step_timers()` runs before movement in `step_prepared_arrays()`.
- Due timers fire in `timer_seq` order.
- The supported PrintManager start timer sets `print_manager_active`, sets
  `print_manager_last_pos`, may insert an important point/body, emits
  `property printing=true`, sets deterministic distance, and clears the timer
  slot.
- Per-step event arrays are already fixed width:
  - `event_count[B] int16`
  - `event_mask[B,L] bool`
  - `event_type[B,L] int16`
  - `event_player[B,L] int16`
  - `event_other[B,L] int16`
  - `event_bool[B,L] int8`
  - `event_value_i[B,L,2] int32`
  - `event_value_f[B,L,2] float64`
  - `event_overflow[B] bool`
  - `event_overflow_attempts[B] int32`
- Supported event type codes are:
  - `EVENT_POSITION = 1`
  - `EVENT_POINT = 2`
  - `EVENT_DIE = 3`
  - `EVENT_SCORE_ROUND = 4`
  - `EVENT_SCORE = 5`
  - `EVENT_ROUND_END = 6`
  - `EVENT_PROPERTY = 7`
- `reset_array_state(target, source)` can copy a prepared initial state into a
  working state for benchmark loops.
- `reset_array_rows(target, source, reset_mask)` can copy selected prepared
  vector rows into a working batch and has focused tests.
- `_bool_row_mask(...)` validates row-mask inputs for vector helpers.
- `final_transition_mask(done, truncated=None)` builds final-row masks for
  terminal or truncated rows. This is a helper, not production autoreset.
- `terminal_transition_snapshot(state, final_mask, array_names=...)` copies
  selected final transition rows before state reset. This is a comparator
  helper for preserving terminal rows, not a production replay contract.
- `reset_array_rows_with_info(...)` snapshots selected terminal rows, delegates
  row copying to `reset_array_rows(...)`, and returns reset metadata arrays:
  `reset_episode_id`, `reset_seed`, `reset_source`, `reset_mask`, and
  `reset_rows`, plus schema ids. It accepts explicit metadata arrays or reads
  post-reset `episode_id`/`reset_seed` defaults when those fields exist in the
  target. This is still a local helper, not production `reset_arrays`,
  `reset_many`, or public autoreset.
- Comparator-local `reset_arrays(...)` now wraps `reset_array_rows_with_info(...)`
  and proves the reset/autoreset mutation order for selected rows:
  - validates `episode_id`, `episode_step`, `env_active`, `reset_pending`,
    `done`, `terminated`, `truncated`, `terminal_reason`, `reset_seed`, and
    `reset_source` row arrays on target and source states;
  - requires an explicit reset seed and accepts scalar or full-row
    `reset_source`;
  - snapshots terminal rows before copying reset-template rows;
  - increments selected `episode_id` from the pre-reset target row;
  - stamps selected `reset_seed` and `reset_source`;
  - resets selected `episode_step`, `env_active`, `reset_pending`, `done`,
    `terminated`, `truncated`, and `terminal_reason`;
  - resets selected `tick` and optional `elapsed_ms` when present;
  - clears selected event rows and optional `timer_fired_count`;
  - returns the same reset/rules/state schema ids plus reset lifecycle arrays.
  This is still comparator-local proof code, not production `reset_many` or
  public autoreset.
- Comparator-local `reset_many(...)` now gives a narrow public-ish wrapper over
  `reset_arrays(...)`. Its focused test proves selected rows only, terminal
  snapshot before reset, `episode_id` increment, `reset_seed`/`reset_source`
  stamping, terminal flag clearing, event row clearing, skipped-row stability,
  and invalid mask rejection. This is still not a production trainer API.
- Terminal reason constants now exist:
  - `TERMINAL_REASON_NONE = 0`
  - `TERMINAL_REASON_SURVIVOR_WIN = 1`
  - `TERMINAL_REASON_ALL_DEAD_DRAW = 2`
  - `TERMINAL_REASON_TIMEOUT_TRUNCATED = 3`
  - `TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED = 4`
  - `TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED = 5`
- `row_lifecycle_arrays(...)` validates source terminal reason rows,
  post-step `episode_step`/`horizon_steps`, and optional event/body overflow
  masks, then returns `terminated`, `truncated`, `done`, and `terminal_reason`
  arrays. Source termination takes precedence over truncation; for non-source
  truncations, event overflow reason takes precedence over body overflow, then
  timeout. This is a helper, not production autoreset or public trainer API.
- `scripts/benchmark_vector_batch_rows.py` can stack supported one-step
  fixtures and includes `timer_advance_ms` in prepared batch metadata.
- `scripts/benchmark_vector_batch_rows.py` has a focused B>1 delayed-start
  pre-step timer helper with debug-event and no-event tests.
- `scripts/benchmark_vector_actor_loop_bridge.py` runs short fixture-reset
  rollout blocks and has debug-only internal autoreset after replay staging.
  This is not a public reset/autoreset contract.
- Separate old-body metadata source support exists through the `age_ms` fixture
  and runner handling.
- `src/curvyzero/env/vector_reset.py` now owns a production-facing
  `reset_arrays(target, reset_template, reset_mask, *, reset_seed,
  reset_source, snapshot_array_names=None)` boundary with local constants. It:
  - validates required row arrays;
  - snapshots selected terminal rows before mutation;
  - copies selected rows from the reset template;
  - increments selected `episode_id` from the pre-reset target row;
  - stamps selected `reset_seed` and `reset_source`;
  - clears selected terminal flags, terminal reasons, `tick`, `elapsed_ms`,
    event rows, and `timer_fired_count` when those arrays are present;
  - leaves skipped rows unchanged;
  - returns reset metadata and the terminal snapshot.
  This is production-facing reset only. It does not create natural spawns,
  generate seeds, maintain row-local RNG history, schedule lifecycle timers,
  or write replay rows by itself.
- `src/curvyzero/env/vector_spawn.py` owns the narrow
  `spawn_round_rows(state, row_mask, *, player_count)` helper. Tests in
  `tests/test_vector_spawn.py` match promoted first-round spawn facts for 2P
  heading retry, 3P reverse spawn order, 3P present/absent, and 4P reverse
  spawn order. The helper uses
  row-local random tape/cursor/draw-count, per-row map size, reverse player
  index order, present masks, heading retry loops, absent-player death-list
  arrays when present, and does not insert or mutate world bodies. It does not
  schedule timers, call `game:start`, run warmup/warmdown, start PrintManager,
  handle next-round, score, terminals, autoreset, final observation, or replay.
- `src/curvyzero/env/vector_lifecycle.py` now also has
  `reset_spawn_warmup_1v1_no_bonus_rows(...)`. It is the first stricter
  1v1/no-bonus warmup slice: selected rows are reset from a template, spawned
  with `spawn_round_rows`, stamped `started=true`, `in_round=true`,
  `world_active=false`, selected round-local world/body/print/death arrays are
  cleared, and exactly one `GAME_START` timer is scheduled.
- `src/curvyzero/env/vector_runtime.py` has the matching strict
  1v1/no-bonus warmup timer advancement helper. It can fire `game:start`,
  schedule reversed PrintManager start timers, fire those PrintManager starts,
  insert important body points, and consume row-local random tape. This is still
  only the supported 1v1/no-bonus warmup slice.
- `src/curvyzero/env/vector_runtime.py::step_many(...)` is the supported
  fixture-backed source-ordered CPU transition kernel. Benchmark wrappers route
  normal calls through it; private `_step_many_kernel(..., phase_timers=...)` is
  diagnostics-only.
- The vector runtime terminal path now marks optional `done`, `terminated`,
  `reset_pending`, `terminal_reason`, `draw`, and `winner` row arrays after
  survivor or draw terminal events when those arrays exist.
- `src/curvyzero/env/vector_lifecycle.py` now also has
  `run_warmup_start_step_1v1_no_bonus_rows(...)`, which composes strict
  reset/spawn/warmup timer advancement and one or more runtime steps. A focused
  test proves the first wall-death terminal path and real vector trainer
  final-observation/reward arrays feeding autoreset planning for that narrow
  slice.
- `src/curvyzero/env/vector_trainer_observation.py` now provides
  `observe_vector_1v1_egocentric_rays_v0(...)` and
  `build_final_trainer_transition_1v1_no_bonus_rows(...)`. It raycasts against
  vector body circles, not fake occupancy, and builds pinned `float32[106]`
  observations plus `float32[B,2,106]` final observation arrays.
- `src/curvyzero/env/vector_autoreset.py::apply_autoreset_rows(...)` now stages
  final observation/reward through `plan_autoreset_rows(...)`, then mutates
  selected rows through `vector_reset.reset_arrays(...)` while preserving
  terminal snapshot-before-reset ordering.

## Not Implemented Yet

These are the coding gaps:

- No full reset-integrated natural spawn/template builder for fresh source-like
  rows. A strict 1v1/no-bonus reset/spawn/warmup helper exists, and the runtime
  can advance that warmup through `game:start` and delayed PrintManager starts,
  but it does not cover broad lifecycle.
- No production `reset_many` or trainer API for source-like vector rows.
- No row-local seed history.
- No production row-local RNG generator/state policy. Fixture tape/cursor
  arrays exist for supported parity paths, but seed/state history is not a
  production reset contract yet.
- No seed generation policy. Callers must provide reset seeds.
- No production reset info schema wired into a public trainer API.
- No public full env autoreset API. The strict helper mutates selected rows
  after final arrays are staged, but it is not a trainer-facing environment.
- No replay-writer integration for the final vector trainer observation/reward
  arrays.
- No integrated horizon truncation policy in step/reset/autoreset state. The
  current row-lifecycle helper only computes surfaces from caller-provided
  arrays.
- No production terminal-info surface beyond the optional row arrays marked by
  the runtime.
- No production state-field wiring for `terminated`, `truncated`,
  `episode_step`, `horizon_steps`, or `terminal_reason` beyond standalone
  helpers, comparator-local reset helpers, and the runtime's narrow terminal
  marker.
- No production replay row contract that separates terminal transition refs
  from the next reset state. A local terminal-row snapshot helper now proves the
  separation pattern for selected arrays.
- No B>1 delayed-start timer default in speed smokes.
- No broad timer-kind support beyond the narrow `GAME_START` and
  `PRINT_MANAGER_START` warmup path.
- No broad lifecycle timer advancement. The strict 1v1/no-bonus warmup path can
  now advance through `game:start` and delayed PrintManager starts, but
  `round:end`, warmdown, next round, broad 3P/4P lifecycle, autoreset, final
  observation, and replay are still missing.

## Target API Split

Continue the boundary work in this order:

1. `reset_arrays(...)`
2. `advance_pre_step_timers(...)`
3. `step_arrays(...)`
4. `build_transition_surfaces(...)`
5. `autoreset_done_rows(...)`

The hot movement step starts after reset and pre-step timers. It should not
schedule start timers, choose new seeds, or overwrite terminal output.

Concrete target signatures can be local helper functions first, except
`reset_arrays(...)`, which now has a production-facing boundary in
`src/curvyzero/env/vector_reset.py`:

```text
reset_arrays(state, row_mask, reset_inputs) -> reset_info_arrays
advance_pre_step_timers(state, timer_advance_ms, event_state) -> counters
step_arrays(state, source_moves, event_state, event_mode) -> counters
build_transition_surfaces(prev_state, state, event_state) -> obs, reward, done, info
autoreset_done_rows(state, done, reset_inputs) -> reset_info_arrays
```

Do not move this straight into a public trainer API. Prove it locally with
tests first.

Current status for step 1: `src/curvyzero/env/vector_reset.py` has the
production-facing masked `reset_arrays(...)` boundary. This is useful reset
boundary code, but it does not generate new seeds, expose a production trainer
API, run public autoreset after transition building, schedule lifecycle timers,
or remove the dependency on natural episode lifecycle and row-local spawn/RNG.
`src/curvyzero/env/vector_lifecycle.py::reset_spawn_warmup_1v1_no_bonus_rows`
now schedules the first strict 1v1/no-bonus `GAME_START` warmup timer after
reset/spawn. `vector_runtime.advance_warmup_1v1_no_bonus_timers(...)` can now
fire it and the delayed PrintManager starts for that narrow slice. The runtime
step can now mark basic terminal lifecycle arrays for terminal rows when those
arrays exist. `run_warmup_start_step_1v1_no_bonus_rows(...)` composes those
pieces through a focused wall-death terminal vector trainer handoff test. The
next missing steps are warmdown/next-round behavior, replay writer integration,
a public full env API, broad bonuses, broad 3P/4P support, the visual renderer,
and performance integration.

## Required State Arrays

Keep these fields in the vector state. Some already exist in the comparator;
some need to be added to production/reset helpers.

Core row state:

| Field | Shape | Dtype | Required behavior |
| --- | ---: | --- | --- |
| `tick` | `[B]` | `int32` | Reset to `0`; increment only for active nonterminal rows after a returned step. |
| `elapsed_ms` | `[B]` | `float64` | Reset to `0`; add `step_ms` or controlled elapsed time after each captured step. |
| `done` | `[B]` | `bool` | True for rows whose returned transition ended the episode or round. |
| `terminated` | `[B]` | `bool` | True for source terminal rows, false for horizon truncation. |
| `truncated` | `[B]` | `bool` | True for horizon/cap/overflow truncation, false for normal source terminal. |
| `env_active` | `[B]` | `bool` | False rows are skipped until reset/autoreset. |
| `started` | `[B]` | `bool` | Source-like game-start flag. |
| `in_round` | `[B]` | `bool` | Round lifecycle flag. |
| `world_active` | `[B]` | `bool` | Point-to-body insertion is allowed only when true. |
| `reset_pending` | `[B]` | `bool` | Internal flag for rows selected for autoreset after output is built. |
| `episode_id` | `[B]` | `int64` | Increment on every reset of that row. |
| `episode_step` | `[B]` | `int32` | Reset to `0`; increment after each returned transition. |
| `horizon_steps` | `[B]` | `int32` | `0` means no horizon cap; otherwise truncate when reached. |
| `reset_seed` | `[B]` | `uint64` | Seed used for the current episode row. |
| `reset_source` | `[B]` | `int16` | `0 manual`, `1 autoreset`, `2 fixture`, `3 replay`. |
| `terminal_reason` | `[B]` | `int16` | `0 none`, `1 survivor_win`, `2 all_dead_draw`, `3 timeout_truncated`, `4 event_overflow_truncated`, `5 body_overflow_truncated`. |

Timer state:

| Field | Shape | Dtype | Required behavior |
| --- | ---: | --- | --- |
| `timer_active` | `[B,T]` | `bool` | Active timer slots. |
| `timer_remaining_ms` | `[B,T]` | `float64` | Subtract row `timer_advance_ms` before movement. |
| `timer_kind` | `[B,T]` | `int16` | `0 none`, `1 print_manager_start`; reserve more only when source fixtures need them. |
| `timer_player` | `[B,T]` | `int16` | Player index or `-1`. |
| `timer_seq` | `[B,T]` | `int32` | Stable fire order within a row. |
| `timer_overflow` | `[B]` | `bool` | Sticky overflow flag. |
| `timer_write_cursor` | `[B]` | `int16` | Next timer slot or append count. |
| `timer_fired_count` | `[B]` | `int16` | Debug counter for the current step. |

Row-local RNG state:

| Field | Shape | Dtype | Required behavior |
| --- | ---: | --- | --- |
| `rng_seed` | `[B]` | `uint64` | Base seed for the current episode row. |
| `rng_counter` | `[B]` | `uint64` | Number of source-style draws consumed in the row. |
| `rng_tape_value` | `[B,R]` optional | `float64` | Deterministic fixture tape for source parity checks. Production can replace this with a generator as long as the call log stays stable. |
| `rng_tape_length` | `[B]` optional | `int16` | Number of valid taped values. |
| `rng_exhausted` | `[B]` | `bool` | True if a fixture tape is exhausted; map to truncation or hard failure in debug mode. |
| `rng_call_count` | `[B]` | `int16` | Per-step debug call count. Reset to zero for each returned step. |
| `rng_call_site` | `[B,R]` optional | `int16` | Debug call-site code: spawn x/y, spawn angle attempt, print start, print stop, print toggle, bonus timeout/type/position. |
| `rng_call_player` | `[B,R]` optional | `int16` | Player for player-scoped random calls, or `-1`. |

Minimum RNG behavior from the promoted lifecycle slice:

- Reset/new-round consumes spawn calls in reverse player order.
- PrintManager start timers consume start-distance calls in reverse player order
  when their timers fire.
- Warmdown followed by next round continues the same row-local stream; it does
  not restart at random index zero.

Step inputs:

| Field | Shape | Dtype | Required behavior |
| --- | ---: | --- | --- |
| `source_moves` | `[B,P]` | `int8` | Internal left/straight/right source moves. |
| `step_ms` | `[B]` or scalar | `float64` | Movement elapsed time. |
| `timer_advance_ms` | `[B]` | `float64` | Controlled timer advancement before movement. Use `[0]` for rows with no timer advance. |
| `autoreset` | scalar or `[B]` | `bool` | If true, reset done rows after transition output is built. |

Reset info arrays:

| Field | Shape | Dtype | Required behavior |
| --- | ---: | --- | --- |
| `reset_episode_id` | `[B]` | `int64` | Episode id after reset. |
| `reset_seed` | `[B]` | `uint64` | Seed used for row. |
| `reset_source` | `[B]` | `int16` | Manual/autoreset/fixture/replay code. |
| `reset_episode_step` | `[B]` | `int32` | Post-reset episode step surface returned by local helper. |
| `reset_env_active` | `[B]` | `bool` | Post-reset active-row surface returned by local helper. |
| `reset_pending` | `[B]` | `bool` | Post-reset pending-reset surface returned by local helper. |
| `reset_done` | `[B]` | `bool` | Post-reset done surface returned by local helper. |
| `reset_terminated` | `[B]` | `bool` | Post-reset terminated surface returned by local helper. |
| `reset_truncated` | `[B]` | `bool` | Post-reset truncated surface returned by local helper. |
| `reset_terminal_reason` | `[B]` | `int16` | Post-reset terminal reason surface returned by local helper. |
| `reset_schema_id` | scalar/ref | string | Stable reset schema id. |
| `rules_schema_id` | scalar/ref | string | Current env rules schema id. |
| `state_schema_id` | scalar/ref | string | Current vector state schema id. |

## Reset Checklist

Code the reset path first.

- Add a helper that resets selected rows by mask, not the whole batch only.
- Clear per-step event arrays for reset rows.
- Clear per-step timer fired counters for reset rows.
- Clear `done`, `terminated`, `truncated`, `reset_pending`, and terminal reason
  for reset rows.
- Set `env_active=true`.
- Set `tick=0`, `episode_step=0`, and `elapsed_ms=0`.
- Increment `episode_id` for each reset row.
- Store `reset_seed` and `reset_source`.
- Rebuild player arrays: position, heading, alive, score fields if the reset
  target requires fresh episode score, body counters, trail state, draw cursor,
  PrintManager state, and materialized body buffer.
- Rebuild `started`, `in_round`, and `world_active` according to the chosen
  narrow target. For the first implementation, keep the existing fixture-like
  active world behavior unless a source fixture says otherwise.
- Schedule source start timers in `timer_*` arrays. For the first target, this
  means PrintManager start timers with `timer_remaining_ms=3000.0`,
  `timer_kind=1`, `timer_player=p`, and stable reverse-player `timer_seq`.
- If timer capacity overflows, set `timer_overflow=true`, `overflow=true`, and
  later map that row to truncation.

Minimum first test:

- Reset one row for the delayed-start PrintManager fixture.
- Assert timer arrays contain one `PRINT_MANAGER_START` timer per player in
  reverse-player sequence.
- Assert no event rows are emitted by reset itself unless the reset spec later
  explicitly says reset events are part of the returned transition.

Current local reset-helper status:

- Done in production-facing reset code: selected-row reset boundary, lifecycle
  validation, terminal snapshot before reset, template copy, `episode_id`
  increment from the pre-reset target, `reset_seed`/`reset_source` stamping,
  selected-row terminal/event/timer-fired clearing, skipped-row stability, and
  reset metadata return.
- Still open: natural spawn/reset-template creation beyond the strict 1v1
  helper, broad delayed-start and warmdown scheduling contracts, production
  `reset_many` or trainer API, seed generation/history, replay row integration,
  visual rendering, and performance integration.

## Pre-Step Timer Checklist

Then make pre-step timers a reusable production helper.

- Accept `timer_advance_ms[B]`, not one scalar for the whole batch.
- Reject negative or non-finite values.
- Skip inactive, done, or overflow rows.
- Subtract each row's advance from active timer slots.
- Find due timers where `timer_remaining_ms <= 0`.
- Fire due timers sorted by `timer_seq`.
- Emit timer side-effect events before movement events in the same event arrays.
- Clear fired timer slots after their side effects.
- Do not carry overshoot into PrintManager distance.
- Keep counters:
  - `pre_step_timer_advances`
  - `pre_step_timer_fires`
  - `print_manager_delayed_start_fires`
  - `print_manager_delayed_start_points`
  - `timer_overflow_attempts`

Minimum first tests:

- B=1 delayed-start trace:
  - step 0 uses `timer_advance_ms=2999`.
  - PrintManager remains inactive.
  - event rows contain only the zero-step position event.
  - step 1 uses `timer_advance_ms=1`.
  - timer fires before movement.
  - event order is point, property, position.
- B>1 delayed-start trace:
  - one row uses `2999`, another row uses `3000`.
  - only the due row fires.
  - event rows remain row-local.

## Autoreset Checklist

Autoreset must happen after output for the terminal transition is built.

Required order:

1. Snapshot previous observation refs if the replay/transition builder needs
   them.
2. Run pre-step timers.
3. Run movement/collision/PrintManager step.
4. Build final `obs`, `reward`, `terminated`, `truncated`, `done`, `info`, and
   event refs for the returned transition.
5. Mark rows needing reset with `reset_pending`.
6. If autoreset is enabled, reset those rows in state for the next call.
7. Return terminal transition data unchanged.

Tests must prove:

- A terminal row returns its final observation and terminal event rows.
- The state for the next call is reset.
- Terminal event arrays or refs from the returned transition are not overwritten
  by the autoreset.
- `episode_id` increments only after the terminal transition has been built.
- `reset_source=1` for autoreset rows.
- Nonterminal rows are not reset.

## Horizon And Overflow Truncation Checklist

Add this after basic autoreset behavior is source-shaped and protected by a
focused claim test.

- If `horizon_steps > 0` and `episode_step + 1 >= horizon_steps`, set:
  - `terminated=false`
  - `truncated=true`
  - `done=true`
  - `terminal_reason=3 timeout_truncated`
- If event overflow is fatal for training rows, set:
  - `terminated=false`
  - `truncated=true`
  - `done=true`
  - `terminal_reason=4 event_overflow_truncated`
- If body/timer capacity overflow is fatal for training rows, set:
  - `terminated=false`
  - `truncated=true`
  - `done=true`
  - `terminal_reason=5 body_overflow_truncated` or add a timer-specific code
    before using it.
- Reward for pure truncation should be zero until the reward spec says
  otherwise.

## Acceptance Commands

Run these after reset, timer, autoreset, vector, or actor bridge changes:

```sh
uv run pytest tests/test_compare_vector_arrays_to_fidelity.py -q
```

Record the exact output in the local run log. Do not use the number as a
fidelity claim.

```sh
uv run pytest tests/test_compare_vector_arrays_to_fidelity.py tests/test_benchmark_vector_batch_rows.py -q
```

Record the exact output in the local run log. Do not use the number as a
fidelity claim.

```sh
uv run pytest tests/test_compare_vector_arrays_to_fidelity.py tests/test_benchmark_vector_batch_rows.py tests/test_benchmark_vector_actor_loop_bridge.py -q
```

Record the exact output in the local run log. Do not use the number as a
fidelity claim.

```sh
uv run ruff check scripts/compare_vector_arrays_to_fidelity.py tests/test_compare_vector_arrays_to_fidelity.py
```

Record the exact output in the local run log. This is lint/static evidence only.

Run the broader focused source/env scenario suite used by the current-truth
docs after touching source runner or scenario code.

Record the exact output in the local run log.

```sh
uv run pytest -q
```

Run this as a broad regression check only when the change justifies it. Record
the exact output in the local run log.

```sh
uv run ruff check
```

For helper-only changes, touched-file ruff is usually enough:

```sh
uv run ruff check scripts/compare_vector_arrays_to_fidelity.py \
  tests/test_compare_vector_arrays_to_fidelity.py
```

Record the exact output in the local run log.

Run the mixed vector comparator:

```sh
uv run python scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_body_canary_batch.json \
  scenarios/environment/source_borderless_wrap_step.json \
  scenarios/environment/source_normal_wall_death_step.json \
  scenarios/environment/source_print_manager_batch.json \
  scenarios/environment/source_trail_gap_batch.json \
  --body-capacity 4 \
  --format plain
```

Record the exact mixed comparator output in the local run log.

Run both quick speed defaults after the implementation keeps the source-claim
preflight passing:

```sh
uv run python scripts/benchmark_vector_batch_rows.py \
  --batch-sizes 32 \
  --repeat 1 \
  --warmup 0 \
  --body-capacity 4 \
  --event-modes debug none \
  --format plain
```

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 16 \
  --rollout-steps 2 \
  --repeat 1 \
  --warmup 0 \
  --body-capacity 4 \
  --event-modes debug none \
  --format plain
```

Record the exact speed-default output in the local run log. Treat it as timing
and plumbing evidence only.

## Do Not Claim Yet

Do not claim any of these after only the first reset/timer/autoreset pass:

- production self-play is ready
- natural spawn or reset-template generation is ready
- production `reset_many` is ready
- public autoreset behavior has landed
- row-local seed history is ready
- lifecycle timer scheduling is ready
- horizon truncation is ready
- final trainer observation schema is ready
- final reward schema is ready
- replay writer/reader compatibility is ready
- broad round lifecycle is ready
- broad timer support is ready
- broad natural trail-gap support is ready
- GPU/JAX env step is ready

The first goal is smaller: a source-backed row reset, pre-step timer, terminal
transition, and autoreset boundary that another worker can build on.
