# Vector State Schema Draft

Status: planning draft plus narrow implemented boundaries for a later
vectorized or GPU-friendly backend.

Scope: fixed-shape state for source-like CurvyTron movement, trails, bodies,
trail gaps, print-manager state, and debug counters. This is not an
implementation request. Fidelity work still owns the source rules.

Reviewed inputs:

- `docs/research/environment/performance_vectorization_plan.md`
- `docs/experiments/environment/2026-05-09-vectorization-background-scout.md`
- `docs/working/environment/source_feature_inventory.md`
- Current `src/curvyzero/fidelity/source_runners.py` mechanics for
  `source-body-canary`, `source-trail-cadence-canary`, and
  `source-trail-gap-canary`.
- `scripts/benchmark_vectorization_prototype.py`, an isolated source-like NumPy
  scaffold for fixed-shape movement, body append, strict overlap scan, and
  own-body latency masks.
- Working notes for body/trail, trail gaps, and print-manager behavior.
- `docs/working/environment/reset_timer_autoreset_plan_2026-05-09.md`, the
  reset/pre-step timer/autoreset implementation checklist.

## Design Goal

The future state should be one structure-of-arrays object with fixed shapes per
run profile. The main shape is:

```text
B = environments in the batch
P = players per environment
K = maximum materialized world bodies per environment
C = optional spatial cells per environment
M = maximum body ids per spatial cell
L = maximum debug events per step
T = maximum pending timers per environment row
R = maximum source Math.random tape values for fixture-seeded rows
```

The hot transition is a backend/wrapper abstraction, not a native CurvyTron
entry point. Native source behavior is real-time player control state advanced
by elapsed-millisecond server frames. A backend may choose a fixed decision
cadence and expose arrays for training/replay, but that cadence is schema
policy, not source truth.

The hot transition should eventually look like:

```text
step_arrays(state, source_moves, rng_state) -> new_state, obs, reward, done, info_arrays
```

The public wrapper can still expose dicts. The hot state should not depend on
Python lists, per-agent objects, hidden global RNG, or variable output shapes.

Use `float64` while source parity is still being pinned. A later performance
backend can test `float32` only after movement, strict collision, trail cadence,
and print-manager fixtures still match.

## Source Order To Preserve

Within one tick, source-like order is the hard constraint:

1. Iterate players in reverse source player order.
2. For each live player, apply move, heading update, and position update.
3. Set the live body number from the player's current body counter.
4. If `printing` is true and the hidden draw cursor is empty or farther than
   the avatar radius, insert a normal point and a materialized body immediately.
5. Check border/wrap and body collision at the endpoint.
6. If the player dies, insert the death point/body, run any active
   PrintManager stop side effects, then emit die/score state.
7. If the player is still alive, update `PrintManager`. A toggle inserts an
   important boundary point/body. A print-to-hole toggle clears visible trail
   state but does not delete old world bodies.

This means batch over environments first. Keep a small reverse loop over
players until profiling proves another ordering is both correct and faster.
The current fixture-seeded comparator exercises this order through normal body
insertion, selected border/wall branches, death-point insertion, one-survivor
scoring, selected post-collision PrintManager toggles, and active PrintManager
death-stop side effects, plus the forced trail-gap body absence, stored-body,
and boundary canaries. It also has narrow delayed-start comparator support for
the promoted PrintManager timer fixture, and one separate scalar comparator
pass for the natural taped multi-step trail-gap crossing. Broader
reset/timer/autoreset, broader natural trail-gap variants, and broader
wall/wrap cases remain schema requirements, not prototype coverage.

## Semantics That Block Vectorization

Do not build a production vector backend until these semantics are pinned by
source fixtures and common traces:

- Source movement units: elapsed milliseconds, speed, angular velocity, and
  floating-point rounding need to be stable enough that strict body and wall
  threshold cases do not drift.
- Reverse player update order: higher-index players can insert same-frame bodies
  before lower-index players collide.
- Normal point cadence: first point draws when the draw cursor is empty; later
  points draw only when distance from the draw cursor is strictly greater than
  avatar radius.
- Visible trail state vs draw cursor: fixtures can force a draw cursor without a
  visible trail point, so one `last_trail_pos` field is not enough.
- Materialized body insertion timing: normal points insert bodies before
  collision, death points insert bodies on death, and print-manager boundary
  points insert bodies after collision for survivors.
- Body collision rule: strict circle overlap only; exact tangent is safe.
- Own-body latency: own bodies are ignored until
  `live_body_num - stored_body_num > trail_latency`; opponent bodies can collide
  immediately.
- Trail gaps: turning printing off clears visual/draw trail state but does not
  delete old materialized bodies.
- Print-manager behavior: active flag, distance subtraction, no overshoot
  carryover, random print/hole distances, post-collision toggle order, and
  death-stop side effects need to stay source-like.
- World lifecycle: point-to-body insertion depends on started/world-active
  state. Setup-time `start()`/`stop()` side effects must be either avoided by
  fixtures or modeled deliberately.
- Death and scoring order: same-tick deaths, terminal draw/winner behavior, and
  score/round-score events should stay source-visible.
- Pending mechanics: bonuses, full round lifecycle, replay/browser messages,
  final observation schema, reward contract, and wrapper reset/autoreset rules
  should not be guessed by a vector backend.

These are not reasons to avoid isolated microbenchmarks. They are reasons to
keep benchmarks source-labeled and prevent a fast backend from becoming the
semantic authority too early.

## Prototype Coverage Status

`scripts/benchmark_vectorization_prototype.py` was the first concrete consumer
of this schema. It is a script-level scaffold, not a backend. Newer narrow
boundaries now also exist in production-shaped modules:
`src/curvyzero/env/vector_reset.py`, `src/curvyzero/env/vector_spawn.py`,
`src/curvyzero/env/vector_lifecycle.py`, and
`src/curvyzero/env/vector_autoreset.py`. Those modules are still narrow helper
boundaries, not a full production vector environment.

Covered in the prototype:

- `B`, `P`, and `K` fixed shapes with structure-of-arrays state.
- Reverse player update order over a vectorized batch axis.
- Fixed-step movement from `source_move` values, where fixed step means a
  wrapper/profile cadence over elapsed-ms source movement.
- Separate `printing`, visible trail last, hidden draw cursor, and materialized
  body buffer arrays.
- Normal point cadence using empty cursor or strict distance `> radius`.
- Append-only normal body writes before collision scan.
- Strict circle overlap with exact-tangent safety.
- Own-body latency masks using `live_body_num - body_num <= trail_latency`.
- Opponent bodies as immediate collision candidates.
- Toy array observation construction and explicit body overflow counters.

Not covered in the prototype:

- Source fixture replay or common-trace equivalence.
- Reverse-order semantics past the collision scan, including death-point
  insertion, scoring, and event order.
- PrintManager death stop, delayed-start timer behavior, broader border/wrap
  interactions, and trail-gap semantics beyond the source-verified toggle
  controls in this isolated synthetic prototype. The fixture comparator covers a
  narrower source-backed subset.
- Normal wall death and broader borderless wrap branches beyond the one simple
  `source_borderless_wrap_step` fixture.
- Bonuses/randomness, round lifecycle, public wrappers, autoreset, and full
  browser/replay trail events.

Local smoke on macOS arm64 with `B=128`, `P=3`, `K=512`, `steps=200`,
`float64`, and straight movement reported about 64.7k synthetic env steps/s. The
collision scan dominated at 0.329231s of a 0.395907s timed loop. These are
source-like synthetic timings only; they are useful for array-shape and scan
scaling decisions, not for source-fidelity or trainer-throughput claims.

Modal CPU smoke now exists in
`src/curvyzero/infra/modal/environment_vector_bench.py`. It runs whole benchmark
scripts in one remote CPU job and writes coarse artifacts to
`curvyzero-runs/environment/vectorization/<run_id>/`. It is for repeatable
speed-lane artifacts, not per-step remote execution.

## Fixture-Seeded Array Step Status

`scripts/compare_vector_arrays_to_fidelity.py` is the first real step from
fixture seeds into array stepping. It is still script-only and narrow.

Real now:

- Reads arrays from `scripts/seed_vector_state_from_fixtures.py`.
- Runs one NumPy tick in reverse player order.
- Covers all six three-player `source-body-canary` batch fixtures. Four are
  marked `python-runner-verified`; the two opponent-body fixtures are checked
  through the same Python source runner during comparison.
- Covers the simple two-player `source_borderless_wrap_step` fixture against
  the Python `source-borderless-wrap` runner.
- Covers the two-player `source_normal_wall_death_step` fixture against the
  Python `source-normal-wall` runner.
- Covers the one-player `source_print_manager_no_toggle_control_step`,
  `source_print_manager_print_to_hole_step`,
  `source_print_manager_hole_to_print_step`, and
  `source_print_manager_exact_zero_toggle_step` fixtures against the Python
  `source-print-manager-canary` runner. This slice updates active PrintManager
  distance and last position, emits the no-toggle position-only event stream,
  emits the important boundary point plus `property printing=true/false` for
  toggles, handles the exact-zero threshold, and clears visible trail/draw
  cursor state for print-to-hole.
- Covers `source_print_manager_delayed_start_timer_step` against the Python
  `source-print-manager-canary` runner. This is a narrow comparator path: it
  models the fixture's pre-step timer/start side effects and zero-step captures,
  but it is not broad reset/timer/autoreset support.
- Covers the three-player `source_print_manager_active_stop_on_death_step`,
  `source_print_manager_active_hole_stop_on_death_step`, and
  `source_print_manager_body_collision_stop_on_death_step` fixtures against the
  Python `source-print-manager-canary` runner. This slice keeps the source row
  order: death point, optional important PrintManager stop point/property, die,
  then `score:round`.
- Covers the three-player `source_trail_gap_hole_space_safe_step`,
  `source_trail_gap_stored_body_still_kills_step`,
  `source_trail_gap_print_to_hole_boundary_kills_step`, and
  `source_trail_gap_hole_to_print_boundary_kills_step` fixtures against the
  Python `source-trail-gap-canary` runner. This slice keeps hole-space free of
  normal point bodies, preserves stored world bodies inside visual gaps, and
  emits the print-to-hole or hole-to-print boundary point/property before a
  later player can hit that new body.
- Covers the separate three-player
  `source_trail_gap_natural_multistep_hole_crossing` full trace against the
  Python `source-trail-gap-canary` runner. This scalar-only comparator path
  uses fixture-seeded source `Math.random` tape arrays to reproduce the
  PrintManager distances that create the natural hole crossing; it is not in
  speed defaults.
- Updates source-move kinematics, opponent seeded-body checks, normal point body
  insertion, strict body overlap, own-body latency, nonterminal body-hit death
  state, death point body insertion, trail count/last point, body counters, and
  world body count.
  For the borderless fixture, it updates movement and applies the source wrap
  branch after movement. For the normal-wall fixture, it updates wall death,
  death point insertion, and the one-survivor score field. For the PrintManager
  fixtures, it updates no-toggle bookkeeping, the source-pinned
  print-to-hole/hole-to-print/exact-zero boundary toggle after collision
  checks, active death-stop side effects, and the narrow delayed-start timer
  fixture.
- Emits fixed-width event rows with `L=16` for the supported source-visible
  event types: `position`, `point`, `die`, `score:round`, `score`, and
  `round:end`, plus the narrow PrintManager `property` event. The rows use
  simple numeric code arrays plus count, mask, and overflow arrays.
- Projects the array result back to common-trace state and event fields and
  compares it against the matching Python source runner.
- Exposes the prepared in-place array transition so repeated timing can exclude
  source runners, common-trace projection, and comparison from the hot loop.

Current local comparator command:

```sh
python3 scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_body_canary_batch.json \
  scenarios/environment/source_borderless_wrap_step.json \
  scenarios/environment/source_normal_wall_death_step.json \
  scenarios/environment/source_print_manager_batch.json \
  scenarios/environment/source_trail_gap_batch.json \
  --body-capacity 4 \
  --format plain
```

This compares state fields plus event rows for the supported body, simple
borderless wrap, normal-wall death, deterministic PrintManager, and forced
trail-gap fixture set. Treat the named source claims and diff output as the
evidence; do not quote pass totals as proof.

Separate natural trail-gap scalar proof:

```sh
python3 scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_trail_gap_natural_multistep_hole_crossing.json \
  --body-capacity 4 \
  --format plain
```

This is intentionally separate from the default mixed comparator and from speed
defaults. Use it as a named natural trail-gap check, not as a pass-count
dashboard.

Full PrintManager batch smoke:

```sh
python3 scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_print_manager_batch.json \
  --body-capacity 4 \
  --format plain
```

Use this as the focused PrintManager batch check. Its value is the source claim
coverage and diff output, not the number of passing rows.

Delayed-start comparator boundary:

The comparator now supports `source_print_manager_delayed_start_timer_step` by
modeling only the fixture's setup, timer advance, start side effects, and
captured zero-step rows. This is not broad reset/timer/autoreset support.

Broader reset/timer contract for delayed-start rows:

The concrete implementation checklist now lives in
`docs/working/environment/reset_timer_autoreset_plan_2026-05-09.md`. The summary
below is the schema-level boundary.

- Reset/init owns source setup side effects. It fills the row state, clears
  per-step event buffers, initializes RNG and counters, and schedules any
  source `Game.onStart` timers. It must not hide timer setup inside the
  movement/collision step.
- Timers are row-local fixed arrays. Before each captured environment step,
  advance pending timers by the row's controlled `timer_advance_ms`; fire due
  timers in stable source order; append any timer side-effect events before
  movement events.
- Pre-step event rows share the same fixed `[B,L]` debug event arrays as normal
  step events. Timer-created point/property rows are appended first, then
  `step_arrays` appends position, collision, score, and post-collision
  PrintManager rows.
- The hot `step_arrays(state, source_moves, rng_state)` contract starts after
  reset/autoreset and pre-step timer advancement for the row. It applies one
  movement/collision/PrintManager-test tick. It does not schedule initial
  `Game.onStart` timers or decide autoreset policy.
- Autoreset is a row-boundary operation between returned trainer transitions.
  Terminal events, rewards, and final observations from a step are returned
  before that row is reset for a later step. Autoreset creates a fresh row state
  and schedules fresh start timers; it must not overwrite the terminal step's
  event rows.
- `source_print_manager_delayed_start_timer_step` needs exactly this boundary:
  reset schedules `PrintManager.start` for `3000 ms`; the first zero-ms
  captured tick after `2999 ms` leaves the manager inactive and emits only the
  zero-step position event; the next tick advances `1 ms`, fires the timer
  before movement, inserts the important start point/body, emits
  `property printing=true`, sets `pm_active=true`, sets `pm_last_pos` to the
  current avatar position, draws the deterministic `39` print distance, then
  emits the zero-step position event. Do not make this pass by starting inside
  post-collision `PrintManager.test()` or by comparing only the second tick.

Batch-row note: `scripts/benchmark_vector_batch_rows.py` now uses the widened
fixture default and includes the broadened `B>1` event preflight before timing.
After the earlier batched event-writer change, the local `--repeat 3000` run at
`B=128` reported about `234k` rows/sec for `P=2,K=4` and `394k` rows/sec for
`P=3,K=4`; fixed event emission and body-hit owner lookup remained visible
costs.

Repeated-step timing uses an older default subset without running source trace
comparison inside the hot loop. Check the script default before quoting a
fixture count:

```sh
python3 scripts/benchmark_vector_array_steps.py \
  --repeat 10000 \
  --warmup 500 \
  --body-capacity 4 \
  --format plain
```

Older local result on macOS arm64, Python 3.11.13, NumPy 2.4.4:

Note: this timing table was recorded before fixed event rows were added to the
array step. Rerun timing before quoting current step/sec.

| Bucket | Value |
| --- | ---: |
| Timed supported transitions | 80,000 |
| Setup | 0.001878s |
| Preflight source trace | 0.001287s |
| Preflight env step | 0.001016s |
| Preflight projection | 0.000053s |
| Preflight comparison | 0.000129s |
| Timed reset-copy bucket | 0.610331s |
| Timed env-step bucket | 6.726476s |
| Env steps/sec, step bucket only | 11,893.3 |
| Env steps/sec, reset-copy plus step | 10,903.9 |
| Hot-loop source/projection/comparison calls | 0 |

This timing is deliberately modest: it repeats B=1 fixture-backed one-step
transitions from prepared arrays. It is not a stacked production batch, not a
policy/search loop, and not a broad env throughput claim.

Still fake or unsupported:

- Natural `Game.newRound()` lifecycle fixtures are deliberately rejected by
  `scripts/seed_vector_state_from_fixtures.py` as unsupported ordinary
  initial-state seeds. That is a guard against fake optimized/vector lifecycle
  claims, not an implementation of lifecycle.
- The lifecycle rejection report carries RNG contract metadata needed for the
  future reset/spawn implementation: call index, site, avatar, value, at-ms,
  expected call count, and capacity pressure. All 20 promoted lifecycle
  fixtures through focused 4P survivor next-round,
  present/absent survivor scoring, and focused 3P multi-round match-end are
  included in this same rejection/metadata guard, including the
  three 2P core fixtures, focused 3P spawn-order fixture, focused 3P
  warmup/PrintManager-start fixture, focused 4P first-round spawn-order
  fixture, focused 4P all-present all-dead next-round fixture, focused 4P
  survivor next-round fixture, focused 3P
  present/non-present first-round and next-round fixtures, focused 2P max-score
  match-end fixture, one focused all-present 3P
  `max_score: 2` match-end path, focused 3P tie-at-max continuation, focused
  3P all-present multi-round match-end, focused
  3P all-dead next-round fixture,
  focused 3P survivor-scoring `round:end` fixture, and focused 3P survivor
  warmdown/next-round fixture.
- `src/curvyzero/env/vector_spawn.py` now has the narrow
  `spawn_round_rows(state, row_mask, *, player_count)` helper for promoted
  first-round spawn facts, including 4P reverse spawn order. It uses row-local
  random tape/cursor/draw-count, per-row map size, reverse player order,
  present masks, heading retry loops, absent-player death-list arrays when
  present, and no world-body insertion.
- `src/curvyzero/env/vector_lifecycle.py` composes reset plus source-shaped
  spawn for selected rows and can stamp narrow 1v1 delayed-start timer metadata
  when the optional arrays exist. It still reports `full_lifecycle=false`.
- `src/curvyzero/env/vector_autoreset.py` has a pure
  `plan_autoreset_rows(...)` helper that validates terminal masks and copies
  final observation/reward/reset metadata for selected rows. It does not mutate
  state or run `reset_arrays(...)`.
- Optimized/vector lifecycle, reset-integrated spawn, round timers,
  `game:start`, warmup/warmdown, PrintManager start, next-round, scoring,
  terminal handling, autoreset integration, final observation, and replay are
  still missing.
- The broad `benchmark_vectorization_prototype.py` speed numbers are still
  synthetic.
- Event support is narrow: `angle` has a reserved code but is not emitted by
  this comparator slice, and `property` is emitted only for the supported
  PrintManager toggle, death-stop, and trail-gap boundary slices. Unsupported
  event detail should stay recorded instead of guessed.
- Normal-wall same-frame draws, broader natural trail-gap variants beyond the
  one scalar taped crossing, borderless
  body-skip and PrintManager-wrap variants, terminal scoring beyond the
  one-survivor wall-death fixture, round lifecycle, bonuses, observations,
  rewards, broad reset/autoreset, policy calls, and MCTS arrays are not part of
  this gate.
- The repeated-step timing table above is `B=1` fixture comparison and B=1
  fixture-step timing. The batch-row benchmark is stacked, but still not a
  production self-play batch.
- The next blockers are turning the narrow delayed-start comparator path into a
  broader reset/timer/autoreset contract, deciding if/when the separate natural
  trail-gap scalar case should enter speed defaults, and covering broader
  wall/wrap semantics. Until those are implemented and checked, the speed lane
  still cannot claim source-visible production semantics.

## Core Environment Fields

These fields are per environment unless noted otherwise.

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `tick` | `[B]` | `int32` | Step count within the current episode or round. |
| `elapsed_ms` | `[B]` | `float64` | Optional accumulated source time for debug/profiling. |
| `step_ms` | `[B]` or scalar | `float64` | Wrapper/profile elapsed-ms frame size; source movement uses elapsed milliseconds. |
| `done` | `[B]` | `bool` | Public episode/round terminal mask. |
| `terminated` | `[B]` | `bool` | Source terminal mask for returned trainer transitions. |
| `truncated` | `[B]` | `bool` | Max-step or profile cap mask. |
| `env_active` | `[B]` | `bool` | False rows are masked until reset/autoreset. |
| `started` | `[B]` | `bool` | Source-like game started flag. |
| `in_round` | `[B]` | `bool` | Round lifecycle flag. |
| `world_active` | `[B]` | `bool` | Bodies are inserted only when world is active. |
| `borderless` | `[B]` | `bool` | Chooses wall death vs wrap. |
| `overflow` | `[B]` | `bool` | Sticky row-level state overflow flag. |
| `overflow_code` | `[B]` | `int16` | Which fixed buffer overflowed first. |
| `reset_pending` | `[B]` | `bool` | Row should be reset after returned transition data is built. |
| `episode_id` | `[B]` | `int64` | Incremented on every row reset. |
| `episode_step` | `[B]` | `int32` | Step count within the current row episode. |
| `horizon_steps` | `[B]` | `int32` | `0` means no horizon cap; otherwise drives truncation. |
| `reset_seed` | `[B]` | `uint64` | Seed used to initialize the current row episode. |
| `reset_source` | `[B]` | `int16` | `0 manual`, `1 autoreset`, `2 fixture`, `3 replay`. |
| `terminal_reason` | `[B]` | `int16` | `0 none`, `1 survivor_win`, `2 all_dead_draw`, `3 timeout_truncated`, `4 event_overflow_truncated`, `5 body_overflow_truncated`. |

`started`, `in_round`, and `world_active` can be constants for early
benchmarks, but they should exist in the schema. Source point events insert
bodies only when the game has started and the world is active.

## Reset And Timer Fields

The current comparator has a narrow delayed-start timer path. Production reset
support should use the same array names where possible, but move them behind a
real reset/pre-step/autoreset boundary.

Timer arrays:

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `timer_active` | `[B,T]` | `bool` | Active row-local timer slots. |
| `timer_remaining_ms` | `[B,T]` | `float64` | Decremented by `timer_advance_ms[B]` before movement. |
| `timer_kind` | `[B,T]` | `int16` | `0 none`, `1 print_manager_start`; add more only from source fixtures. |
| `timer_player` | `[B,T]` | `int16` | Player index for player-owned timers, or `-1`. |
| `timer_seq` | `[B,T]` | `int32` | Stable fire order within one row. |
| `timer_overflow` | `[B]` | `bool` | Sticky timer capacity overflow flag. |
| `timer_write_cursor` | `[B]` | `int16` | Next timer slot or append count for reset scheduling. |
| `timer_fired_count` | `[B]` | `int16` | Debug count for the current captured step. |

Step input arrays:

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `source_moves` | `[B,P]` | `int8` | Internal control-state move ids after wrapper action conversion. |
| `timer_advance_ms` | `[B]` | `float64` | Controlled timer advancement before movement. |
| `autoreset` | scalar or `[B]` | `bool` | Reset done rows after terminal transition output is built. |

Reset info arrays:

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `reset_episode_id` | `[B]` | `int64` | Episode id after reset. |
| `reset_seed` | `[B]` | `uint64` | Seed used by reset. |
| `reset_source` | `[B]` | `int16` | Manual/autoreset/fixture/replay code. |
| `reset_schema_id` | scalar/ref | string | Reset contract id. |
| `rules_schema_id` | scalar/ref | string | Rules contract id. |
| `state_schema_id` | scalar/ref | string | Vector state contract id. |

Required ordering:

1. `src/curvyzero/env/vector_reset.py::reset_arrays` is now the
   production-facing masked reset boundary. It snapshots selected terminal
   rows, copies selected rows from the reset template, increments selected
   `episode_id` from the pre-reset target, stamps `reset_seed`/`reset_source`,
   clears terminal/event/timer-fired fields where present, and leaves skipped
   rows untouched. Comparator tape metadata follows the same template-copy
   rule: reset rows get template cursor/exhaustion/draw-count values, and
   skipped rows stay untouched.
2. `advance_pre_step_timers` subtracts `timer_advance_ms[B]`, fires due timers
   in `timer_seq` order, and appends timer side-effect events before movement.
3. `step_arrays` runs movement, body insertion, collision, scoring, and
   post-collision PrintManager work.
4. Transition surfaces are built from the final step state and event rows.
5. Autoreset resets done rows only after final observations, rewards, done
   flags, info, and event refs for the returned transition are preserved.

The delayed-start fixture is the first guardrail for the future timer split:
reset/timer integration must schedule the `3000 ms` PrintManager start timer,
`2999 ms` must not fire it, and the next `1 ms` must fire it before the
zero-step position event. The new reset boundary does not schedule lifecycle
timers yet.

## Player Fields

These are per environment and player.

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `player_id_num` | `[P]` or `[B,P]` | `int16` | Stable source order, not a dynamic list. |
| `update_order` | `[P]` | `int16` | Usually `[P-1, ..., 0]`; fixed profile field. |
| `pos` | `[B,P,2]` | `float64` | Current endpoint position. |
| `prev_pos` | `[B,P,2]` | `float64` | Optional scratch for swept/debug benchmarks. |
| `heading` | `[B,P]` | `float64` | Source angle in radians. |
| `source_move` | `[B,P]` | `int8` | Internal move id, usually left/straight/right as `-1/0/1`. |
| `alive` | `[B,P]` | `bool` | Player can update only when true. |
| `death_tick` | `[B,P]` | `int32` | `-1` while alive. |
| `radius` | `[B,P]` or scalar | `float64` | Default source avatar radius is `0.6`. |
| `speed` | `[B,P]` or scalar | `float64` | Default source speed is `16` units/s. |
| `angular_velocity` | `[B,P]` or scalar | `float64` | Source angular velocity per ms. |
| `trail_latency` | `[B,P]` or scalar | `int16` | Source own-body latency is point-number based. |
| `score` | `[B,P]` | `int32` | Total score. |
| `round_score` | `[B,P]` | `int32` | Current round score. |
| `killer` | `[B,P]` | `int16` | Last killer player, `-1` for none/wall. |

Public trainer action ids should be converted to `source_move` at the wrapper
boundary. The hot transition should not parse agent strings or action dicts.

## Trail And Print State

The source has at least three related concepts. Do not collapse them:

- `printing`: whether normal per-radius points may be emitted.
- Visible trail state: what the trail point list currently contains.
- Draw cursor state: the hidden `lastX/lastY` used by `isTimeToDraw()`.

The first cadence fixtures already need visible state and draw cursor state to
be independent.

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `printing` | `[B,P]` | `bool` | Controls normal point insertion. |
| `visible_trail_count` | `[B,P]` | `int32` | Count of visible trail points in the current visual segment. |
| `has_visible_trail_last` | `[B,P]` | `bool` | Whether `visible_trail_last_pos` is meaningful. |
| `visible_trail_last_pos` | `[B,P,2]` | `float64` | Last visible point, matching common-trace `lastTrailPoint`. |
| `has_draw_cursor` | `[B,P]` | `bool` | Source `trail.lastX !== null`. |
| `draw_cursor_pos` | `[B,P,2]` | `float64` | Cursor used by `isTimeToDraw()`. |
| `trail_clear_count` | `[B,P]` | `int32` | Debug counter for print-to-hole clears. |
| `normal_point_count` | `[B,P]` | `int32` | Debug counter for non-important points. |
| `important_point_count` | `[B,P]` | `int32` | Debug counter for print-manager/death boundary points. |

Normal insertion mask:

```text
should_draw = alive
  & printing
  & (~has_draw_cursor
     | distance(pos, draw_cursor_pos) > radius)
```

When a normal point is inserted, update visible trail state, draw cursor state,
player body counters, and the materialized body buffer in the same player-order
step.

When a print-to-hole toggle happens, insert one important body point, then clear
visible trail state and the draw cursor. Do not clear materialized bodies.

## Materialized Body Buffer

The collision body state should be a fixed append-only buffer per environment
for the first vector backend. A spatial index can be added after the semantics
are pinned.

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `body_active` | `[B,K]` | `bool` | Slot is valid. |
| `body_pos` | `[B,K,2]` | `float64` | Body center. |
| `body_radius` | `[B,K]` | `float64` | Usually avatar radius. |
| `body_owner` | `[B,K]` | `int16` | Owner player index. |
| `body_num` | `[B,K]` | `int32` | Source `AvatarBody.num`. |
| `body_insert_tick` | `[B,K]` | `int32` | Debug/profiling only. |
| `body_insert_kind` | `[B,K]` | `int8` | `0 normal`, `1 important`, `2 death`, `3 seeded`. |
| `body_write_cursor` | `[B]` | `int32` | Next free slot. |
| `world_body_count` | `[B]` | `int32` | Monotonic source-style insert count. |
| `body_overflow` | `[B]` | `bool` | Sticky overflow flag for `K`. |

Per-player body counters:

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `body_count` | `[B,P]` | `int32` | Next body number to assign on point insertion. |
| `live_body_num` | `[B,P]` | `int32` | Current live body number used during collision checks. |

For each player update, set `live_body_num[:, p] = body_count[:, p]` before
normal point insertion and collision. Inserted bodies use the current
`body_count[:, p]`, then increment `body_count[:, p]`.

Collision candidate mask for player `p`:

```text
active_body = body_active
own_body = body_owner == p
own_too_young = own_body & (live_body_num[:, p] - body_num <= trail_latency[:, p])
candidate = active_body & ~own_too_young
hit = distance(pos[:, p], body_pos) < radius[:, p] + body_radius
```

The strict `<` matters. Exact tangent should stay safe.

## Optional Spatial Index

Start with full-buffer scans for correctness and simple benchmarks. Add a fixed
spatial index only when scans are measured as a bottleneck.

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `cell_body_ids` | `[B,C,M]` | `int32` | Body slot ids by spatial cell. |
| `cell_body_count` | `[B,C]` | `int16` | Valid ids per cell. |
| `cell_overflow` | `[B,C]` | `bool` | Sticky cell capacity overflow. |
| `body_cell_minmax` | `[B,K,4]` | `int16` | Optional debug of inserted bounds. |

For source parity, a spatial index is only an acceleration structure. The source
collision rule is still circle overlap plus owner/latency filtering.

## Print Manager Fields

Print-manager state is per player. It updates after collision and only for
survivors.

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `pm_active` | `[B,P]` | `bool` | Inactive managers do nothing. |
| `pm_distance` | `[B,P]` | `float64` | Remaining distance until toggle. |
| `pm_last_pos` | `[B,P,2]` | `float64` | Previous manager sample point. |
| `pm_toggle_count` | `[B,P]` | `int32` | Debug counter. |
| `pm_last_toggle_tick` | `[B,P]` | `int32` | Debug counter. |

Update:

1. Subtract distance from `pm_last_pos` to current `pos`.
2. Set `pm_last_pos` to current `pos`.
3. If `pm_distance <= 0`, toggle `printing`.
4. Insert one important body point.
5. If toggled to hole, clear visible trail state and draw cursor.
6. Draw the next print/hole distance from the explicit RNG state.

There is no overshoot carryover in the source behavior.

## RNG State

Use explicit per-environment RNG arrays, not global RNG.

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `rng_key` | `[B,2]` or `[B,4]` | `uint32`/`uint64` | Backend-specific key words. |
| `rng_counter` | `[B]` | `uint64` | Optional draw counter for audit. |
| `rng_stream_id` | `[B]` | `uint32` | Distinguish reset, print gaps, bonuses, curricula. |
| `random_tape_values` | `[B,R]` | `float64` | Fixture-seeded source `Math.random` values for comparator rows. |
| `random_tape_length` | `[B]` | `int32` | Valid tape entries for each row; `0` keeps the deterministic `0.5` default. |
| `random_tape_cursor` | `[B]` | `int32` | Next tape entry consumed by source-order PrintManager distance draws. |
| `random_tape_exhausted` | `[B]` | `bool` | Debug flag for fixture tape exhaustion. |
| `random_tape_draw_count` | `[B]` | `int32` | Per-row draw-call count for fixture tape/default-distance draws. |

For early microbenchmarks, deterministic `0.5` random values can be a profile
mode. The current fixture comparator consumes non-empty source tape values only
for PrintManager distance draws in the separate natural trail-gap scalar case.
Other supported PrintManager toggle fixtures keep `random_tape_length == 0`,
use the deterministic default distance, and increment `random_tape_draw_count`
without advancing `random_tape_cursor`. The schema should still reserve real
RNG fields because spawn variation and future bonuses need reproducible streams
beyond fixed fixture tapes.

Current comparator reset contract for these tape fields is intentionally small:
`reset_array_rows_with_info` snapshots selected terminal rows first, then
`reset_array_rows` copies all selected rows from the reset template. That means
`random_tape_values`, `random_tape_length`, `random_tape_cursor`,
`random_tape_exhausted`, and `random_tape_draw_count` after reset are exactly
the source/template row values. A fresh reset template should keep cursor and
draw count at `0` and exhaustion at `false`. Rows outside the reset mask are not
changed. This is not full lifecycle RNG support and does not cover optimized or
GPU paths.

## Timer Queue Rule

Use the timer arrays listed in `Reset And Timer Fields`. Keep timers minimal and
row-local. The first required timer kind is delayed PrintManager start; round
lifecycle and bonuses can add kinds only after source fixtures pin them.

`timer_advance_ms[B]` is an input to the pre-step timer phase, not trainer
policy state. Source fixtures may set it directly; production profiles can use a
fixed wrapper clock, but the native source model remains elapsed-ms server
updates over current controls.

## Events And Debug Counters

Training does not need full event objects in the hot path, but fidelity and
benchmarks need fixed debug outputs.

At the start of a captured step, clear `event_count` and append pre-step timer
events first. Movement, collision, scoring, and post-collision PrintManager
events append after that in source order.

Per-step debug buffers:

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `event_count` | `[B]` | `int16` | Events emitted this step. |
| `event_mask` | `[B,L]` | `bool` | Valid event slot mask. |
| `event_type` | `[B,L]` | `int16` | Position, point, property, die, score, round end. |
| `event_player` | `[B,L]` | `int16` | Primary player. |
| `event_other` | `[B,L]` | `int16` | Killer/winner/related player or `-1`. |
| `event_bool` | `[B,L]` | `int8` | `-1 null`, `0 false`, `1 true`. |
| `event_value_i` | `[B,L,2]` | `int32` | Score, round score, property code, or small integer payload. |
| `event_value_f` | `[B,L,2]` | `float64` | Point/position payload when needed. |
| `event_overflow` | `[B]` | `bool` | Sticky event buffer overflow. |
| `event_overflow_attempts` | `[B]` | `int32` | Count of event writes that exceeded `L`. |

Counters worth keeping:

- `movement_updates[B,P]`
- `normal_points_inserted[B,P]`
- `important_points_inserted[B,P]`
- `death_points_inserted[B,P]`
- `body_scan_count[B,P]`
- `body_candidate_count[B,P]`
- `body_hit_slot[B,P]`
- `wall_hit[B,P]`
- `print_toggles[B,P]`
- `trail_clears[B,P]`
- `wrapper_dict_items[B]`

These counters make microbenchmarks easier to interpret and help catch silent
short-circuiting.

## Masks And Overflow Policy

Use masks everywhere a Python implementation might otherwise append, remove, or
skip:

- `env_active[B]`
- `alive[B,P]`
- `player_update_mask[B,P]`
- `body_active[B,K]`
- `should_draw[B,P]`
- `body_insert_mask[B,P]`
- `collision_candidate[B,K]` inside each player update
- `pm_update_mask[B,P]`
- `timer_active[B,T]`
- `timer_fire_mask[B,T]`
- `obs_valid[B,P]`
- `reset_mask[B]`

Overflow policy must be explicit before this becomes a training backend:

- Fidelity/debug profile: fail the row immediately and surface
  `overflow_code`.
- Benchmark profile: count the overflow, mark the row done, and exclude it from
  steady-state throughput summaries.
- Training profile: choose either large enough `K` for the rollout horizon or a
  tested truncation/autoreset policy. Do not silently wrap body slots because old
  bodies matter for trail-gap collision behavior.

Suggested `overflow_code` values:

| Code | Meaning |
| ---: | --- |
| `0` | No overflow. |
| `1` | Body buffer full. |
| `2` | Spatial cell body list full. |
| `3` | Event buffer full. |
| `4` | Observation/debug scratch full. |
| `5` | Timer queue full. |

## Observation And Public Output

Keep the hot observation as arrays:

| Field | Shape | Dtype | Notes |
| --- | ---: | --- | --- |
| `obs` | `[B,P,*obs_shape]` | `float32` or `float64` | Backend-specific observation tensor. |
| `reward` | `[B,P]` | `float32` | Public reward array. |
| `terminated` | `[B,P]` | `bool` | Per-agent terminal mask. |
| `truncated_agent` | `[B,P]` | `bool` | Per-agent truncation mask. |
| `info_code` | `[B,P]` | `int16` | Compact info/debug status. |

The dict/PettingZoo/Gymnasium wrappers should convert arrays to public objects
after the hot step. Benchmark wrapper conversion separately.

First debug packing benchmark:

- `scripts/benchmark_vector_obs_reward_packing.py` now packs fixed debug arrays
  from the current fixture-seeded vector state: `obs[B,P,9]`, `reward[B,P]`,
  legal action masks, ego row/env/player ids, done/truncated masks, and
  per-agent terminal/truncation masks.
- The observation is a privileged scalar debug surface: normalized position,
  heading sin/cos, alive, printing, score, round score, and normalized map size.
  It is not the final ego observation for training.
- The reward is placeholder/narrow: `score_delta + round_score_delta -
  died_this_step`. The death term uses compact `die` event rows when present and
  falls back to alive-transition inference if not. The score terms still come
  from state deltas.
- The benchmark proves runnable fixed shapes and pack-only timing on the current
  fixture slice. It does not prove reset/autoreset, replay chunks, policy/MCTS
  connection, or the final observation/reward contract.

## Variable-Length State For Now

The future hot backend should be fixed-shape, but several source-facing objects
should stay variable-length or offline until fidelity work needs them:

- Python source-runner `world_bodies` lists remain in the fidelity runners. The
  vector target mirrors them with fixed `body_*[B,K]` arrays and overflow masks.
- Full visible trail point lists stay out of the hot state for now. The proposed
  state keeps only visible count, visible last point, and draw cursor. Full
  replay/client trails can be reconstructed offline from point events if needed.
- Raw event objects, common-trace JSON, browser messages, and replay payloads
  stay offline. The hot state can expose a capped debug event buffer, but it
  should not allocate Python event objects per tick.
- Scenario action scripts and fixture definitions can remain variable-length
  files. A benchmark should compile them into fixed rollout lengths or scripted
  workload generators before timing.
- Public dict outputs, info dicts, agent string lists, Gymnasium/PettingZoo
  wrappers, and artifact manifests stay at the API/debug edge.
- Spatial indexes are optional. A first schema can scan `K` fixed body slots;
  fixed `cell_body_ids[B,C,M]` should be added only after scan cost is measured.
- Bonuses and power-up stacks are not in the first hot schema. Add fixed caps
  only after source bonus behavior is pinned.

If a future trainer needs more history, add capped arrays and masks deliberately.
Do not smuggle variable Python lists into the hot transition.

## What Can Batch Now

These operations can batch over environments now, even with a small player loop:

- Public trainer action ids to internal `source_move` arrays.
- Heading and position math for one player across all `B` rows.
- Wall or borderless-wrap checks for one player across all `B` rows.
- Normal point mask computation for one player across all `B` rows.
- Body append for all rows where that player emitted a point.
- Full-buffer or spatial-index body collision scan for one player across all
  `B` rows.
- Own-body latency masking from `live_body_num`, `body_num`, and
  `trail_latency`.
- Print-manager distance updates and toggle masks for one player across all
  `B` rows.
- Observation construction from arrays.
- Reset/autoreset by row mask.

The leading batch axis is the main win. Removing the `P` loop is not required
for the next useful backend.

## What Probably Keeps A Small Player Loop

Keep reverse player order explicit for now:

- Same-frame point insertion can kill a lower-index player later in the same
  tick.
- A player's own normal point must exist before that player's collision check.
- Print-manager toggles happen after that player's collision check.
- Death point insertion and scoring/events are order-sensitive.
- Event order is source-visible in common traces.

With `P` usually small, this loop is a correctness tool, not a performance
failure.

## What Should Stay CPU Or Offline

Keep these out of a first GPU/vector hot path:

- JS oracle execution and common-trace diffs.
- Raw event timeline formatting and artifact writes.
- Browser/replay/client visual checks.
- Public dict/wrapper object creation until measurements prove it must move.
- Modal queue/storage work.
- Scenario generation and source fixture management.
- Full debug event payloads for every training rollout.

GPU stepping is not the next move. The branchy source order and body checks may
be better served first by NumPy batching or compiled CPU kernels.

## Fixture-To-Array Bridge

The first bridge is `scripts/seed_vector_state_from_fixtures.py`.

It reads one or more scenario files, or a batch manifest with a `scenarios`
list, and writes JSON with one `B=1` seed per fixture. It is a compiler from
fixture setup into array-shaped state. It is not an environment.

Natural `source_lifecycle_*` fixtures that call real `Game.newRound()` are not
ordinary initial-state seeds. The bridge now fails those fixtures honestly as
unsupported and records the RNG contract metadata the future reset/spawn path
will need: call index, site, avatar, value, at-ms, expected call count, and
capacity pressure. This protects the docs and speed lane from claiming
optimized/vector lifecycle before reset/spawn exists. All 20 promoted lifecycle
fixtures through focused 4P survivor next-round, present/absent survivor
scoring, and focused 3P multi-round match-end are included in the
rejection/metadata guard, including the
three 2P core fixtures, focused 3P spawn-order fixture, focused 3P
warmup/PrintManager-start fixture, focused 4P first-round spawn-order fixture,
focused 4P all-present all-dead next-round fixture, focused 4P survivor
next-round fixture, focused 3P present/non-present first-round,
survivor-scoring, and next-round fixtures, focused 2P
max-score match-end fixture, one focused all-present 3P `max_score: 2`
match-end path, focused 3P tie-at-max continuation, focused 3P all-present
multi-round match-end, focused 3P all-dead
next-round fixture, focused 3P survivor-scoring `round:end` fixture, and
focused 3P survivor warmdown/next-round fixture.

The narrow vector spawn helper now lives in `src/curvyzero/env/vector_spawn.py`.
`spawn_round_rows(state, row_mask, *, player_count)` is not a seed compiler and
not a lifecycle step. It mutates selected rows with promoted first-round spawn
facts: 2P heading retry, 3P reverse spawn order, 3P present/absent, and 4P
reverse spawn order. Tests in `tests/test_vector_spawn.py` protect that helper.
It consumes row-local random-tape values through cursor/draw-count metadata,
uses per-row map size, iterates players in reverse source order, honors present
masks, records absent players in death-list arrays when those arrays exist, and
leaves world bodies untouched.

Fields it can seed today:

- positions, previous positions, headings, alive flags, death ticks, scores, and
  round scores
- source movement scalars: speed, angular velocity, radius, trail latency, and
  per-step distance/angle deltas
- map size and normal-wall vs borderless mode
- source move schedules in player order
- printing flags
- visible trail count and last visible point
- hidden draw cursor state
- seeded world body buffers: active mask, position, radius, owner, body number,
  insert kind, write cursor, and world body count
- per-player `body_count` and `live_body_num`
- initial PrintManager active flag, distance, and last position when the fixture
  provides them
- source `Math.random` tape arrays when the fixture provides
  `source_setup.random.math_random_sequence`, plus row-local tape cursor,
  exhaustion, and draw-count metadata initialized for reset/template copying

Fields it does not seed yet:

- natural lifecycle/reset/spawn from `Game.newRound()` fixtures, including all
  28 promoted lifecycle fixtures through focused 4P survivor next-round,
  present/absent survivor scoring, and focused 3P multi-round match-end
- reset/autoreset rows for many live games
- production per-env RNG state for random spawn, bonuses, and domain variation
  beyond fixed source tape arrays
- observations, rewards, terminal flags, truncation flags, and public `info`
  payloads
- compact event arrays for point, die, score, round-end, and property events
- death-point insertion state, score resolution state, round lifecycle timers,
  bonus stacks, and replay/browser output
- policy inputs, model outputs, replay-buffer records, or search-tree state

The seed script itself does not step the arrays. It only says: here is the
initial state a vector lane would receive for this fixture. The comparison step
now lives in `scripts/compare_vector_arrays_to_fidelity.py`: run one pure array
transition from this seed, project the result to common-trace fields, and diff it
against the existing JS/Python common trace.

For batched self-play, this bridge is only the first input edge. Still missing:

- optimized/vector lifecycle and reset-integrated spawn
- lifecycle timer setup, `game:start`, PrintManager start, warmup/warmdown, and
  next-round behavior
- a full self-play-ready `step_arrays(state, source_moves, rng)` transition that
  preserves source order and updates every array above
- a batched reset path that can create many fresh games without reading fixture
  JSON in the hot loop
- fixed-capacity policy for body/event overflow when many self-play games run
  for long horizons
- trainer-facing observation and reward arrays for all live rows
- row masks for done, truncated, env-active, and autoreset behavior
- a policy-call boundary that batches observations and returns actions without
  Python per-agent dict work in the hot path
- a model-call boundary for MuZero representation, dynamics, and prediction
  functions
- MCTS/search arrays for roots, child priors, values, visits, rewards, recurrent
  states, and legal-action masks
- a way to keep real environment rollouts, learned model rollouts, and trace
  fixtures separate so speed code does not become the rule authority

Do not overbuild the trainer API around this script. Use it to prove that the
same scenario facts can initialize the future vector state. Build the batched
self-play API only after the array transition matches the verified fixture set.

## Next Microbenchmark Plan

Create a source-aware timing scaffold before changing production layout. It can
be an isolated script that fills the schema-like arrays directly and runs fixed
scripted workloads.

Current scaffold status: `scripts/benchmark_source_fidelity.py` now times the
existing source-fidelity runner surface over the source kinematics, body canary,
trail cadence, trail gap, and PrintManager batch manifests. It gives real
source-fidelity timings for inclusive runner calls, common-trace projection, and
payload/output wrapping only. Movement, point insertion, collision scan, and
PrintManager internals remain explicitly scaffolded as `not_measured` because
the current runner surface does not expose those spans.

The same script also has an opt-in `--profile` mode. Its `cProfile`/`pstats`
summary is scoped to the measured source-fidelity runner-surface loop and
excludes warmup plus manifest collection. Treat its top functions as
runner-surface guidance only: scenario loading, inclusive runner calls,
common-trace projection, payload/JSON wrapping, and benchmark overhead. It is
not a production environment profile and not a source-internal bucket split.

Current vector prototype status: `scripts/benchmark_vectorization_prototype.py`
fills schema-like arrays directly and reports setup, warmup, source-like
movement, normal point mask, append-only body insertion, full-buffer collision
scan, toy observation, memory, overflow counters, and optional JAX/PyTorch module
detection. It is labeled source-like synthetic and records the missing semantics
in its JSON output.

Report setup/warmup separately from steady-state time. Sweep at least:

- `B`: `1`, `16`, `128`, `512`
- `P`: `1`, `3`, `4`
- `K`: `0`, `64`, `512`, `4096`
- Dtype: `float64` first, optional `float32` comparison only after parity checks

Timing buckets:

1. Source-like movement:
   action conversion, reverse player loop, heading update, and position update.
2. Normal point insertion:
   `isTimeToDraw` mask, visible trail update, draw cursor update, and
   `body_count/live_body_num` handling. The current prototype splits this into
   `normal_point_mask` and `body_append`.
3. Body insertion:
   fixed-buffer append, `world_body_count`, body owner/num/radius fields, and
   optional spatial index update.
4. Collision scan:
   strict circle overlap, opponent bodies, own-body latency mask, and first hit
   selection. Run both full-buffer scan and spatial-index variants only if the
   index exists.
5. Print-manager update:
   distance subtraction, last-position update, toggle mask, important point
   insertion, visual clear, and next-distance RNG draw.
6. Observation construction:
   array-only observation for all live/done rows.
7. Dict/wrapper output:
   agent id lookup, observation/reward/terminated/truncated/info dict creation,
   and any copies at the public API edge.

The current prototype covers buckets 1 through 4 and a toy version of bucket 6.
Buckets 5 and 7 should wait until the fidelity lane pins the remaining source
semantics and public contract.

Workloads:

- Long no-death movement with no bodies.
- Normal trail point every tick.
- Below-radius movement with no point.
- Same-frame higher-index point insertion that can kill a lower-index player.
- Own-body latency safe and kill cases.
- Visual hole crossing with no body.
- Visual hole crossing over an old body.
- Print-to-hole and hole-to-print toggles.

Each benchmark result should record the schema profile, commands, seed policy,
runtime versions, hardware label, equivalence fixture status, body/event
overflow counts, and memory per environment.

Next concrete GPU/vector path:

1. Keep the NumPy script as the shape and timing oracle for `B/P/K` sweeps.
2. Use `scripts/seed_vector_state_from_fixtures.py` to seed fixture-backed arrays
   before adding new vector semantics.
3. Extend only semantics that are already source-pinned: wall/wrap masks,
   PrintManager post-collision toggles, death body insertion, and scoring/events.
4. Once a pure array transition matches scripted fixtures, make an optional
   script-level JAX or PyTorch spike using the same state: batch with `vmap` or a
   leading tensor axis, roll out fixed horizons with `scan` or an equivalent
   loop, and keep the small reverse `P` loop.
5. Measure CPU first, then GPU only for large `B/K` profiles where the
   full-buffer scan remains the measured bottleneck after avoiding host/device
   transfer churn.
6. Use Modal GPU later for batched env, policy, and MCTS/search experiments that
   stay inside one remote process and write coarse artifacts. Do not use Modal
   as a per-step environment service.
