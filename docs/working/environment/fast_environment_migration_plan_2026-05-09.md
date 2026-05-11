# Fast Environment Migration Plan

Date: 2026-05-09
Status: aggressive migration note, docs only

## Direction

The runtime goal is one fast faithful surface under hardening:
`VectorMultiplayerEnv`. The scalar source env and JS oracle path are
executable proof/oracle tools used to keep that runtime honest. They do not run
product training.

Current correction: `vector_runtime.py` now owns step input normalization,
validation, runtime event constants, zeroed counter construction, PrintManager
row-local random distance assignment, final state-derived counter normalization,
the supported fixture-backed source-ordered CPU batch step, terminal lifecycle
row marking, and the first strict 1v1/no-bonus warmup timer advance helper.
That helper can fire `game:start`, schedule reversed PrintManager starts, start
PrintManagers, insert important body points, and consume row-local random tape.
The strict 1v1/no-bonus vector handoff now also has real final trainer
observation/reward staging and a narrow autoreset apply helper, but not a full
trainer env API.

`src/curvyzero/env/vector_runtime.py::step_many` is now the supported
fixture-backed transition-kernel boundary. `scripts/benchmark_vector_batch_rows.py`
routes normal calls through `step_many`; its private
`_step_many_kernel(..., phase_timers=...)` path is only for benchmark
diagnostics. The dead duplicate old benchmark body has been removed.

The immediate runtime move is hardening `VectorMultiplayerEnv` rather
than treating strict 1v1 or raw `vector_runtime.step_many` as the product. Keep
fixture seeding, comparator projection, source-runner calls, phase timing, and
benchmark reporting in scripts while the intended runtime grows those surfaces.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Fastest Existing Bits

Production-facing but still narrow:

- `src/curvyzero/env/vector_reset.py`: masked row reset, terminal snapshot, row
  lifecycle stamping, reset seed/source metadata.
- `src/curvyzero/env/vector_spawn.py`: source-shaped natural spawn for selected
  rows from row-local random tape.
- `src/curvyzero/env/vector_lifecycle.py`: reset plus spawn composition, plus
  narrow 1v1/no-bonus warmup timer scheduling and a strict
  reset/spawn/warmup/timer/runtime step composition helper.
- `src/curvyzero/env/vector_runtime.py::advance_warmup_1v1_no_bonus_timers`:
  narrow 1v1/no-bonus warmup timer advancement through game start and delayed
  PrintManager starts.
- `src/curvyzero/env/vector_autoreset.py`: pure public autoreset planning
  contract plus `apply_autoreset_rows(...)`, which stages final
  observation/reward through the planner and then calls
  `vector_reset.reset_arrays(...)` for selected rows.
- `src/curvyzero/env/vector_trainer_observation.py`: narrow 1v1/no-bonus vector
  trainer handoff. It raycasts against vector body circles, produces pinned
  `float32[106]` observations and `float32[B,2,106]` final observation arrays,
  and derives sparse final reward maps from terminal lifecycle rows.

Runtime-owned fast path plus script wrappers:

- `src/curvyzero/env/vector_runtime.py::step_many(...)` owns the supported
  fixture-backed source-ordered CPU transition kernel. It supports B-stacked
  rows, reverse player order, movement, point/body updates, collision, wall,
  PrintManager slices, delayed-start timer slices, and event/no-event modes.
- `scripts/benchmark_vector_batch_rows.py::step_batched_arrays(...)` is now a
  benchmark wrapper around the transition-kernel boundary. Normal calls go through
  `vector_runtime.step_many`; private `_step_many_kernel(..., phase_timers=...)`
  is reserved for phase-timing diagnostics.
- `scripts/compare_vector_arrays_to_fidelity.py::step_prepared_arrays(...)` is
  the B=1 source-ordered kernel plus comparator/oracle glue. It is useful as a
  parity oracle for the extracted runtime.
- `scripts/benchmark_vector_actor_loop_bridge.py` is the best current
  end-to-end shape scout: vector step, debug obs/reward packing, synthetic
  policy/search, action remap, replay staging, and debug-only autoreset.

Only scripts, fixtures, or synthetic evidence:

- Fixture seeding and cycling are not production reset/autoreset.
- The actor bridge's policy/search is a local NumPy stand-in, not learned model
  inference, MCTS, Mctx, or GPU work.
- Debug obs/reward packing is not the final trainer observation/reward schema.
- In-memory replay chunk staging is not the production replay stream.
- Modal/JAX/Mctx runs are boundary/runtime evidence only. They do not step
  CurvyTron and should not drive env-over-GPU work yet.
- `scripts/benchmark_vectorization_prototype.py` is useful shape timing only; it
  is not source-fidelity throughput.

## Runtime Hardening Target

Use `src/curvyzero/env/vector_multiplayer_env.py::VectorMultiplayerEnv`
as the intended runtime boundary under hardening. `vector_runtime.step_many` is
the lower-level supported fixture-backed transition kernel it can build on; it
is not the product runtime by itself.

Keep moving script dependencies toward runtime-owned modules:

1. Update actor/debug call sites that still call benchmark wrappers to use the
   runtime boundary when they do not need benchmark reporting.
2. Keep the source-ordered scalar helper only as a test/comparison wrapper if needed;
   do not keep a second independent implementation.
3. Move fixed array constants and event/timer row helpers currently duplicated in
   `scripts/compare_vector_arrays_to_fidelity.py`, but only after they are used
   by the runtime module.

Leave in scripts:

- Scenario/fixture loading.
- Source common-trace projection and mismatch reports.
- Benchmark CLI output, phase timing summaries, and sample replay files.
- Synthetic policy/search stand-ins and Modal boundary samples.

The remaining cleanup should be boring: keep behavior in the runtime kernel,
trim duplicate script code, and require the same comparator and B>1 preflight
to pass. Do not change physics, event order, timers, or reset semantics during
the cleanup.

Known runtime blockers: the supported fixture-backed batch step is extracted,
and the terminal path now marks optional `done`, `terminated`, `reset_pending`,
`terminal_reason`, `draw`, and `winner` arrays after survivor or draw terminal
events when those arrays exist. The runtime is still not a full trainer
environment. The strict 1v1/no-bonus vector final-observation/reward handoff is
real, but remaining urgent gaps are replay writer integration from vector
batches, a public full env API, broad lifecycle coverage for warmdown/next
round/3P/4P/bonuses, the visual renderer/LightZero adapter, production
reset/seed history, and whole-loop performance integration.

## Modal Pattern

Use Modal for coarse jobs around the runtime, not inside the runtime. Good first
jobs are CPU correctness/timing smokes for `vector_runtime` batches, 1v1
no-bonus terminal/autoreset loops, bounded seed/batch-size sweeps, and artifact
storage. Keep exact env steps, action selection, MCTS nodes, and replay-row
writes local inside one Modal container. Later GPU work should run whole
self-play/search/train jobs in a container, write chunky replay/checkpoint
artifacts to a Volume, and return compact refs.

## First Benchmark

Use the P2 no-event actor bridge as the first production speed guard because it
is closest to the hot training loop while keeping debug event tax out of the
way.

Current clean P2 no-event readout to preserve or explain:

- `actor_step_p50` about `0.557 ms`
- `env_step_p50` about `0.271 ms`
- `synthetic_policy_p50` about `0.060 ms`
- env step about `48.5%` of the current fake loop

Run the same benchmark before and after extraction. A successful extraction is
parity-first: same supported fixtures, same B>1 preflight, same no-event state
semantics, and no meaningful p50/p95 regression.

## Amdahl Guardrail

Do not optimize GPU/JAX env stepping yet. Measure env step against real
MCTS/search/model cost first.

Today, env step looks large because the policy/search bucket is fake and cheap.
With `env_step_p50 ~= 0.271 ms`:

- If real MCTS/search is around `1.9 ms`, env share may fall near `11%`. Even an
  impossible infinitely fast env would only improve the full loop by about
  `1 / (1 - 0.11) = 1.12x`.
- If real search is around `0.5 ms`, env may still be roughly `25-35%` of the
  loop. Then env work can matter, but the maximum payoff is still bounded until
  search, observation packing, replay, reset, and transfer are measured too.
- In the current fake loop, env share around `48.5%` gives an env-only upper
  bound near `1.94x`; that number must not be used to justify GPU env work
  until real search timing replaces the synthetic stand-in.

Every speed report should carry env step, observation/reward packing,
MCTS/search/model, action mapping, reset/autoreset, replay staging, and p50/p95
action latency. Rows/sec alone is not enough.

## First Parity Target

Extraction parity target:

1. Existing `compare_vector_arrays_to_fidelity.py` supported fixture set remains
   green when it imports the new runtime step.
2. Existing `benchmark_vector_batch_rows.py` B>1 preflight remains green for
   `debug-event` and `no-event`.
3. Existing actor bridge sample still produces the same fixed-shape debug
   surfaces after importing the runtime step.

First production parity target after extraction:

- 1v1/no-bonus vector runtime covers reset/spawn/warmup, timer advancement,
  source-ordered step, terminal masks, final observation/reward handoff, narrow
  autoreset apply ordering, and replay-v0-compatible metadata for the current
  source-backed 1v1 slice.
- The new `vector_lifecycle.run_warmup_start_step_1v1_no_bonus_rows` helper
  already composes the strict reset/spawn/warmup/timer/runtime step slice, and
  a focused test proves wall-death terminal state plus real vector trainer
  final-observation/reward arrays feeding autoreset planning. It does not prove
  replay writer integration, a public full env API, broad lifecycle, bonuses,
  visual rendering, performance integration, or trainer readiness.
- The scalar source env remains the oracle harness until the vector runtime can
  replay promoted lifecycle traces without fixture cycling.

## Delete Or Postpone

Already done:

- `scripts/benchmark_vector_batch_rows.py` routes normal step calls through
  `vector_runtime.step_many`.
- The dead duplicate old benchmark body after the runtime wrapper return was
  removed.

Delete later:

- Script-local duplicate reset helpers in `compare_vector_arrays_to_fidelity.py`
  that overlap with `src/curvyzero/env/vector_reset.py`, once call sites import
  the production helper.

Postpone:

- GPU/JAX environment stepping.
- Numba/native rewrites.
- Mixed-P padded production batching.
- Broad bonuses and 3P/4P production training claims.
- Production replay writer and trainer API exposure until the vector trainer
  handoff, reset/autoreset, and seed/RNG metadata are wired through vector batch
  replay and the public env API.

Keep:

- Source env and JS oracle as executable specs.
- Fixture/comparator scripts as parity gates.
- Debug-event mode for investigations, but prefer no-event mode for the hot
  training path unless a sampled event stream is explicitly required.

## Concrete Next Steps

1. Update `scripts/benchmark_vector_actor_loop_bridge.py` and other actor/debug
   call sites that do not need benchmark reporting to call the runtime step
   boundary directly.
2. Wire the strict 1v1/no-bonus vector trainer handoff into replay row writing
   and the public full env API.
3. Run comparator parity and B>1 preflight in both `debug-event` and `no-event`
   modes.
4. Rerun the P2 no-event actor bridge and report p50/p95 actor step, env step,
   synthetic policy, replay, and Amdahl buckets.
5. Replace synthetic policy/search timing with calibrated real MCTS/search or a
   measured boundary profile before choosing GPU/JAX env work.
6. Add broad lifecycle coverage only as separate claims: warmdown, next round,
   3P/4P, bonuses, the visual renderer, and the LightZero adapter.
