# GPU Observation Investigation

Date: 2026-05-15

Purpose: understand why the first GPU observation integration was slower than
CPU, and define what to build next so GPU rendering actually helps the stock
LightZero CurvyTron loop.

## Current Facts

The semantic policy surface is fixed:

```text
browser_lines + simple_symbols -> [4,64,64]
```

The current reliable training backend is:

```text
policy_observation_backend=cpu_oracle
```

The experimental scalar backend is:

```text
policy_observation_backend=jax_gpu
```

That scalar backend is wired end to end and reaches stock
`lzero.entry.train_muzero`, but it is slower:

| backend | steps | wall | steps/s | obs mean |
| --- | ---: | ---: | ---: | ---: |
| `cpu_oracle` | 512 | `15.54s` | `32.94` | `4.42ms` |
| scalar `jax_gpu` | 512 | `63.73s` | `8.03` | `80.31ms` |

Subprocess scalar `jax_gpu` fails before collection with JAX CUDA init errors in
env workers.

The isolated batched renderer is fast:

| renderer | batch | GPU | end-to-end | frames/s |
| --- | ---: | --- | ---: | ---: |
| `block_704_gray64` | 1 | H100 | `8.35ms` | `119.8` |
| `block_704_gray64` | 64 | H100 | `34.54ms` | `1852.8` |
| `block_704_gray64` | 256 | H100 | `104.78ms` | `2443.2` |

Plain read: the GPU math is not the obvious problem. The connection point is.

Scalar component profiles now explain the trainer result:

| scalar component shape | GPU | two-view step |
| --- | --- | ---: |
| `trail_slots=256` | H100 | `~8.57ms` |
| `trail_slots=1024` | H100 | `20.77ms` |
| `trail_slots=2048` | H100 | `39.43ms` |
| `trail_slots=4096` | H100 | `72.20ms` |
| full trainer scalar `jax_gpu` | H100 | `~80ms` observation mean |

Critical code fact: the full trainer's scalar JAX path sets `trail_slots` from
`state["visual_trail_active"].shape[1]`. The trainer-created
`VectorMultiplayerEnv` does not pass `body_capacity`, so it inherits the source
default `4096`. The scalar component benchmark builds its production env with
`body_capacity=trail_slots`; the earlier fast profile used `256`. The block
renderer loops over all slots, active or not, so the capacity mismatch is now
the primary scalar slowdown explanation. The remaining `~8ms` gap is small
enough to attribute to trainer wrapper overhead, stack copies, scheduling, and
possible CUDA sync attribution until proven otherwise.

Capacity safety caveat: `VectorMultiplayerEnv.body_capacity` currently sizes
both collision body arrays and browser-style `visual_trail_*` arrays. Body
overflow is terminal/truncated telemetry, but visual-trail overflow is only a
flag on state. Lowering capacity can silently shorten rendered history unless
`visual_trail_overflow`, `visual_trail_write_cursor`, and `body_write_cursor`
are explicitly surfaced/asserted in the canary.

Safest near-term fix is renderer-side compaction to the active prefix needed
for the current row, with a configurable minimum shape, while leaving env
capacity unchanged. Exposing `body_capacity` is still useful for explicit
canaries and benchmarks, but it should default to existing source behavior.

Current patch direction: do not lower the real env capacity as the first fix.
Keep the full env state, but make the scalar GPU renderer choose its JAX render
shape from the active visual-trail prefix instead of allocated capacity. The
first implementation uses a minimum bucket based on `2 * max_source_ticks`,
rounds to a power of two, grows if active slots need more room, and fails rather
than silently truncating active trails. This is still a canary, not the
production batched backend.

Fidelity update: the current JAX `browser_lines` renderer is not
oracle-equivalent. It connects each trail point to the immediately previous slot,
but the CPU visual oracle connects to the previous active same-owner
visual-trail point. Parity rows that differ by only a few gray values are not
enough proof because overlapping caps can hide this semantic gap. Treat GPU
timings as renderer economics only until this is fixed and covered by targeted
parity tests.

Full-loop patched canary:

| backend | steps | wall | steps/s | obs mean |
| --- | ---: | ---: | ---: | ---: |
| `cpu_oracle` | 512 | `15.61s` | `32.80` | `4.50ms` |
| scalar `jax_gpu` active-prefix bucket | 512 | `34.46s` | `14.86` | `24.39ms` |

Plain read: active-prefix bucketing fixed the 4096-slot disaster but did not
make scalar per-env GPU competitive. It reduced observation time by about `3.3x`
versus the original scalar `jax_gpu` profile, but CPU remains about `5.4x`
faster on this 512-step stock profile. The next real speed path is not more
scalar polishing unless it is very small; it is batched/fused observation or
staying with the CPU oracle.

## Latest Harness Fixes

Red-team review found that the isolated GPU benchmark could overstate parity:
for `state_source=real_env_rollout`, verification rebuilt a production-like
state from compact arrays instead of using the original `VectorMultiplayerEnv`
state. That could hide `avatar_color`, write-cursor, and stale-tail mistakes.

Fixes now in the worktree:

- Real-env verification keeps the original production state for the CPU oracle.
- Compact GPU state masks active visual/body trail slots at or after the
  relevant write cursor.
- Compact state carries `trail_write_cursor` so reconstructed synthetic
  references do not invent cursor values from `sum(active)`.
- The scalar trainer-side GPU slot profile also ignores active bits after
  `visual_trail_write_cursor`.
- The direct-gray CPU verifier now uses previous active same-owner slots rather
  than the stale raw `slot - 1` rule.
- The scalar fused benchmark now reports CPU-oracle parity for the two-player
  fused output, not only "fused JAX equals separate JAX."

Fresh H100 lab results after these harness fixes:

| row | shape | end-to-end | device render | parity |
| --- | --- | ---: | ---: | --- |
| B64, player 0, S1024, real_env512 | one view | `137.35ms` | `134.38ms` | 11 mismatched pixels in 2 checked rows, max diff 23 |
| B64, player 1, S1024, real_env512 | one view | `137.60ms` | `134.16ms` | 11 mismatched pixels in 2 checked rows, max diff 23 |
| B1 scalar, S1024, real_env512 | fused two views | `29.78ms` fused vs `53.66ms` separate | `23.97ms` fused render | fused JAX matches separate JAX exactly, but CPU parity fails on 11 pixels, max diff 18 |

Plain read at this point: the same-owner connectivity fix and fused two-view
shape were real, but the GPU renderer still differed from the CPU oracle. The
leading suspect was owner-layer/RGB overwrite composition: CPU browser-lines
draws owner layers in a fixed owner order on RGB, while the JAX block renderer
was writing luma in slot order.

Follow-up fix: the block GPU renderer now carries a per-subpixel owner-priority
buffer for trail pixels. Priority matches CPU owner draw order: invalid owners
first, then valid owners descending, which means player `0` wins two-player
trail overlaps even when its luma is lower. Bonuses and heads still draw after
trails.

Fresh H100 lab results after the owner-priority fix:

| row | shape | end-to-end | device render | parity |
| --- | --- | ---: | ---: | --- |
| B64, player 0, S1024, real_env512 | one view | `211.85ms` | `208.57ms` | exact parity on 2 checked rows |
| B64, player 1, S1024, real_env512 | one view | `212.98ms` | `207.83ms` | exact parity on 2 checked rows |
| B1 scalar, S1024, real_env512 | fused two views | `28.13ms` fused vs `50.41ms` separate | `24.72ms` fused render | exact CPU parity for both views |
| B256, player 0, S1024, real_env512 | one view | `735.48ms` | `728.92ms` | exact parity on 2 checked rows |
| B64, player 0, S256, real_env128 | one view | `59.14ms` | `54.75ms` | exact parity on 2 checked rows |

Plain read now: the lab renderer is finally matching the CPU oracle on these
real-env parity rows. The cost increased for the one-view S1024 rows because
exact owner-priority composition carries an extra high-resolution priority
buffer. The B256/S1024 H100 row is only about `348` frames/sec end to end, while
the B64/S256 row is about `1082` frames/sec. This is still not a trainer
backend, but the next gate has moved from "fix obvious semantics" to "broaden
adversarial parity and decide whether the exact GPU economics beat CPU in a
real batched boundary."

Critique update: matching a few render rows is not enough. Before any trainer
integration, the GPU path needs a production-contract parity gauntlet that
checks observations, rewards, done/truncated, final/reset observations, frame
stack order, legal/action ordering, bonus/head/trail overlap, cursor wrap,
stale-tail masking, controlled-player perspective, and color/avatar effects.
The safest next systems experiment is not scalar `jax_gpu` and not a service
yet; it is a profile-only vector-env facade that owns many CPU envs in one
parent process, gathers render state, performs one batched GPU render, updates
`[4,64,64]` stacks, and reports LightZero-shaped observations without touching
stock training defaults.

Adversarial fixture update: the benchmark now has
`state_source=adversarial_fixture`. It builds hand-authored 3-player production
states with interleaved owners, inactive holes, `break_before`, radius changes,
cursor-zero and stale-tail cases, invalid owners, dead/absent players,
head/bonus/trail overlap, all 12 `simple_symbols`, and non-identity
`avatar_color` rows: swapped, duplicated, and high-index color ids. The compact
GPU state now carries `avatar_color`, and the JAX policy-grayscale palette uses
the same controlled-color mapping as the CPU oracle.

Fresh H100 adversarial fixture results after the avatar-color fix:

| controlled player | batch | players | trail slots | bonuses | parity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4 | 3 | 10 | 12 | exact, `0` mismatches |
| 1 | 4 | 3 | 10 | 12 | exact, `0` mismatches |
| 2 | 4 | 3 | 10 | 12 | exact, `0` mismatches |

Plain read: this is a much stronger renderer proof than the earlier smoke rows,
but it is still not trainer promotion. It checks newest-frame render parity for
a small adversarial corpus; it does not yet check stack/reset/final-observation
contract parity or a batched LightZero boundary.

Concrete fidelity hole to keep pinned: an adjacent-slot JAX `browser_lines`
implementation connects a trail slot only to `slot - 1`, while the CPU oracle
connects to the previous active visual-trail point for the same owner. Real
vector runtime appends visual trail points while iterating players in reverse
order, so two-player rows naturally interleave owner slots. A canary with
owners `[0, 1, 0]`, positions `[(20,32), (32,8), (44,32)]`, radius `1`, and
`break_before=[1,1,0]` changes 72 downsampled pixels: the CPU oracle draws the
owner-0 segment through `(32,32)`, while the adjacent-slot rule leaves that
midpoint as background. The current worktree has a previous-owner helper in the
JAX renderer, but this canary still needs to be promoted to a regression test
and direct synthetic verification should not be trusted as the oracle.

Exact tests to add before trusting GPU `browser_lines`:

- CPU oracle canary in `tests/test_vector_visual_observation.py`: interleaved
  visual-trail owners `[0,1,0]` must render the same-owner midpoint, and setting
  `break_before` on the third point must remove it.
- Benchmark conversion canary: compacting that production state through
  `_production_to_benchmark_source_state` must preserve order, active holes,
  owners, and `break_before`.
- GPU parity canary in the benchmark test surface: `block_704_gray64` and
  `direct_gray64` must match `render_source_state_canvas_gray64` on the
  interleaved case, not the benchmark's current adjacent-slot CPU verifier.
- Real-env rollout canary: after two printing players have at least two visual
  points each, compact/render one row and assert CPU oracle equality for both
  controlled-player palettes.

Minimal implementation shape: compute previous active same-owner trail metadata
for the copied prefix, or scan it inside the JIT as the current worktree does,
and have every JAX render surface consume that instead of `slot - 1`. Keep
`break_before` as the segment gate for the current slot.

Additional red-team risks that shape/hash checks will miss:

- Owner draw order: CPU browser lines draw grouped owners in its fixed owner
  order; a slot-order JAX overwrite can flip crossing-line pixels.
- Cursor discipline: CPU ignores visual trail slots at or after
  `visual_trail_write_cursor`; active-prefix logic must not render stale active
  bits beyond cursor.
- Bonus symbols: direct gray64/simple-symbol economics paths can preserve shape
  while losing inner-symbol identity. Parity must cover all 12 types, clipped
  edges, subpixel centers, and trail/bonus/head overlap order.
- Palette/perspective: policy views are controlled-player self/other luma and
  may route through `avatar_color`; GPU compaction cannot silently assume owner
  index equals color index when all-color effects are active.
- Stack/reset order: reset must zero old FIFO frames, then render exactly one
  reset frame in the newest channel; per-row tournament resets must clear dirty
  caches only for reset rows.
- Timing contract: trainer and tournament must agree on
  `decision_source_frames`, `source_physics_step_ms`, max source ticks, and
  profile/no-death settings before comparing policies.

## What Might Be Wrong

### Hypothesis A: per-step launch/copy overhead dominates

The scalar hook sends one tiny env state to JAX on every env step, renders one
or two tiny frames, then copies back to NumPy. GPU work likes batches; this path
feeds it crumbs.

Expected symptom: isolated batch-1 is much worse than batch-64/batch-256, and
trainer scalar observation time stays high even after warmup.

Status: true, but secondary to capacity. Scalar B=1 overhead is still enough to
make the `256`-slot H100 path slower than the CPU oracle, but it does not
explain the `~80ms` result by itself.

### Hypothesis B: rendering both player views doubles work

The scalar hook calls one compiled JAX renderer per controlled player, so a
two-player observation does two render launches and two readbacks in a row.

Expected symptom: one-view benchmark is much faster than two-view scalar
trainer path. A fused two-view render should cut overhead.

Status: bounded. Two views are already included in both scalar component rows.

### Hypothesis C: production-state conversion is expensive

`_production_to_benchmark_source_state` builds compact NumPy arrays from the
live env state every step. If that copy/reshape/filter path is expensive, GPU
render will not help until state is already in compact/vector form.

Expected symptom: a component timer shows CPU conversion is a large fraction of
the `80ms` scalar observation.

Status: mostly bounded. Component conversion was about `0.10ms` at `256` slots;
even if larger at `4096`, render capacity dominates.

### Hypothesis D: trainer shape is 4096 trail slots, benchmark shape is 256

The trainer derives JAX `trail_slots` from live env state shape. Its env factory
does not expose/pass `body_capacity`, so source default capacity is `4096`.
The scalar component helper accepts `trail_slots` and uses it as
`body_capacity`; default is `256`. The JAX block renderer uses a `lax.fori_loop`
over all slots, not just active slots.

Expected symptom: component profile with `trail_slots=4096` lands near the
trainer's `~80ms`; a trainer experiment with `body_capacity=256` should land
much closer to the component's `~8-10ms`.

Status: confirmed primary explanation for scalar slowdown.

### Hypothesis E: JAX/PyTorch/process topology is wrong

LightZero subprocess env workers and JAX CUDA do not mix cleanly. Multiple env
processes may each initialize JAX/CUDA and reserve memory, or fail if CUDA was
initialized before fork.

Expected symptom: base env manager works slowly; subprocess env manager fails
or explodes memory.

Status: true for the current scalar canary.

### Hypothesis F: wrong JAX API shape

The scalar hook imports a Modal benchmark module and uses helper functions that
were built for benchmark rows, not production trainer hot loops. It may miss
preallocation, donation, persistent device arrays, or a vectorized player axis.

Expected symptom: a minimal production-shaped JAX function is much faster than
the current wrapper.

Status: plausible; needs toy experiments.

### Hypothesis G: observation timing is the first hard GPU sync

The env's own `observation_sec` wraps `_lightzero_observation`, and the scalar
JAX path calls `block_until_ready()` plus `np.asarray()` inside that window.
LightZero policy/MCTS/model work runs on PyTorch CUDA in the same process. If
PyTorch work is still asynchronous when env stepping reaches observation, the
JAX readiness/readback point may absorb unrelated GPU backlog.

Expected symptom: enabling CUDA synchronization around policy/MCTS timing moves
time out of `observation_sec` and into policy/search timers without changing
wall time much.

Status: plausible secondary suspect. Still test because it can distort phase
timing, but it is no longer needed to explain the order-of-magnitude gap.

### Hypothesis H: the fast JAX renderer is semantically wrong

The current JAX browser-line path connects against adjacent trail slots. The CPU
oracle renders from compact visual-trail records and connects to the previous
active same-owner point, with `break_before` preserving discontinuities.

Expected symptom: sparse/inactive interleaved slots, owner alternation, wrapped
cursor rows, or blanked opponent slots produce missing or extra connector
segments even when broad real-rollout parity looks close.

Status: confirmed fidelity blocker. Fix before batching, fusion, service
plumbing, or training use.

## Runtime Anchors

- [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html):
  JAX preallocates 75% of GPU memory on first operation by default. Use
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=.XX`,
  or, for diagnosis only, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
- [JAX async dispatch](https://docs.jax.dev/en/latest/async_dispatch.html):
  host timing can lie until `block_until_ready()` or host readback. Our scalar
  path blocks and reads back inside observation, so it measures real render plus
  any GPU work queued ahead of it.
- [PyTorch CUDA timing and allocator notes](https://docs.pytorch.org/docs/main/notes/cuda.html):
  CUDA work is asynchronous; precise timings need synchronization or CUDA
  events. PyTorch also uses a caching allocator, configurable with
  `PYTORCH_ALLOC_CONF` / `PYTORCH_CUDA_ALLOC_CONF`.
- [PyTorch multiprocessing notes](https://docs.pytorch.org/docs/2.8/notes/multiprocessing.html):
  CUDA with `fork` is a poison-fork hazard; CUDA subprocesses require `spawn` or
  `forkserver`.
- [DI-engine env manager overview](https://di-engine-test.readthedocs.io/en/latest/feature/env_manager_overview_en.html):
  `BaseEnvManager` is serial in one process; subprocess env managers run envs in
  child processes via multiprocessing and IPC.

## Near-Term Questions

1. If `body_capacity` is capped to the needed policy-history capacity, how close
   does full trainer scalar `jax_gpu` get to the `8-10ms` component row?
2. Does pre-observation CUDA synchronization move any residual scalar time out
   of observation and into PyTorch policy/search/learner buckets?
3. Can a fused two-player JAX render reduce scalar canary cost enough to keep it
   cheap for base-mode regression checks?
4. Can we batch across envs without rewriting stock LightZero too deeply?
5. If stock LightZero only returns NumPy observations, what is the cleanest
   boundary for a GPU render service?
6. Is a CPU compiled/vectorized renderer simpler and good enough while the
   batched GPU architecture is built?

## Investigation Plan

- [x] Code-path audit: the scalar trainer path derives `trail_slots` from live
  state shape and likely uses `4096`; the scalar component default is `256`.
- [x] Toy component benchmark: conversion vs device copy vs one-view render vs
  two-view render vs host readback is bounded at `~8.57ms` for the tested shape.
- [x] Run the scalar component profile with `trail_slots=4096`: H100 two-view
  scalar is `72.20ms`, matching the trainer's `~80ms` observation mean.
- [ ] Run a trainer canary with renderer `render_trail_slots` capped to
  `1024` for the 512-step/two-player profile; then compare an explicit
  env `body_capacity=1024` canary only after overflow assertions are present.
- [ ] Add the safe plumbing as an explicit env/config knob, defaulting to the
  existing source default. For exact full visual history, conservative capacity
  is `max_source_ticks * player_count`; shorter canaries can choose smaller
  values only with overflow assertions.
- [ ] Add renderer timing sidecar fields: `trail_slots`, active trail count,
  JIT cache key/miss, conversion, host-to-device, p0 render/readback,
  p1 render/readback, stack/copy, and optional pre-observation CUDA sync debt.
- [ ] Add targeted GPU-vs-CPU parity fixtures for previous active same-owner
  browser-line connectivity, inactive-slot holes, owner alternation,
  `break_before`, compaction buckets, reset/cursor regression, and visual-trail
  overflow guards.
- [ ] Process-topology audit: explain why subprocess failed and what process
  shapes can safely own a JAX GPU context.
- [ ] Batched design sketch: minimal interface for `[B, P, 1,64,64]` renderer
  that could feed LightZero without old trainer rewrites.
- [ ] Add docs and guardrails so no one mistakes scalar `jax_gpu` for the
  production GPU backend.

## Topology Recommendation

Keep scalar `jax_gpu` as a **base-mode canary only**, after the trainer capacity
is explicitly sized. It is useful because it proves JAX CUDA availability,
observation parity, host/device transfer plumbing, and LightZero integration in
one process. It should not be the default trainer backend because even the
correctly-sized `256`-slot scalar component is around `8.57ms`, while the CPU
oracle training observation mean was `4.42ms`.

Do not use scalar `jax_gpu` with subprocess env workers. In subprocess mode, each
worker can become a separate JAX/CUDA owner, and forked CUDA initialization is a
known bad topology. `spawn`/`forkserver` are worth one controlled proof only if
set before importing Torch/JAX/LightZero, but they do not fix the core problem:
multiple B=1 renderers fighting over one H100 and each needing its own JAX
runtime/allocator state.

The env vars still matter, but as guardrails rather than the main slowdown fix:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: first diagnostic for JAX/PyTorch
  coexistence in one Modal H100 process.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=.10` or `.20`: safer long-running canary
  setting if preallocation remains enabled.
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`: debugging only; useful to prove memory
  ownership problems, expected slower.
- `JAX_LOG_COMPILES=1`: catch accidental retracing after resets/capacity changes.
- `JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache`: optional canary ergonomics, set
  before first compile.
- `JAX_TRANSFER_GUARD=log` or direction-specific transfer guard logging: use on a
  tiny run to expose surprise host/device traffic.
- `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`: experiment only; compare against
  PyTorch's default allocator after JAX memory fraction is constrained.
- `CUDA_LAUNCH_BLOCKING=1`: tiny diagnostic only; useful for attribution, not a
  throughput setting.

The batched-GPU recommendation does **not** change. The new fact actually makes
it stronger: scalar render cost scales with reserved trail capacity, while the
renderer throughput win came from batching. The production direction should be a
single GPU-owning batched renderer fed compact rows from CPU env workers or a
base-mode collector boundary, with one JAX runtime, stable shapes, explicit
capacity, and one host readback per batch. Stock LightZero can keep CPU envs and
a GPU learner for now; GPU observation should cross the process boundary in
batches, not by putting JAX inside every env worker.

## Current Working Recommendation

Keep training on `cpu_oracle` for now. Keep scalar `jax_gpu` as a carefully
sized base-mode regression canary. Do not scale it wide. The production GPU path
should be batched, single-owner, and capacity-explicit.
