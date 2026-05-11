# Environment Performance And Vectorization Plan

Status: active speed/vectorization note plus local toy-v0 scout data, an
isolated source-like NumPy vector prototype, fixture-backed array comparison,
and CPU/Modal policy-search evidence. Not a source-fidelity benchmark report.

## Short Answer

The core scalar/source slice is trustworthy enough to push speed now. Keep
source-fidelity fixtures as the guardrail, and optimize only measured paths that
can be compared back to verified behavior.

The API should stay stable across implementations: local Python reference first,
NumPy batch next, and production PyTorch/JAX/GPU backends only after profiling
shows they are needed. Small Modal/JAX/Mctx sweeps are active now as
speed/runtime probes, not as CurvyTron rollout proof. Each faster backend must
match the same scenario fixtures, common-trace projections, config hash inputs,
and trainer-facing reset/step contract.

Do not infer final simulator performance from the current smoke benchmarks. They
are useful regression checks for simplified code, not evidence that source-derived
movement, trails, collisions, observations, wrappers, or training throughput are
fast enough. The local toy-v0 smoke is in the rough 25k to 35k steps/s range on
the local macOS arm64 machine, depending on run shape and local noise. Treat that
as a baseline sanity check only.

The new NumPy vector prototype is also not a fidelity result. It is useful
because it keeps the GPU/vector lane warm around fixed-shape arrays, source-like
reverse player order, draw-cursor cadence, append-only body buffers, strict
overlap scans, and own-body latency masks without touching production env code.
Treat its timings as source-like synthetic inner-loop scout data only.

The speed scout's likely bottlenecks are trail/collision writes, observation
generation, and Python dict/object overhead. Add targeted benchmarks for those
paths before changing production code or data layout.

The policy/search lane has a separate local constraint: JAX and Mctx are not
installed in this checkout. Local policy/search timing is therefore limited to
NumPy shape stand-ins unless dependencies are added or the existing Modal Mctx
benchmark is used. Do not blur those categories.

## Principles

- Preserve source fidelity before changing data layout or control flow for speed.
- Keep trainer-facing imports on `curvyzero.env`; source runners and trace tools
  remain evidence machinery.
- Measure before optimizing, and separate compile/setup time from steady-state
  stepping.
- Prefer fixed-shape state and batch-friendly arrays, but do not force the narrow
  Python fidelity runner to become the production vectorized backend.
- Keep future backend shapes JAX-friendly without making the current backend JAX:
  pure array transition functions, explicit RNG state, fixed rollout lengths for
  scans, and no Python object mutation in any future compiled hot path.
- Keep Modal, queues, storage, and artifact writes outside per-tick and MCTS hot
  loops.
- Treat backend changes as equivalence work first and performance work second.

## Local Scout: 2026-05-09

Scope: current `curvyzero-v0` single-env smoke only, not CurvyTron fidelity.

Commands and full details are recorded in
[2026-05-09 Toy-v0 Performance Scout](../../experiments/environment/2026-05-09-toy-v0-performance-scout.md).

A follow-up background scout folds in the current trail cadence frontier and the
data shape needed for later vector/GPU-readiness:
[2026-05-09 Vectorization Background Scout](../../experiments/environment/2026-05-09-vectorization-background-scout.md).
The key constraint is simple: visual trail state and collision body state are
different. A future batch backend must represent printed trail cursor state,
materialized world bodies, per-body owner/number/radius, own-body latency, and
print-manager gap state explicitly before speed claims mean much.

A first source-aware fixed-shape schema draft now lives in
[Vector State Schema Draft](../../working/environment/vector_state_schema.md).
It is planning only, but it names the arrays and masks a future batch backend
should carry before NumPy, compiled CPU, JAX, PyTorch, or GPU work begins.

Sequential local runs with `PYTHONPATH=src python3 scripts/benchmark_env.py`
showed:

| Command shape | Steps | Steps/sec | Notes |
| --- | ---: | ---: | --- |
| `--episodes 100 --max-steps 500 --format json` | 2,370 | 31,131.4 | Short sanity run. |
| `--episodes 1000 --max-steps 500 --format json` | 23,423 | 34,962.5 | Best current local scout. |
| `--episodes 1000 --max-steps 2000 --format json` | 23,423 | 24,707.3 | Same step count, slower wall time; local noise or measurement side effect. |

Profiler scout on 500 episodes and 11,785 steps showed the main cost inside
`CurvyTronEnv.step`:

| Function | Cumulative time | Meaning |
| --- | ---: | --- |
| `core.py:65(step)` | 0.580s | Whole trainer-facing step call. |
| `core.py:96(_physics_tick)` | 0.438s | Movement, collision check, trail writes. |
| `core.py:156(_draw_segments)` | 0.327s | Segment/trail raster path. |
| `core.py:163(_mark_segment)` | 0.311s | Per-segment sampling and cell writes. |
| `numpy linspace` | 0.094s | Sampling cells along segments. |
| `core.py:169(_mark_cell)` | 0.055s | Python cell writes into occupancy. |
| `core.py:176(_observations)` | 0.049s | Tiny global-vector observation and copies. |
| `core.py:38(agents)` | 0.033s | Rebuilding agent ids repeatedly. |

Interpretation:

- The current toy env already stores state in arrays, which keeps the later path
  open.
- The current hot path is still Python-heavy: dict outputs, repeated agent-list
  construction, per-segment loops, `np.linspace`, cell writes, and per-agent
  observation copies.
- The benchmark scaffold now records a manifest and coarse external timers, but
  it does not claim exact movement, segment/trail, collision, observation, or
  wrapper/dict split timings.
- The next optimization-lane work should add source-level spans or isolated
  microbenchmarks before reporting those internal timing buckets.
- Do not optimize this toy-v0 implementation as production code. Use the scout to
  design better measurement and future state shape.

## Benchmark Scaffold Update: 2026-05-09

`scripts/benchmark_env.py` still measures only the toy-v0 single-env random-action
smoke. It now adds a JSON `manifest` with command argv, working directory,
`PYTHONPATH`, backend label, source-fidelity claim, workload, env config, agents,
observation shapes, git revision/status when available, Python/NumPy/platform
runtime, and timer definitions.

The measured timers remain intentionally coarse:

| Timer | Meaning |
| --- | --- |
| `reset` | Inclusive `env.reset`; includes state allocation, initial occupancy marks, and reset observations. |
| `action_sample` | Benchmark harness random joint-action dict construction. |
| `step` | Inclusive `env.step`; includes physics, observations, rewards, terminal/truncation/info dicts, and `StepResult`. |
| `loop_overhead` | Benchmark elapsed time not covered by the three coarse timers. |

The JSON also carries `requested_split_timer_status` so future readers do not
misread missing data as measured data. Movement, trail/collision writes,
observation generation, and wrapper/dict output overhead are marked
`not_measured`; reset/autoreset is `partially_measured` because the smoke only
times explicit resets and has no autoreset wrapper.

Small local smoke after the manifest update:

```sh
PYTHONPATH=src python3 scripts/benchmark_env.py --episodes 10 --max-steps 100 --format json
```

Result on the local macOS arm64 machine: 232 steps in 0.0071s, or about 32,861
steps/s. Coarse timers were reset 0.00018s, action sampling 0.00052s, step
0.00630s, and loop overhead 0.00006s. This is a repeatability smoke, not a speed
claim.

## Source-Fidelity Runner Surface Scaffold: 2026-05-09

`scripts/benchmark_source_fidelity.py` is a small, isolated benchmark scaffold
for the current source-fidelity scenario runner surface. It reads existing batch
manifests and scenario fixtures, runs the matching Python source-fidelity runner,
then times payload wrapping, common-trace projection, and JSON encoding. It does
not edit or instrument `source_runners.py`, scenario fixtures, tests, or source
fidelity behavior.

Default smoke command:

```sh
python3 scripts/benchmark_source_fidelity.py --repeat 1 --warmup 0 --format json
```

Small profiling smoke command:

```sh
python3 scripts/benchmark_source_fidelity.py --repeat 25 --warmup 3 --profile --profile-limit 15 --format plain
```

Local smoke over the current default source batches covered 20 scenarios and
500 measured scenario iterations in 0.169521s. Measured source-fidelity
runner-surface timers were: scenario load 0.041321s, inclusive runner calls
0.039538s, payload wrapping 0.010929s, common-trace projection 0.035631s, JSON
encoding 0.015010s, and loop overhead 0.027092s.

The attached `cProfile`/`pstats` summary is scoped to the measured scenario loop
only. It excludes warmup and manifest collection. Top cumulative entries were
the benchmark `_run_one` wrapper, `_timed_call`, `load_scenario`, the inclusive
runner-call lambda, `project_common_trace`, JSON encoding, and path resolution.
This is useful runner-surface evidence, but it is not production environment
stepping and still does not split source-internal movement, point insertion,
collision scan, or PrintManager work.

These numbers are real CurvyTron source-fidelity runner-surface measurements for
the selected fixtures. They are not movement, point-insertion, collision-scan, or
PrintManager split timings. The script records those requested buckets as
`not_measured` and covered only by the inclusive `runner_call` until source-level
spans or an isolated source-like microbenchmark exist.

## Source-Like NumPy Vector Prototype: 2026-05-09

`scripts/benchmark_vectorization_prototype.py` is an isolated prototype scaffold.
It imports NumPy but does not import `curvyzero` production environment modules,
source runners, tests, or scenario fixtures. It fills fixed-shape schema-like
arrays directly and times a synthetic inner loop.

What it covers:

- Fixed-shape structure-of-arrays state with `B` environments, `P` players, and
  `K` materialized bodies per environment.
- Reverse player update order over a leading batch axis.
- Fixed-step source-like movement from `source_move` values `-1/0/1`.
- Separate `printing`, visible trail last, hidden draw cursor, and materialized
  body buffer arrays.
- Normal point cadence from empty cursor or strict distance `> radius`.
- Normal body append before that player's collision scan.
- Full-buffer strict circle overlap with exact-tangent safety.
- Own-body latency mask by `live_body_num - stored_body_num <= trail_latency`;
  opponent bodies are immediate.
- Toy array observation construction and explicit body-buffer overflow counters.

What it is missing:

- Source fixture replay and common-trace equivalence.
- PrintManager post-collision distance updates, toggles, boundary body insertion,
  random print/hole distances, and visual clears.
- Borderless PrintManager wrap branch and normal wall death branch.
- Death-point insertion, same-tick scoring/events, terminal winner/draw
  lifecycle, bonuses/randomness, public wrappers, dict/info outputs, autoreset,
  and full browser/replay trail events.

Smoke commands:

```sh
python3 -m py_compile scripts/benchmark_vectorization_prototype.py
python3 scripts/benchmark_vectorization_prototype.py --batch 32 --players 3 --body-capacity 128 --steps 50 --warmup 5 --format json
python3 scripts/benchmark_vectorization_prototype.py --batch 128 --players 3 --body-capacity 512 --steps 200 --warmup 20 --format plain
```

Modal CPU smoke command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_vector_bench --kind cpu-smoke --run-id env-vector-YYYYMMDD-001
```

That Modal wrapper is intentionally coarse. It runs whole benchmark commands in
one CPU Modal Function, writes a run under `curvyzero-runs` at
`environment/vectorization/<run_id>/`, and returns exact artifact refs. It does
not call Modal from `env.step`, from the NumPy inner loop, or from MCTS/search.

Local macOS arm64 smoke for `B=128`, `P=3`, `K=512`, `steps=200`, `float64`,
straight movement:

| Metric | Value | Trust |
| --- | ---: | --- |
| Elapsed timed loop | 0.395907s | Source-like synthetic only. |
| Env steps/sec | 64,661.7 | Not source-faithful; no wrappers. |
| Player updates/sec | 193,985.1 | Source-like reverse player loop. |
| Fixed collision slots/sec | 119,434,570.2 | Full fixed-buffer scan, CPU NumPy. |
| Movement timer | 0.015243s | Synthetic movement bucket. |
| Normal point mask timer | 0.005501s | Draw-cursor cadence bucket. |
| Body append timer | 0.017655s | Append-only body buffer bucket. |
| Collision scan timer | 0.329231s | Dominant synthetic inner-loop cost. |
| Observation timer | 0.016879s | Toy array observation only. |
| Loop overhead | 0.011397s | Python/timer overhead. |

The run inserted 25,728 normal bodies, found 0 hits, used no overflow rows, and
allocated about 18,939 state bytes per environment. Built-in sanity checks passed
for strict tangent safety, strict overlap hit, own latency masking at delta 3,
own collision candidacy at delta 4, and immediate opponent candidacy.

Optional tensor backend detection on this machine found PyTorch available and
JAX unavailable. The script only detects those modules with `importlib`; neither
backend is imported or required, and no dependency was added.

Interpretation: this is now a useful scout for array shape, bucket boundaries,
and collision-scan scaling. It is not evidence that the source-fidelity runner or
future trainer-facing environment can step at these rates.

## Policy/Search Batch Stand-In: 2026-05-09

`scripts/benchmark_policy_search_batch_standin.py` is a local CPU NumPy scout for
the bridge between environment batches and future policy/search batches. It does
not import CurvyZero env code, JAX, Mctx, or PyTorch. It does not run MCTS. It is
only a fixed-shape batch and copy benchmark.

What it covers:

- Packs `[B_env, P, obs_dim]` observation arrays into padded
  `[B_policy, obs_dim]` ego rows.
- Carries active-row masks, env ids, and player ids.
- Runs a synthetic representation/prediction root model.
- Runs repeated recurrent-model-like matrix work plus fake visit counts.
- Builds action weights and root values, then maps selected actions back to
  `[B_env, P]`.
- Optionally copies inputs and search outputs through NumPy host arrays.

What it is missing:

- Real Mctx tree selection, backup, Gumbel sampling, or JAX compile behavior.
- GPU kernels, device transfer, CUDA memory behavior, or `.block_until_ready()`.
- Real CurvyTron observations, rewards, environment stepping, autoreset, replay,
  or training.

Local macOS arm64 smoke with JAX missing, Mctx missing, PyTorch detected but not
imported:

| Profile | Elapsed | Policy rows/sec | Main bucket | Hidden tree lower bound |
| --- | ---: | ---: | ---: | ---: |
| `B_env=256`, `P=2`, rows 512, hidden 64, sims 16 | 0.035542s | 288,109.2 | recurrent loop 0.033583s | 2,228,224 bytes |
| `B_env=512`, `P=3`, rows 1536, hidden 128, sims 32 | 0.133663s | 114,915.9 | recurrent loop 0.130137s | 25,952,256 bytes |
| `B_env=256`, `P=3`, rows 768, live 75%, hidden 64, sims 16 | 0.022509s | 341,203.2 | recurrent loop 0.021413s | 3,342,336 bytes |

Interpretation: this stand-in says the batch shape and hidden-memory scaling are
worth tracking. It does not say Mctx will be this fast, and it does not say GPU
transfer is cheap. The local NumPy copy buckets were small in these profiles,
but they are host-array copies only.

## Likely Stages

### Stage 0: Local Python Fidelity Runner

Purpose: make behavior auditable.

- Keep the source-derived Python runner close to the common-trace fixtures.
- Add mechanics one slice at a time: elapsed-ms movement, self-delay, opponent
  trails, print cadence, bonuses, and round lifecycle.
- Avoid layout rewrites unless they reduce ambiguity or make tests clearer.
- Benchmark only as a regression smoke, not as an optimization target.

Exit gate: source-derived fixtures cover enough state/events that a second
implementation has something meaningful to match.

### Stage 1: Stable Environment API Boundary

Purpose: let faster backends swap in without changing users.

- Define the public reset/step contract, config hash inputs, observation schema,
  action schema, reward semantics, terminal/truncation handling, and seed rules.
- Keep single-env wrappers thin over the same semantics expected from batch mode.
- Define `reset_many`/`step_many` shape expectations before implementing broad
  vectorization.
- Keep Gymnasium, PettingZoo, LightZero, Mctx, and debug adapters outside the
  inner environment state transition.

Exit gate: tests can assert the same scenarios through the public API and the
fidelity/common-trace path.

### Stage 2: NumPy Batch Backend

Purpose: remove per-environment Python dispatch while preserving semantics.

- Use structure-of-arrays state for positions, angles, alive flags, scores, ticks,
  done masks, RNG state, and occupancy/trail grids.
- Prefer fixed batch, player, map, and observation shapes per run profile.
- Use masks and autoreset rules instead of removing finished rows.
- Keep collision updates two-phase so same-tick deaths and trail writes stay
  explicit.
- Measure movement, collision, observation, reset/autoreset, wrapper overhead, and
  memory separately.

Exit gate: NumPy batch matches fidelity fixtures and has measured throughput and
memory profiles for representative scripted and random-action workloads.

### Stage 3: Targeted CPU Optimizers

Purpose: accelerate measured CPU bottlenecks without changing the public API.

- Try Numba or equivalent compiled kernels only for the measured hot sections,
  likely collision rasterization, occupancy updates, or observation extraction.
- Keep Python/NumPy fixtures as the equivalence oracle.
- Report first-call compile time separately from steady-state throughput.

Exit gate: profiler data shows the compiled path improves the real bottleneck and
does not make fixture debugging opaque.

### Stage 4: PyTorch, JAX, Or GPU Backends

Purpose: test integration-specific speedups after CPU evidence exists.

- Consider PyTorch tensor stepping only if a PyTorch training path wins and
  host-device transfer or observation movement is measured as costly.
- Consider a JAX-native environment only if JAX/Mctx self-play is blocked by CPU
  stepping or an all-JAX actor loop materially simplifies the measured system.
- Consider GPU-resident stepping only if CPU rollout workers cannot keep the
  model/search/trainer fed.
- Keep Mctx synthetic benchmarks separate from real environment backend choices;
  MuZero search does not require the source simulator to be JAX-native.

Exit gate: a backend-specific spike records correctness coverage, compile/setup
cost, steady-state throughput, memory, device utilization where relevant, and the
reason it beats the simpler CPU path.

## Future JAX/GPU Friendliness

This section is shape guidance for later. It is not approval to rewrite the env
in JAX or run a GPU simulator now.

- Design the eventual compiled transition as a pure function:
  `step_arrays(state, action, rng) -> new_state, obs, reward, done, info_arrays`.
  The current mutable Python env can remain the readable wrapper/reference path.
- Keep all hot data as arrays with fixed shapes per run profile: batch, players,
  map/grid size, max trail/body buffer length, observation shape, and action
  count.
- Keep trail cadence source-friendly: normal point insertion happens before
  collision, print-manager toggles happen after collision for survivors, and
  visual trail clearing does not remove already materialized collision bodies.
  Future state needs separate arrays for trail cursor/printing state and world
  body state.
- Carry explicit per-env RNG state. Future print gaps, trail holes, spawn/domain
  variation, and stochastic curricula should use state-carried keys or seed words,
  not hidden global RNG.
- Treat JAX `vmap` as a future batch shape check: one single-env array transition
  should be mappable over a leading batch axis.
- Treat JAX `lax.scan` as a future fixed-rollout shape check: the state carry
  must keep the same structure, shape, and dtype every step.
- Keep Mctx separate from the real simulator. Mctx's MuZero search uses learned
  model embeddings and batched JAX recurrent functions; the real CurvyTron rollout
  env does not have to become JAX-native immediately.
- Keep EnvPool-like CPU C++/threadpool execution as a later escape hatch if
  Python/NumPy/Numba stepping bottlenecks before GPU stepping is justified.
- Use gymnax and Brax as design-pattern references for explicit state, params,
  RNG, `jit`, `vmap`, and `scan`; do not add them as dependencies for CurvyTron.
- Keep equivalence gates ahead of speed claims: source scenarios, JS/Python common
  trace diffs, rules/config hash, observation schema hash, benchmark manifest, and
  exact command/results.

## Fixed-Shape State Draft: 2026-05-09

Detailed draft:
[Vector State Schema Draft](../../working/environment/vector_state_schema.md).

Key decisions:

- Use a structure-of-arrays state with fixed `B` environments, `P` players, and
  `K` materialized world-body slots per run profile. Optional spatial-index and
  debug-event buffers also need fixed caps.
- Keep source player order explicit. Batch over environments first, and keep a
  small reverse loop over players until same-frame point insertion, body
  collision, death events, print-manager toggles, and scoring are proven safe in
  a different order.
- Store positions, headings, alive masks, death ticks, scores, round scores,
  source moves, player ids/order, avatar radius/speed/angular velocity, and
  per-player trail latency as arrays. Convert public action ids and agent
  strings at the wrapper boundary.
- Keep visual trail state, hidden draw cursor state, and collision body state
  separate. The current source runners already need separate visible
  `lastTrailPoint` and hidden `isTimeToDraw` cursor behavior.
- Represent materialized bodies with append-only fixed buffers:
  body active mask, position, radius, owner, body number, insert tick/kind,
  write cursor, and monotonic world body count. Per player, carry both
  `body_count` for the next inserted point and `live_body_num` for collision
  latency checks.
- Preserve source collision rules: strict circle overlap, opponent bodies can
  collide immediately, and own bodies collide only when
  `live_body_num - body_num > trail_latency`.
- Carry print-manager arrays for active, remaining distance, last sampled
  position, toggle counters, and last toggle tick. Print-manager toggles happen
  after collision for survivors; print-to-hole clears visible/draw trail state
  only, not old bodies.
- Carry explicit per-env RNG key/counter/stream fields. Deterministic `0.5`
  random values can stay a benchmark mode, but hidden global RNG should not enter
  a future hot state transition.
- Use masks for env activity, alive players, body slots, insertion, collision
  candidates, print-manager updates, observations, and reset/autoreset rows.
  Overflow must be explicit. Do not silently wrap body slots because old bodies
  matter for visual-hole collision behavior.
- Keep fixed debug/event arrays optional but shaped: event count/type/player,
  compact payloads, event overflow, and counters for movement, point insertion,
  body scans, wall/body hits, print toggles, and wrapper output.

Semantics that still block production vectorization until fidelity is pinned:

- Elapsed-ms movement, source angular velocity, speed, and floating-point
  threshold behavior.
- Reverse player update order and same-frame point/body insertion.
- Strict normal point cadence: empty draw cursor draws, later cursor distance
  must be `> radius`.
- Separate visible trail state, hidden draw cursor state, and materialized world
  body state.
- Strict circle overlap, exact-tangent safety, opponent immediate collision, and
  own-body latency by body number.
- Print-manager active/distance/last-position state, no overshoot carryover,
  random print/hole distances, and post-collision toggle order.
- Death-point insertion, same-tick death/scoring order, round lifecycle, bonuses,
  final observation/reward contracts, and public reset/autoreset behavior.

Variable-length state that should stay outside the hot backend for now:

- Full source-runner body lists, full visual trail point lists, raw event
  objects, common-trace JSON, browser/replay messages, scenario files, public
  dict outputs, wrapper info objects, artifact manifests, and bonus stacks.
- The vector target should mirror the gameplay-critical parts with capped arrays
  and masks. If a cap overflows, fail or end the row explicitly; do not silently
  wrap old body slots.

Batching guidance:

- Batch now over environments for action conversion, heading/position movement,
  normal point masks, body appends, wall/wrap checks, body scans, own-latency
  masks, print-manager distance/toggle updates, observations, and row
  reset/autoreset.
- Keep the reverse player loop for source-visible ordering. This is especially
  important for same-frame higher-index point insertion that can kill a
  lower-index player in the same tick.
- Keep JS oracle runs, common-trace diffs, raw event formatting, browser/replay
  checks, artifact writes, Modal/storage work, scenario generation, and public
  dict/wrapper conversion CPU/offline until measurements justify moving any of
  them.

Next microbenchmark plan:

- Use `scripts/benchmark_vectorization_prototype.py` as the first isolated
  source-like timing scaffold. It fills schema-like arrays directly and records
  setup/warmup separately from steady-state stepping. Label every result as
  source-like microbench scaffolding unless it runs through the actual
  source-fidelity runner fixtures.
- Sweep `B` (`1`, `16`, `128`, `512`), `P` (`1`, `3`, `4`), and `K`
  (`0`, `64`, `512`, `4096`) with `float64` first. Try `float32` only after
  parity checks pass.
- Time movement, normal point mask, body insertion, collision scan, and toy array
  observation with the current scaffold. Add PrintManager update and public
  dict/wrapper output only after fidelity pins the missing source semantics.
- Include scripted workloads for long no-death movement, normal point every
  tick, below-radius no-point movement, same-frame point insertion, own-latency
  safe/kill cases, visual-hole safe crossing, visual-hole old-body collision,
  and print-to-hole/hole-to-print toggles.
- Record schema profile, command, seed policy, runtime/hardware labels,
  equivalence fixture status, body/event overflow counts, and memory per
  environment with every result.
- Use `scripts/benchmark_policy_search_batch_standin.py` to keep policy-row
  packing, padded-row cost, recurrent batch shape, action unmapping, and
  replay-target copy budgets visible while local JAX/Mctx is unavailable.

Next concrete GPU/vector path:

- Keep the NumPy prototype as the shape oracle, not a backend.
- Keep Modal CPU as a whole-job runner for repeatable source/vector benchmark
  artifacts and later profile-sharded sweeps.
- Add a scenario-to-array compiler under `scripts/` that can seed the prototype
  state from existing fidelity fixtures without importing it into production env
  code.
- Extend the prototype only where source fidelity is already pinned: wall/wrap
  masks after border fixtures, PrintManager post-collision toggles after gap and
  wrap fixtures, and scoring/events after lifecycle fixtures.
- Once a pure array step matches enough scripted source fixtures, make an
  optional JAX or PyTorch script-level spike with the same state fields:
  `vmap`/batch over `B`, `scan`/fixed rollout over time, keep the small reverse
  `P` loop, and compare CPU first before trying GPU.
- Replace the NumPy policy/search stand-in with real Mctx/JAX timing when the
  dependency/runtime is available. The real report must separate compile time,
  steady-state runtime, device transfer, decisions/sec, simulations/sec, output
  validity, and memory.
- Put future GPU runs on batched env/policy/MCTS work that stays inside one
  remote process: tensor rollouts, batched policy/recurrent inference, MCTS
  batches, or a combined env+policy+search benchmark after the separate pieces
  have CPU artifacts.
- Only consider spatial indexing, compiled CPU kernels, or GPU-resident stepping
  after the full-buffer `K` scan sweeps show the collision bucket is still the
  measured bottleneck at realistic rollout sizes.

External source pointers:

- [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
- [JAX automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html)
- [JAX lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [jax.random](https://docs.jax.dev/en/latest/jax.random.html)
- [JAX pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html)
- [Mctx README](https://github.com/google-deepmind/mctx)
- [EnvPool NeurIPS 2022 abstract](https://papers.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html)
- [EnvPool docs](https://envpool.readthedocs.io/)
- [gymnax README](https://github.com/RobertTLange/gymnax)
- [Brax README](https://github.com/google/brax)

## Benchmark TODOs

- Current local smoke scaffold: `scripts/benchmark_env.py` reports fixed-seed
  single-env `CurvyTronEnv` step throughput in plain text or JSON, with coarse
  reset/action-sampling/step timers and a manifest. Treat it as simplified
  toy-v0 regression evidence, not an optimization target. The 2026-05-09 local
  scout recorded roughly 25k to 35k toy-v0 steps/s and a cProfile signal that
  segment/trail rasterization is the largest current cost.
- [x] Add a local benchmark manifest format that records command, git revision or
      source snapshot availability, config hash, backend, seed, batch size,
      player count, map size, observation schema, action repeat, scenario set,
      and local machine/runtime shape. Modal shape still belongs in Modal runs.
- [x] Add source-level instrumentation or isolated microbenchmarks before
      reporting movement, trail/collision, scoring/events, observation,
      reset/autoreset, wrapper/adaptor overhead, or artifact-writing split
      timers. First isolated source-like scaffold added; scoring/events,
      reset/autoreset, wrappers, and artifact-writing remain unmeasured.
- [x] Add a source-aware runner-surface scaffold:
      `scripts/benchmark_source_fidelity.py` times existing source-fidelity
      scenario runner calls, payload wrapping, common-trace projection, and JSON
      encoding while marking movement, point insertion, collision scan, and
      PrintManager internals as not measured.
- [x] Add an opt-in local `cProfile`/`pstats` summary to
      `scripts/benchmark_source_fidelity.py`; it profiles the measured
      source-fidelity runner-surface loop only and excludes warmup plus manifest
      collection.
- [x] Add a repeatable toy-v0 smoke command around the current ~33.5k steps/s
      result, labeled as simplified-environment throughput.
- [ ] Add isolated microbenchmarks for trail/collision grid writes, observation
      generation, and Python dict/object allocation or lookup overhead.
- [x] Add a CPU NumPy policy/search batch stand-in for fixed policy-row packing,
      recurrent model-like work, action unmapping, padded rows, and host-array
      copy buckets. It is not Mctx/GPU evidence.
- [ ] Replace the policy/search stand-in with a real JAX/Mctx benchmark when the
      dependency/runtime is available locally or through Modal.
- [ ] Add a source-aware trail cadence microbenchmark that times movement,
      `isTimeToDraw`, point insertion, body materialization, own-body latency
      checks, print-manager gap toggles, and circle collision lookup separately.
      Current NumPy prototype covers movement, draw-cursor cadence, body
      materialization, own-body latency masks, and circle scan; PrintManager gap
      toggles are still missing.
- [x] Draft the fixed-shape batch state schema for materialized bodies, visual
      trail cursor state, print-manager state, body counters, masks, and explicit
      body-buffer overflow policy. See
      [Vector State Schema Draft](../../working/environment/vector_state_schema.md).
- [ ] Complete the source-aware timing scaffold from the schema draft: the first
      NumPy prototype has movement, normal point mask, body insertion, collision
      scan, and toy observation timers. Print-manager update and dict/wrapper
      output still need source-pinned semantics before they become useful.
- [ ] Add scripted microbenchmarks for normal-wall death, source borderless wrap,
      self-trail delay, opponent-trail collision, head-head/same-frame death, and
      long no-death movement.
- [ ] Add random-action smoke benchmarks only after the scripted set exists, so
      throughput changes can be interpreted against known mechanics.
- [ ] Report memory per rollout for state arrays, occupancy grids, observations,
      scratch buffers, and replay staging.
- [ ] Separate local CPU, Modal CPU, and Modal GPU results; do not compare them
      without recording hardware and dependency versions.
- [ ] Track compile/import/setup time separately for Numba, JAX, PyTorch, Modal
      images, and any GPU warmup.
- [ ] Add equivalence checks that run before each benchmark profile and fail fast
      if the backend no longer matches the fidelity fixtures.
- [x] Store the first local toy-v0 scout under `docs/experiments/environment/`
      with exact commands/results.
- [ ] Store future benchmark outputs under `docs/experiments/` or an artifact
      directory with a short interpretation note, not only terminal output.

## Risks

- Premature vectorization can freeze the wrong semantics before body collisions,
  trail gaps, bonuses, and round lifecycle are source-verified.
- A stable-looking `step_many` API can still hide policy choices such as wall
  inclusivity, death-trail writes, tie handling, autoreset, and observation timing.
- NumPy speedups may be erased by Python raster loops, per-agent observation work,
  or variable-length trail history.
- Occupancy grids can diverge from source geometry if resolution, trail thickness,
  and swept movement are not explicit fixture inputs.
- GPU stepping can compete with model/search work for bandwidth and may be slower
  for tiny branch-heavy kernels.
- JAX/PyTorch backends can pull environment design toward framework convenience
  instead of source behavior.
- Benchmark noise can lead to false confidence unless setup time, warmup, batch
  size, hardware, and scenario mix are recorded.

## Links

- Broader synthesis: [Simulator Performance And Vectorization](../performance_vectorization.md)
- Local scout: [2026-05-09 Toy-v0 Performance Scout](../../experiments/environment/2026-05-09-toy-v0-performance-scout.md)
- State schema draft: [Vector State Schema Draft](../../working/environment/vector_state_schema.md)
- Modal/vector merge plan: [Modal Vectorization Integration Plan](../../working/environment/modal_vectorization_integration_plan.md)
- Fidelity loop: [Environment Iteration Loop Review](../../working/environment/iteration_loop_review.md)
- Stable environment map: [Environment Fidelity Lane](../../design/environment/README.md)
- Modal hot-loop boundary: [Modal Architecture](../../design/modal_architecture.md)
