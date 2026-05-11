# Modal Vectorization Integration Plan

Status: architecture plan plus first coarse CPU Modal wrapper
Date: 2026-05-09

Scope: the source-fidelity lane remains the semantic authority. This plan covers
how the isolated vector/GPU prototype lane should merge with fidelity evidence
and become coarse Modal benchmark jobs. It does not authorize production
environment rewrites.

## Top Decision

Do not make Modal, JAX, PyTorch, or GPU execution the environment authority.

The merge should be:

```text
verified source fixtures
  -> fixture-to-array seeding
  -> local array equivalence and B/P/K sweeps
  -> Modal CPU smoke and Modal CPU sweep artifacts
  -> optional tensor backend smoke
  -> optional Modal GPU tensor benchmark
```

Modal is the job and artifact layer. It should run whole benchmark, sweep, or
equivalence jobs. It should not be called per environment step, per player
update, per collision scan, per MCTS node, or per trace row.

GPU is justified only for a tensor/JAX/PyTorch benchmark that keeps work on the
device long enough to measure steady-state throughput. GPU is not justified for
the current NumPy prototype, source-fidelity runners, JS oracle probes, JSON
trace diffs, or public wrapper/object allocation.

## Current Split Critique

The current split is directionally right, but the interfaces between lanes need
to be tighter.

### Fidelity Runners

Strengths:

- They preserve source behavior as readable evidence.
- Current batches cover kinematics, border, body, PrintManager, trail,
  trail-gap, and first collision-order behavior.
- `scripts/benchmark_source_fidelity.py` times the runner surface without
  changing source semantics.

Problems:

- Runner-surface timing is not inner-loop timing. Movement, point insertion,
  collision scan, and PrintManager buckets are still nested inside
  `runner_call`.
- The fidelity runners output common traces, not fixed-shape arrays. A vector
  lane still needs a deliberate fixture-to-array compiler.
- The source runner should not become the vector backend. It is the oracle edge,
  not the optimized implementation.

Architecture response:

- Keep source runners local/readable and fixture-first.
- Add an offline seeding layer that reads only verified scenarios and produces
  fixed array state plus action scripts for the vector prototype.
- Make every vector benchmark record the fidelity fixture status it matched, or
  explicitly say `equivalence_not_run`.

### Source-Like Vector Prototype

Strengths:

- `scripts/benchmark_vectorization_prototype.py` already exercises useful
  fixed-shape ideas: `B/P/K` state, reverse player order, strict overlap,
  append-only body buffers, own-body latency masks, and separate draw cursor vs
  visible trail state.
- It times the collision bucket separately enough to guide future sweeps.
- It is isolated from production env code, which is exactly right at this stage.

Problems:

- It is not fixture seeded and does not compare to common traces.
- For that isolated prototype, missing source semantics are not small: broad
  timer/autoreset behavior, natural trail-gap cases, broader border/wrap
  branches, death-point insertion, same-tick scoring, round lifecycle, bonuses,
  wrappers, and autoreset are still outside its coverage. The fixture
  comparator has narrower support than this synthetic prototype.
- CPU NumPy timing is useful for array shape, but it says little about GPU
  usefulness. A GPU lane needs a tensor implementation, not remote NumPy.

Architecture response:

- Keep the NumPy prototype as a shape and benchmark scaffold, not a backend.
- Extend it only where fidelity is verified. For example, add wall/wrap after
  border fixtures, PrintManager gap toggles after gap fixtures, and death/scoring
  after collision-order/lifecycle fixtures.
- Add JAX or PyTorch only as a separate script-level spike with the same state
  fields. Run CPU first, then GPU only when the same code can avoid host/device
  churn.

### Modal Boundary

Strengths:

- Existing docs already encode the hot-loop rule.
- Existing Modal smoke files prove Python remote execution, simple GPU
  visibility, Mctx/JAX dependency smoke, and Volume artifact writes.
- The `curvyzero-runs` Volume pattern is already present.

Problems:

- The vector lane has a first named Modal app/function shape, but fixture
  equivalence and sweep functions are not standardized yet.
- Artifact layout for vector sweeps is not standardized.
- There is no explicit gate saying Modal CPU comes before Modal GPU for
  environment vectorization.

Architecture response:

- Keep `curvyzero-env-vector-bench` as a coarse-function app only.
- Write benchmark outputs as coarse shard files under immutable run/attempt
  paths, then publish manifests last.
- Use Modal CPU for reproducible remote sweeps before GPU. GPU should enter only
  through a tensor/JAX/PyTorch function with `@app.function(gpu=...)`.

## Modal Shape

Use two environment apps, not one large mixed-purpose app:

| App | Purpose | Status |
| --- | --- | --- |
| `curvyzero-env-fidelity` | JS/Python source trace batches and first-mismatch artifacts. | Already designed in `docs/design/environment/modal_fidelity_jobs.md`; full batch job is future. |
| `curvyzero-env-vector-bench` | Coarse CPU source/vector benchmark smoke now; fixture-seeded array equivalence, broader sweeps, and optional tensor/GPU smokes later. | First CPU smoke wrapper exists in `src/curvyzero/infra/modal/environment_vector_bench.py`. |

Current module:

```text
src/curvyzero/infra/modal/environment_vector_bench.py
```

Keep scripts under `scripts/` as the local source of truth for benchmark logic
until a real reusable package boundary appears. The Modal file is a thin
wrapper: it copies source, scenarios, and scripts into an image, runs whole
commands inside the container, writes artifacts, commits the Volume, and returns
a compact summary. It must stay outside env steps and MCTS loops.

### CPU Functions

| Function | Runs | Inputs | Outputs |
| --- | --- | --- | --- |
| `cpu_smoke` | One `benchmark_source_fidelity.py` command and one `benchmark_vectorization_prototype.py` command. | Repeat/warmup, one vector `B/P/K/steps/dtype/action` profile, run id. | One attempt directory with source summary, vector profile shard, sweep summary, complete file, and manifest. |
| `source_fidelity_benchmark_batch` | `scripts/benchmark_source_fidelity.py` over selected batch manifests. | Scenario batch refs, repeat/warmup, run id. | One runner-surface timing JSON plus manifest. |
| `vector_numpy_sweep` | `scripts/benchmark_vectorization_prototype.py` for many `B/P/K/dtype/action` profiles. | Sweep spec JSON. | One result shard per profile group plus sweep summary. |
| `fixture_array_equivalence_batch` | Future remote wrapper around the local fixture-to-array seeding and comparison scripts. | Verified scenario refs, backend/profile, fixed horizon. | Equivalence report, unsupported-semantics list, first mismatch refs. |
| `package_vector_artifacts` | Manifest validation and optional export packing. | Run id and artifact refs. | Small export manifest, not a full tree download. |

All of these are CPU functions. They should use `Function.map` or deployed
function calls only at the profile-shard or scenario-batch level.

### GPU Functions

| Function | Runs | Inputs | Outputs |
| --- | --- | --- | --- |
| `tensor_backend_smoke` | JAX or PyTorch import/device check plus one tiny array rollout. | Backend, dtype, tiny profile, GPU type. | Device info, compile/import time, first/second run timing. |
| `tensor_vector_sweep` | Future JAX/PyTorch vector prototype over large `B/K` profiles. | Sweep spec, backend, GPU type. | Compile time, steady-state timing, memory/device utilization notes, equivalence status. |
| `batched_policy_or_mcts_smoke` | Future batched policy/recurrent inference or MCTS/search benchmark, not environment stepping by remote call. | Backend, batch/search profile, checkpoint or dummy model, GPU type. | Import/compile/setup time, first/steady timing, device info, artifact refs. |

Use Modal GPU only through `@app.function(gpu=...)`. Modal's current GPU docs
list GPU types such as `T4`, `L4`, `A10`, `L40S`, `A100`, `H100`, `H200`, and
`B200`; multi-GPU is requested with suffixes like `:n`, and larger GPU counts
can increase wait times. For this lane, start with one `L4` or `T4` smoke, then
one exact GPU type for benchmark stability. Do not use multi-GPU until a trainer
or search benchmark, not the environment prototype, proves it needs it.

Future GPU jobs should be for batched work that can stay local to one remote
process: tensor env rollouts, batched policy/recurrent inference, MCTS/search
batches such as Mctx experiments, or a combined env+policy+search benchmark.
They should write the same coarse artifacts as CPU jobs. They should not make
one Modal call per environment step, model call, tree node, or trace row.

## Bridge From Synthetic Mctx To CurvyTron Tensors

The current Mctx benchmark is useful because it proves the Modal/JAX/Mctx
runtime path, not because it resembles CurvyTron yet. The next bridge should be
plain and mechanical:

Current bridge status: `mctx_synthetic_benchmark.py` has a `curvytron_debug`
mode that builds synthetic host `obs[B,P,9]`, plus a less-fake mode that consumes
fixture-seeded CPU debug-packer output before timing device-resident Mctx search.
The synthetic Modal L4 run on 2026-05-09 reported `B=4`, `P=2`, `obs_dim=9`,
`num_simulations=4`, `8` live ego rows, `8` search roots, host observation setup
`0.0003469869999999098s`, host-to-device placement `0.21319387700000014s`, and
steady search median `0.0024705955000010604s`. The fixture-seeded run used real
debug-packer output but still remains boundary evidence only: no real env rollout
loop, no learned dynamics, no final reward, no replay, no trainer, and no
source-fidelity claim.

1. State arrays: start from the fixture-seeded structure-of-arrays state
   already used by the vector lane. Keep shapes explicit, such as `B`, `P`, and
   `K`, with alive/done masks, body arrays, trail/print-manager fields, and a
   per-row RNG stream. First gate: supported fixture states still compare
   against source traces before any search code sees them.
2. Observation arrays: project state arrays into fixed observation tensors and
   action masks. The first version can be a compact numeric observation, not the
   final public wrapper observation. It should record its shape, dtype, player
   axis, and exactly which source fields are included or omitted.
3. Model function: replace the synthetic linear toy model with a JAX model API
   that accepts observation arrays and returns MuZero-style `prior_logits`,
   `value`, and an embedding. Start with deterministic dummy weights in the real
   observation shape, then swap in a checkpoint only after the shape smoke is
   stable.
4. Mctx recurrent function: wrap the model dynamics as
   `recurrent_fn(params, rng_key, action, embedding)` and return Mctx
   `reward`, `discount`, `prior_logits`, `value`, and `next_embedding`. This is
   where the synthetic benchmark becomes a real shape benchmark: Mctx should see
   the same batch/action dimensions that self-play will use.
5. Batched env step: after Mctx returns batched actions, call the local
   `step_arrays` transition for the whole batch, produce the next observation,
   reward, done, and info/event arrays, then loop inside one process. Modal can
   run this whole loop as one job or one profile shard; it must not be called
   once per environment step.

Only the last version is a self-play loop. The intermediate bridge benchmarks
should stay labeled by what they actually measure: observation packing, model
shape, Mctx search, or batched env stepping.

## No Per-Step Modal Calls

The allowed shape is:

```text
local client submits one sweep spec
  Modal function starts
  load scenarios and profile shards
  run local Python loops inside the container
  write coarse JSON/NPZ artifacts
  commit Volume
  return compact summary and exact refs
```

The forbidden shape is:

```text
for each step:
  call Modal
for each player:
  call Modal
for each collision row:
  call Modal
```

Do not use Modal Queues, Dicts, `.remote()`, web endpoints, or deployed function
lookups inside `env.step()`, the vector step function, the source runner tick
loop, MCTS search, model inference batches, or trace diff loops. Use Modal to
parallelize whole scenario batches or profile shards.

## Image Layering

Modal Images define the container environment your code runs in. Build them
deliberately and keep frequently changing project code late in the layer chain.
The official Modal Images guide recommends defining environments with
`modal.Image`, pinning dependencies tightly, and using local file/source adders
when the container needs project code.

Use separate images by dependency class:

| Image | Dependencies | Uses | Do not include |
| --- | --- | --- | --- |
| `env_bench_cpu_image` | Python 3.11, NumPy, project source, scenarios, scripts. | Source-fidelity surface timing and NumPy sweeps. | JAX, PyTorch, CUDA, Node unless needed. |
| `env_fidelity_reference_image` | Node plus reference oracle files. | JS oracle trace batches only. | Tensor libraries and training deps. |
| `env_vector_numba_image` | CPU base plus pinned Numba or similar. | Later compiled CPU spike after measured need. | GPU dependencies. |
| `env_tensor_cpu_image` | CPU base plus pinned JAX CPU or PyTorch CPU. | Tensor backend CPU smoke and compile/shape checks. | CUDA packages. |
| `env_tensor_gpu_image` | Pinned CUDA-compatible JAX or PyTorch stack. | GPU smoke and tensor sweep only. | JS oracle, browser, broad dev tooling. |

Layer order:

1. Base Python/OS image.
2. Stable system packages.
3. Stable Python packages with exact versions.
4. Optional framework packages in separate tensor images.
5. Local `src`, `scripts`, selected `scenarios`, and minimal docs needed for
   provenance.

Do not copy replay trees, old experiment outputs, browser builds, or generated
benchmark artifacts into images. Mount the Volume for artifacts.

## Storage And Artifacts

Use the existing `curvyzero-runs` Volume mounted at `/runs` for active benchmark
artifacts. Modal's Volume reference says containers must commit changes for
other containers to see them and reload to see external commits; it also warns
against concurrent writes to the same file. The Volume guide also notes file
count and traversal limits, so this lane should prefer coarse shard files over
one file per tick, step, or trace row.

Proposed layout:

```text
/runs/environment/vectorization/<run_id>/
  manifest.json
  spec.json
  attempts/
    <attempt_id>/
      attempt.json
      source_fidelity/
        batch-000.summary.json
      vector_numpy/
        profile-shard-000.json
        profile-shard-001.json
        sweep_summary.json
      fixture_array/
        scenario-shard-000.equivalence.json
        first_mismatch.jsonl
      tensor_cpu/
        smoke.json
      tensor_gpu/
        gpu-smoke-L4.json
        profile-shard-000.json
      compare/
        source-vs-vector-summary.json
      complete.json
```

Rules:

- Write immutable attempt and shard paths.
- Each shard file should contain many profile or scenario results.
- Use compressed `.npz` only for array snapshots that are genuinely needed for a
  mismatch; do not save every step by default.
- Write payloads first, then shard summaries, then `complete.json`, then the run
  manifest.
- Commit the Volume after a completed shard or completed attempt.
- Return only run id, attempt id, pass/fail counts, and exact artifact refs.
- Never let multiple workers write the same manifest or shard path. Use
  per-shard paths and a single finalizer.

## Inputs And Outputs

### Sweep Spec

Use a small JSON spec for both local and Modal runs:

```json
{
  "schema": "curvyzero_env_vector_sweep_spec/v1",
  "run_id": "env-vector-YYYYMMDD-001",
  "scenario_batches": [
    "scenarios/environment/source_kinematics_batch.json",
    "scenarios/environment/source_body_canary_batch.json"
  ],
  "profiles": [
    {"batch": 128, "players": 3, "body_capacity": 512, "steps": 200, "dtype": "float64"}
  ],
  "warmup": 20,
  "repeat": 1,
  "backend": "numpy-prototype",
  "equivalence_required": false
}
```

### Result Summary

Every benchmark result should include:

- schema version and benchmark id
- code ref, git status summary, Modal app/function/image label when remote
- backend and dependency versions
- local/Modal/GPU hardware label
- scenario batch refs and fixture status
- profile fields: `B`, `P`, `K`, dtype, steps, warmup, action pattern
- setup/import/compile time separated from steady-state timing
- timing buckets, memory per environment, overflow counts, and error counters
- equivalence status: `not_run`, `passed`, `failed`, or `unsupported`
- exact artifact refs for summaries and first mismatch files

## Fidelity Comparison Contract

The vector lane merges with fidelity through fixtures, not through hand-copied
source behavior.

1. Read a verified scenario batch manifest.
2. Run the current source-fidelity runner and common-trace projection.
3. Compile the scenario into fixed arrays:
   positions, heading, alive mask, body buffers, draw cursor, visible trail
   state, PrintManager fields, body counters, RNG state if needed, and fixed
   action script.
4. Run the array transition for the same fixed horizon.
5. Project debug arrays back into the common-trace fields that the backend
   claims to support.
6. Compare exact event/state fields and numeric tolerances.
7. If a source field is not represented in arrays, mark the scenario
   `unsupported`, not `passed`.

Do not seed from unverified mechanics. If a fixture is only source-read or
JS-pinned, it can guide future design, but it should not be a vector equivalence
gate yet.

The first equivalence-set sketch was intentionally narrow. Treat this list as
shape guidance, not current status:

- fixed-60Hz straight and turn kinematics
- normal wall basics
- stored-body strict overlap and own-body latency
- normal trail cadence
- verified trail-gap/body canaries
- first collision-order death-point fixture once Python/common-trace parity is
  promoted

## Benchmark Flow

Local first:

```sh
python3 scripts/benchmark_source_fidelity.py --repeat 1 --warmup 0 --format json
python3 scripts/benchmark_vectorization_prototype.py --batch 128 --players 3 --body-capacity 512 --steps 200 --warmup 20 --format json
```

Current local equivalence smoke:

```sh
python3 scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_body_canary_batch.json \
  scenarios/environment/source_borderless_wrap_step.json \
  scenarios/environment/source_normal_wall_death_step.json \
  scenarios/environment/source_print_manager_batch.json \
  scenarios/environment/source_trail_gap_batch.json \
  --format json
```

This exists now. It seeds each fixture through
`scripts/seed_vector_state_from_fixtures.py`, runs one narrow NumPy array tick,
and compares supported state and event fields to the matching Python source
runner. The current mixed smoke protects the body batch plus simple borderless
wrap, normal-wall death, the full source PrintManager batch, and the four
forced trail-gap fixtures. It emits fixed event rows for the currently
supported event types, including the narrow delayed-start timer/start fixture.
It does not write `.npz` artifacts yet, cover broader natural trail-gap
variants, implement broad reset/timer/autoreset, or prove production batching.

Modal CPU smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_vector_bench --kind cpu-smoke --run-id env-vector-YYYYMMDD-001
```

This exists now. It runs one whole source-fidelity benchmark command and one
whole NumPy vector prototype command inside a single CPU Modal Function, then
writes:

```text
/runs/environment/vectorization/<run_id>/
  spec.json
  manifest.json
  attempts/<attempt_id>/
    attempt.json
    source_fidelity/batch-000.summary.json
    vector_numpy/profile-shard-000.json
    vector_numpy/sweep_summary.json
    complete.json
```

Tiny remote smoke verified on 2026-05-09:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_vector_bench \
  --kind cpu-smoke \
  --run-id env-vector-cpu-smoke-20260509-worker-c \
  --attempt-id attempt-000001 \
  --source-repeat 1 \
  --source-warmup 0 \
  --vector-batch 4 \
  --vector-players 2 \
  --vector-body-capacity 16 \
  --vector-steps 5 \
  --vector-warmup 0
```

Result: completed. It covered 20 source scenarios and wrote the complete file at
`environment/vectorization/env-vector-cpu-smoke-20260509-worker-c/attempts/attempt-000001/complete.json`.
The tiny vector profile reported about 18.5k synthetic env steps/s, but that is
only a plumbing smoke and not a speed claim.

Future Modal CPU sweep:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_vector_bench --kind cpu-sweep --spec docs/experiments/environment/env-vector-sweep-smoke.json --run-id env-vector-YYYYMMDD-002
```

Future Modal GPU smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_vector_bench --kind gpu-smoke --backend jax --gpu L4 --run-id env-vector-YYYYMMDD-003
```

Deploy only after `modal run` is boring:

```sh
uv run --extra modal modal deploy -m curvyzero.infra.modal.environment_vector_bench
uv run python -m scripts.modal_run_env_vector_bench --spec docs/experiments/environment/env-vector-sweep-smoke.json --run-id env-vector-YYYYMMDD-004
```

The local client should call a deployed Function by name and fetch exact refs
only. It should not download the entire run tree by default.

## What Stays Local

Keep these local until they are boring:

- new source claim writing and scenario authoring
- single-scenario debug loops
- first mismatch archaeology
- fixture-to-array compiler development
- source/vector projection code
- small `B/P/K` smoke sweeps
- docs, interpretation notes, and benchmark spec drafting

Local work is cheaper and faster while semantics are still moving.

## What Runs On Modal CPU

Run these on Modal CPU:

- reproducible source-fidelity runner-surface benchmark batches
- NumPy prototype sweeps over many `B/P/K` profiles
- fixture-to-array equivalence batches once the local compiler is wrapped remotely
- artifact packaging and manifest validation
- broad regression sweeps before using numbers in design decisions

Modal CPU proves the container environment, command contract, and artifact
layout. It also gives a clean separation from local macOS noise before GPU is in
the conversation.

## What Runs On Modal GPU

Run these on Modal GPU only after the CPU path is boring:

- JAX or PyTorch import/device smoke with pinned dependencies
- tensor array transition compile/first-run/steady-state timing
- large `B/K` tensor sweeps that keep rollout work on the device
- batched policy/recurrent inference when the trainer has a real model path
- batched MCTS/search experiments when the search code can keep batches in the
  remote process
- combined env+policy+MCTS benchmarks only after the separate pieces have CPU
  artifacts and a reason to share one GPU job
- Mctx or model/search benchmarks that actually need accelerator hardware

Do not run source-fidelity JS/Python runners, JSON trace diffs, public wrapper
conversion, or pure NumPy prototypes on GPU. Do not use multi-GPU for the
environment vector lane unless a later trainer/search benchmark has already
shown a single GPU is the bottleneck.

## Merge Path

### Phase 1: Fixture-To-Array Seeding

`scripts/seed_vector_state_from_fixtures.py` is the first local tool for this
phase. It reads scenario files or batch manifests and emits JSON array seeds for
the vector lane. It keeps one `B=1` seed per fixture and records which fixtures
are JS-pinned or Python-runner-verified.

`scripts/compare_vector_arrays_to_fidelity.py` now takes the first real step
after seeding. It supports the current twenty fixture-backed transitions: six
three-player body-canary fixtures, the simple two-player borderless wrap
fixture, the two-player normal-wall death fixture, all eight source
PrintManager batch fixtures, and the four forced trail-gap fixtures. It runs
one array tick, with a narrow pre-step path for the delayed-start fixture, and
compares projected state fields plus compact event rows to the common trace.

This is not batched self-play yet. It does not call a policy, build
observations, produce rewards, autoreset rows, run a learned model, hold an MCTS
tree, implement broad reset/timer/autoreset, or handle broader natural
trail-gap event detail. It is only a first equivalence edge from fixture JSON to
vector-shaped state to one checked state transition.

Before this phase can feed many self-play games, the vector lane still needs:

- a full self-play-ready array transition with source-order movement, trail,
  body, border, print-manager, death, forced gap, and scoring updates
- reset/autoreset arrays for many live rows
- explicit per-row RNG streams
- observation and reward arrays that are stable enough for training
- a policy/model call boundary that batches rows without per-agent dict work
- fixed search-tree arrays for MuZero/MCTS roots, children, values, visits,
  rewards, recurrent states, and legal-action masks
- common-trace comparison for broader wall/wrap, PrintManager, natural trail-gap,
  terminal scoring, and event-buffer outputs

Do not make Modal or a trainer API depend on this seed JSON as a final contract.
Use it to get equivalence evidence first.

Exit gate: done for the current supported fixture-seeded state/event cases. The
full PrintManager vector batch is part of that supported fixture set. Keep
timing clearly separated from source trace comparison, with unsupported
scenarios still separated from failures.

### Phase 2: Local Equivalence And Sweeps

Run the seeded prototype locally against the narrow verified fixture set. Keep
the existing synthetic `B/P/K` sweeps, but do not mix synthetic speed numbers
with fixture-equivalence numbers.

Exit gate: local report has both `equivalence` and `synthetic_sweep` sections,
with unsupported scenarios separated from failures.

### Phase 3: Modal CPU Smoke

Done for the current coarse smoke in
`src/curvyzero/infra/modal/environment_vector_bench.py`. It wraps the local
scripts in one Modal CPU Function, runs one source benchmark and one tiny vector
sweep, writes one attempt directory, commits the Volume, and returns exact refs.

This does not prove source/vector equivalence. It only proves the remote command
contract and artifact layout for the speed lane.

Exit gate: returned summary points to a complete manifest and one shard summary.

### Phase 4: Modal CPU Sweep

Use Modal parallelism only over profile shards or scenario batches. Keep each
worker independent and let a single finalizer publish the run manifest.

Exit gate: broad CPU sweep records hardware/runtime, dependency versions, setup
time, steady-state timing, memory, overflow counters, and equivalence status.

### Phase 5: Modal GPU Smoke

The first concrete GPU smoke now exists at
`src/curvyzero/infra/modal/mctx_gpu_dependency_smoke.py`.

Run it from the repository root:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

It uses pinned JAX/Mctx dependencies, requests one `L4` or `T4`, records import
status, package versions, JAX backend/devices, `nvidia-smi`, compile plus first
run, and a second steady-state run for one tiny `mctx.gumbel_muzero_policy`
search. Full pass/fail details live in
[Modal Mctx GPU Smoke Runbook](modal_mctx_gpu_smoke_runbook.md).

Exit gate: one tiny tensor rollout works on one GPU and prints a JSON report.
Worker U ran this on 2026-05-09. It passed on Modal app
`ap-k2iRqzGbvLshqsZW8jDVav` with `ok: true`, JAX backend `gpu`, device
`cuda:0`, `mctx==0.0.6`, `jax==0.7.0`, `jaxlib==0.7.0`, `numpy==2.4.4`, and
an NVIDIA L4. The tiny search recorded `3.364952897s` compile plus first run
and `0.0025687180000000254s` for the second run. This is runtime/dependency
evidence only, not environment speed or source-fidelity proof. Cost note: the
run did allocate a remote Modal GPU briefly, so normal Modal billing may apply.

### Phase 6: Tensor Prototype Sweep

Port only the array transition pieces that have fixture equivalence. Run CPU
tensor first. Then run Modal GPU on large profiles where transfer overhead is
controlled.

Exit gate: a written comparison shows whether GPU beats Modal CPU for the same
supported semantics and explains why.

## Overengineering Traps

Do not move these to Modal yet:

- per-step environment calls
- source-fidelity runner internals
- trace diff inner loops
- JS oracle execution per scenario step
- scenario authoring and mismatch archaeology
- public browser/server hosting
- screenshot/video checks as gameplay evidence
- Queue/Dict hot-loop coordination
- one file per step, event, or profile sample
- CloudBucketMount archives before Volume pressure exists
- memory snapshots before import/JIT startup is measured as painful
- multi-GPU environment sweeps
- spatial indexes before full-buffer `K` scans are measured and fixture-safe
- production `reset_many`/`step_many` or trainer-facing vector API

The main trap is polishing remote infrastructure around a backend that has not
yet proven equivalence. The second trap is letting GPU availability pull the
environment design toward framework convenience instead of source behavior.

## Implementation Order

1. Done: add `scripts/seed_vector_state_from_fixtures.py` as a script-only
   compiler for verified scenarios, with an explicit unsupported-semantics
   report.
2. Done: add a local comparison script that projects array outputs back to
   common-trace fields and fails before timing if equivalence is required.
3. Done: add the Modal CPU wrapper and Volume layout for one tiny smoke. It uses
   the existing image and Volume patterns from `smoke.py` and
   `fidelity_smoke.py`.
4. Add profile-sharded Modal CPU sweeps with one final manifest writer.
5. Add a pinned tensor CPU/GPU dependency smoke only after a script-level JAX or
   PyTorch prototype exists.

## Sources

Local docs:

- `docs/research/modal_patterns.md`
- `docs/design/modal_architecture.md`
- `docs/design/environment/modal_fidelity_jobs.md`
- `docs/runbooks/modal_environment_fidelity.md`
- `docs/research/environment/performance_vectorization_plan.md`
- `docs/working/environment/vector_state_schema.md`
- `docs/working/environment/full_fidelity_execution_plan.md`
- `docs/decisions/0002-modal-hot-loop-locality.md`

Official Modal docs used for current Modal facts:

- [Modal GPU acceleration](https://modal.com/docs/guide/gpu)
- [Modal Images](https://modal.com/docs/guide/images)
- [Modal Volumes guide](https://modal.com/docs/guide/volumes)
- [modal.Volume reference](https://modal.com/docs/reference/modal.Volume)
