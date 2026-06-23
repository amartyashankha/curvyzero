# Subagent GPU Renderer Boundary

Date: 2026-05-21

Scope: code and docs inspection only. No trainer defaults, live runs,
checkpoints, tournaments, or production code were changed.

## Bottom Line

The current promising GPU observation path is batched, but only across the
render boundary. It is not end-to-end GPU simulation or end-to-end GPU
LightZero collection.

The real path today is:

```text
CPU VectorMultiplayerEnv state
-> CPU compact/pack/trail-slot selection
-> host-to-device JAX copy
-> one fused batched GPU render for B rows x 2 views
-> device-to-host readback
-> CPU row-major conversion
-> CPU float32 stack update [B,2,4,64,64]
-> CPU/Python scalar LightZero timestep dictionaries
-> stock LightZero policy/search/learner path
```

So the GPU path is genuinely batched for the renderer, and the profile manager
keeps a batched CurvyTron surface alive. The remaining overhead is mostly host
state conversion, device synchronization/readback, host stack writes, scalar
env-manager object traffic, and the one-process manager shape.

## Current CPU/GPU Boundaries

### Stock CPU-Oracle Training Path

Trusted training defaults still use `cpu_oracle` policy observations. The
source of truth is the observation contract in
`src/curvyzero/env/observation_surface_contract.py`: default backend is
`cpu_oracle`, scalar `jax_gpu` is explicitly experimental, and the production
direction is named as batched GPU observation rather than scalar `jax_gpu`.

In the scalar trainer env
`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`,
the normal CPU path renders source-state gray64 frames on the host, normalizes
them into host float32 stacks, and returns LightZero-shaped NumPy payloads. On
GPU compute, LightZero model/search/learner tensors may live on GPU, but the
environment observation generation remains CPU unless the scalar `jax_gpu`
backend is explicitly selected.

### Scalar `jax_gpu` Path

The scalar `jax_gpu` backend is not the speed path. In
`CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv._update_stack`, it renders
one env row's two player perspectives, copies raw frames back to NumPy, then
updates the same host float32 stacks. This path still has CPU env state, a
per-env render call shape, device synchronization, D2H readback, and host stack
mutation. It is useful as a diagnostic, not as the batched optimizer target.

### Batched Profile Boundary

The candidate boundary is in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
The exact crossing sequence is:

```text
_production_to_benchmark_source_state(...)      # CPU NumPy compact state
_pack_compact_trails_in_owner_draw_order(...)   # CPU NumPy reorder
_select_render_trail_slots(...)                 # CPU control/stats
_truncate_compact_trails_for_render(...)        # CPU slice/copy
_copy_state_to_device(...)                      # jax.device_put + block
render_fn(device_state).block_until_ready()     # GPU render + sync
np.asarray(output_device)                       # D2H readback + sync
_view_major_to_row_major_frames(...)            # CPU reshape/transpose
_push_row_major_frames_into_stack(...)          # CPU normalized stack write
```

`_copy_state_to_device` blocks every copied JAX array. The render timer also
blocks on `output_device.block_until_ready()`. Then `np.asarray(output_device)`
is the explicit GPU-to-CPU boundary.

The profile timing fields mirror these boundaries:
`production_to_compact_sec`, `owner_ordered_pack_sec`, `host_to_device_sec`,
`device_render_sec`, `device_to_host_sec`, `view_major_to_row_major_sec`,
`stack_sec`, `lightzero_scalarize_sec`, and optional RND/pickle buckets.

### Trainer-Surface Boundary

`SourceStateMultiplayerTrainerSurface` in
`src/curvyzero/training/multiplayer_source_state_trainer_surface.py` owns a
batched `VectorMultiplayerEnv`. With `renderer_backed_profile`, it uses
`_RendererBackedSourceStateGray64Stack4`.

This is the important shape:

```text
surface.step(joint_action[B,2])
-> env.step(...) on CPU
-> stack.update(env, copy=False)
-> _RendererBackedSourceStateGray64Stack4._render_rows(...)
-> SourceStateBatchedObservationRenderer.render(...)
-> host stack [B,2,4,64,64]
-> _surface_step(...) packages policy rows
```

The stack adapter asks for all rows and both player views in row-major order on
the hot full-row path: `[(row0,p0), (row0,p1), ...]`. It validates returned
frames as `uint8` and then normalizes into a NumPy float32 stack.

### Stock-LightZero Profile Hook

The stock-loop profile hook lives in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
For `env_manager_type=curvyzero_batched_profile`, it patches LightZero's
`create_env_manager` and builds:

```text
SourceStateMultiplayerTrainerSurface(batch_size=env_num/2, player_count=2)
-> _DynamicJaxBatchedObservationRenderer
-> BatchedLightZeroScalarActionBridge
-> BatchedLightZeroStockEnvManagerAdapter
-> stock train_muzero collector sees scalar env ids
```

It refuses hidden fallback if LightZero asks for another env-manager type. The
zero-observation profile manager uses the same bridge and adapter but swaps the
renderer for a host zero-fill renderer, making it a useful Amdahl floor for the
same one-process/scalar object shape.

## Is The GPU Path Actually Batched?

Yes, but with a specific meaning.

Already batched:

- `VectorMultiplayerEnv` holds `B` physical rows with 2 players.
- The bridge commits one joint action array shaped `[B,2]`.
- The renderer-backed stack requests all `B*2` player views in one renderer
  call on normal full-row steps.
- `_make_jax_two_view_render_fn` returns both controlled-player views from one
  JIT path.
- The trainer stack is `[B,2,4,64,64]`.
- When all policy rows are live and row-major, `policy_observation` can be a
  reshape instead of per-row gather/copy.

Still not batched or not resident:

- Source state is CPU NumPy, not device-resident.
- Compacting, owner packing, trail-slot selection, and row-major conversion are
  CPU work.
- Frames are read back to host every step.
- Stack state is host float32.
- LightZero-facing actions and timesteps are Python mappings keyed by scalar
  env id.
- Stock adapter emits per-env `BaseEnvTimestep`-like objects.
- The manager is one process, so it does not preserve subprocess actor
  parallelism.
- Torch policy/search later sees host observations and performs its own normal
  tensor movement.

One caveat: `_DynamicJaxBatchedObservationRenderer` handles partial row/player
requests by rendering the full batch, then gathering requested frames. That is
correctness-friendly, but normal-death/autoreset rows can make it look more
batched than the requested subset actually needed.

## Remaining Host Overhead

The important host overhead buckets are:

- CPU env stepping in `VectorMultiplayerEnv`.
- CPU production-state to compact-state conversion.
- CPU owner-order trail packing.
- Dynamic render-width/trail-slot stats and truncation.
- H2D copy and explicit synchronization.
- D2H readback and explicit synchronization.
- View-major GPU output to row-major trainer order.
- Host float32 stack update and reset-row stack writes.
- Terminal `final_observation` copy into host arrays/info.
- Reward, masks, live-row filtering, and policy-row packaging.
- Action dict sorting and conversion into `[B,2]`.
- `ready_obs` and `timestep_by_env_id` Python dict construction.
- Per-env stock `BaseEnvTimestep` conversion and reward/done scalar coercion.
- Optional `pickle.dumps(...)` payload measurement.
- Optional RND collect/train/estimate calls after materialization.

The current zero-observation rows are the clean warning: removing real render
does not remove the manager, env, stack, scalarization, policy/search, replay,
learner, or RND floor.

## What This Means For A Central GPU Observation Service

A central service could beat the current one-process manager only if it solves
both sides of the trade:

1. Preserve CPU actor parallelism from multiple workers/processes.
2. Batch enough observation requests per GPU render to amortize H2D/JIT/render/D2H.

It will not win if it simply adds another IPC hop around the same
CPU-compact -> H2D -> render -> D2H -> CPU-stack sequence for small batches.
The service has to aggregate requests from multiple actors, return row/player
frames fast enough, and avoid forcing every actor to block on a global render
barrier.

The smallest useful service target is probably "central batched renderer", not
"central environment". Keep env stepping in CPU actors, ship compact render
state or a packed render request to the service, batch render on GPU, return
`uint8 [N,1,64,64]` frames, and let actors/manager perform their existing
stack and LightZero packaging. That isolates whether shared GPU batching beats
the current one-process manager before attempting end-to-end trainer surgery.

## Smallest No-Training Prototypes

### 1. In-Process Batch-Size Ceiling

Run the existing profile env-manager canary without LightZero training and
without RND. Sweep `B={32,64,128,256,512}` with `direct_gray64`,
`dynamic_render_trail_slots=True`, and zero/normal death variants.

Question answered: where does the current one-process batched manager saturate,
and how much of that is render versus host stack/scalarization?

Success signal: throughput stops improving while `device_render_sec` is no
longer dominant, or manager host buckets dominate. That defines the bar a
central service must clear.

### 2. Actor-Fan-In Renderer Microservice

Build a no-training local/Modal prototype with `K` CPU actor loops. Each actor
owns a small `VectorMultiplayerEnv` batch, steps random actions, sends compact
render requests to one GPU service loop, receives `uint8` frames, and updates a
local stack. No LightZero, no replay, no learner.

Measure:

- actor env-step time;
- compact/pack time in actor versus service;
- request queue wait;
- service batch size distribution;
- H2D/render/D2H;
- response latency p50/p95;
- aggregate env rows/sec.

Question answered: can actor parallelism plus central GPU batching beat the
one-process manager before stock LightZero object overhead enters the picture?

### 3. Service Placement Split

Use the same prototype but compare two request payloads:

- actors send production `VectorMultiplayerEnv.state` slices and service does
  compact/pack;
- actors send already compact owner-packed render arrays and service only does
  H2D/render/D2H.

Question answered: whether the CPU compact/pack work belongs near actors or
near the GPU. If service-side compacting serializes too much CPU work, it will
erase the actor-parallel benefit.

### 4. Readback-Avoidance Counterfactual

Add a benchmark mode that leaves renderer output on device and performs only a
synthetic downstream device consumer, compared against the current D2H plus
host stack update. This should stay no-training and can use a dummy JAX/Torch
consumer.

Question answered: how much of the central-service upside requires avoiding
readback entirely rather than merely batching render.

### 5. Scalar-LightZero Object Floor Replay

Feed precomputed or zero frames into `BatchedLightZeroStockEnvManagerAdapter`
with random scalar actions, no trainer. Compare dict/object materialization
against a vectorized payload shape.

Question answered: after a service makes render cheap, is stock scalar
LightZero packaging still large enough that the service cannot matter?

## Recommendation

Do not touch production trainer defaults. Keep real training on `cpu_oracle`
until the batched path has complete semantic gates and a full-loop proof.

For optimizer work, the next useful branch is not another scalar `jax_gpu`
renderer. It is a no-training service prototype that preserves CPU actor
parallelism while batching render requests. The decisive comparison is:

```text
current one-process batched manager
vs
K CPU actor loops + one central GPU renderer service
```

Use zero-observation and direct-render rows as controls. If the service cannot
beat the one-process manager before LightZero training is added, it will not
beat it after adding stock collector/search/replay/learner overhead.
