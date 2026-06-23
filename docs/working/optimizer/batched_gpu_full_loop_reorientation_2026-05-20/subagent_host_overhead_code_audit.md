# Subagent Host Overhead Code Audit

Date: 2026-05-21

Scope: code/dataflow audit only. No live runs, trainer defaults, launch
defaults, checkpoints, tournaments, or active profile rows were touched.

## Bottom Line

The current batched GPU observation lane is genuinely batched through env rows,
both player views, direct GPU render, and the trainer surface stack. It still
crosses back to host every step and then scalarizes into Python/NumPy
LightZero-shaped timesteps. The C512 profile numbers are consistent with that
shape:

- real batched GPU row: about `1439.84 steps/s`;
- zero observation row: about `1805.22 steps/s`;
- full-loop C512 real row: policy about `119.54s`, manager step about
  `178.18s`, render aggregate about `83.52s`.

Plain read: render/stack/pack still matter, but renderer-only upside is now
bounded. The biggest uncertainty is not "can GPU render faster?" It is how much
wall is still caused by host-side stack/scalar timestep/object construction and
the one-process manager shape after render gets cheaper.

## Current Stock-Boundary Dataflow

The stock profile launcher path is
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
For `env_manager_type=curvyzero_batched_profile` or
`curvyzero_batched_zero_obs_profile`, it installs a profile-only hook around
LightZero's `create_env_manager` and registers profile manager classes in
DI-engine's `ENV_MANAGER_REGISTRY`.

The hooked manager construction is:

```text
stock lzero.entry.train_muzero
-> patched create_env_manager
-> SourceStateMultiplayerTrainerSurface(batch_size=env_num/2, player_count=2)
-> renderer_backed_profile stack
-> _DynamicJaxBatchedObservationRenderer or ZeroObservationRenderer
-> BatchedLightZeroScalarActionBridge
-> BatchedLightZeroStockEnvManagerAdapter
-> stock LightZero collector sees scalar env ids again
```

This is still profile-only and fail-closed:

- even scalar env count required, because two scalar env ids map to one
  physical CurvyTron row;
- explicit renderer backend required for `renderer_backed_profile`;
- hidden fallback to scalar env managers is refused;
- command-level `policy_observation_backend=cpu_oracle` is not the meaningful
  backend identity for these rows. The meaningful identity is the manager type,
  surface stack backend, and renderer backend telemetry.

## Candidate Render Boundary

The boundary profile in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
still has a very explicit host/device shape:

```text
VectorMultiplayerEnv.step(joint_actions)        # CPU/NumPy env
-> _production_to_benchmark_source_state        # CPU compact copy/reshape
-> _pack_compact_trails_in_owner_draw_order     # CPU owner-order packing
-> _select_render_trail_slots / truncation      # CPU stats/control
-> _copy_state_to_device                        # H2D JAX device_put
-> render_fn(device_state).block_until_ready()  # GPU render plus sync
-> np.asarray(output_device)                    # D2H readback and sync
-> _view_major_to_row_major_frames              # CPU transpose/copy
-> _push_row_major_frames_into_stack            # CPU float32 stack update
-> materialize LightZero payload / RND / pickle # CPU Python boundary
```

The profile timing buckets match that sequence:
`production_to_compact_sec`, `owner_ordered_pack_sec`, `host_to_device_sec`,
`device_render_sec`, `device_to_host_sec`, `view_major_to_row_major_sec`,
`stack_sec`, `lightzero_scalarize_sec`, `lightzero_payload_pickle_sec`, and RND
method buckets.

The major synchronization points are explicit:

- `device_put` in `_copy_state_to_device`;
- `output_device.block_until_ready()` inside the device-render timer;
- `np.asarray(output_device)` for device-to-host readback;
- host stack writes after readback;
- Python object materialization and optional `pickle.dumps(...)`.

## Trainer Surface Dataflow

`SourceStateMultiplayerTrainerSurface` in
`src/curvyzero/training/multiplayer_source_state_trainer_surface.py` owns the
batched `VectorMultiplayerEnv`. With `renderer_backed_profile`, its stack is
`_RendererBackedSourceStateGray64Stack4`.

Hot step:

```text
surface.step(joint_action)
-> env.step([B,2])                         # CPU vector env
-> stack.update(env, copy=False)           # render all rows/player views
-> _survival_plus_bonus_reward             # CPU reward array
-> _surface_step(...)                      # package policy rows/info
```

What is already batched:

- one `VectorMultiplayerEnv` with `B` physical rows and 2 players;
- one joint action array shaped `[B,2]`;
- one renderer-backed stack shaped `[B,2,4,64,64]`;
- full-row render requests use cached row/player arrays:
  `[row0,p0], [row0,p1], ...`;
- policy observations can use reshape instead of gather/copy when live policy
  rows are full row-major.

What is still host/CPU:

- env physics and source state live in NumPy arrays;
- stack storage is float32 NumPy on host;
- latest frame normalization is a NumPy write into host stack;
- reward, masks, live-row filtering, final-observation maps, and info payloads
  are host arrays/dicts;
- terminal final observation copies stack slices on host.

Important current detail: for full-row updates, `_write_latest(...)` avoids a
temporary float array by using `np.multiply(..., out=...)`. For partial row
updates, it still assigns `frames[:, :, 0].astype(np.float32, copy=False) *
1/255`, which may allocate and is worth measuring in normal-death rows.

## Scalar LightZero Boundary

`src/curvyzero/training/source_state_batched_observation_mock_collector.py`
contains the local bridge pieces:

- `BatchedLightZeroScalarActionBridge` maps scalar env ids to `(row, player)`,
  validates actions, commits one batched step, and returns scalar timesteps;
- `BatchedLightZeroProfileEnvManager` is the profile manager facade;
- `BatchedLightZeroStockEnvManagerAdapter` is the stock-shaped adapter used by
  the LightZero hook;
- materializers produce `MockBaseEnvTimestep` or DI-engine
  `BaseEnvTimestep`-like payloads.

What remains scalar/Python:

- LightZero actions arrive as `Mapping[int, Any]`;
- action keys are sorted and turned into NumPy arrays each step;
- `action_by_env_id` is looped over to fill `[B,2]` joint action;
- `ready_obs` is a dict keyed by scalar env id;
- `timestep_by_env_id` is a dict keyed by scalar env id;
- stock adapter converts each scalar timestep into a `BaseEnvTimestep` and
  coerces reward/done to plain `float`/`bool`;
- `info` is a Python list/dict per policy row, with terminal
  `final_observation` copies when present.

Recent normal-death fixes mean the bridge can now omit complete physical rows
when LightZero stops asking for that row's scalar env ids, while still
rejecting half-row omissions. That preserves simultaneous-action semantics, but
it means partial render/update paths are now correctness-relevant and should be
timed separately.

## Zero Observation Lane

The zero-observation manager in the launcher uses the same
`SourceStateMultiplayerTrainerSurface`, scalar bridge, stock adapter, and
LightZero collector/search/replay/learner boundary. It swaps only the renderer
for `ZeroObservationRenderer`, which fills `request.out` with zeros on host and
reports near-zero render telemetry.

That makes zero-observation a strong Amdahl probe for:

- stock LightZero collection/search/learner still active;
- scalar timestep payload boundary still active;
- one-process batched manager still active;
- real env stepping still active;
- render pixels removed.

It is not a production observation, and it does not remove host stack or scalar
payload work entirely. At C512 it still spends about `108.17s` in manager step,
with about `63.33s` in surface env step and `11.63s` in stack update. That is
the "observation-free-ish but not Python-free" floor for this architecture.

## CPU/GPU Crossings

Current real batched GPU render crosses CPU/GPU every manager step:

1. CPU source state arrays are compacted and packed.
2. Compact arrays are copied to device.
3. JAX render produces device frames in view-major order.
4. Render blocks until ready.
5. Frames are copied back to NumPy.
6. CPU transposes/copies view-major to row-major.
7. CPU writes normalized float32 frames into the trainer stack.
8. CPU scalarizes stack rows into LightZero payloads.
9. Torch policy/search later moves observation tensors back to GPU through
   LightZero's normal model path.

That last point is important: this lane is not end-to-end GPU RL. It is
GPU-render-then-host-payload-then-GPU-policy. The profile can still win because
it batches render work, but every step has synchronization and host payload
work in the middle.

## Already Batched Versus Still Scalar

Already batched:

- CPU env physics across `B` rows in one `VectorMultiplayerEnv`;
- actions as one `[B,2]` array inside the bridge/surface;
- renderer request for all rows and both player views on the hot no-death path;
- direct GPU renderer for both player views;
- stack storage/update as `[B,2,4,64,64]`;
- policy observation array can be flat `[B*2,4,64,64]` when all rows are live;
- RND meter can consume a flat policy observation batch after materialization.

Still scalar or Python-object heavy:

- LightZero-facing `ready_obs` and `timestep_by_env_id` dicts;
- per-env `BaseEnvTimestep` objects in the stock adapter;
- per-row/player `info` dict/list construction;
- action dict sorting and looping;
- terminal final-observation insertion into `info`;
- optional pickle proxy over timestep payloads;
- stock collector/game-segment/replay object flow;
- manager hook/profiler counters are Python-side and aggregate after each step;
- normal-death partial row handling can leave full-batch render plus gather
  inside `_DynamicJaxBatchedObservationRenderer`.

## Host Sync And Copy Suspects

Most useful suspects after this audit:

1. **Host stack and scalar payload floor.** Zero-observation still has a large
   manager step. The note in `uint8_payload_design_note.md` is directionally
   right: float32 stack payload is huge, but uint8 requires an explicit model
   contract and cannot be a hidden env flip.
2. **Full-batch render for partial requests.** `_DynamicJaxBatchedObservationRenderer`
   now handles partial row/player requests by rendering the full batch
   internally and gathering requested frames. This is safe and low-risk for
   correctness, but normal-death rows need to quantify how often partial
   requests happen and how costly they are.
3. **CPU compact/pack duplication.** Production state is converted to benchmark
   compact state, owner-ordered, possibly truncated, then copied to device every
   step. At C256/C512 the pack and H2D buckets are not dominant, but they are
   persistent and scale with width/active trail slots.
4. **Device render synchronization.** The render timer includes
   `block_until_ready()`, which is good for honest timings but guarantees no
   overlap with host stack/scalarization.
5. **View-major to row-major copy.** The boundary renderer still returns
   view-major and host-transposes to row-major. Avoiding this would be low-risk
   only if done behind the existing parity/semantic gates.
6. **Policy path re-upload.** Rendered frames come back to CPU, then LightZero
   policy inference sends observation tensors to GPU. This is architectural,
   not a small renderer patch.

## What Is Already Instrumented

The current lane already records useful aggregates:

- Modal boundary: env step, compact, owner pack, H2D, device render, D2H,
  view-major conversion, stack, final obs, autoreset, reset render/stack,
  scalarize, pickle, RND method timers, and trail stats.
- Surface facade: surface step total, renderer render/device/H2D/D2H/pack,
  non-renderer, env step, stack update, reward, package, scalarize/pickle/RND.
- Stock LightZero hook: manager reset/step, surface env step, surface stack
  update, reward/package, renderer render/device/H2D/D2H/pack, policy forward,
  MCTS, learner, replay, RND, GPU sampling, and effective step denominator.
- Result compaction preserves the batched manager counters and timers in
  `scripts/run_curvytron_optimizer_profile_manifest.py` and
  `_compact_train_result_for_output(...)`.

The biggest missing pieces are not top-level timers; they are splits inside the
host-heavy buckets.

## Low-Risk Next Instrumentation

Do these only in profile paths and result summaries, not trainer defaults.

1. Split manager step host overhead:
   - action dict sort/validation;
   - joint action fill;
   - surface step;
   - ready_obs construction;
   - timestep materialization;
   - stock `BaseEnvTimestep` conversion loop.

2. Split surface package time:
   - live mask and policy row selection;
   - policy observation reshape versus gather/copy;
   - policy action mask copy;
   - final-observation allocation/copy;
   - `_info(...)` dict construction.

3. Split renderer-backed stack update:
   - FIFO shift;
   - renderer call;
   - latest-frame normalization/write;
   - full-row versus partial-row write path;
   - `copy=True` versus `copy=False` result copy, if any.

4. Add partial-request counters:
   - requested output row count;
   - full-row-major hot path count;
   - partial render request count;
   - omitted complete physical row count;
   - partial autoreset row count.

5. Add payload byte estimates without changing payload:
   - observation array bytes;
   - action mask bytes;
   - reward/done bytes;
   - `info` final-observation bytes;
   - pickle bytes/time for `ready_obs` and `timestep_by_env_id` separately in
     profile rows.

6. Add CUDA/JAX sync attribution only as an optional diagnostic:
   - keep current honest `block_until_ready` render timer;
   - add a separate "launch-only" experimental timer only if clearly labeled,
     because it is not comparable to current wall timers.

## Low-Risk Prototypes

1. **Zero-observation host-floor probe with splits.** Repeat C256/C512 zero obs
   after adding manager/surface/stack split timers. This answers how much of the
   zero floor is scalar payload versus env step versus stack/object churn.

2. **Row-major device output canary.** Add a profile-only renderer output order
   that directly matches `[B,2,1,64,64]` or flat row-major and eliminates
   `_view_major_to_row_major_frames`. Gate it with existing row/player parity
   tests. This is a small copy/sync reduction, not a rewrite.

3. **Profile-only uint8 payload canary.** Keep stacks internally as current
   float32 or add explicit cast points, but measure a named uint8 payload
   contract separately. Do not change stock model inputs silently.

4. **Partial-row renderer stress.** Run a normal-death/profile-only row that
   reports partial request frequency and cost. If partial requests are common,
   prototype row-masked render/update. If rare, keep the safe full-render gather
   path.

5. **Hybrid manager toy canary.** Follow `subagent_hybrid_manager_plan.md`:
   start with zero observations and actor subprocesses sending compact source
   state/reward/done metadata to a parent. The first question is whether actor
   parallelism survives without shipping rendered stacks over IPC.

## Follow-up: smallest instrumentation patch

Goal: split the C512/C768 host floor without changing trainer defaults, manager
semantics, payload shape, or live-run behavior. The smallest patch should only
add timers/counters under existing profile-only paths and then expose the new
fields in compact profile output.

Files/functions to touch:

1. `src/curvyzero/training/source_state_batched_observation_mock_collector.py`
   - `BatchedLightZeroScalarActionBridge.step(...)`: time
     `action_env_ids` sort, `_joint_action_from_scalar_actions(...)`,
     `loop.step(...)`, terminal `surface.reset(row_mask=...)`, and
     `_output_from_loop_step(...)`. Store a plain dict on the returned
     `ScalarActionBridgeOutput`, for example `profile_timing_sec`, plus counts
     for provided scalar actions, omitted complete rows, terminal rows, and
     autoreset rows.
   - `BatchedLightZeroScalarActionBridge._joint_action_from_scalar_actions(...)`:
     count missing ids, extra ids, omitted complete physical rows, rejected
     partial rows, and scalar-to-joint loop count. This answers whether action
     dict handling is visible at C512/C768.
   - `BatchedLightZeroScalarActionBridge._output_from_loop_step(...)`: time
     policy env-id construction, `ready_obs` dict construction,
     timestep materialization/reuse, and `_split_timestep_by_env_id(...)`.
     This is the direct scalarization/object churn split.
   - `BatchedLightZeroStockEnvManagerAdapter.step(...)`: time the per-env
     `BaseEnvTimestep` conversion loop separately from the bridge step. Record
     converted timestep count and terminal timestep count.

2. `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`
   - `_RendererBackedSourceStateGray64Stack4.update(...)`: time FIFO shift,
     `_render_rows(...)`, latest-frame write, and optional return copy.
   - `_RendererBackedSourceStateGray64Stack4.reset_rows(...)`: same split for
     reset/autoreset rows, plus reset row count.
   - `_RendererBackedSourceStateGray64Stack4._render_rows(...)`: record
     requested physical row count, requested output row count, full-row-major
     hot-path flag, partial-request flag, and renderer call time.
   - `_RendererBackedSourceStateGray64Stack4._write_latest(...)`: split
     full-row write versus partial-row write time. This targets the normal-death
     partial-row ambiguity.
   - `SourceStateMultiplayerTrainerSurface._surface_step(...)`: time live-mask
     and policy-row selection, policy observation reshape/gather, action-mask
     copy, final-observation allocation/copy, `_info(...)`, and dataclass
     construction. Keep these inside `trainer_surface_profile_timing` only when
     `observation_stack_backend == renderer_backed_profile`.

3. `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
   - `_install_batched_profile_env_manager_hook(...).profiled_manager_step(...)`:
     harvest `bridge_output.profile_timing_sec` and any adapter timing dict into
     `profiler.add_time("batched_profile_bridge_<field>_sec", ...)` and
     `profiler.add_sample(...)` for partial/action counts. Existing harvesting
     already collects `trainer_surface_profile_timing` and renderer telemetry,
     so new surface/stack keys should flow through with little code.
   - `_compact_train_result_for_output(...)`: add compact timer fields for the
     new bridge, stack, surface-package, and partial-row splits. This keeps the
     result reader from having to dig through raw phase-profile JSON.

4. `scripts/summarize_curvytron_optimizer_profile_results.py`
   - `_profile_row(...)` and `_markdown_table(...)`: add optional columns only
     for the handful of new C512/C768 ambiguity buckets:
     bridge action handling, bridge scalarization/object conversion, stack
     shift/write/render, surface package, partial request count, policy
     forward, and MCTS. This is a summarizer-only convenience, not a runtime
     change.

First profile rows to use it:

| row | manager path | width | death | observation | purpose |
|---|---|---:|---|---|---|
| `instr-c512-real-nodeath-sim2` | `curvyzero_batched_profile` | C512 | no-death | real direct GPU | Rebaseline the known `1439.84 steps/s` regime with split manager/stack/scalar timers. |
| `instr-c512-zero-nodeath-sim2` | `curvyzero_batched_zero_obs_profile` | C512 | no-death | zero obs | Split the `1805.22 steps/s` host floor into env step, stack, scalarization, action dict, policy/search. |
| `instr-c768-real-nodeath-sim2` | `curvyzero_batched_profile` | C768 | no-death | real direct GPU | Test whether C768 is render/stack limited or policy/search/manager saturated. |
| `instr-c768-zero-nodeath-sim2` | `curvyzero_batched_zero_obs_profile` | C768 | no-death | zero obs | Check the non-render ceiling and policy/search saturation at C768. |
| `instr-c256-real-normaldeath-sim2` | `curvyzero_batched_profile` | C256 | normal | real direct GPU | Exercise partial rows/autoreset safely before trusting normal-death C512/C768. |

Readout logic:

- If C512/C768 real and zero both spend most new time in policy forward plus
  MCTS, stop renderer-only work and move to search/topology/manager
  architecture.
- If zero rows spend large time in bridge scalarization, object conversion, or
  payload bytes, prioritize payload slimming before render kernels.
- If real rows have large stack write/reshape/gather time but zero rows do not,
  prioritize stack layout/output-order cleanup.
- If normal-death rows show frequent partial requests, measure row-masked render
  before wider normal-death runs. If partial requests are rare, keep the
  current full-render gather path for correctness.
- If action dict handling is below noise at C512/C768, do not spend engineering
  time on action mapping yet.

## Recommendation

Keep current Coach/trainer defaults on trusted CPU-oracle paths. Keep the
batched GPU manager as the best optimizer profile lane at wide C512-style
topologies, but treat C512 real versus zero as a stopping-rule pair:

- if real rows approach zero within roughly `25%`, move effort to
  manager/policy/search architecture;
- if real rows remain far below zero, spend only targeted effort on
  render/stack/pack copies with the split timers above;
- for any 10x target, start designing around loop architecture, because the
  current zero-observation ceiling is far below an order-of-magnitude win.
