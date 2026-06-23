# Compact Closed-Loop Iteration Dataflow Deep Dive, 2026-05-22

Status: docs-only subagent note. I read the relevant profile code and working
docs, and did not touch live Coach training runs, Modal volumes, checkpoints,
evals, GIFs, tournaments, or source code.

Files inspected first:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_batched_observation_profile.py`
- current docs in `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/`

## Current Ground Truth

The production-trusted visual surface remains:

```text
browser_lines + simple_symbols + cpu_oracle
```

The current fast profile-only surface is:

```text
jax_gpu_persistent_policy_framebuffer_profile + direct_gray64
```

That profile surface is for optimization evidence. It preserves the current
policy observation shape and the browser-line/simple-symbol intent, but it is
not the same claim as promoting the CPU oracle replacement into Coach training.

The clean latest cross-sim compact profile denominator in the working notes is:

```text
H100, B1024/P2, loop24, actor_count=1, native actor buffer,
body_capacity=4096, hidden_dim=64, max_depth=16, rollout_steps=4,
borrow_single_actor_render_state=True,
resident GPU stack,
compact_root_copy_observation=False,
closed-loop replay-index on,
explicit resident-stack sync off.
```

Rows:

| sims | active roots/sec | total loop sec | env frac | search frac | observation | GPU draw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `50,357` | `0.9761s` | `53.7%` | `15.6%` | `0.2996s` | `0.0045s` |
| 32 | `43,340` | `1.1341s` | `40.9%` | `32.9%` | `0.2652s` | `0.0040s` |

The async internal renderer device-only flag did not improve total wall
(`50,239` sim16 and `40,909` sim32), so the current read is not "raw draw wait
is the wall." The wall is the broader next-search-input handoff plus search and
residual.

## One Iteration

In the compact MCTX profile loop, one closed-loop iteration starts with a
current `HybridCompactBatch` and ends with the next `HybridCompactBatch` plus
optional replay-index rows:

```text
current compact batch
-> build compact root batch
-> prepare resident/host search input and legal mask
-> JAX/MCTX search
-> read selected actions, visit policy, and root values
-> validate compact search result
-> make CPU joint action
-> step CPU CurvyTron actor/env
-> render/update next observation
-> update resident GPU stack
-> build compact replay-index rows
-> next compact batch
```

This is profile-only. It does not call `train_muzero`.

## State Residency At The Start

| State | Current owner | Residency | Size read |
| --- | --- | --- | --- |
| CurvyTron mechanics state | `VectorMultiplayerEnv` inside the actor | CPU NumPy arrays | Large, especially trail/body state |
| Borrowed render state | actor `env.state` mapping | CPU NumPy references | Large if copied, cheap if borrowed |
| Compact sidecars | `HybridBatchedObservationProfileManager._compact` and `HybridCompactBatch` | CPU NumPy | Mostly small |
| Persistent renderer layer | `_PersistentJaxPolicyFramebufferRenderer._layer_device` | JAX device | Large persistent framebuffer/layer |
| Latest rendered frame | renderer `last_output_device` | JAX device | Medium: `[B,2,1,64,64]`, about 8 MiB at B1024/P2 uint8 |
| Resident stack | `compact_visual_resident_device_stack` | JAX device | Large: `[B,2,4,64,64]`, about 32 MiB at B1024/P2 uint8 |
| Root sidecars | `CompactRootBatchV1` | CPU NumPy | Small except observation field |
| Search tree/model state | MCTX/JAX synthetic benchmark | JAX device | Medium/large, depends on sims/depth/hidden |
| Search output arrays | MCTX output, then NumPy copies | GPU first, CPU after readback | Small by bytes, important as sync |
| Replay-index rows | compact bridge builders | CPU NumPy | Small in current rows |

The important asymmetry is that the environment is still CPU, while the current
observation stack used by search is intended to be resident on the GPU. That
makes selected actions the legitimate CPU/GPU crossing point.

## Step-By-Step Dataflow

### 1. Build `CompactRootBatchV1`

Code path: `build_compact_root_batch_v1(loop_batch, ...)` in the closed-loop
section of `mctx_synthetic_benchmark.py`.

Inputs:

- CPU `HybridCompactBatch`
- CPU observation field, legal masks, active-root mask, env rows, players,
  rewards, done/final-observation sidecars

Outputs:

- CPU `CompactRootBatchV1`
- `legal_mask`, `active_root_mask`, `env_row`, `player`, metadata

Large movement:

- Host observation is large if copied.
- In the current best row, `copy_observation=False` avoids the expensive root
  observation copy. The root batch still has a host observation field for
  validation/shape compatibility, but the hot search input is the resident
  JAX stack.

Small movement:

- masks, row/player ids, rewards, dones, root metadata.

Likely residual:

- root-batch validation and `np.asarray(loop_root_batch.observation)` still run
  every loop. With no-copy and resident search input this should be much
  smaller than the old root-copy path, but it is still compatibility work.

### 2. Prepare Search Input

Code path:

```text
if resident:
    loop_obs = compact_visual_resident_device_stack.reshape(root_count, 4, 64, 64)
else:
    loop_obs = jax.device_put(loop_obs_host)
loop_invalid = jax.device_put(loop_invalid_host)
loop_obs.block_until_ready()
loop_invalid.block_until_ready()
```

Resident mode:

- observation stack stays on JAX device;
- reshape is a device view/logical reshape;
- legal/invalid mask still crosses CPU to JAX every loop.

Host mode:

- full root observation stack moves host to device every loop.

Large movement:

- host mode H2D is large: `[roots,4,64,64]`.
- resident mode avoids that full observation H2D.

Small movement:

- invalid mask is `[roots,3]`, tiny by bytes.

Synchronization read:

- Even with explicit resident-stack sync off, `loop_obs.block_until_ready()` can
  become the next consumer wait. Sync removal only helps if it overlaps real
  work or disappears from total wall, not if it simply moves from
  `resident_stack_update_sec` to `h2d_sec` or `search_sec`.

### 3. Run MCTX Search

Code path: JIT `run_search(...)`, then `loop_output.action_weights.block_until_ready()`.

Inputs:

- JAX observation tensor
- JAX invalid-action mask
- JAX PRNG key
- fixed-shape synthetic MuZero/MCTX model/search parameters

Outputs:

- selected action
- action weights / visit policy
- search tree/root values

Large movement:

- search internals are device-resident.

Small movement:

- final action, visit policy, and root value arrays are small.

Synchronization read:

- `action_weights.block_until_ready()` makes `search_sec` a synchronized search
  wall bucket. That is useful for attribution and necessary before CPU action
  use, but it means this profile bucket includes any previous deferred JAX work
  that was first demanded by the search output.

### 4. Read Search Outputs To CPU

Code path:

```text
loop_actions = np.asarray(loop_output.action)
loop_action_weights = np.asarray(loop_output.action_weights)
loop_root_values = _extract_mctx_root_values(loop_output)
```

Large movement:

- none by bytes.

Small movement:

- actions `[roots]`
- action weights `[roots,3]`
- root values `[roots]`

Synchronization read:

- small byte count does not mean small wall. These reads force the CPU to wait
  for search and any deferred producer work. This is a safe sync in the current
  CPU-env loop because the next stage needs CPU joint actions.

### 5. Validate Compact Search Result

Code path: `validate_compact_search_result_v1(...)`.

Inputs:

- CPU root batch
- CPU actions, visit policies, root values
- active-root indices

Outputs:

- CPU `CompactSearchResultV1`

Large movement:

- none expected.

Small movement:

- per-active-root compact arrays.

Likely residual:

- correctness checks and compact object construction are CPU work. This is not
  currently the main wall, but it is part of the compatibility surface.

### 6. Build CPU Joint Action

Code path:

```text
loop_joint_action = np.full((B, P), 1, dtype=np.int16)
loop_joint_action[env_row, player] = selected_action
```

Inputs:

- CPU compact search result

Outputs:

- CPU joint action `[B,2]`

Size:

- tiny.

Synchronization read:

- this stage is exactly where a CPU env must have CPU actions. A sync here is
  safe and semantically real until the env itself becomes device-resident.

### 7. Step The CurvyTron Actor/Env

Code path: `HybridBatchedObservationProfileManager.step(loop_joint_action)`.

Inputs:

- CPU joint action
- actor-owned CPU `VectorMultiplayerEnv` state

Outputs:

- updated CPU env state
- CPU compact reward/done/action-mask/identity sidecars
- optional terminal/autoreset sidecars
- next `HybridCompactBatch`

What is actually game mechanics:

- `actor_env_runtime_sec` is the closest current leaf to true mechanics.
- In the current notes, true mechanics are small compared with the inclusive
  `env_step_sec` bucket.

What else is included:

- public-step preparation;
- reward and post-runtime bookkeeping;
- public info/batch packaging;
- compact sidecar writes;
- render-state handling;
- observation update and reset/final-observation handling;
- capture probe for the compact batch.

Borrowed render-state mode:

- with `borrow_single_actor_render_state=True`, the renderer receives the
  actor `env.state` mapping directly instead of copying large
  `visual_trail_*` arrays into parent native buffers;
- this is why `actor_render_state_write_sec` drops to zero in the best rows;
- it is profile-only and fails closed on terminal rows, because terminal
  final-observation snapshots would otherwise be ambiguous after autoreset.

Large movement avoided:

- copied render-state mode can move large visual-trail arrays every step.
  Borrowed mode removes that parent-buffer copy in the no-terminal canary.

### 8. Render/Update The Next Observation

Code path: manager `_update_observation(...)` calls
`_PersistentJaxPolicyFramebufferRenderer.render(...)`.

In resident mode the render request uses:

```text
device_only=True
synchronize_device=not defer_renderer_device_sync
```

Persistent renderer CPU work:

- build compact state from production/borrowed CPU state;
- find live trail slots;
- build delta segments/reset masks;
- allocate compact/delta arrays;
- update previous cursor/owner metadata.

Persistent renderer GPU work:

- copy delta and compose state to JAX device;
- update persistent trail layer;
- compose latest player-view frame;
- store `last_output_device`.

Host stack behavior:

- host-stack mode shifts host stack and writes latest frame into host stack;
- resident mode sets `update_host_observation_stack=False`, so it skips the
  hot host FIFO stack update and avoids frame D2H for the main path.

Large movement:

- delta/compose H2D can still be material.
- full frame D2H is avoided in device-only resident mode.

Small movement:

- renderer telemetry and request metadata.

Likely residual:

- `_persistent_visual_compact_state_from_production_fast(...)` still does CPU
  slicing, dtype checks, and some copies;
- `_persistent_delta_state(...)` still allocates and loops over rows/slots to
  build segment arrays;
- `_copy_state_to_device(...)` still transfers delta and compose fields every
  step;
- `_compose_fn(...)` raw draw is already tiny in the latest rows.

### 9. Update The Resident GPU Stack

Code path in `mctx_synthetic_benchmark.py`:

```text
latest_device = renderer.last_output_device
device_stack = jnp.concatenate((device_stack[:, :, 1:], latest_device), axis=2)
if compact_visual_resident_sync:
    device_stack.block_until_ready()
```

Inputs:

- previous resident stack on JAX device
- latest rendered frame on JAX device

Outputs:

- next resident stack on JAX device

Large movement:

- this is a full device-side FIFO stack rewrite/logical concat. At B1024/P2
  uint8, the stack is about 32 MiB. No host copy is required, but device memory
  bandwidth and allocation can still matter.

Synchronization read:

- current best row has explicit resident-stack sync off.
- turning off this block is safe only if the next consumer naturally waits at a
  better point or real overlap occurs. The async renderer falsifier suggests
  raw wait movement alone does not buy much.

### 10. Build Compact Replay-Index Rows

Code path: `build_compact_replay_index_rows_v1_from_search_result(...)`.

Inputs:

- previous compact batch
- root batch
- compact search result
- next reward/done/final-observation mask
- next joint action

Outputs:

- CPU compact replay-index rows

Size:

- small relative to observation stacks and render state.

Read:

- current working notes put replay-index rows far below the main wall. Keep
  them correct, but do not make them the primary optimization target unless new
  same-denominator rows show otherwise.

### 11. Carry The Next Batch

The next loop iteration uses:

- `loop_batch = loop_next_step.compact_batch` on CPU;
- `compact_visual_resident_device_stack` on JAX device;
- updated actor/env CPU state;
- renderer persistent device layer and CPU previous-cursor metadata.

This is the actual loop closure. The important state is split across CPU env
ownership and GPU observation/search ownership.

## Large Versus Small Payloads

Large or potentially large:

- `[B,2,4,64,64]` observation stack:
  - about 32 MiB at B1024/P2 as uint8;
  - about 128 MiB as float32.
- latest frame `[B,2,1,64,64]`, about 8 MiB at B1024/P2 uint8.
- render trail/body state and visual trail arrays, especially when body
  capacity is 4096.
- persistent renderer delta arrays when many rows/slots advance or reset.
- Torch/JAX latent pools and tree/search tensors in LightZero/MCTX probes.

Small by bytes, still important by synchronization:

- action mask `[roots,3]`;
- selected actions `[roots]`;
- visit policy `[roots,3]`;
- root values `[roots]`;
- rewards/dones/final masks;
- row/player/env ids;
- replay-index rows.

The current wall is not explained by payload bytes alone. Several small arrays
are expensive because they force host/device ordering.

## Synchronization Map

Safe or currently necessary sync points:

- search output to CPU actions before CPU env step;
- compact search result validation and replay-index construction when those
  consumers are CPU;
- CPU CTree root/backprop boundaries in direct LightZero probes, because CTree
  currently requires CPU/list arrays;
- sampled parity/readback checks;
- explicit synchronized profile rows used for attribution, as long as they are
  paired with total-wall throughput rows.

Dangerous or suspect sync points:

- full frame D2H followed by H2D into search;
- blocking after resident stack update when search is the first real consumer;
- blocking on resident stack in `h2d_sec` merely to label a bucket, if total
  wall does not improve;
- async renderer flags that only move waits from render to stack/search;
- materializing host root observations in a resident-stack hot path for
  validation every loop;
- per-simulation LightZero direct-CTree `.detach().cpu().numpy()` and `.tolist()`
  calls;
- RND or metric CPU readbacks on the hot cadence.

Rule of thumb: synchronize where ownership truly crosses to CPU env, CPU CTree,
CPU replay validation, or sampled correctness checks. Avoid synchronizing just
to inspect data that the next device stage could consume.

## What The Current Timers Include

Top-level compact-loop buckets in `mctx_synthetic_benchmark.py`:

| Timer | Includes | Does not include |
| --- | --- | --- |
| `root_build_sec` | `CompactRootBatchV1` construction, metadata, masks, validation, optional observation copy | search, env step |
| `h2d_sec` | host observation H2D in host mode; resident stack readiness plus mask H2D in resident mode | MCTX search itself |
| `search_sec` | JAX `run_search(...)` and `action_weights.block_until_ready()` | CPU action scatter, env step |
| `d2h_sec` | `np.asarray` of actions/weights and root-value extraction | replay-index rows |
| `env_step_sec` | `manager.step(...)` plus resident stack update in the compact loop | root build, top-level H2D, search, D2H, replay-index |
| `replay_index_sec` | compact replay-index row construction | full learner target materialization |
| `total_sec` | full per-iteration wall | setup, manager init, JIT compile/warmup |

Nested actor/env timers:

- `actor_step_wall_sec`: wall around sequential in-process actor stepping.
- `actor_env_runtime_sec`: closest leaf to real game mechanics.
- `actor_env_public_prepare_sec`, `actor_env_reward_sec`,
  `actor_env_public_info_sec`, `actor_env_batch_pack_sec`: CPU packaging and
  public surface work around the runtime.
- `actor_compact_write_sec`: compact sidecar writes.
- `actor_render_state_write_sec`: parent render-state writes, zero in borrowed
  profile rows.

Nested observation/render timers:

- `observation_sec`: inclusive manager observation update plus reset/final
  observation handling.
- `renderer_stack_update_sec`: duplicate/inclusive label for observation update,
  not an extra exclusive span.
- `renderer_render_sec`: persistent renderer inclusive span:
  production-to-compact, delta pack, H2D, persistent update, compose/draw, and
  D2H if enabled.
- `renderer_production_to_compact_sec`: CPU conversion from production state to
  render compact fields.
- `renderer_persistent_delta_pack_sec`: CPU delta/reset segment construction.
- `renderer_host_to_device_sec`: H2D for delta/compose/render fields.
- `renderer_persistent_update_sec`: JAX persistent trail-layer update.
- `renderer_device_render_sec`: raw JAX frame composition/draw.
- `renderer_device_to_host_sec`: frame readback, zero in true device-only path.
- `resident_stack_update_sec`: JAX FIFO stack update from `last_output_device`,
  measured outside `_update_observation` but inside top-level `env_step_sec`.

LightZero/direct-CTree timers, when those probes are used:

- `lightzero_consumer_input_prepare_sec`: host tensor wrapping, pinning, or
  prenormalization.
- `lightzero_consumer_h2d_sec` / `lightzero_array_ceiling_h2d_sec`: host stack
  to Torch device.
- `lightzero_mcts_arrays_boundary_initial_inference_sec`: real MuZero initial
  inference.
- `lightzero_mcts_arrays_boundary_model_output_d2h_sec`: root or recurrent
  model output transfer to CPU for CTree.
- `lightzero_mcts_arrays_boundary_root_prepare_sec`: CPU root setup, legal
  action lists, noise, policy logits lists, `roots.prepare(...)`.
- `lightzero_mcts_arrays_boundary_search_sec`: CTree search wall.
- `lightzero_mcts_arrays_boundary_output_assembly_sec`: compact arrays or
  public output assembly.
- `lightzero_mcts_arrays_boundary_non_model_sec`: total minus model time, a
  useful measure of CPU/tree/list/search boundary tax.

Many LightZero timers deliberately synchronize CUDA. They are attribution
tools, not proof that every fence is required by the best future algorithm.

## Likely Remaining Expensive Copies And Materializations

1. Persistent render compact/delta construction on CPU:
   `_persistent_visual_compact_state_from_production_fast(...)` and
   `_persistent_delta_state(...)` still allocate, slice, copy, and loop over
   CPU arrays every step. Borrowing actor state removed the parent render-state
   copy, but the renderer still rebuilds its compact/delta input.

2. Per-step delta/compose H2D:
   `_copy_state_to_device(...)` sends delta state and compose state to JAX every
   render. Raw draw is tiny, so the transfer and input construction around it
   are more suspicious than the kernel itself.

3. Resident FIFO stack update:
   `jnp.concatenate((stack[:, :, 1:], latest), axis=2)` rewrites or reallocates
   a large device stack each step. This is better than D2H/H2D, but it is not
   free, and sync-off can hide its wait in the next search-input bucket.

4. Root batch compatibility work:
   no-copy root observations removed the obvious large host copy, but
   `CompactRootBatchV1` construction, validation, and host observation
   `np.asarray(...)` still happen in the loop. In resident mode this is mostly
   a validation/sidecar contract rather than the real search input.

5. Mask H2D and readiness fencing:
   the mask is tiny, but `jax.device_put(loop_invalid_host)` plus
   `block_until_ready()` can serialize the loop. Persistent device masks or
   fused mask production should be measured, not assumed.

6. Search-output readback:
   actions, weights, and values are small, but `np.asarray(...)` is the main
   CPU handoff before env step and replay. This is safe while the env is CPU,
   but it also absorbs any deferred device work.

7. CPU compact result and replay-index object construction:
   currently small, but still CPU materialization. It should remain a measured
   guardrail because it will become more visible as observation handoff shrinks.

8. LightZero direct-CTree boundary, for LightZero probes:
   direct GPU-latent keeps latent storage on device longer, but CTree still
   needs CPU/list reward, value, policy, root prep, traverse/backprop, and
   selected-output arrays. This is the main reason direct-CTree is a useful
   incremental profile lane rather than a full GPU MCTS architecture.

9. Scalar/public LightZero materialization, when enabled:
   `materialize_lightzero_scalar_timestep(...)`, public output decoding,
   `BaseEnvTimestep`-like objects, and payload pickling are intentionally
   disabled or avoided in the best compact row. They remain expensive edges if
   reintroduced into a Coach-facing loop.

10. Terminal/autoreset final-observation snapshots:
    borrowed render-state mode currently fails closed on terminals. A
    terminal-safe version needs a pre-reset snapshot protocol, which will add a
    real copy/readback/ownership point unless designed carefully.

## Plain Read

The current compact loop is not blocked by raw GPU drawing. It is not blocked
primarily by CurvyTron mechanics. It is blocked by turning the just-advanced
CPU game state into the next search input while preserving compact replay,
legal-mask, row/player, and final-observation semantics.

The most important ownership boundary is now:

```text
CPU actor/env state
-> CPU compact render/delta state
-> JAX persistent framebuffer
-> JAX resident observation stack
-> JAX search input
-> CPU selected action for next env step
```

The best current profile row succeeds because it removes two obvious costs:

- parent render-state copies, via borrowed single-actor render state;
- root observation copies, via no-copy root batches and resident search input.

The remaining wall is likely the residual around compact/delta construction,
per-step H2D/update, resident stack update/fencing, root sidecar validation,
mask transfer, search, and CPU action/readback control.

## Five Concrete Next Measurements

1. Add exclusive timers inside persistent render input construction: split
   live-slot scan, fast compact-state slicing/copying, delta row loop, delta
   array allocation, compose-state assembly, avatar/color cursor bookkeeping,
   and `_copy_state_to_device(...)` for delta versus compose.

2. Measure resident stack FIFO cost without changing semantics: compare current
   `jnp.concatenate` stack update against a preallocated/double-buffered device
   stack update, with explicit sync on, explicit sync off, and total-wall-only
   rows at sim16 and sim32.

3. Split resident search-input `h2d_sec`: report resident-stack readiness wait,
   mask `device_put`, mask readiness wait, and root-batch host validation
   separately so sync-off gains cannot hide in one bucket.

4. Run a sampled-validation root path: build full `CompactRootBatchV1`
   observation only every N loops while keeping masks/sidecars every loop, then
   compare total roots/sec and sampled parity against the current no-copy
   resident row.

5. Add a LightZero/direct-CTree companion row on the same compact batch shape
   that reports input tensor prep, initial inference, root output D2H,
   root prepare/listify, per-simulation recurrent D2H/listify, CTree
   traverse/backprop, and compact output assembly, so the MCTX profile wall can
   be compared to the real LightZero search boundary on the same denominator.
