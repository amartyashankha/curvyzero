# Host Overhead And Sync Critique, 2026-05-22

Scope: read-only critique of the current profile-only compact visual loop. I
read:

- `world_model.md`
- `task_board.md`
- `experiment_log.md`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- renderer telemetry surface in
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

No live Coach/training run, checkpoint, eval, GIF, tournament artifact, or
source file was touched. This file is the only edit.

## Current Read

The old largest named host copy was real:

```text
actor env.state -> parent render-state buffers -> renderer compact adapter
```

The borrowed single-actor canary validates that diagnosis. In the current docs,
`borrow_single_actor_render_state=True` makes
`actor_render_state_write_sec == 0` and moves total roots/sec, not just a label.
That means the visual-trail parent-buffer copy was not noise.

The current remaining wall is therefore narrower and nastier:

```text
borrowed actor env.state
-> renderer production/compact/delta pack
-> renderer H2D/update/draw
-> resident stack FIFO/update and action-mask H2D
-> root/search result validation and compact replay-index sidecars
-> Python object/materialization residual
```

Raw GPU draw is not the wall. Actual env mechanics is not the wall. The risk is
that the next few rows will look like "observation" or "env_step" while the real
cost is host packing, validation copies, and synchronization relocation.

## Biggest Hidden Sync / Host Suspects

### 1. Renderer compact/delta pack is still CPU-shaped

In the persistent renderer, `render(...)` still starts by converting borrowed or
copied production state into compact renderer state, then building deltas:

- `_PersistentJaxPolicyFramebufferRenderer.render(...)`
- `_persistent_compact_state_from_production(...)`
- `_persistent_visual_compact_state_from_production_fast(...)`
- `_persistent_delta_state(...)`

Code references:

- `source_state_batched_observation_boundary_profile.py:2855`
- `source_state_batched_observation_boundary_profile.py:2863`
- `source_state_batched_observation_boundary_profile.py:2879`
- `source_state_batched_observation_boundary_profile.py:3034`
- `source_state_batched_observation_boundary_profile.py:3226`

Why I distrust it:

- The fast visual adapter avoids rebuilding the full old arrays, but it still
  walks rows and live trail ranges on the host.
- `_persistent_delta_state(...)` has Python loops over rows and trail slots.
- Even in borrowed mode, this is not "state already resident"; it is "borrowed
  host state repacked every step."

Already covered by timers:

- `renderer_production_to_compact_sec`
- `renderer_persistent_delta_pack_sec`
- `renderer_render_sec`
- nested inside `observation_sec`
- nested inside closed-loop `env_step_sec`

Missing fields:

- `renderer_compact_state_bytes`
- `renderer_delta_bytes`
- `renderer_delta_slot_count`
- `renderer_live_cursor_max`
- `renderer_delta_pack_row_loop_sec`
- `renderer_delta_pack_slot_loop_sec`
- `renderer_compose_state_pack_sec`
- compact-state fast-path mode: `borrowed_production`, `parent_copy`, or
  `already_compact`

### 2. Renderer H2D/update timings are synchronized, but attribution can move

The persistent renderer copies delta/compose state to device and blocks in
`_copy_state_to_device(...)`, then blocks after persistent update and compose.

Code references:

- `source_state_batched_observation_boundary_profile.py:2892`
- `source_state_gpu_render_benchmark.py:1741`
- `source_state_batched_observation_boundary_profile.py:2907`
- `source_state_batched_observation_boundary_profile.py:2912`

Already covered by timers:

- `renderer_host_to_device_sec`
- `renderer_persistent_update_sec`
- `renderer_device_render_sec`
- `renderer_device_to_host_sec`

Why I still distrust it:

- These are real synchronized spans, but H2D wait, update wait, and compose wait
  can move if one `block_until_ready()` is removed or deferred.
- `device_render_sec` is tiny in the latest rows, so if total wall is still high
  the useful question is not "draw faster"; it is "why are we repacking and
  synchronizing this state every decision?"

Missing fields:

- explicit sync count per closed-loop step
- explicit renderer device-put byte count split by delta state vs compose state
- queue-only versus wait time for renderer H2D/update/render
- number of device arrays copied per render call

### 3. Resident stack update may be a synchronization tax

Resident stack mode avoids host stack readback for search input, but the
benchmark maintains a separate JAX FIFO stack with:

```text
jnp.concatenate((device_stack[:, :, 1:], latest_device), axis=2)
```

and, by default, blocks immediately.

Code references:

- `mctx_synthetic_benchmark.py:801`
- `mctx_synthetic_benchmark.py:1147`
- `mctx_synthetic_benchmark.py:1166`
- `mctx_synthetic_benchmark.py:2597`
- `mctx_synthetic_benchmark.py:2615`

Already covered by timers:

- `resident_stack_update_sec`
- part of closed-loop `env_step_sec`
- `h2d_sec` later blocks on the resident stack before search

Why I distrust it:

- If `compact_visual_resident_sync=False`, the same wait may simply move from
  `resident_stack_update_sec` into `h2d_sec` or `search_sec`.
- If total wall improves when the sync is deferred, the current row has a real
  avoidable serialization point.
- If total wall does not improve, the timer was mostly attribution relocation.

Missing fields:

- resident stack queue time vs wait time
- whether `h2d_sec` waited on an already-pending resident stack update
- resident stack bytes logically shifted/written
- separate action-mask H2D time in resident mode

### 4. Root/search-input prep still materializes more host objects than search consumes

The closed loop builds `CompactRootBatchV1` every iteration even when resident
mode feeds MCTX from `compact_visual_resident_device_stack`, not from
`loop_root_batch.observation`.

Code references:

- `mctx_synthetic_benchmark.py:2485`
- `compact_policy_row_bridge.py:124`
- `compact_policy_row_bridge.py:224`
- `mctx_synthetic_benchmark.py:2504`
- `mctx_synthetic_benchmark.py:2521`

Already covered by timers:

- `root_build_sec`
- `h2d_sec`
- root contract metadata includes `observation_copied`
- `compact_root_copy_observation=False` removes the biggest observation copy

Why I distrust it:

- Even no-copy root observation still validates and copies sidecars:
  `legal_mask`, `active_root_mask`, `to_play`, `env_row`, `player`,
  `policy_env_id`, `target_reward`, `done_root`, and masks.
- In resident mode, the host observation is still materialized for validation
  and shape checks even though it is not the hot search input.
- Root build is now smaller, but the remaining work is exactly the kind of
  Python/NumPy object churn that can become visible after borrowed state.

Missing fields:

- `root_observation_view_sec`
- `root_observation_copy_sec`
- `root_sidecar_validate_sec`
- `root_sidecar_copy_sec`
- `root_metadata_sec`
- `root_batch_python_object_count`
- `root_sidecar_bytes_copied`

### 5. Search-result D2H is under-labeled

Closed-loop `d2h_sec` measures action and action-weight readback:

- `np.asarray(loop_output.action)`
- `np.asarray(loop_output.action_weights)`

Then `_extract_mctx_root_values(...)` may perform another `np.asarray(...)` on
MCTX search-tree values outside `d2h_sec`.

Code references:

- `mctx_synthetic_benchmark.py:2550`
- `mctx_synthetic_benchmark.py:2554`
- `mctx_synthetic_benchmark.py:641`

Already covered by timers:

- `d2h_sec` for action/action weights only
- residual indirectly captures root-value extraction plus validation

Missing fields:

- `d2h_action_sec`
- `d2h_visit_policy_sec`
- `d2h_root_value_sec`
- `search_result_validate_sec`
- `joint_action_build_sec`

### 6. `env_step_sec` includes profile glue that is not in the nested split

Inside `HybridBatchedObservationProfileManager.step(...)`, the compact batch and
capture probe are timed as `batched_stack_probe_sec`, but the MCTX closed-loop
`next_step_timings_sec` whitelist does not include that field. Return-side
copies also lack an explicit field.

Code references:

- `source_state_hybrid_observation_profile.py:889`
- `source_state_hybrid_observation_profile.py:1001`
- `source_state_hybrid_observation_profile.py:1664`
- `mctx_synthetic_benchmark.py:2638`

Already covered by timers:

- top-level closed-loop `env_step_sec`
- `batched_stack_probe_sec` exists in the manager timing dictionary

Why I distrust it:

- The closed-loop grouped stdout can make `observation_sec` and render leaves
  look like the whole `env_step_sec` story.
- Some of the compact-batch sidecar assembly, capture-probe dispatch, returned
  `HybridObservationProfileStep` copies, and `_copy_compact_payload_for_step`
  bytes are still only visible as residual.

Missing fields:

- `compact_batch_build_sec`
- `capture_probe_sec`
- `hybrid_step_return_copy_sec`
- `compact_payload_copy_sec`
- `closed_loop_env_step_unattributed_sec`
- `closed_loop_total_residual_sec` already exists, but it needs a sibling
  residual inside `env_step_sec`.

## Timer Fields That Are Already Good

These fields are useful and should stay:

- `actor_env_runtime_sec`, `actor_env_reward_sec`,
  `actor_env_post_runtime_bookkeeping_sec`: the strict mechanics group.
- `actor_render_state_write_sec` plus
  `actor_render_state_write_visual_trail_sec`,
  `actor_render_state_write_player_sec`,
  `actor_render_state_write_bonus_sec`,
  `actor_render_state_write_other_sec`: already proved the parent visual-trail
  copy.
- `observation_sec`: real inclusive manager observation/update wall.
- `renderer_production_to_compact_sec`,
  `renderer_persistent_delta_pack_sec`,
  `renderer_host_to_device_sec`,
  `renderer_persistent_update_sec`,
  `renderer_device_render_sec`,
  `renderer_device_to_host_sec`: the right renderer leaf vocabulary.
- `root_build_sec`, `h2d_sec`, `search_sec`, `d2h_sec`,
  `replay_index_sec`: the right top-level closed-loop buckets.
- `resident_stack_update_sec`: valuable, as long as we remember it can be a
  synchronization-placement artifact.
- `residual_sec` and `residual_fraction_of_total`: essential; do not hide this
  behind prettier grouped labels.

Fields to treat carefully:

- `renderer_stack_update_sec` is currently equal to `observation_sec`, so it is
  a duplicate label, not an additive bucket.
- `renderer_render_sec` is inclusive of renderer leaves.
- `actor_idle_wait_sec` is residual math in the in-process manager, not proof of
  real async actor idle time.
- grouped `observation_handoff_leaf_sec` is diagnostic; some leaves are nested.

## What Is Missing Before The Next Big Claim

Minimum missing accounting for the borrowed/resident row:

```text
closed_loop_env_step_unattributed_sec =
  env_step_sec
  - actor_step_wall_sec
  - observation_sec
  - resident_stack_update_sec
  - known step-side extras
```

Then split the unattributed part into:

- compact batch sidecar build
- capture probe dispatch
- return-side copies
- final-observation copy
- root/search-result validation
- joint-action assembly
- root-value D2H

Minimum missing byte counters:

- render-state visual-trail bytes avoided or copied
- compact renderer state bytes
- renderer delta bytes
- renderer compose-state H2D bytes
- root sidecar bytes copied
- root observation bytes copied or viewed
- action-mask H2D bytes
- MCTX policy readback bytes
- root-value readback bytes

Minimum missing sync counters:

- renderer H2D array count and sync count
- renderer update/render sync count
- resident stack sync count
- search output sync count
- D2H readback count
- any `np.asarray(JAX value)` outside named D2H fields

## Two Small Profile-Only Falsifiers

### Experiment 1: Resident-stack sync relocation

Run the current best borrowed resident denominator twice:

```text
H100
B1024/P2
actor_count=1
native_actor_buffer=True
borrow_single_actor_render_state=True
compact_visual_observation_source=resident_gpu
compact_root_copy_observation=False
closed_loop_replay_index=True
closed_loop_steps=16 or 24
sim16 and sim32 if cheap
```

Only toggle:

```text
compact_visual_resident_sync=True
compact_visual_resident_sync=False
```

Falsifies my view if:

- total active roots/sec is unchanged and time merely moves from
  `resident_stack_update_sec` into `h2d_sec` or `search_sec`; then resident
  stack sync is mostly attribution, not a serial wall.

Supports my view if:

- total active roots/sec improves meaningfully with sync deferred, and
  `h2d_sec + search_sec` does not absorb the whole delta.

Keep rule:

- judge total closed-loop roots/sec, not the resident-stack timer alone.

### Experiment 2: Residual accounting patch, no behavior change

Add profile-only timers only, then rerun the same borrowed resident row. Do not
change dataflow.

Add fields around:

- `_make_compact_batch(...)`
- `_run_batched_stack_probe(...)`
- `HybridObservationProfileStep(...)` construction and payload copies
- `_extract_mctx_root_values(...)`
- `validate_compact_search_result_v1(...)`
- `loop_joint_action` construction
- `build_compact_replay_index_rows_v1_from_search_result(...)` internal
  validation versus final array construction

Falsifies my view if:

- the new residual split shows less than about 5% total wall in these Python
  materialization/validation edges and the unexplained residual collapses.
  Then the next patch should focus almost entirely on renderer residency/search.

Supports my view if:

- sidecar/object/validation/root-value/joint-action fields account for a
  double-digit fraction of total wall after borrowed state. Then the next state
  owner must remove host materialization, not just move renderer arrays around.

Keep rule:

- this is instrumentation only; no trainer defaults, no live training, no
  reward/search/replay semantics change.

## Plain Recommendation

Treat the borrowed-state win as proof that ownership is the right axis, but do
not overfit to the old `actor_render_state_write_sec` leaf. In the current best
profile rows, the next hidden walls are renderer state repacking/H2D, resident
stack synchronization placement, root/search sidecar validation, and Python
object materialization around the compact batch/result edge.

The next useful patch is not a faster draw kernel. It is an exclusive
borrowed/resident timing split plus one sync-relocation falsifier. If the split
shows real residual/object cost, the next state owner should make compact
render/search state persistent and sampled-valid, not mandatory-host-valid on
every decision.
