# Current Code Dataflow And Sync Map, 2026-05-22

Status: docs-only sidecar note. I inspected the requested source and tests
first, then used current working docs only as helper context. I did not touch
live Coach training runs, checkpoints, evals, GIFs, tournaments, Modal volumes,
or source code.

Requested files inspected first:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`
- `tests/test_source_state_batched_observation_boundary_profile.py`
- `tests/test_source_state_hybrid_observation_profile.py`

## Plain Read

The current profile code has three nested dataflow layers:

1. **Boundary render profile:** CPU env step, CPU source-state compaction,
   JAX render, read back frames, update a host stack, optionally scalarize into
   LightZero-shaped rows.
2. **Hybrid manager profile:** batched CPU actor/env state, compact sidecars,
   optional renderer-backed stack, optional pre-scalar `HybridCompactBatch`
   consumer.
3. **Compact MCTX closed loop:** `HybridCompactBatch` to `CompactRootBatchV1`,
   JAX/MCTX search, CPU selected actions, next env step, optional compact replay
   index rows.

The code supports host-stack and resident-stack variants. The latest helper
rows say raw GPU drawing is already tiny. The wall is mostly the handoff around
CPU env state, compact/render-state ownership, stack/root-input preparation,
and MCTX search as simulations rise.

## Doc Freshness Notes

The current working docs are useful but not source of truth.

- Several helper docs have stale line numbers because the source has moved.
  This note names functions and timing fields instead of relying on those line
  numbers.
- Two 2026-05-22 helper notes disagree on exact "latest" compact resident
  borrowed numbers. One reports about `48.6k/36.0k` roots/sec for sim16/sim32;
  another reports about `50.4k/43.3k`. Both agree on the important read:
  observation/search-input handoff and search dominate; raw GPU draw is only
  about `4-6 ms` in the row family.
- Older docs that talk about profile-only direct CTree as the active optimizer
  lane are still relevant for stock LightZero critique, but the requested MCTX
  sidecar code now has a cleaner compact closed-loop map with explicit buckets.

## One Iteration: Boundary Render Profile

This is the lower-level profile in
`source_state_batched_observation_boundary_profile.py`.

1. Random CPU actions are generated as `joint_actions[B,2] int16`.
2. `VectorMultiplayerEnv.step(joint_actions)` advances CPU env state.
3. `_render_candidate_frames(...)` reads `env.state`.
4. `_production_to_benchmark_source_state(...)` converts production state into
   compact render arrays: trail x/y/radius/owner/active/break, head state,
   avatar color, and bonus state.
5. `_pack_compact_trails_in_owner_draw_order(...)` repacks trail slots in CPU
   owner draw order for the block surface.
6. `_select_render_trail_slots(...)` chooses the active/bucketed render width;
   `_truncate_compact_trails_for_render(...)` trims arrays to that width.
7. `_copy_state_to_device(...)` `jax.device_put`s every compact render array
   and usually blocks each device value ready.
8. The JAX two-view render function draws both player views and blocks on the
   output.
9. `np.asarray(output_device)` copies rendered frames back to CPU.
10. `_view_major_to_row_major_frames(...)` converts JAX output order from
    `[p0 rows, p1 rows]` into `[B,2,1,64,64]`.
11. `_push_row_major_frames_into_stack(...)` shifts host stacks and writes the
    newest frame into `[B,2,4,64,64]`.
12. If rows are done, the code snapshots `final_observation`, calls CPU
    autoreset, renders reset frames, and resets selected stack rows.
13. Optional compatibility work materializes a LightZero-shaped scalar
    timestep, pickles it, and can run RND probes.
14. Optional CPU oracle parity renders the same rows and compares frames/stacks.

Main timing fields for this layer:

| Edge | Code action | Timing bucket |
| --- | --- | --- |
| CPU mechanics | `env.step(...)` | `env_step_sec` |
| CPU source to compact arrays | `_production_to_benchmark_source_state` | `production_to_compact_sec` |
| CPU owner-order trail pack | `_pack_compact_trails_in_owner_draw_order` | `owner_ordered_pack_sec` |
| CPU compact arrays to JAX | `_copy_state_to_device` | `host_to_device_sec` |
| JAX draw plus ready wait | `render_fn(...).block_until_ready()` | `device_render_sec` |
| JAX frame to CPU | `np.asarray(output_device)` | `device_to_host_sec` |
| View-major to row-major | `_view_major_to_row_major_frames` | `view_major_to_row_major_sec` |
| Host FIFO stack write | `_push_row_major_frames_into_stack` | `stack_sec`, `reset_stack_sec` |
| Terminal snapshot | copy done rows into final obs | `final_obs_sec` |
| CPU autoreset | `env.autoreset_done_rows(...)` | `autoreset_sec` |
| Reset render | second candidate render after reset | `reset_render_sec` |
| Scalar LightZero edge | `materialize_lightzero_scalar_timestep` | `lightzero_scalarize_sec` |
| Pickle compatibility payload | `pickle.dumps(timestep)` | `lightzero_payload_pickle_sec`, bytes |
| RND probe | `collect_data`, `train_with_data`, `estimate` | `rnd_*_sec` |

## One Iteration: Hybrid Manager Step

This is the profile/train-like batched manager in
`source_state_hybrid_observation_profile.py`.

1. Caller provides `joint_action[B,2] int16`.
2. Each in-process actor steps a `VectorMultiplayerEnv` slice.
3. In normal payload mode, each actor returns copied arrays: reward, done,
   episode step, elapsed ms, round id, alive, action mask, joint action, terminal
   rows, autoreset rows, and optional render state.
4. In `native_actor_buffer=True`, actors write directly into parent compact
   buffers and optional parent render-state buffers.
5. In `borrow_single_actor_render_state=True`, the manager borrows one actor's
   `env.state` arrays instead of copying render state into parent buffers. The
   tests require this mode to fail closed on terminal rows.
6. The manager merges payloads or reuses the native compact buffers.
7. `_update_observation(...)` shifts host stack if enabled.
8. If no renderer is installed, the latest stack frame is zero-filled.
9. If a renderer is installed, the manager builds
   `SourceStateBatchedRenderRequest` with row-major rows/players and an output
   buffer shaped `[B*2,1,64,64]`.
10. The renderer returns frames and telemetry. Host-stack mode writes the latest
    frame into `[B,2,4,64,64]`; uint8 stacks keep bytes, float32 stacks
    normalize by `1/255`.
11. If rows are done, the manager snapshots final observations before reset and
    renders reset frames for autoreset rows.
12. `_make_compact_batch(...)` builds `HybridCompactBatch`, including
    observation, masks, rewards, dones, row/player ids, active-root mask,
    final-observation sidecars, episode metadata, alive, and joint action.
13. If a `batched_stack_probe` exists, it receives the compact batch before
    scalar materialization.
14. If `materialize_scalar_timestep=True`, the manager creates
    `MockBaseEnvTimestep` and flattened obs `[B*2,4,64,64]`.
15. If `pickle_payload=True`, it pickles the compact payload for size/tax
    measurement.

Main timing fields for this layer:

| Edge | Code action | Timing bucket |
| --- | --- | --- |
| Actor call wall | parent loops over actors | `parent_send_receive_sec`, `actor_step_wall_sec` |
| Actor env work | `VectorMultiplayerEnv.step` and internal timers | `actor_step_sec`, `actor_env_*_sec` |
| Native compact writes | actor writes parent compact buffers | `actor_compact_write_sec` |
| Render-state copy | actor/local render arrays into parent buffers | `actor_render_state_write_sec` and visual/player/bonus/other splits |
| Payload merge | `_merge_payloads(...)` | `gather_merge_sec` |
| Observation update inclusive | `_update_observation(...)` plus reset render if any | `observation_sec`, `renderer_stack_update_sec` |
| Renderer leaves | renderer telemetry | `renderer_production_to_compact_sec`, `renderer_persistent_delta_pack_sec`, `renderer_host_to_device_sec`, `renderer_persistent_update_sec`, `renderer_device_render_sec`, `renderer_device_to_host_sec` |
| Host stack movement | shift and latest write | `stack_shift_sec`, `stack_latest_update_sec` |
| Compact batch build | `_make_compact_batch(...)` | `compact_batch_build_sec` |
| Pre-scalar consumer | `_run_batched_stack_probe(...)` | `batched_stack_probe_*_sec` |
| Scalar LightZero compatibility | `materialize_lightzero_scalar_timestep` | `scalar_materialization_sec` |
| Compact pickle tax | `pickle.dumps(compact)` | `compact_payload_pickle_sec` |

## One Iteration: Compact MCTX Closed Loop

This is the most useful current sidecar map in
`mctx_synthetic_benchmark.py`. It does not call `train_muzero`.

One closed-loop iteration starts with a CPU `HybridCompactBatch`, a resident or
host visual stack, and a previous CPU joint action. It ends with a next
`HybridCompactBatch`, updated resident stack, search output arrays, and optional
compact replay-index rows.

1. Build `CompactRootBatchV1` from the current `HybridCompactBatch`.
2. Build CPU root sidecars: observation field, legal mask, active root mask,
   env row, player, reward/done/final sidecars, metadata.
3. In resident mode, validate row-major order and reshape
   `compact_visual_resident_device_stack` to `[R,4,64,64]`.
4. In host mode, copy `loop_root_batch.observation` to JAX.
5. Copy `invalid_actions[R,3] bool` to JAX.
6. The code blocks both `loop_obs` and `loop_invalid` ready before search.
7. `run_search(...)` normalizes uint8 visual roots, applies the small visual
   encoder, builds `mctx.RootFnOutput`, and calls
   `mctx.gumbel_muzero_policy(...)`.
8. Replay-valid rows block on `loop_output.action_weights`; action-only rows
   block only on `loop_output.action`.
9. The code reads `action[R]` to CPU.
10. Replay-valid rows also read `action_weights[R,3]` and root values. Root
    values now try direct search-tree fields before a summary fallback.
11. `validate_compact_search_result_v1(...)` checks selected actions, visit
    policy, root values, active roots, and root identity.
12. Selected actions are scattered into CPU `joint_action[B,2] int16`.
13. `HybridBatchedObservationProfileManager.step(joint_action)` advances the
    next CPU env state and prepares the next compact batch.
14. In resident visual mode, the benchmark appends
    `observation_renderer.last_output_device` into the resident stack with
    `jnp.concatenate((stack[:,:,1:], latest), axis=2)`.
15. If replay indexing is enabled, `build_compact_replay_index_rows_v1_from_search_result`
    builds compact target rows without copying full observations.
16. The next loop starts with `loop_next_step.compact_batch`.

Main closed-loop timing fields:

| Edge | Code action | Timing bucket |
| --- | --- | --- |
| Root object/sidecar build | `build_compact_root_batch_v1` | `root_build_sec` |
| CPU sidecar arrays | `np.asarray` obs/mask/active, active indices | `root_sidecar_sec` |
| Search input H2D/ready | device stack reshape or `device_put`, mask `device_put`, ready waits | `h2d_sec` |
| MCTX search | `run_search(...)` and output ready wait | `search_sec` |
| Search outputs to CPU | `np.asarray(action/action_weights)` | `d2h_sec` |
| Root value extraction | `_extract_mctx_root_values(...)` | `root_value_extract_sec` |
| Compact search result | `validate_compact_search_result_v1` | `search_result_validate_sec` |
| Action scatter | CPU selected roots to `joint_action[B,2]` | `joint_action_build_sec` |
| Next env/obs step | manager step plus resident stack update | `env_step_sec` |
| Replay index | compact replay index rows | `replay_index_sec` |
| Deferred full payload | delayed action weights/root values | `deferred_search_payload_flush_sec` |
| Overlapped payload wait | wait for payload worker result | `overlapped_search_payload_wait_sec` |

Important: `env_step_sec` in the closed MCTX loop is an inclusive top-level
bucket. Its nested `next_step_timings_sec` split contains actor/env mechanics,
render-state handoff, renderer leaves, observation update, resident stack
update, and compact batch build. Do not add top-level `env_step_sec` and the
nested leaves as if they were exclusive.

## Large And Small Data

Use `B` for env rows, `P=2`, `R=B*P`, `A=3`.

| Data | Shape | Rough size at B1024/P2 | Read |
| --- | --- | ---: | --- |
| Joint action | `[B,2] int16` | `4 KiB` | Small, CPU, required for env step. |
| Selected actions | `[R] int32` | `8 KiB` | Small, GPU to CPU, required today. |
| Legal/invalid mask | `[R,3] bool` | `6 KiB` | Small bytes, currently H2D every loop. |
| Visit policy | `[R,3] float32` | `24 KiB` | Small, needed for replay targets. |
| Root values | `[R] float32` | `8 KiB` | Small, needed for replay targets. |
| Latest frame | `[B,2,1,64,64] uint8` | `8 MiB` | Medium, should stay device-resident. |
| Uint8 stack | `[B,2,4,64,64] uint8` | `32 MiB` | Large, resident mode keeps it on GPU. |
| Float32 stack | `[B,2,4,64,64] float32` | `128 MiB` | Large, avoid in hot loop. |
| Host stack shift | `3/4` of stack | `24 MiB uint8` or `96 MiB float32` | Avoid when resident. |
| Root observation copy | `[R,4,64,64] uint8` | `32 MiB` | Current no-copy rows avoid the copy. |
| Boundary render output | `[B*2,1,64,64] uint8` | `8 MiB` at B1024, `512 KiB` at B64 | Host readback in host-stack modes. |
| Boundary float32 policy stack | `[B,2,4,64,64] float32` | `128 MiB` at B1024, `8 MiB` at B64 | Compatibility stack. |
| Visual trail pos | `[B,4096,2] float32` | `32 MiB` | Large render state. |
| Visual trail radius | `[B,4096] float32` | `16 MiB` | Large render state. |
| Visual trail owner | `[B,4096] int32` | `16 MiB` | Large render state. |
| Visual trail active/break | two `[B,4096] uint8` arrays | `8 MiB` | Large-ish render state. |
| Compact scalar sidecars | reward/done/ids/masks/alive | usually KiB to low MiB | Small next to stacks/trails. |

The byte story is clear: actions, masks, visit policies, and values are tiny.
They hurt when they force host/device ordering too often. The large payloads
are stacks, latest frames, trail/render state, and root observation copies.

## Unavoidable Transfers And Syncs Today

These boundaries are semantic in the current CPU-env design:

1. **GPU selected action to CPU before env step.** The CPU env cannot advance
   until it has `joint_action[B,2]`.
2. **CPU env state to render/search sidecars.** As long as
   `VectorMultiplayerEnv` owns mechanics on CPU, the current observation must be
   derived from CPU state.
3. **Renderer output before search.** Search must consume the current frame.
   This needs a device dependency, though not necessarily a host-visible block.
4. **Terminal final observation before autoreset.** Done rows must snapshot the
   terminal stack before reset mutates state.
5. **Replay target payload before replay commit.** If replay index rows are
   committed in the hot loop, visit policies and root values must be available
   for those active roots.
6. **Explicit profiling barriers.** `block_until_ready()` is needed when the
   purpose is honest bucket attribution.

## Accidental Or Optimizable Boundaries

These are not fundamental to the training semantics:

1. **Full frame D2H then H2D.** Host-stack paths copy rendered frames to CPU and
   later search/model code copies observations back to GPU.
2. **Host stack shift/update.** The stack FIFO is a large memory movement when
   search can consume a resident stack.
3. **Parent render-state copies.** Copying visual trail arrays into parent
   buffers is large. Borrowed single-actor state removes it in the no-terminal
   canary, but not yet for general multi-actor/terminal cases.
4. **Root observation copy.** `copy_observation=False` already removes a
   visible `32 MiB` copy at B1024/P2.
5. **Per-loop legal mask H2D.** The mask is tiny, but persistent device masks or
   sparse updates would remove a repeated ordering point.
6. **Resident `loop_obs.block_until_ready()`.** In resident mode the reshape is
   not a host data transfer. The wait may be useful for profiling, but an
   optimized production lane should let search become the first real consumer.
7. **Immediate visit-policy/root-value D2H for action-only stepping.** The env
   only needs selected actions. Visit policy and root value can be deferred or
   overlapped when replay indexing is not being committed immediately.
8. **Scalar timestep materialization and pickle.** These are compatibility
   measurements, not required by a compact pre-scalar search service.
9. **CPU oracle parity rendering every interval.** Correct for validation rows,
   not for the hot path.
10. **MCTX root-value summary fallback.** Current code tries direct tree fields
    first. Falling back to summary can accidentally materialize more than needed.

## Latest Row Mapping

Treat these as helper-doc readings, not code facts. The source code defines the
buckets; the docs carry the recent run numbers.

Latest compact MCTX helper row family:

```text
H100, B1024/P2, body4096, hidden64, depth16, loop24,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack, no root observation copy,
closed-loop replay-index on.
```

One 2026-05-22 note reports:

| Simulations | Active roots/sec | Loop wall | Env frac | Search frac | Observation | GPU draw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `50,357` | `0.9761s` | `53.7%` | `15.6%` | `0.2996s` | `0.0045s` |
| 32 | `43,340` | `1.1341s` | `40.9%` | `32.9%` | `0.2652s` | `0.0040s` |

Another same-day helper note reports older or alternate rows around:

| Simulations | Resident borrowed roots/sec | Read |
| ---: | ---: | --- |
| 16 | `48,579` | actor render-state write removed, observation still large |
| 32 | `36,041` | search grows to a larger share |

The stable mapping is:

- `search_sec` is MCTX `run_search(...)` plus synchronized output readiness.
- `env_step_sec` is the next CPU manager step plus observation/search-input
  handoff, not pure game mechanics.
- Nested `actor_env_runtime_sec`, `actor_env_reward_sec`, and
  `actor_env_post_runtime_bookkeeping_sec` are the actual mechanics leaf.
- Nested `actor_render_state_write_sec`, `observation_sec`,
  `renderer_*_sec`, `resident_stack_update_sec`, and
  `compact_batch_build_sec` are the handoff leaf.
- `renderer_device_render_sec` is raw GPU draw and is tiny in the latest row
  family, about `4-7 ms`.

Stock/full-loop helper rows are a separate denominator. The matched H100
C64/sim16 docs report about `433 steps/sec` stock and `566 steps/sec` direct
output-fast without RND, and about `351` versus `449 steps/sec` with the
hash-fixed RND meter. Those rows include stock `train_muzero` collection,
replay, learner, and object boundaries; do not compare them directly to compact
MCTX roots/sec.

## Design Critique

### 1. Make render/search input state single-owner

Change:

```text
CPU actor env state
-> renderer-owned compact/delta state
-> no parent render-state copy
-> terminal-safe final-observation snapshot
```

Why:

- Borrowed single-actor state already proves the parent render-state copy is a
  real wall.
- The remaining renderer path still rebuilds or packs CPU compact state before
  H2D.

Risks:

- Terminal/autoreset must snapshot final observation before reset.
- Multi-actor ownership needs a real slice protocol, not borrowed references
  that race or alias wrong rows.
- CPU oracle parity and row-major player order must stay exact.

### 2. Keep latest frame, FIFO stack, root observations, and masks resident

Change:

```text
JAX renderer latest frame
-> device FIFO stack
-> root view [R,4,64,64]
-> MCTX search
-> CPU selected actions only
```

Why:

- The large payload is the visual stack/root observation, not the action array.
- Resident rows and no-copy root batches already move the right direction.

Risks:

- Removing explicit sync can just move the wait into `h2d_sec` or `search_sec`;
  judge total wall, not bucket labels.
- JAX/Torch interop and validation sampling need a clean owner.
- Device memory lifetime and shape reuse must be boring and deterministic.

### 3. Promote compact search/replay/RND as an array contract, not a scalar adapter

Change:

```text
HybridCompactBatch
-> CompactRootBatchV1 sidecars
-> array-native search output
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> learner/RND materialization at controlled edges
```

Why:

- Stock LightZero object fanout is useful for compatibility but expensive in
  the hot path.
- Replay-index rows are already cheap; the next win is avoiding full
  observation/target/RND materialization every collection tick.

Risks:

- Must preserve legal masks, `to_play`, root noise, support transforms, value
  targets, final observations, RND latest-frame semantics, checkpoints, and eval
  behavior.
- This is a replacement lane with validation adapters, not a small renderer
  patch.
- A profile-only MCTX denominator is not Coach training speed until learner and
  replay integration are proven.

## Bottom Line

In the actual code, the hot dataflow is not "draw frame on GPU." It is:

```text
CPU action
-> CPU env state
-> compact/render state
-> GPU latest frame / stack
-> root sidecars and mask
-> MCTX search
-> CPU action and replay payload
-> next CPU env step
```

The unavoidable CPU/GPU crossing today is selected actions back to the CPU env.
The optimizable crossings are the large visual/render-state and root-input
copies, plus compatibility scalarization and immediate replay payload
materialization. The cleanest architecture direction is one compact resident
hot lane with stock LightZero-style objects only at validation, replay, learner,
and Coach compatibility edges.
