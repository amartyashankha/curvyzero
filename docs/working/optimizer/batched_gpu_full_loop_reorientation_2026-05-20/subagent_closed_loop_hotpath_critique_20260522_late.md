# Closed Compact Loop Hot-Path Critique, Late 2026-05-22

Scope: read-only code inspection of:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`

No live Coach/training run was touched. No shared source file was edited.

## What `env_step_sec` Contains

In the repeated closed compact loop, `env_step_sec` is the wall-clock span from
just before:

```text
loop_next_step = compact_visual_manager_for_replay.step(loop_joint_action)
```

through the optional resident GPU stack update, ending just before replay-index
construction. In code, that is `loop_step_started` through `loop_step_sec` in
`mctx_synthetic_benchmark.py`.

So `env_step_sec` includes:

1. `HybridBatchedObservationProfileManager.step(...)`.
2. Actor stepping for every in-process actor.
3. With `native_actor_buffer=True`, `actor.step_into(...)`, which calls
   `VectorMultiplayerEnv.step(...)`, writes scalar compact sidecars into
   parent buffers, and writes renderer state rows into parent render buffers.
4. With `native_actor_buffer=False`, actor payload object creation, render-state
   copying, and parent `_merge_payloads(...)`.
5. `_update_observation(...)`, including the persistent GPU renderer call.
6. Terminal final-observation host copies if any row is done.
7. Autoreset observation reset/render work if any terminal row autoresets.
8. Compact root sidecar assembly in `_make_compact_batch(...)` and the capture
   probe call.
9. Return-side copies in `HybridObservationProfileStep`: reward, done,
   action_mask, compact payload, policy ids.
10. In `resident_gpu` compact visual mode only: after `manager.step(...)`, the
    benchmark calls `update_resident_compact_visual_stack(...)`, does a JAX
    FIFO-style `jnp.concatenate((stack[:, :, 1:], latest), axis=2)`, and blocks
    on the resulting resident stack.

It does not include:

- `root_build_sec`: `build_compact_root_batch_v1(...)`.
- `h2d_sec`: device_put/ready of root observation or resident stack handle plus
  invalid-action mask.
- `search_sec`: `run_search(...)`.
- `d2h_sec`: action/action-weight readback.
- `replay_index_sec`: `build_compact_replay_index_rows_v1_from_search_result(...)`.
- Most loop glue before stepping: `validate_compact_search_result_v1(...)`,
  root-value extraction, and `loop_joint_action` construction are in residual,
  not named buckets.

## Host vs Resident Stack Meaning

`compact_visual_observation_source=host`:

- `HybridBatchedObservationProfileManager` uses
  `update_host_observation_stack=True`.
- `_update_observation(...)` shifts the host `[B, P, 4, 64, 64]` uint8 stack.
- The persistent renderer reads `output_device` back to host.
- `_update_observation(...)` copies the latest host frame into the stack.
- The closed loop later device-puts `loop_obs_host` for MCTX search.

`compact_visual_observation_source=resident_gpu`:

- The manager is created with `update_host_observation_stack=False`.
- `_update_observation(...)` still calls the persistent renderer, but with
  `device_only=True`.
- The renderer still performs production-to-compact conversion, delta packing,
  host-to-device copies of compact/delta state, persistent layer update, and
  device compose.
- The renderer stores `last_output_device`.
- The manager's host observation remains the zero/placeholder stack, but the
  benchmark maintains a separate resident JAX stack from `last_output_device`.
- MCTX search consumes `compact_visual_resident_device_stack.reshape(...)`.

Important consequence: resident mode avoids the hot observation D2H/H2D bounce
for the search input, but it does not remove the renderer's CPU production
conversion, GPU render/update work, action-mask H2D, search readback, or replay
edge.

## Bucket Reality Check

The top-level closed-loop buckets are mostly real wall spans:

- `root_build_sec`: real CPU wall, but in resident mode it is partly a
  validation artifact because `build_compact_root_batch_v1(...)` still copies
  the host observation into a flattened root observation even though MCTX uses
  the resident device stack.
- `h2d_sec`: real synchronized wall. In host mode it includes observation plus
  invalid-mask transfer. In resident mode it mostly becomes invalid-mask
  transfer and synchronization on an already-resident stack.
- `search_sec`: real GPU wall because the code blocks on
  `loop_output.action_weights`.
- `d2h_sec`: real host readback of action and action weights.
- `env_step_sec`: real inclusive wall, but it is not "physics". It is mostly
  actor stepping plus render-state write plus renderer/observation/stack work,
  and in resident mode also the benchmark's resident stack FIFO update.
- `replay_index_sec`: real CPU wall when enabled. It is a useful collection-edge
  cost, but the current implementation is validation-heavy and copies checked
  arrays, so it may overstate a production optimized writer.

The nested `next_step_timings_sec` are attribution fields, not an additive
exclusive profile:

- `actor_step_wall_sec` is real wall over the in-process actor loop. It includes
  actor env runtime, compact writes, render-state writes, and autoreset work.
- `actor_env_*` fields are real internal timings from
  `VectorMultiplayerEnv.step(...)`, but they are leaves inside
  `actor_step_wall_sec`, not separate top-level wall.
- `actor_render_state_write_sec` and `actor_compact_write_sec` are real CPU
  copy/write time inside `actor_step_wall_sec`.
- `actor_idle_wait_sec` is residual math. In this in-process manager it is not
  true async actor idle time; treat it as uninstrumented overhead/noise.
- `observation_sec` is real wall around `_update_observation(...)` plus reset
  observation handling.
- `renderer_stack_update_sec` is set equal to `observation_sec`; it is a
  duplicate label, not an additional bucket.
- `renderer_render_sec` is inclusive renderer telemetry. It contains
  `renderer_production_to_compact_sec`,
  `renderer_persistent_delta_pack_sec`,
  `renderer_host_to_device_sec`,
  `renderer_persistent_update_sec`,
  `renderer_device_render_sec`, and sometimes
  `renderer_device_to_host_sec`.
- `renderer_production_to_compact_sec` is real CPU wall and is currently one of
  the most suspicious remaining costs.
- `renderer_host_to_device_sec`, `renderer_persistent_update_sec`, and
  `renderer_device_render_sec` are real synchronized JAX/GPU spans because the
  renderer blocks after device_put/update/compose.
- `renderer_device_to_host_sec` is real in host mode and intentionally zero in
  resident mode.
- `stack_shift_sec` and `stack_latest_update_sec` are real host stack copies in
  host mode. In resident mode they should be near-zero placeholders because the
  manager is not updating the host stack.
- `resident_stack_update_sec` is real current wall, but it is outside
  `HybridBatchedObservationProfileManager.step(...)` and inside
  `env_step_sec`. It also includes an explicit `block_until_ready`, so it may
  be more of a synchronization artifact than a required serial edge.

The residual is not ignorable. It can contain validation and Python glue:

- `validate_compact_search_result_v1(...)`.
- Root-value extraction.
- `loop_joint_action = np.full(...)` plus indexed assignment.
- Loop bookkeeping.

Those are outside the named bucket sum.

## Smallest Change Most Likely To Move Roots/Sec By More Than 20%

Do not spend the next optimization on MCTX search or replay-index toggling. In
the repeated B1024/sim16/native rows, search is single-digit percent and
replay-index is not the dominant path. Deleting either is not a credible 20%+
closed-loop win.

The smallest change with a credible 20%+ upside is:

```text
Add a persistent-renderer fast path where the manager/native actor buffer
supplies already-compact render state, so
_PersistentJaxPolicyFramebufferRenderer.render(...) does not rebuild compact
render state from production state on every closed-loop step.
```

Concretely, add a profile-only compact-state input contract beside the current
production-state path:

```text
native actor buffer writes/maintains:
  trail/head/avatar/bonus compact arrays in the renderer's expected layout

renderer.render(request):
  if request.state advertises already-compact persistent policy state:
    use it directly
  else:
    fall back to _persistent_compact_state_from_production(...)
```

Why this is the highest-probability >20% lever:

- Current rows already showed that actual env runtime is tiny compared with
  observation/render handoff.
- The persistent renderer still calls `_persistent_compact_state_from_production`
  every render.
- The matched H100 timing split had production-to-compact at about `0.517s` in
  the B1024/sim16/loop16/native row, while total closed-loop wall was around
  two seconds. Removing most of that one bucket is large enough to plausibly
  move total roots/sec by more than 20%.
- This attacks real wall, not just a reporting denominator.

There are two smaller benchmark-only cleanups worth doing, but I would not bet
on either alone for 20%:

- In resident mode, do not copy/materialize `CompactRootBatchV1.observation`
  for the hot loop when the search input is the resident stack. Keep sidecars
  and sampled validation only.
- Defer the explicit `resident_stack_update_sec` block until the stack is
  actually consumed, allowing CPU replay/root validation to overlap with the JAX
  stack FIFO update. This may improve wall time, but it can also simply move the
  wait into `h2d_sec` or `search_sec`.

## Validation Risks

- Host/resident stack parity can silently break. Keep the existing row-major
  guard and add sampled comparisons of resident stack rows against host stack
  rows on warmup, after several FIFO shifts, and after reset rows.
- Terminal/autoreset semantics are fragile. Resident latest-frame update must
  match terminal final-observation and autoreset stack reset policy, not just
  no-death clean loops.
- Row-major identity matters. Resident reshape assumes `[env row, player]`
  ordering exactly matches `CompactRootBatchV1.env_row/player`.
- Active-root accounting can hide work. Report env rows, root rows, active
  roots, inactive roots, terminal rows, and padded roots.
- Replay-index off is a different denominator. It is useful for isolating step
  and search, but it cannot be used as the collection throughput claim unless a
  replay writer replacement is also measured.
- JAX async timing can lie by relocation. Removing a `block_until_ready` may
  move time between `resident_stack_update_sec`, `h2d_sec`, and `search_sec`.
  Validate on total closed-loop wall, not just bucket movement.
- `root_build_sec` currently includes validation copies. If hot-loop root
  observation materialization is skipped, keep a sampled validation path so the
  speedup is not just deleting contract checks.
- Persistent renderer compact-state fast path must prove equivalence to
  `_persistent_compact_state_from_production(...)` for avatar color changes,
  bonus state, trail cursor wrap, trail breaks, dead/alive flags, and dynamic
  live-prefix lengths.
- Current profile mode uses `DEATH_MODE_PROFILE_NO_DEATH` in the actor harness.
  A real claim needs death/autoreset rows too.
- B1024/H100 rows can be noisy. Require A/B rows for host vs resident,
  replay-index on/off, and at least one longer loop repeat before calling a
  20% win real.

## Plain Read

`env_step_sec` is the closed-loop feed-the-next-search bucket. It includes real
actor/env work, but most of the current wall is observation/render/state
handoff, not game physics. Resident stack mode is directionally right because it
lets search consume a GPU stack, but the renderer still rebuilds compact render
state and synchronizes several boundaries each step.

The next high-value optimizer cut is to make the persistent renderer consume
state that is already in its compact layout. After that, resident stack plus
sampled host validation becomes a much cleaner denominator.
