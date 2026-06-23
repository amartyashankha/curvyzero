# Device-Resident Observation Boundary Reorientation

Date: 2026-05-22

Scope: profile-only CurvyTron compact closed-loop / MCTX path. This is not live
Coach training advice, and no source files were edited for this note.

## Current Dataflow

The current repeated compact MCTX loop is still:

```text
CPU/in-process compact actors
-> parent native render-state buffers
-> JAX persistent policy framebuffer render
-> np.asarray(output_device) device-to-host uint8 frames
-> host [B,2,4,64,64] uint8 stack shift/update
-> HybridCompactBatch / CompactRootBatchV1
-> jax.device_put(obs_host) for MCTX search
-> device search
-> host actions/action_weights/root values
-> CPU env/observation step and compact replay-index rows
```

The recent H100 rows say this boundary is now the wall, not search. With
`native_actor_buffer=True`, repeated closed-loop rows still spend `74.4%` to
`81.3%` in `env_step_sec`; native sub-buckets show observation/stack update at
`0.575s` of `0.893s` for B512 and `0.860s` of `1.493s` for B1024, with renderer
render inside that bucket at `0.516s` and `0.705s`. Search is only `2.9%` to
`4.9%`.

The specific double bounce is visible in code:

- `HybridBatchedObservationProfileManager.step()` calls `_update_observation()`,
  builds `HybridCompactBatch`, then runs the optional batched probe in
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:587` and
  `:687`.
- `_update_observation()` shifts `self._zero_stack` on host, calls
  `observation_renderer.render(...)`, converts `result.frames` with
  `np.asarray`, and copies latest uint8 frames into the host stack at
  `source_state_hybrid_observation_profile.py:835`, `:856`, `:857`, `:867`.
- `_PersistentJaxPolicyFramebufferRenderer.render()` already has the desired
  producer value: `output_device = self._compose_fn(...)`, stores
  `self.last_output_device = output_device`, then immediately reads it back via
  `frames = np.asarray(output_device)` at
  `source_state_batched_observation_boundary_profile.py:2912`, `:2914`, `:2918`.
- `mctx_synthetic_benchmark.py` then rebuilds roots from host stack and does
  `obs = jax.device_put(obs_host)` for the fresh boundary at `:856`, `:882`;
  the repeated closed loop repeats the same pattern at `:2070`, `:2079`,
  `:2097`.
- There is already a synthetic precedent:
  `_JaxHybridBatchedStackProbe` can use a renderer `device_latest_provider`,
  keep a JAX `_device_stack`, and avoid host stack H2D for the synthetic probe at
  `source_state_batched_observation_boundary_profile.py:3522`, `:3562`, `:3569`.
  That path is not the current MCTX closed loop.

## Smallest Credible Experiment

Add a profile-only MCTX device-observation side path that leaves all host
contracts in place for validation/replay, but feeds MCTX from a resident JAX
device stack built from `_PersistentJaxPolicyFramebufferRenderer.last_output_device`.

Concretely:

```text
renderer.last_output_device [B,2,1,64,64] uint8
-> resident JAX stack FIFO [B,2,4,64,64] uint8
-> reshape to [B*2,4,64,64]
-> pass that device array to run_search(...)

Host CompactRootBatchV1 still supplies:
  legal_mask / active_root_mask
  env_row/player identity
  compact replay/replay-index validation
```

This avoids the large observation leg:

```text
GPU render -> host uint8 stack -> JAX device_put(obs_host)
```

while keeping the small, safer host pieces:

```text
legal/action mask device_put
actions/action_weights/root-value readback
CPU env step and replay-index proof
```

## Ranked Minimal Experiments

1. **MCTX device-stack input, host metadata unchanged**

   Hook in `mctx_synthetic_benchmark.py` only, behind a flag such as
   `--device-resident-observation` for
   `curvytron_hybrid_compact_visual_sample`.

   Use `compact_visual_manager_for_replay.observation_renderer.last_output_device`
   after each `manager.step(...)`. Maintain a local JAX resident stack with the
   same FIFO semantics as `_JaxHybridBatchedStackProbe`: initialize zeros,
   concatenate `stack[:, :, 1:]` with latest frame, reshape to root-major, and
   pass directly to `run_search`. Still build `CompactRootBatchV1` from the host
   `HybridCompactBatch` for masks, root identity, validation, and replay rows.

   Expected speedup: highest near-term. It removes renderer D2H observation
   readback, host stack shift/latest copy for the MCTX input, and obs H2D. It
   will not remove actor stepping or replay-index work, so a 1.2x-1.6x closed-loop
   win would be a strong result; larger would mean stack traffic was hiding more
   than expected.

   Risk: medium. It can silently diverge from host stack semantics around first
   frames, autoreset rows, terminal `final_observation`, and row-major ordering.
   Keep host `CompactRootBatchV1` as the oracle and compare sampled device stack
   rows against host stack on warmup and after any reset row.

2. **Promote the existing synthetic `device_latest_provider` probe to an MCTX canary**

   Hook in `source_state_batched_observation_boundary_profile.py` first, not
   MCTX production-like loop. Add a new batched probe variant beside
   `_JaxHybridBatchedStackProbe` that consumes the resident device stack and runs
   the same toy MCTX/JAX search shape used by `mctx_synthetic_benchmark.py`.

   Expected speedup: medium as a falsifier, because it isolates whether the
   resident stack path is actually fast before touching replay/search contracts.

   Risk: low to medium. It may overstate the win because it avoids the real
   `CompactRootBatchV1` and replay-index edge. Treat it only as a plumbing
   canary.

3. **Renderer-owned device stack plus optional host mirror**

   Move the FIFO stack into `_PersistentJaxPolicyFramebufferRenderer` or a small
   wrapper returned by `_make_profile_observation_renderer`, so `render()` can
   update `last_stack_device` next to `last_output_device`. Host `_zero_stack`
   remains as a sampled parity mirror, not the search input.

   Expected speedup: similar to experiment 1, but cleaner if it passes.

   Risk: higher. This starts changing ownership semantics: reset/autoreset,
   terminal final observations, partial render requests, and stack dtype all
   become renderer responsibilities. It is probably the second implementation
   after experiment 1 proves the denominator moves.

## What Could Go Wrong

- **Reset/final-observation drift.** Current host code captures
  `final_observation` before autoreset, then zeroes/reset-renders rows. A device
  FIFO must reproduce that exact ordering or only claim no-death rows.
- **Row/player order mismatch.** MCTX roots are `[B*P,4,64,64]`; renderer latest is
  `[B,2,1,64,64]`. The reshape must match `policy_env_row = repeat(rows)` and
  `policy_player = tile(players)`.
- **Host validation can hide stale device input.** If host `CompactRootBatchV1`
  drives replay but MCTX searched a stale device stack, legality still passes.
  Add checksum/parity telemetry for sampled root observations.
- **Partial render/reset requests.** The persistent renderer validates full
  row-major requests. Reset-only renders currently allocate separate host `out`
  buffers. The first experiment should either disallow terminal/autoreset rows
  or explicitly patch resident rows after reset render.
- **JAX compile/cache churn.** A device stack path must keep static shapes:
  B, P, stack depth, and 64x64 fixed. Do not add dynamic active-root compaction
  to the first slice.
- **Mask remains host.** That is acceptable for the first experiment because mask
  is tiny compared with observations, but the report should show
  `obs_h2d_bytes=0` separately from `mask_h2d_bytes`.

## Recommendation

Do experiment 1 first. It is the smallest slice that answers the real question:
does avoiding GPU render -> host stack -> JAX device_put move the repeated
closed-loop denominator? Keep host compact batches and replay validation intact,
but make MCTX consume the renderer-backed resident JAX stack. If the closed-loop
roots/sec does not move materially, the next bottleneck is actor/env stepping or
replay-index materialization, not observation transfer.
