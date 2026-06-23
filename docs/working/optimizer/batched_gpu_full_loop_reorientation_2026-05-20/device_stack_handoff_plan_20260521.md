# Device Stack Handoff Plan

Date: 2026-05-21

## Current Truth

The persistent GPU renderer and cursor-bound collision patch changed the
Amdahl picture:

- B512/100 surface profile: dynamic direct64 `~81ms` -> persistent direct64
  `~45ms`.
- B512/500 persistent before env patch: total `~79ms`, render `~16ms`, env
  `~39ms`, stack `~39ms`.
- B512/500 persistent after env patch: total `~66ms`, render `~16ms`, env
  `~14ms`, stack `~45ms`.

Plain read: the renderer is no longer the whole problem. The next large local
wall is host stack/materialization.

## Smallest Safe Next Seam

Use the hybrid profile canary, not the production trainer surface:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
HybridBatchedObservationProfileManager.step()
-> _update_observation()
-> optional HybridBatchedStackProbe.run(observation, action_mask)
-> optional scalar materialization
```

Why this seam:

- It is already profile-only.
- It already supports `stack_storage_dtype="uint8"`.
- It can run a pre-scalar consumer with `hybrid_batched_stack_probe_simulations`.
- It can skip scalar materialization with
  `hybrid_materialize_scalar_timestep=False`.
- It does not touch stock LightZero, Coach launcher defaults, tournament/eval,
  or live runs.

Why not start in `SourceStateMultiplayerTrainerSurface`:

- That surface currently promises dense chronological float32 NumPy
  observations.
- A host-only ring buffer would likely move the copy from stack update into
  packaging/policy materialization.
- Changing that contract before the profile seam proves a win would be
  unnecessary churn.

## First Probe

Run H100 hybrid canary rows with:

```text
--hybrid-observation-canary
--observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile
--render-surface direct_gray64
--hybrid-stack-storage-dtype uint8
--hybrid-materialize-scalar-timestep false
--hybrid-batched-stack-probe-simulations > 0
```

Compare against the same shape with `hybrid_materialize_scalar_timestep=true`
and/or `stack_storage_dtype=float32`.

The first goal is not trainer speed. The first goal is to answer:

```text
Did we remove the host stack/materialization wall, or did we move it into
host_to_device / normalize / gather / readback?
```

## First Probe Results

H100, B512/A16, 100 measured steps, 20 warmup steps, persistent direct64
renderer, synthetic batched stack probe:

| stack dtype | scalar materialization | steps/sec | observation sec | stack probe H2D sec | scalarization sec |
|---|---:|---:|---:|---:|---:|
| `uint8` | off | `16309.79` | `2.748s` | `0.178s` | `0.000s` |
| `uint8` | on | `9801.46` | `2.947s` | `0.196s` | `3.590s` |
| `float32` | off | `9208.00` | `5.978s` | `0.876s` | `0.000s` |

Read:

- `uint8` is the right compact stack dtype for this profile seam.
- Scalar materialization is expensive enough to be a named edge cost.
- The profile now has a plausible 5-10x-class architecture direction, but only
  if policy/search consumes the compact batched stack before scalarization.
- This is still synthetic and profile-only. It is not LightZero MCTS and not a
  live-training setting.

## Device-Latest Variant

A first explicit profile-only device-latest flag was added:

```text
--hybrid-batched-stack-probe-device-latest
```

It requires `jax_gpu_persistent_policy_framebuffer_profile` and reads the
renderer's latest JAX device frame inside the synthetic stack probe.

Result:

| row | steps/sec | probe H2D sec | observation sec |
|---|---:|---:|---:|
| host `uint8` pre-scalar probe | `16309.79` | `0.178s` | `2.748s` |
| device-latest pre-scalar probe | `11595.65` | `0.042s` | `4.039s` |

Read:

- The flag proves the H2D portion can be reduced.
- It is not a win yet because the manager still updates the host stack and the
  probe maintains an additional device stack.
- The real next gate is to skip host stack update entirely when scalar
  materialization is off and a device consumer is present.

## Telemetry Required

Keep or add fields that make hidden copies visible:

- `stack_storage_dtype`
- `stack_shape`
- `stack_bytes_per_step`
- `stack_shift_sec`
- `stack_latest_update_sec`
- `renderer_device_to_host_sec`
- `renderer_device_to_host_bytes`
- `batched_stack_probe_host_to_device_sec`
- `batched_stack_probe_normalize_sec`
- `batched_stack_probe_device_sec`
- `batched_stack_probe_readback_sec`
- `batched_stack_probe_input_dtype`
- `batched_stack_probe_input_shape`
- `batched_stack_probe_input_bytes_total`
- `scalar_materialization_sec`
- `materialized_timestep_count`

Persistent renderer rows should also keep:

- `persistent_delta_pack_sec`
- `persistent_update_sec`
- `persistent_delta_slot_count`
- `persistent_reset_row_count`
- `persistent_partial_render_request`

## Guardrails

- No trainer defaults.
- No tournament/eval changes.
- No live runs.
- No hidden CPU fallback.
- Do not call this LightZero MCTS; the batched stack probe is synthetic.
- Do not promote this until row/player order, newest-frame order, terminal
  final-observation, and skipped-scalar behavior stay covered by tests.
