# Mock Search-Service Ceiling Plan

Date: 2026-05-22

Status: first sidecar mode implemented. No trainer defaults, live Coach runs,
checkpoints, evals, GIFs, or tournaments are touched by this plan or the first
implementation.

Implementation update:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
mode: mock_search_service
```

The first implementation reuses the existing compact masked-policy array path,
but gives it explicit search-service-ceiling identity and telemetry. It uses
real batched CurvyTron observations, real legal masks, and real scratch MuZero
`initial_inference`; it performs zero CTree calls and zero recurrent rollout
calls.

## Plain Goal

We need a small experiment that answers one question:

```text
If search returned compact batched arrays cheaply, would the rest of the
CurvyTron/LightZero loop have enough headroom for a 5-10x architecture win?
```

This is not a learning experiment and not a MuZero replacement. It is a ceiling
test for the architecture thesis:

```text
Small wrapper patches give about 1.3x.
Big wins require owning a compact batched search boundary.
```

## Why This Experiment

The latest matched full-loop rows show:

- no-RND: stock `433.17` steps/sec -> `direct_ctree_gpu_latent + output-fast`
  `566.19`, about `1.31x`;
- `rnd_meter_v0`: stock `351.02` -> direct `448.52`, about `1.28x`.

The H100 compile spike did not open the next door:

```text
sim8:  dense compile 10298 roots/sec vs direct_ctree_gpu_latent 7567
sim16: dense compile 4872 roots/sec vs direct_ctree_gpu_latent 6154
```

So the next clean falsifier is not another wrapper polish. It is a mock
search-service ceiling: keep the real batched CurvyTron input boundary, remove
real CTree/search, and measure the best practical compact-output path.

## Insertion Point

Use the existing profile-only sidecar:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
```

Do not add anything to the live trainer launcher except optional docs later.
The trainer launcher:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

is only a denominator/reference source for current full-loop timings. The mock
service should not call `train_muzero`.

Smallest code shape when implemented:

1. Add one array-ceiling mode near the existing constants:

   ```python
   LIGHTZERO_ARRAY_CEILING_MODE_MOCK_SEARCH_SERVICE = "mock_search_service"
   ```

   Added to `LIGHTZERO_ARRAY_CEILING_MODES`.

2. Reuse the existing route:

   ```text
   --hybrid-observation-canary
   --hybrid-lightzero-array-ceiling-probe
   --hybrid-lightzero-array-ceiling-mode mock_search_service
   ```

3. Implement it inside `_LightZeroArrayCeilingStackProbe.run(...)`, after real
   observation tensor prep and real `model.initial_inference(obs_tensor)`.
   Done.

4. Add one optional flag only if needed:

   ```text
   --hybrid-lightzero-mock-service-materialize-public-output
   ```

   Implemented 2026-05-22. It builds LightZero-shaped public collect dicts
   from compact mock search arrays and reports explicit public-output timing,
   count, byte, and checksum fields. It is still profile-only and does not
   touch the trainer.

This keeps the experiment in the current profile harness:

```text
real VectorMultiplayerEnv rows
-> real GPU renderer / stack update
-> real batched [B,2,4,64,64] observation
-> real legal action mask
-> real scratch MuZero policy initial_inference
-> fake compact search-service arrays
-> optional public-output materialization at the edge
```

## What Is Real

The experiment must keep these real:

- real batched CurvyTron env stepping from `run_hybrid_observation_profile`;
- real no-death profile rows, actor partitioning, autoreset scaffold, and
  legal masks;
- real renderer backend and stack dtype from the profile config;
- real `[B, 2, 4, 64, 64]` uint8 policy observation stack;
- real action masks with binary validation and zero-mask filtering;
- real scratch LightZero MuZero policy/model construction;
- real CUDA H2D, normalization, and `model.initial_inference`;
- real compact output byte counts, illegal-action checks, checksums, and
  optional LightZero-shaped output fanout cost.

## What Is Faked

The experiment deliberately fakes these:

- no `policy.collect_mode.forward`;
- no `_mcts_collect.search`;
- no CTree `roots`, `batch_traverse`, or `batch_backpropagate`;
- no recurrent search rollout;
- no Dirichlet root noise;
- no tree backup;
- no real searched value.

Output meaning:

```text
predicted_value = root value from initial_inference
searched_value  = predicted_value
visit_policy    = masked softmax(policy_logits)
action          = deterministic best legal action from visit_policy
```

If public output is materialized, use the compact arrays to build dicts shaped
like LightZero collect output:

```text
action
visit_count_distributions
visit_count_distribution_entropy
searched_value
predicted_value
predicted_policy_logits
```

For fake visit counts, use `visit_policy * num_simulations` or an equivalent
fixed scale. The point is output shape and cost, not search semantics.

## CLI Rows

Use the same safe sidecar as the current direct/search rows. Example H100
compact row:

```bash
uv run --extra modal modal run --detach \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --hybrid-observation-canary \
  --compute gpu-h100 \
  --batch-size 512 \
  --actor-count 16 \
  --steps 60 \
  --warmup-steps 15 \
  --trail-slots 1024 \
  --body-capacity 1024 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --render-surface direct_gray64 \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-lightzero-array-ceiling-probe \
  --hybrid-lightzero-array-ceiling-mode mock_search_service \
  --hybrid-lightzero-array-ceiling-input-mode host_uint8_pinned \
  --hybrid-lightzero-consumer-num-simulations 16 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

Optional public-output edge row:

```bash
uv run --extra modal modal run --detach \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --hybrid-observation-canary \
  --compute gpu-h100 \
  --batch-size 512 \
  --actor-count 16 \
  --steps 60 \
  --warmup-steps 15 \
  --trail-slots 1024 \
  --body-capacity 1024 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --render-surface direct_gray64 \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-lightzero-array-ceiling-probe \
  --hybrid-lightzero-array-ceiling-mode mock_search_service \
  --hybrid-lightzero-array-ceiling-input-mode host_uint8_pinned \
  --hybrid-lightzero-consumer-num-simulations 16 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0 \
  --hybrid-lightzero-mock-service-materialize-public-output
```

The public-output flag is now present. Use it only to price the scalar/object
edge; do not confuse it with real MCTS or replay integration.

Reference denominator rows should use the same batch, actor count, render
surface, stack dtype, steps, warmup, and H100:

```text
direct_ctree_gpu_latent sim16
recurrent_toy sim16
stock_facade sim16, optional
```

The most important denominator is:

```text
hybrid_lightzero_mcts_arrays_boundary_impl=direct_ctree_gpu_latent
```

because it is the current best real-search profile boundary.

## Metrics

Reuse existing high-level profile fields:

- roots/sec or scalar steps/sec from the profile result;
- `batched_stack_probe_sec`;
- observation/render/stack timings;
- materialized scalar timestep count;
- `touches_live_runs=false`;
- `calls_train_muzero=false`;
- `trainer_defaults_changed=false`.

Add mock-service-specific telemetry:

```text
lightzero_array_ceiling_mode = mock_search_service
mock_search_service_total_sec
mock_search_service_tensor_prepare_sec
mock_search_service_h2d_sec
mock_search_service_normalize_sec
mock_search_service_initial_inference_sec
mock_search_service_mask_softmax_sec
mock_search_service_compact_output_sec
mock_search_service_readback_sec
mock_search_service_input_bytes
mock_search_service_compact_output_bytes
mock_search_service_active_roots
mock_search_service_zero_mask_roots
mock_search_service_requested_simulations
mock_search_service_recurrent_inference_calls
mock_search_service_real_ctree_calls
mock_search_service_illegal_action_count
mock_search_service_action_checksum
mock_search_service_value_checksum
mock_search_service_visit_shape
mock_search_service_semantics =
  mock_search_service_compact_arrays_profile_not_mcts
```

Also keep the existing array-ceiling aliases populated where practical:

```text
lightzero_array_ceiling_total_sec
lightzero_array_ceiling_initial_inference_sec
lightzero_array_ceiling_search_update_sec = 0.0
lightzero_array_ceiling_recurrent_inference_sec = 0.0
lightzero_array_ceiling_output_assembly_sec
lightzero_array_ceiling_readback_sec
lightzero_array_ceiling_illegal_action_count
```

That lets the existing summarizers compare it against `policy_arrays`,
`recurrent_toy`, and dense compile rows without a separate parser first.

## Validation Checks

This is not exact parity with MuZero search. The checks should instead prove
that the row is a clean ceiling:

1. The row must fail closed unless `hybrid_observation_canary=true`.
2. The row must fail closed unless `hybrid_stack_storage_dtype=uint8`.
3. The row must consume binary legal masks and filter zero-mask roots.
4. `illegal_action_count` must be zero.
5. Every selected action must be legal under the consumed mask.
6. Visit distributions must have shape `[active_roots, 3]`.
7. Visit distributions must have zero mass on illegal actions.
8. Visit distributions must sum to `1.0` for every active root.
9. `searched_value` and `predicted_value` shapes must match active roots.
10. Compact output bytes and optional public output bytes must be reported.
11. Public-output edge rows must report output count, bytes, seconds, and a
    checksum so rows cannot silently skip the materialization they claim.
12. Result must say:

    ```text
    calls_train_muzero=false
    stock_lightzero_integrated=false
    touches_live_runs=false
    trainer_defaults_changed=false
    ```

## How It Avoids Live Runs

It only runs through:

```text
source_state_batched_observation_boundary_profile.py
```

That sidecar is already profile-only. It does not call `train_muzero`, does not
write checkpoints, does not spawn eval/GIF jobs, does not use Coach run ids,
and should continue to return:

```text
profile_only=true
calls_train_muzero=false
touches_live_runs=false
trainer_defaults_changed=false
```

Run it with `modal run --detach` only as an isolated profile app. Do not use
the Coach launcher for this mock.

## Validate Or Falsify

Validate the radical architecture thesis if the compact mock-service row is
much faster than the current real-search denominator on the same H100 shape:

```text
mock_search_service compact sim16 >= 3x direct_ctree_gpu_latent sim16
```

Stronger signal:

```text
mock_search_service compact sim16 approaches recurrent_toy/policy_arrays rates
and stays stable as batch increases.
```

That would mean the remaining real-search boundary is large enough to justify a
MiniZero/KataGo-style search service or array-native search ownership.

Falsify or weaken the thesis if:

```text
mock_search_service compact sim16 is only <= 1.5x direct_ctree_gpu_latent sim16
```

on a clean matched row. That would mean the rest of the current profile loop
already dominates, so replacing search alone cannot produce the desired
multiplier. The next target would then be collector/replay/RND topology, not
search service first.

The public-output edge row gives a second decision:

```text
compact -> public output costs a lot:
  keep future service compact through replay/target writing.

compact -> public output is cheap:
  prioritize the search loop/API itself before replay output format.
```

## Minimal Experiment Grid

First wave:

| row | purpose |
| --- | --- |
| `mock_search_service`, compact, sim16 label | ceiling for perfect compact search boundary |
| `mock_search_service`, public-output edge, sim16 label | price per-env dict fanout at the edge |
| `direct_ctree_gpu_latent`, sim16 | current best real-search denominator |
| `recurrent_toy`, sim16 | model-call-heavy no-CTree reference |

Implementation note, 2026-05-22:

```text
The public-output edge switch is wired through the Modal entrypoint and the
hybrid profile-grid builder. Focused local validation:

uv run pytest -q -p no:cacheprovider tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_source_state_batched_observation_boundary_profile.py -k "mock_search_service or hybrid_profile_grid_can_emit_mock or validate_boundary_config_accepts_mock or rejects_public_output"
-> 7 passed, 104 deselected, 2 warnings

ruff and py_compile passed for the touched files.
```

Second wave only if first wave is promising:

| axis | values |
| --- | --- |
| batch size | `256`, `512`, `1024` |
| compute | H100 first, L4/T4 after |
| scalar timestep edge | off first, on only if compact result is strong |
| RND | off for ceiling; RND is a separate later denominator |

## Recommendation

Implement this as a sidecar-only array-ceiling mode, not a trainer hook. If it
shows a large gap over `direct_ctree_gpu_latent`, the next real design should
be a compact batched search service or array-native search owner. If it does
not show a large gap, stop saying search service is the 10x lane and move the
architecture critique to collector/replay/RND topology.
