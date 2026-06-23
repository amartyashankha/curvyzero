# LightZero Boundary Validation Plan - 2026-05-21

Scope: review and test plan only. I did not touch live training runs, launch
Modal jobs, write run artifacts, or edit production code. The target boundary is
the new profile-only path that feeds pre-scalar `[B,2,4,64,64]` `uint8` stacks
into LightZero collect-forward and initial-inference probes.

## Current Boundary Read

The LightZero profile probes are intentionally profile-only. The boundary module
states it does not import or modify trainers, tournaments, checkpoints, eval,
Modal Volumes, or live runs, and the hybrid result reports
`profile_only=True`, `calls_train_muzero=False`, `stock_lightzero_integrated=False`,
and `touches_live_runs=False`.

The collect-forward probe flattens row/player roots from `[B,2,4,64,64]` to
`[N,4,64,64]`, where `N` is only the legal-root count after zero-mask filtering.
It passes NumPy float32 masks shaped `[N,3]`, `ready_env_id=np.arange(N)`, and
fixed-opponent `to_play=[-1] * N` into `policy.collect_mode.forward(...)`.
It normalizes only `uint8` stacks by dividing by `255`, decodes each returned
root, and raises if a decoded action is illegal for that root's mask.

The initial-inference probe mirrors the same shape, mask-filtering, H2D, and
normalization edge, but calls `policy._model.initial_inference(obs_tensor)`.
It deliberately excludes action selection, MCTS/tree search, `to_play`, and
output action decoding. Its output contract is a compact shape/device/dtype
summary for `value`, `reward`, `policy_logits`, `latent_state`, `hidden_state`,
and `value_prefix` when present.

The config surface is mostly pinned: the profile policy builder uses the public
`build_visual_survival_configs(...)` source-state fixed-opponent path, fixed
straight opponent, `disable_death_for_profile=True`, model shape `[4,64,64]`,
action count `3`, and surface labels for env variant, observation shape,
policy observation backend, trail render mode, and bonus render mode.

## Existing Local Coverage

Already covered:

- Fixed-opponent collect-forward `to_play=-1`: `test_lightzero_collect_forward_stack_probe_flattens_roots_and_decodes` asserts every forwarded root receives `-1`.
- Action-mask filtering: `test_lightzero_collect_forward_stack_probe_filters_zero_mask_roots` proves all-zero roots are dropped, `ready_env_id` is compacted, and telemetry reports total versus filtered roots.
- Batched dict-of-arrays decode: `test_policy_output_row_from_plain_handles_batched_mapping_outputs` covers row slicing for `action`, value, and visit distributions.
- Scalar materialization edge: hybrid tests prove `uint8` stack storage scalarizes to float32, terminal final observations scalarize, scalar materialization can be skipped only when a batched consumer exists, and skipping without a consumer fails closed.
- No-live-run metadata: hybrid profile tests assert `profile_only=True`, `calls_train_muzero=False`, `stock_lightzero_integrated=False`, `trainer_defaults_changed=False`, and `touches_live_runs=False`.
- Policy observation surface labels: config-builder tests pin `policy_observation_contract_id`, perspective schema, perspective owner, seat mapping, backend, trail render mode, and bonus render mode.

## Proposed Tests

### P0. Initial-Inference Shape/Device/Output Summary

Add a local fake-torch/fake-model unit next to the collect-forward probe tests.
Import `_LightZeroInitialInferenceStackProbe` and drive it with a `[2,2,4,64,64]`
`uint8` stack plus a mask that drops one root.

Assertions:

- `model.initial_inference` is called once with `[3,4,64,64]`.
- Input min/max is within `[0,1]` after `uint8` normalization.
- Telemetry reports `lightzero_total_root_count=4`,
  `lightzero_filtered_zero_mask_root_count=1`, `lightzero_root_count=3`,
  `lightzero_roots_per_call=3`, and `simulations=0`.
- `lightzero_initial_inference_policy_surface` preserves the supplied surface
  labels.
- `lightzero_initial_inference_output_summary` includes expected shape/dtype/device
  entries for fake `policy_logits`, `value`, `reward`, and `latent_state`.
- `lightzero_initial_inference_output_key_sample` includes the summary keys,
  not a full latent readback.

This is the smallest local gap in the current tree: collect-forward has fake
consumer unit coverage; initial-inference currently has implementation and config
plumbing, but no equivalent cheap local test pinning the public telemetry.

### P0. Initial-Inference Config Surface And Modal Routing

Add config validation tests for `hybrid_lightzero_initial_inference_probe`.

Assertions:

- It is accepted only with `hybrid_observation_canary=True`,
  `observation_renderer_backend='jax_gpu_persistent_policy_framebuffer_profile'`,
  `render_surface='direct_gray64'`, and `hybrid_stack_storage_dtype='uint8'`.
- It rejects combination with collect-forward, resident chunk, or synthetic
  batched-stack consumers.
- The public Modal entrypoint surface exposes the initial-inference flag and
  routes it to the LightZero image functions, same as collect-forward.

Tiny gap found: the internal config path recognizes
`hybrid_lightzero_initial_inference_probe`, but the public `run_boundary_profile_entrypoint`
signature/config currently exposes `hybrid_lightzero_collect_forward_probe` only.
A local test should catch that before someone tries to launch an initial-inference
row through the CLI and silently lands on the non-LightZero image path.

### P0. Collect-Forward Illegal-Action Fail-Closed

Add one fake collect-mode test where the mask permits only action `1`, but the
fake policy returns action `2`.

Assertions:

- `_LightZeroCollectForwardStackProbe.run(...)` raises `ValueError` with
  `"decoded illegal actions"`.
- No success telemetry is returned for the bad batch.

This pins the most important safety edge of action-mask consumption: masks are
not just forwarded to LightZero; decoded actions are checked before the profile
result is considered valid.

### P1. Output Decoding Variants

Broaden local tests for `_policy_output_row_from_plain` and
`_extract_eval_action_from_plain`.

Cases:

- mapping keyed by string ready ids: `{"0": {"action": 1}, "1": {"action": 2}}`;
- list output: `[{"action": 0}, {"action": 2}]`;
- nested root output: `{0: {"selected_action": [1]}}`;
- dict-of-arrays with `selected_actions`, `visit_count_distributions`, and
  `predicted_value`;
- missing action raises a clear `ValueError`.

These tests should stay pure Python/NumPy and not instantiate LightZero.

### P1. Scalar Materialization Edge For `uint8` Final Observations

Add a direct `materialize_lightzero_scalar_timestep(...)` test with `uint8`
`step_observation` and `uint8` `final_observation`.

Assertions:

- Flat observation is contiguous float32 `[B*2,4,64,64]`.
- Latest-channel sentinel values are divided by `255`.
- `to_play` is a vector of `-1`.
- Batch/action-mask row-major order remains `[row0 p0, row0 p1, row1 p0, row1 p1]`.
- Terminal `final_observation` is attached only for done rows and is float32
  normalized.

The hybrid tests cover this indirectly through the manager; this direct unit
would pin the scalar materializer's own boundary.

### P1. Profile-Only/No-Live-Write Attestation

Add a compact-result test for the LightZero probe rows that does not build
LightZero. Use a fake batched-stack probe with the same telemetry keys and run
`run_hybrid_observation_profile(...)`.

Assertions:

- Result still reports `profile_only=True`, `calls_train_muzero=False`,
  `stock_lightzero_integrated=False`, `touches_live_runs=False`, and
  `trainer_defaults_changed=False`.
- `batched_stack_probe_backend_name` and `batched_stack_probe_semantics` are
  preserved.
- When `materialize_scalar_timestep=False`, `materialized_timestep_count=0` and
  LightZero probe timings are still counted.

This is mainly an anti-regression guard against a future optimizer row crossing
from profile-only probes into trainer/run-management code.

### P1. Policy Surface Labels In Probe Telemetry

Add a fake-policy probe test that supplies a complete `policy_metadata["surface"]`
dict and asserts the telemetry carries it unchanged for both collect-forward and
initial-inference probes.

Minimum labels:

- `env_variant`
- `observation_shape`
- `policy_observation_backend`
- `policy_trail_render_mode`
- `policy_bonus_render_mode`

This keeps speed rows self-describing enough to compare across future backend or
surface changes.

## Promotion Gate

Before using these rows as more than optimizer/Amdahl evidence, require one
focused local suite to pass:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_lightzero_config_builder.py
```

Do not treat any probe row as a live-training or stock-loop proof unless a
separate stock `train_muzero` gate explicitly says it called the stock entrypoint.
