# Precomputed Recurrent-Output CTree Falsifier Plan

Date: 2026-05-22

Scope: profile-only CurvyTron optimizer sidecar. Do not touch live training
runs, checkpoints, Modal volumes, trainer defaults, or Coach launch surfaces.
The implementation should stay inside the hybrid observation/profile boundary
unless and until it produces a clear falsifier result.

## Read

Rendering is no longer the likely wall. The current hot boundary is the
LightZero collect/search topology: CTree object/list APIs, per-simulation
Python control, recurrent launches, GPU-to-CPU recurrent outputs, and compact
output assembly.

The next falsifier should be a new direct CTree impl that keeps the same root
construction and the same CTree traverse/backprop loop as
`direct_ctree_gpu_latent`, but replaces measured `model.recurrent_inference`
calls with resident synthetic recurrent-output tensors. This prices the
remaining CTree/list/control path while keeping the LightZero CTree semantics
around roots, traversal, min/max stats, and backprop unchanged.

## 1. Where To Add The Mode

Add this as an MCTS arrays-boundary implementation, not as an array-ceiling
mode:

```python
LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT_PRECOMPUTED_RECURRENT = (
    "direct_ctree_gpu_latent_precomputed_recurrent"
)
```

Edit:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  - Add the constant beside `LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT`.
  - Include it in `LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPLS`.
  - Add a semantics string such as
    `LIGHTZERO_MCTS_ARRAYS_BOUNDARY_GPU_LATENT_PRECOMPUTED_RECURRENT_SEMANTICS =
    "lightzero_mcts_arrays_direct_ctree_gpu_latent_precomputed_recurrent_profile"`.
  - Extend `_LightZeroCollectForwardStackProbe.__init__` to set
    `backend_name = "lightzero_mcts_arrays_direct_ctree_gpu_latent_precomputed_recurrent_consumer"`
    and the new semantics for this impl.
  - Extend the direct dispatch in `run(...)` so the new impl calls
    `_run_direct_mcts_arrays(...)` with `keep_latents_on_device=True` plus a new
    boolean like `precompute_recurrent_outputs=True`.
  - Extend `_run_direct_mcts_arrays(...)` to pass that boolean into
    `_run_direct_ctree_gpu_latent_search(...)`.

Why this location: the new falsifier still uses real LightZero CTree roots,
`roots.prepare(...)`, `tree_muzero.batch_traverse(...)`, and
`tree_muzero.batch_backpropagate(...)`. It is a variant of
`direct_ctree_gpu_latent`, not a fake search service and not dense Torch MCTS.

Also edit the grid builder:

- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
  - Add the new impl string to `MCTS_ARRAYS_BOUNDARY_IMPL_CHOICES`.
  - Do not add it to `NEXT_DIRECT_CTREE_COMPARISON_IMPLS` by default unless the
    next manifest is specifically the falsifier wave; the default direct
    comparison grid currently means stock/direct/GPU-latent only.

## 2. Resident Synthetic Recurrent Outputs

Add a small helper near `_run_direct_ctree_gpu_latent_search(...)`:

```python
def _make_precomputed_recurrent_output_pool(
    *,
    torch: Any,
    latent_state_roots: Any,
    reward_roots: Any,
    pred_values: Any,
    policy_logits: Any,
    num_simulations: int,
) -> dict[str, Any]:
    ...
```

Use the already-real `initial_inference` outputs as the shape/dtype/device
contract:

- `next_latent_state`: same shape/dtype/device as `latent_state_roots`; the
  loop can copy the selected `latent_states` or `latent_state_roots` into
  `latent_pool[simulation_index + 1]`. No model call is needed for shape.
- `reward`: same shape/dtype/device as `reward_roots`.
- `value`: same shape/dtype/device as `pred_values`.
- `policy_logits`: same shape/dtype/device as `policy_logits`.

Implementation shape:

```python
reward_pool = reward_roots.detach().unsqueeze(0).expand(
    num_simulations, *reward_roots.shape
).contiguous()
value_pool = pred_values.detach().unsqueeze(0).expand(
    num_simulations, *pred_values.shape
).contiguous()
policy_logits_pool = policy_logits.detach().unsqueeze(0).expand(
    num_simulations, *policy_logits.shape
).contiguous()
```

Keep the tensors on `latent_state_roots.device`. Use `.contiguous()` so the
resident pool is real storage and indexing cost is representative. A constant
pool is acceptable: this is a falsifier for boundary cost, not a semantic
equivalence path. Report it as synthetic in telemetry.

Do not call `model.recurrent_inference` in the measured loop. If a one-call
shape probe is later desired, make it an explicit debug flag and report it
outside `search_sec`; the P0 version does not need it because the LightZero
MuZero contract makes recurrent reward/value/policy shapes match the root
heads.

## 3. Preserve CTree Semantics While Skipping Recurrent

Keep these pieces unchanged:

- `_run_direct_mcts_arrays(...)` still runs real `model.initial_inference(...)`.
- `mz_network_output_unpack(...)` is still used for the root outputs.
- Root value/logit D2H and `policy_logits_np.tolist()` stay as-is for
  `roots.prepare(...)`.
- `legal_actions`, `noises`, `roots = type(mcts).roots(...)`, and
  `roots.prepare(...)` stay as-is.
- `_run_direct_ctree_gpu_latent_search(...)` still allocates `latent_pool`,
  creates `MinMaxStatsList`, and calls the real LightZero
  `tree_muzero.batch_traverse(...)` and `tree_muzero.batch_backpropagate(...)`
  once per simulation.

Only replace this measured block inside `_run_direct_ctree_gpu_latent_search`:

```python
network_output = model.recurrent_inference(latent_states, last_actions_tensor)
next_latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)
```

with the precomputed path when enabled:

```python
reward = recurrent_pool["reward"][simulation_index]
value = recurrent_pool["value"][simulation_index]
policy_logits = recurrent_pool["policy_logits"][simulation_index]
next_latent_state = latent_states
```

Then keep the existing transform/D2H/list/backprop path:

```python
reward_plain = inverse_scalar_transform(reward).reshape(batch_size, 1)
value_plain = inverse_scalar_transform(value).reshape(batch_size, 1)
policy_logits_plain = policy_logits.to(dtype=torch.float32).reshape(batch_size, -1)
model_output_np = torch.cat(...).detach().cpu().numpy()
reward_batch = reward_np.reshape(-1).tolist()
value_batch = value_np.reshape(-1).tolist()
policy_logits_batch = policy_logits_np.tolist()
tree_muzero.batch_backpropagate(...)
```

This is the important control: CTree still receives CPU Python lists just as it
does today. The only removed thing is the recurrent model launch/work inside
the simulation loop.

## 4. Metrics Needed

Add telemetry in `_run_direct_ctree_gpu_latent_search(...)` and pass it through
the existing `lightzero_consumer_*` and `lightzero_mcts_arrays_boundary_*`
fields in `_run_direct_mcts_arrays(...)`.

Required fields:

- `lightzero_mcts_arrays_boundary_precomputed_recurrent_enabled`
- `lightzero_mcts_arrays_boundary_precomputed_recurrent_setup_sec`
- `lightzero_mcts_arrays_boundary_synthetic_output_index_sec`
- `lightzero_mcts_arrays_boundary_synthetic_output_d2h_sec`
- `lightzero_mcts_arrays_boundary_synthetic_output_d2h_bytes`
- `lightzero_mcts_arrays_boundary_synthetic_output_listify_sec`
- `lightzero_mcts_arrays_boundary_ctree_batch_traverse_sec`
- `lightzero_mcts_arrays_boundary_ctree_batch_traverse_calls`
- `lightzero_mcts_arrays_boundary_ctree_batch_backpropagate_sec`
- `lightzero_mcts_arrays_boundary_ctree_batch_backpropagate_calls`
- `lightzero_mcts_arrays_boundary_search_sec`
- `lightzero_mcts_arrays_boundary_root_prepare_sec`
- `lightzero_mcts_arrays_boundary_output_assembly_sec`
- existing aggregate `roots`, `total_sec`, `lightzero_roots_per_call`, and
  whatever summary computes as roots/sec.

Also add or keep these existing direct fields meaningful:

- `lightzero_consumer_model_recurrent_inference_sec == 0.0`
- `lightzero_consumer_model_recurrent_inference_calls == 0.0`
- `lightzero_consumer_gpu_latent_enabled == True`
- `lightzero_consumer_mcts_search_non_model_sec` should equal the measured
  CTree/synthetic path, not subtract nonexistent recurrent time.

The falsifier read should compare same-denominator rows:

```text
direct_ctree_gpu_latent:
  recurrent_inference_sec + model_output_d2h_sec + ctree traverse/backprop + list/control

direct_ctree_gpu_latent_precomputed_recurrent:
  synthetic_output_index_sec + synthetic_output_d2h/listify + ctree traverse/backprop + list/control
```

If the precomputed row is still close to direct CTree, the wall is not recurrent
launch. If it jumps near `mock_search_service`, recurrent launch/output transfer
was a major remaining tax. If it lands in the middle, the deltas above tell
whether D2H/listify or CTree traverse/backprop is the next wall.

## 5. Focused Tests

Edit `tests/test_source_state_batched_observation_boundary_profile.py`:

1. Config/constant acceptance
   - Add a test like
     `test_validate_boundary_config_accepts_precomputed_recurrent_direct_ctree_impl`.
   - Assert `_validate_boundary_config(...)` accepts
     `hybrid_lightzero_mcts_arrays_boundary_impl="direct_ctree_gpu_latent_precomputed_recurrent"`.

2. Local fake helper test
   - Add a fake `lzero.mcts.tree_search.mcts_ctree` module with
     `tree_muzero.MinMaxStatsList`, `ResultsWrapper`, `batch_traverse`, and
     `batch_backpropagate`.
   - Use real CPU `torch` if available.
   - Give `FakeModel.recurrent_inference(...)` an `AssertionError`.
   - Call `_run_direct_ctree_gpu_latent_search(..., precompute_recurrent_outputs=True, ...)`.
   - Assert recurrent calls are zero, traverse/backprop calls equal
     `num_simulations`, synthetic D2H/list/index telemetry is present, and
     backprop receives `batch_size` rewards/values/policy rows.

3. Probe-level fake test
   - Extend the existing fake direct CTree test pattern around
     `test_lightzero_mcts_arrays_boundary_direct_ctree_returns_compact_arrays`.
   - Instantiate `_LightZeroCollectForwardStackProbe(...,
     arrays_boundary_impl="direct_ctree_gpu_latent_precomputed_recurrent")`.
   - Patch the helper or fake CTree enough to avoid real LightZero.
   - Assert:
     - backend/semantics names are the new precomputed strings;
     - `model.recurrent_inference` was not called;
     - `lightzero_mcts_arrays_boundary_gpu_latent_enabled is True`;
     - `lightzero_mcts_arrays_boundary_precomputed_recurrent_enabled is True`;
     - compact action/visit/value shapes match the existing direct path.

4. Real policy smoke, CPU only
   - Do not add the precomputed mode to the existing stock-value equivalence
     loop; it is synthetic and should not match searched values.
   - Add a separate `pytest.importorskip("lzero")` CPU test with
     all-actions-legal masks that asserts legal actions, normalized visit rows,
     nonempty root values, zero recurrent calls, and no illegal actions.

Edit `tests/test_curvytron_hybrid_observation_profile_grid_builder.py`:

5. Grid builder support
   - Add a row test that
     `lightzero_mcts_arrays_boundary_impl="direct_ctree_gpu_latent_precomputed_recurrent"`
     appears in the emitted command after
     `--hybrid-lightzero-mcts-arrays-boundary-impl`.
   - Assert the label contains
     `-lzmctsarr-direct_ctree_gpu_latent_precomputed_recurrent-inhost_uint8`.

Useful focused command:

```bash
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py \
  -k "precomputed_recurrent or direct_ctree_returns_compact_arrays or real_policy_cpu_matches_stock_values_and_masks or hybrid_profile_grid"
```

## Modal Profile Command

Prefer the durable manifest runner so the JSON result survives local capture:

```bash
uv run python scripts/build_curvytron_hybrid_observation_profile_grid.py \
  --experiment-id opt-precomputed-recurrent-h100-20260522a \
  --computes gpu-h100 \
  --batch-sizes 512 \
  --actor-count 16 \
  --steps 60 \
  --warmup-steps 15 \
  --probe-simulations 16 \
  --materialize-scalar-timestep false \
  --lightzero-mcts-arrays-boundary-probe \
  --lightzero-mcts-arrays-boundary-impls direct_ctree_gpu_latent,direct_ctree_gpu_latent_precomputed_recurrent \
  --lightzero-mcts-arrays-boundary-input-mode host_uint8 \
  --lightzero-consumer-root-noise-weight 0.0

uv run python scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  --manifest artifacts/local/curvytron_hybrid_observation_profile_manifests/opt-precomputed-recurrent-h100-20260522a/manifest.json \
  --output-root artifacts/local/curvytron_hybrid_observation_profile_results/opt-precomputed-recurrent-h100-20260522a \
  --parallel 1
```

Tiny direct Modal smoke after local tests:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --hybrid-observation-canary \
  --compute gpu-h100 \
  --batch-size 64 \
  --actor-count 4 \
  --steps 3 \
  --warmup-steps 1 \
  --trail-slots 256 \
  --body-capacity 256 \
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --hybrid-batched-stack-probe-simulations 0 \
  --hybrid-batched-stack-probe-channels 16 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-lightzero-mcts-arrays-boundary-probe \
  --hybrid-lightzero-mcts-arrays-boundary-impl direct_ctree_gpu_latent_precomputed_recurrent \
  --hybrid-lightzero-mcts-arrays-boundary-input-mode host_uint8 \
  --hybrid-lightzero-consumer-num-simulations 4 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

Acceptance read:

- The row reports `touches_live_runs=false`, `calls_train_muzero=false`, and
  `profile_only=true`.
- `lightzero_mcts_arrays_boundary_impl` is
  `direct_ctree_gpu_latent_precomputed_recurrent`.
- `lightzero_consumer_model_recurrent_inference_calls == 0`.
- CTree traverse/backprop calls equal the requested simulation count on every
  nonzero-root measured step.
- Compare total roots/sec against same-denominator `direct_ctree_gpu_latent`,
  `recurrent_toy`, and `mock_search_service` rows before choosing the next
  implementation lane.

