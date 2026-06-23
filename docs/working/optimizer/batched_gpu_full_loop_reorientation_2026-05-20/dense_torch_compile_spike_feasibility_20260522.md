# Dense Torch Compile Spike Feasibility

Date: 2026-05-22

Scope: read-only sidecar. I inspected
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`,
especially `_run_dense_torch_mcts`. No live runs or production code changed.

## Plain Answer

A bounded `torch.compile` / CUDA-graphs / Triton-style spike is realistic, but
only as a fixed-shape profile experiment. It is not realistic to wrap the
current `_run_dense_torch_mcts` function and expect a clean win.

The useful target is narrower:

```text
Keep B512/A16/root-noise0 fixed, preallocate the search buffers, remove Python
fallback paths, and compile or graph-capture only the dense search/update body.
```

That could answer whether eager dense Torch is losing sim16 to Python/kernel
launch overhead. It should not be treated as a LightZero training replacement.

## Current Evidence

Fresh H100 profile-only ladder, B512/A16, 60 measured steps, 15 warmup,
root-noise0:

| row | sim8 roots/sec | sim16 roots/sec | read |
| --- | ---: | ---: | --- |
| `stock_facade` | `2430` | `2094` | public LightZero boundary |
| `direct_ctree_arrays` | `5009` | `3448` | compact direct CTree arrays |
| `direct_ctree_gpu_latent` | `7547` | `6145` | best practical LightZero-shaped row |
| `dense_torch_mcts` after semantic fix | `8288` | `4294` | wins sim8, fails sim16 scaling |
| `recurrent_toy` ceiling | `12834` | `9191` | not MCTS; NVL caveat |

Plain read:

```text
Dense Torch has enough sim8 evidence to justify one compile/fusion spike.
Dense eager Torch has already failed the sim16 practical gate.
The gap is not rendering and probably not neural inference. It is the search
control/update shell.
```

## Why The Current Function Will Break Or Recompile

The current `_run_dense_torch_mcts` is Python-shaped around Torch tensors.
Specific blockers:

- Dynamic root count enters from `active_root_count`. In `run`, zero-mask roots
  can be filtered by NumPy boolean indexing before dense search, so shapes can
  change unless the spike forces all roots legal.
- Fresh tensors are allocated on every call: `edge_child`, `edge_visit`,
  `edge_value_sum`, `edge_reward`, `edge_prior`, `latent_pool`,
  `node_latent_slot`, `next_node_index`, min/max buffers, and path-history
  buffers.
- More fresh tensors are allocated inside every simulation:
  `current_node`, `active`, `leaf_parent`, and `leaf_action`.
- The recurrent action input has a runtime `try/except` shape fallback. That is
  useful for probing LightZero model conventions, but it is poison for capture.
- Helper calls inspect arbitrary Python objects:
  `_network_output_field`, `_policy_inverse_scalar_value`, and
  `_recurrent_action_input`. They use attribute checks, mappings, exception
  fallbacks, and policy-owned inverse transforms.
- Timers and synchronizations live inside the function:
  `time.perf_counter()` around recurrent calls/update work, plus final CUDA
  sync. Those must sit outside any compiled or graphed region.
- The root-noise branch uses `torch.distributions.Dirichlet(...).sample`. The
  proposed spike should keep `root_noise_weight=0.0`.
- The loop nest is fixed by `num_simulations`, but the inner depth loop is
  triangular: `range(simulation_index + 1)`. It may compile for a fixed sim
  count, but sim8 and sim16 should be separate compiled functions or separate
  graph captures.
- `model.recurrent_inference` returns a LightZero network-output object, not a
  simple tuple owned by this probe. That object boundary is a likely graph
  break unless wrapped.

So: compile is plausible after reshaping the code, not before.

## What Would Need To Change

For a bounded spike, keep this profile-only and add a separate dense-search
helper rather than replacing the current probe.

Minimum code shape:

1. Force the fixed denominator:
   `B=512`, `roots=1024`, `A=3`, fixed sim count, all roots legal,
   `root_noise_weight=0.0`, CUDA device, no dynamic zero-root filtering.
2. Resolve recurrent action shape once before measurement. Use only the winning
   shape in the captured path, probably flat unless the model requires column.
3. Wrap the model output into simple tensors:
   `latent_state`, `reward`, `value`, `policy_logits`. Do not call
   `_network_output_field` inside the compiled/search body.
4. Preallocate all tree and scratch buffers once per probe object and zero/fill
   them in-place per call.
5. Move all telemetry, `perf_counter`, checksums, illegal-action checks,
   readback, and Python dict assembly outside the captured region.
6. Split model calls from search-update if needed:
   compile/capture the selection and backup tensor kernels first, then decide
   whether recurrent inference can be included safely.

CUDA graphs require the strictest version of this: static tensor addresses,
static shapes, no allocation during replay, no exception path, and stable model
execution. `torch.compile(mode="reduce-overhead")` is a looser first probe.
Triton is the fallback if selection/backup remain many small eager kernels after
compile.

## Exact First Experiment

Add one profile-only mode, for example:

```text
dense_torch_mcts_compile_spike
```

First implementation should compile only a pure tensor helper:

```text
select_and_backup_fixed(
    edge_child, edge_visit, edge_value_sum, edge_reward, edge_prior,
    node_latent_slot, min_value, max_value,
    flat_mask_tensor, recurrent_reward, recurrent_value, recurrent_policy,
    simulation_index
) -> leaf_parent, leaf_action
```

Keep `model.recurrent_inference` eager at first. That isolates whether the
sim16 loss is mostly search/update launch overhead. Run only the matched H100
profile-only ladder:

```text
B512 / A16 / 60 measured / 15 warmup / root-noise0
dense_torch_mcts_compile_spike sim8
dense_torch_mcts_compile_spike sim16
direct_ctree_gpu_latent sim8/sim16 as the comparator
recurrent_toy sim8/sim16 as the ceiling sanity row
```

No trainer promotion, no live runs, no root-noise row until root-noise0 works.

## Expected Ceiling

Reasonable expectation, not a claim:

- Sim8: likely small upside over `8288`, maybe `8.5k-9.5k roots/sec`, because
  dense already wins and recurrent/model work still exists.
- Sim16: the real test. A useful compile/fusion result should recover from
  `4294` to at least `6.1k+`, matching or beating `direct_ctree_gpu_latent`.
- Strong result: `7k-8.5k` sim16. That would show the eager Torch shell was the
  main problem.
- Hard practical ceiling with the current model path is probably below the
  `recurrent_toy` row (`9191` sim16), because real MCTS still does selection,
  expansion, backup, and visit output.

I would not forecast a clean `10k+` sim16 dense-MCTS row from compile alone.

## Falsifier

Stop this lane if the fixed-shape spike does not beat the practical CTree row:

```text
sim16 dense_torch_mcts_compile_spike <= direct_ctree_gpu_latent sim16
```

Using the fresh denominator, that means failure if it stays at or below roughly
`6145 roots/sec` after compile warmup is excluded.

Also stop if any of these happen:

- graph breaks force most of the inner loop back to eager mode;
- sim8 improves but sim16 remains below CTree;
- the spike needs dynamic masks, root noise, or exception fallbacks inside the
  captured region to run;
- parity/forced-case checks fail for legal masks, single legal action, no-noise
  deterministic rows, visit distributions, or backed values.

## Decision

Do one bounded compile/fusion spike because it directly tests the current
correct bottleneck: the LightZero collect/search boundary. Keep
`direct_ctree_gpu_latent` as the practical baseline while the spike is
profile-only. If the spike fails sim16, switch attention back to array-native
CTree or a more explicit Triton/CUDA search kernel instead of polishing eager
Torch.

## Implementation Note

Added profile-only mode `dense_torch_mcts_compile_spike` in
`source_state_batched_observation_boundary_profile.py`.

The mode keeps `model.recurrent_inference` eager and attempts
`torch.compile(mode="reduce-overhead", fullgraph=True)` only for two pure tensor
helpers: fixed-shape selection/path recording and expansion/backup. It compiles
only when the probe is on CUDA, `torch.compile` is present, all roots and all
actions are legal, and `root_noise_weight == 0.0`. Otherwise it runs the same
eager dense Torch MCTS path and reports the fallback through
`lightzero_array_ceiling_compile_status`,
`lightzero_array_ceiling_compile_enabled`, and
`lightzero_array_ceiling_compile_reason`.

## H100 Commands

These are profile-only hybrid observation sidecar rows. They do not call
`train_muzero`, live Coach runs, checkpoints, evals, GIFs, or tournaments.
They use H100, B512, actor_count 16, 60 measured steps, 15 warmup steps,
fresh host uint8 input, and root noise forced to zero.

`dense_torch_mcts_compile_spike` sim8:

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
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-batched-stack-probe-simulations 0 \
  --hybrid-batched-stack-probe-channels 16 \
  --hybrid-lightzero-array-ceiling-probe \
  --hybrid-lightzero-array-ceiling-mode dense_torch_mcts_compile_spike \
  --hybrid-lightzero-array-ceiling-input-mode host_uint8 \
  --hybrid-lightzero-consumer-num-simulations 8 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

`dense_torch_mcts_compile_spike` sim16:

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
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-batched-stack-probe-simulations 0 \
  --hybrid-batched-stack-probe-channels 16 \
  --hybrid-lightzero-array-ceiling-probe \
  --hybrid-lightzero-array-ceiling-mode dense_torch_mcts_compile_spike \
  --hybrid-lightzero-array-ceiling-input-mode host_uint8 \
  --hybrid-lightzero-consumer-num-simulations 16 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

`direct_ctree_gpu_latent` comparator sim8:

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
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-batched-stack-probe-simulations 0 \
  --hybrid-batched-stack-probe-channels 16 \
  --hybrid-lightzero-mcts-arrays-boundary-probe \
  --hybrid-lightzero-mcts-arrays-boundary-impl direct_ctree_gpu_latent \
  --hybrid-lightzero-mcts-arrays-boundary-input-mode host_uint8 \
  --hybrid-lightzero-consumer-num-simulations 8 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

`direct_ctree_gpu_latent` comparator sim16:

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
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-batched-stack-probe-simulations 0 \
  --hybrid-batched-stack-probe-channels 16 \
  --hybrid-lightzero-mcts-arrays-boundary-probe \
  --hybrid-lightzero-mcts-arrays-boundary-impl direct_ctree_gpu_latent \
  --hybrid-lightzero-mcts-arrays-boundary-input-mode host_uint8 \
  --hybrid-lightzero-consumer-num-simulations 16 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

`recurrent_toy` ceiling sim8:

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
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-batched-stack-probe-simulations 0 \
  --hybrid-batched-stack-probe-channels 16 \
  --hybrid-lightzero-array-ceiling-probe \
  --hybrid-lightzero-array-ceiling-mode recurrent_toy \
  --hybrid-lightzero-array-ceiling-input-mode host_uint8 \
  --hybrid-lightzero-consumer-num-simulations 8 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

`recurrent_toy` ceiling sim16:

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
  --render-surface direct_gray64 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --hybrid-stack-storage-dtype uint8 \
  --no-hybrid-materialize-scalar-timestep \
  --hybrid-batched-stack-probe-simulations 0 \
  --hybrid-batched-stack-probe-channels 16 \
  --hybrid-lightzero-array-ceiling-probe \
  --hybrid-lightzero-array-ceiling-mode recurrent_toy \
  --hybrid-lightzero-array-ceiling-input-mode host_uint8 \
  --hybrid-lightzero-consumer-num-simulations 16 \
  --hybrid-lightzero-consumer-root-noise-weight 0.0
```

## Compile Telemetry Contract

The compile-spike fields are in compact output under:

```text
batched_stack_probe_last_telemetry
```

Actual compile happened only if all of these are true on a measured row:

```text
lightzero_array_ceiling_mode == "dense_torch_mcts_compile_spike"
lightzero_array_ceiling_compile_enabled == 1.0
lightzero_array_ceiling_compile_status in {"compiled", "compiled_cached"}
lightzero_array_ceiling_compile_helper == "select_leaf+expand_backup"
lightzero_array_ceiling_all_roots_legal_fast_path == 1.0
lightzero_array_ceiling_all_actions_legal_fast_path == 1.0
```

Because warmup steps also run the probe, the last measured telemetry will
usually show `compiled_cached`, not `compiled`; that is still a successful
compile/cached execution. `lightzero_array_ceiling_compile_signature` should
begin with `[1024, 8, ...]` or `[1024, 16, ...]` for this fixed denominator.

Fallback is proven by:

```text
lightzero_array_ceiling_compile_enabled == 0.0
```

Then inspect:

```text
lightzero_array_ceiling_compile_status
lightzero_array_ceiling_compile_reason
lightzero_array_ceiling_compile_attempted
```

Expected fallback statuses/reasons include:

```text
fallback_precondition / requires_cuda_device
fallback_precondition / torch_cuda_unavailable
fallback_precondition / torch_compile_unavailable
fallback_precondition / requires_all_roots_legal
fallback_precondition / requires_all_actions_legal
fallback_precondition / requires_root_noise_zero
fallback_compile_failed / <exception text>
not_attempted_zero_roots / no_active_roots
```

Comparator sanity fields:

```text
direct_ctree_gpu_latent:
batched_stack_probe_semantics == "lightzero_mcts_arrays_direct_ctree_gpu_latent_profile"
batched_stack_probe_last_telemetry.lightzero_mcts_arrays_boundary_impl == "direct_ctree_gpu_latent"
batched_stack_probe_last_telemetry.lightzero_mcts_arrays_boundary_gpu_latent_enabled == true

recurrent_toy:
batched_stack_probe_last_telemetry.lightzero_array_ceiling_mode == "recurrent_toy"
batched_stack_probe_last_telemetry.lightzero_array_ceiling_recurrent_inference_calls == 8.0 or 16.0
```

Use `batched_stack_probe_total_roots / measured_sec` for matched roots/sec.

## Launch Status 2026-05-22

Launched and completed after this planning note. These were profile-only hybrid
observation sidecar rows. They did not call `train_muzero`, touch live Coach
runs, save checkpoints, run eval, write GIFs, or update tournaments.

The earlier manifest-runner concern still matters for future tooling: the
existing manifest runner is shaped for the training profile entrypoint, not the
hybrid observation sidecar. For this falsifier, the main thread recorded the
Modal app ids and stdout JSON directly.

| row | app id | sim8 roots/sec | sim16 roots/sec | read |
| --- | --- | ---: | ---: | --- |
| `dense_torch_mcts_compile_spike` | `ap-dMjPlGmbGGtFrf1JJMysOW` / `ap-cOoZ5pLWOJhN38qans8r8o` | `10298.01` | `4872.70` | wins sim8, fails sim16 |
| `direct_ctree_gpu_latent` | `ap-hNe9labJXf6Z17LN5GNoJF` / `ap-OVE29tvXDfUEAGgVEdi37t` | `7567.35` | `6153.95` | practical baseline still wins sim16 |
| `recurrent_toy` ceiling | `ap-d5dqTBaaVlh6p7KZQT5lvG` / `ap-BkIEqGLr28TuETnXOb19Tf` | `9524.57` | `8969.89` | not MCTS; model-call ceiling sanity row |

Compile evidence:

- sim8 reported `lightzero_array_ceiling_compile_enabled=1.0` and
  `compile_status=compiled_cached`.
- compile rows emitted skipped-CUDA-graph warnings because tree buffers are
  mutated.
- sim16 emitted Dynamo recompile-limit warnings tied to `simulation_index` and
  the triangular path loop.

Decision:

```text
Stop polishing this exact compile-spike helper.
It fails the sim16 gate against direct_ctree_gpu_latent.
```

## Integration Validation

Focused local validation after integration:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_lightzero_phase_profiler.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "collect_search_hook or profile_attestation or array_ceiling"
```

Result: `20 passed, 88 deselected`. Matching ruff and py_compile checks passed
for the touched optimizer profile/test files.

This only proves the sidecar wiring and guardrails. It does not prove a speed
win and does not promote the mode to training.

## Measurement Rule

Rows only count as compile-spike evidence if:

```text
lightzero_array_ceiling_compile_enabled == 1.0
lightzero_array_ceiling_compile_status in {"compiled", "compiled_cached"}
```

Rows with `fallback_precondition`, `fallback_compile_failed`, or
`compile_unavailable` are still useful controls, but they do not answer whether
compiled dense search is faster.

The matched ladder has now run:

```text
B512 / A16 / 60 measured / 15 warmup / root-noise0 / all actions legal
dense_torch_mcts_compile_spike sim8
dense_torch_mcts_compile_spike sim16
direct_ctree_gpu_latent sim8
direct_ctree_gpu_latent sim16
recurrent_toy sim8
recurrent_toy sim16
```

Outcome: sim8 is not enough. Sim16 is the practical gate, and the compile spike
failed it.
