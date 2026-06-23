# Next Profile Tooling Plan, 2026-05-23b

Scope: read-only tooling scout plus this report. I did not launch Modal jobs,
touch live Coach runs, write manifests, collect results, or modify training
state.

## Short Answer

Use the hybrid observation durable runner for the next optimizer sidecar wave:

```text
scripts/build_curvytron_hybrid_observation_profile_grid.py
-> artifacts/local/curvytron_hybrid_observation_profile_manifests/<id>/manifest.json
-> scripts/run_curvytron_hybrid_observation_profile_manifest.py --manifest ...
```

That path is the cleanest profile-only launch route because the manifest and
runner both require:

```text
profile_only=true
calls_train_muzero=false
touches_live_runs=false
--hybrid-observation-canary
curvyzero.infra.modal.source_state_batched_observation_boundary_profile
no --detach
```

Do not run `commands.sh` directly for these rows. Do not add `--detach`. The
runner exists to keep blocking stdout JSON and local result files durable.

The awkward part: the exact comparison we now want spans three tooling families.
Direct CTree, service-tax, and dense/compiled sidecar rows fit the hybrid
runner. Fixed-A3 currently lives in the stock `train_muzero` profile hook, not
the hybrid boundary runner. MCTX lives in `mctx_synthetic_benchmark.py`, also
outside the hybrid manifest runner. Treat those as separate profile currencies
unless we add a unifying runner.

## Current Tooling Read

`scripts/run_curvytron_hybrid_observation_profile_manifest.py` is the safest
runner for profile-only optimizer probes. It preflights the manifest schema,
rejects manifests that call `train_muzero`, rejects live-run contact, rejects
detached commands, runs blocking Modal commands, captures stdout, extracts the
last profile JSON, and writes:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/manifest.json
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/row_<id>_stdout.log
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/row_<id>_result.json
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/rows.jsonl
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/collected_results.json
```

`scripts/build_curvytron_hybrid_observation_profile_grid.py` emits the right
sidecar commands and labels. It supports direct CTree boundary impls
(`stock_facade`, `direct_ctree_arrays`, `direct_ctree_gpu_latent`,
`direct_ctree_gpu_latent_precomputed_recurrent`) and array-ceiling modes
(`mock_search_service`, `service_tax_probe`, `dense_torch_mcts`,
`dense_torch_mcts_compile_spike`, `compact_torch_search_service`, etc.).

`scripts/summarize_curvytron_optimizer_profile_results.py` is for stock
full-loop optimizer profiles. It expects `called_train_muzero=true` and checks
semantic identity fields, direct CTree fallback counts, and flat-A3 runtime
proofs. It is not a hybrid sidecar summarizer.

`docs/working/optimizer/modal_profile_tooling_2026-05-16.md` is still directionally
right: hybrid boundary rows should use the durable runner, while stock profile
grids use the stock manifest runner. The doc's old detached/manual examples are
not the preferred pattern for the hybrid sidecar anymore.

## Required Labels And Guardrails

For hybrid sidecar rows, require these at manifest top level:

```json
{
  "schema_id": "curvyzero_hybrid_observation_profile_manifest/v0",
  "profile_only": true,
  "calls_train_muzero": false,
  "touches_live_runs": false
}
```

Require these on every row:

```json
{
  "schema_id": "curvyzero_hybrid_observation_profile_row/v0",
  "profile_only": true,
  "calls_train_muzero": false,
  "touches_live_runs": false,
  "fixed_denominator": {
    "batch_size": 512,
    "actor_count": 16,
    "steps": 80,
    "warmup_steps": 20,
    "input_mode": "host_uint8_pinned",
    "materialize_scalar_timestep": false
  }
}
```

Require these command flags:

```text
uv run --extra modal modal run --quiet
-m curvyzero.infra.modal.source_state_batched_observation_boundary_profile
--hybrid-observation-canary
--compute gpu-h100
--batch-size 512
--actor-count 16
--steps 80
--warmup-steps 20
--render-surface direct_gray64
--observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile
--hybrid-stack-storage-dtype uint8
--hybrid-batched-stack-probe-simulations 0
--no-hybrid-materialize-scalar-timestep
--hybrid-compact-service-replay-proof
--hybrid-lightzero-consumer-root-noise-weight 0.0
```

For readout and promotion blocking, carry or derive:

```text
promotion_eligible=false
promotion_blocker=profile_only_boundary_probe
stock_lightzero_integrated=false
trainer_defaults_changed=false
semantics=<mode-specific profile semantics>
fallback calls/counts == 0 where applicable
compact_service_replay_proof_enabled=true
```

For stock full-loop profile rows, use different language: they are
`stock_full_loop_profile`, not pure sidecar rows. They may call `train_muzero`,
but only with:

```text
--mode profile
--output-detail compact
--skip-lightzero-eval-in-profile
--no-background-eval-enabled
--no-background-gif-enabled
--save-ckpt-after-iter 999999
```

Flat-A3 must also prove:

```text
collect_search_backend=direct_ctree_gpu_latent
collect_search_ctree_backend=flat_a3
search_backend_proof.observed_collect_search_ctree_backends includes flat_a3
search_backend_proof.flat_payload_timer_present=true
```

MCTX sidecar rows must keep their own warning labels:

```text
compact_search_service_profile.profile_only=true
compact_search_service_profile.not_lightzero_ctree=true
compact_search_service_profile.not_train_muzero=true
```

## Exact Next Small Wave

Run H100 only first. Skip L4 until the H100 ordering is clean. Use sim16 and
sim32 because sim16/sim32 is where eager dense/compiled search either survives
or dies.

### A. Hybrid Sidecar, Direct CTree Baseline

Build:

```bash
uv run python scripts/build_curvytron_hybrid_observation_profile_grid.py \
  --experiment-id next-profile-direct-h100-20260523b \
  --computes h100 \
  --batch-sizes 512 \
  --actor-count 16 \
  --steps 80 \
  --warmup-steps 20 \
  --probe-simulations 16,32 \
  --materialize-scalar-timestep false \
  --device-latest false \
  --compact-service-replay-proof \
  --lightzero-mcts-arrays-boundary-probe \
  --lightzero-mcts-arrays-boundary-impl direct_ctree_gpu_latent \
  --lightzero-mcts-arrays-boundary-input-mode host_uint8_pinned \
  --lightzero-consumer-root-noise-weight 0.0 \
  --quiet
```

Run:

```bash
uv run python scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  --manifest artifacts/local/curvytron_hybrid_observation_profile_manifests/next-profile-direct-h100-20260523b/manifest.json
```

### B. Hybrid Sidecar, Service-Tax Ceiling

Build:

```bash
uv run python scripts/build_curvytron_hybrid_observation_profile_grid.py \
  --experiment-id next-profile-servicetax-h100-20260523b \
  --computes h100 \
  --batch-sizes 512 \
  --actor-count 16 \
  --steps 80 \
  --warmup-steps 20 \
  --probe-simulations 16,32 \
  --materialize-scalar-timestep false \
  --device-latest false \
  --compact-service-replay-proof \
  --lightzero-array-ceiling-probe \
  --lightzero-array-ceiling-mode service_tax_probe \
  --lightzero-array-ceiling-input-mode host_uint8_pinned \
  --lightzero-consumer-root-noise-weight 0.0 \
  --quiet
```

Run:

```bash
uv run python scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  --manifest artifacts/local/curvytron_hybrid_observation_profile_manifests/next-profile-servicetax-h100-20260523b/manifest.json
```

### C. Hybrid Sidecar, Compiled Dense Spike

This is available as `dense_torch_mcts_compile_spike`, but previous sim16
evidence failed against direct CTree. Rerun only if we need the newer compact
replay proof and ledger fields on the same denominator.

Build:

```bash
uv run python scripts/build_curvytron_hybrid_observation_profile_grid.py \
  --experiment-id next-profile-compile-spike-h100-20260523b \
  --computes h100 \
  --batch-sizes 512 \
  --actor-count 16 \
  --steps 80 \
  --warmup-steps 20 \
  --probe-simulations 16,32 \
  --materialize-scalar-timestep false \
  --device-latest false \
  --compact-service-replay-proof \
  --lightzero-array-ceiling-probe \
  --lightzero-array-ceiling-mode dense_torch_mcts_compile_spike \
  --lightzero-array-ceiling-input-mode host_uint8_pinned \
  --lightzero-consumer-root-noise-weight 0.0 \
  --quiet
```

Run:

```bash
uv run python scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  --manifest artifacts/local/curvytron_hybrid_observation_profile_manifests/next-profile-compile-spike-h100-20260523b/manifest.json
```

### D. Fixed-A3 Profile Pair

Fixed-A3 is available, but not in the hybrid sidecar builder. Use the stock
profile manifest runner and label this as stock full-loop profile evidence, not
hybrid sidecar evidence.

Build:

```bash
uv run python scripts/build_curvytron_profile_grid.py \
  --experiment-id next-profile-flat-a3-h100-20260523b \
  --family flat_a3_profile_only \
  --run-prefix flat3h1000523b \
  --attempt-prefix profile \
  --seed 310 \
  --computes gpu-h100-cpu40 \
  --env-manager-types subprocess \
  --collect-search-backends direct_ctree_gpu_latent \
  --collect-search-ctree-backends lightzero,flat_a3 \
  --collectors 128 \
  --batch-sizes 64 \
  --num-simulations 16,32 \
  --source-max-steps 512 \
  --max-train-iter 96 \
  --max-env-step 200000 \
  --save-ckpt-after-iter 999999 \
  --stop-after-learner-train-calls 3 \
  --env-telemetry-stride 256 \
  --disable-death-for-profile
```

Run:

```bash
uv run python scripts/run_curvytron_optimizer_profile_manifest.py \
  --manifest artifacts/local/curvytron_optimizer_profile_manifests/next-profile-flat-a3-h100-20260523b.json \
  --action launch-and-collect
```

Summarize:

```bash
uv run python scripts/summarize_curvytron_optimizer_profile_results.py \
  --results-dir artifacts/local/curvytron_optimizer_profile_results/next-profile-flat-a3-h100-20260523b \
  --require-attestation
```

### E. MCTX Comparator

MCTX has no manifest runner today. If a fresh comparator is needed, run the two
blocking sidecar commands and capture the compact JSON. Do not mix it into the
hybrid result table as LightZero-equivalent.

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --observation-mode curvytron_hybrid_compact_visual_sample \
  --compute h100 \
  --batch-size 512 \
  --player-count 2 \
  --body-capacity 1024 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --num-simulations 16 \
  --hidden-dim 64 \
  --max-depth 16 \
  --warmup-runs 2 \
  --steady-runs 5 \
  --closed-loop-steps 24 \
  --native-actor-buffer \
  --no-compact-root-copy-observation \
  --no-compact-visual-resident-sync \
  --no-emit-full-json
```

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --observation-mode curvytron_hybrid_compact_visual_sample \
  --compute h100 \
  --batch-size 512 \
  --player-count 2 \
  --body-capacity 1024 \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --num-simulations 32 \
  --hidden-dim 64 \
  --max-depth 16 \
  --warmup-runs 2 \
  --steady-runs 5 \
  --closed-loop-steps 24 \
  --native-actor-buffer \
  --no-compact-root-copy-observation \
  --no-compact-visual-resident-sync \
  --no-emit-full-json
```

## Existing Artifact Context

Recent same-denominator hybrid artifacts already show the key sidecar shape:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_direct_20260523a
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_compacttorch_20260523a
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_servicetax_20260523a
artifacts/local/curvytron_hybrid_observation_profile_results/dataflow_wave2_mock_20260523a
```

H100 dataflow wave read:

| sims | direct CTree steps/sec | compact Torch steps/sec | service-tax steps/sec | mock steps/sec |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 5467 | 4047 | 7812 | 7462 |
| 32 | 3137 | 2674 | 5192 | 9171 |

Plain read: eager compact Torch is not the next lane. Service-tax and mock still
show headroom, so the next useful work is a better search/data owner, fixed-A3
integration, or MCTX/JAX-style comparator.

Fixed-A3 no-model evidence exists and is promising but not yet same-currency
with hybrid sidecar rows:

```text
artifacts/local/ctree_no_model_microbench_20260522b.jsonl
```

H100 no-model flat-A3 gate passed deterministic parity and produced roughly
`1.69x` all3 and `1.66x` mixed_2of3 speedup versus list CTree. The next question
is whether that survives the real full-loop profile hook.

MCTX sidecar evidence also exists:

```text
curvytron_hybrid_compact_visual_sample, H100, B512/P2/body1024
sim16: 27635 active roots/sec
sim32: 22219 active roots/sec
```

That is architecture evidence only: toy JAX model/search, not LightZero CTree
and not Coach launch advice.

## Tooling Gaps

1. The hybrid builder cannot build one mixed-mode manifest. Direct CTree,
   service-tax, compiled dense, and mock rows require sibling manifests or a
   hand-authored composite manifest.
2. MCTX has no durable manifest runner, no local `rows.jsonl` collection path,
   and no summarizer. We still paste commands/app ids into docs.
3. `summarize_curvytron_optimizer_profile_results.py` summarizes stock
   full-loop profiles only. Hybrid sidecar result comparison is still mostly
   `jq`, ad hoc tables, and doc prose.
4. Fixed-A3 is split across no-model microbench and stock full-loop profile
   tooling; it is not available as a hybrid boundary mode beside service-tax.
5. Result currencies are easy to mix: Coach training speed, stock full-loop
   profile speed, hybrid sidecar steps/sec, raw probe roots/sec, and MCTX active
   roots/sec are all different.
6. Some existing artifacts have null or uneven ledger fields, especially
   root-noise and array-ceiling byte fields. The runner now has better fallback
   behavior, but old summaries remain dirty.
7. The builder still writes `commands.sh`, even though the current best practice
   is "do not run commands.sh" for hybrid rows.
8. Promotion-blocking fields are strongest in runner summaries, not uniformly
   present in every generated manifest row. Add `promotion_eligible=false` and
   `promotion_blocker` to generated rows to make the lock visible before run
   time.
9. There is no single artifact index answering "what was the latest clean row
   for this denominator?" We discover it by directory names and docs.

## Recommended Tooling Patch After This Wave

Add a tiny `summarize_curvytron_hybrid_observation_profile_results.py` that
reads one or more hybrid result dirs, emits a same-denominator table, and fails
closed if:

```text
profile_only is not true
calls_train_muzero is not false
touches_live_runs is not false
promotion_eligible is not false
root_noise_weight is missing when forced
compact replay proof is missing for service rows
fallback counts are nonzero
summary currency is mixed
```

That would remove most of the current fumble factor without touching training.
