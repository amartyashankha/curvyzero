# Next Backend Gap, 2026-05-23c

Scope: read-only source/doc inspection plus this report. I did not touch live
training, Modal, checkpoints, evals, tournaments, GIFs, volumes, or source code.

## Short Answer

The best base for the next stronger fixed-shape backend is the compact service
contract plus the standalone `CompactTorchSearchServiceV1` harness, with direct
CTree kept as the semantic oracle. Do not promote the current eager compact Torch
or boundary-embedded dense Torch code as the backend itself. They are scaffolding
and donor logic. The next meaningful experiment needs a new fixed-shape search
body behind `CompactSearchServiceV1`, not another wrapper-only change.

There is one small code change that would make a follow-up experiment more honest:
the standalone compact Torch service records compile eligibility, but its tree
search still calls eager helpers. Either wire actual compiled helpers under strict
preconditions or rename/guard the telemetry so "compile enabled" cannot imply the
service used compiled code. That is a measurement fix, not the likely speed win.

## Existing Modes

| Candidate | Mode / impl | Where implemented | What it actually does |
| --- | --- | --- | --- |
| Direct CTree | `direct_ctree_arrays`, `direct_ctree_gpu_latent`, `direct_ctree_gpu_latent_precomputed_recurrent` | Boundary impl constants and validation: `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:193-209`, `:973-984`. Probe names: `:4990-5032`. Main direct path: `:5891-6004`; compact arrays: `:6153-6163`; GPU-latent loop: `:6436-6454`. Compact-service adapter: `:4916-4941`. | Real LightZero CTree MCTS. `direct_ctree_gpu_latent` keeps latent storage on device, but CTree traversal/backprop and recurrent output payloads still cross CPU/list boundaries. This is the semantic baseline. |
| Service-tax | `service_tax_probe` | Mode constant: `:164-181`. Semantics: `:7005-7014`. Recurrent tax loop: `:7152-7243`. Compact arrays: `:7278-7286`. Telemetry: `:7392-7421`. Adapter contract path: `:7539-7569`. | Real initial inference plus one recurrent inference per requested simulation, but no real MCTS/CTree update. It prices model/service/control tax and emits validated compact arrays. |
| Mock | `mock_search_service` | Mode constant: `:164-181`. Semantics: `:7005-7014`. Compact arrays: `:7278-7286`. Public-output pricing: `:7291-7348`. Telemetry: `:7362-7391`. | No real search and no recurrent calls. It uses initial policy/value output, stores compact arrays, and optionally materializes LightZero-shaped public output to price object/scalar overhead. |
| Dense Torch | `dense_torch_mcts` | Mode constant: `:167`. Dense dispatch: `:7128-7150`. Dense search body: `:7921-8200`. Compact replay allowance: `:2240-2256`. | Profile-only dense Torch PUCT-like search tensors on the model device. It is not LightZero CTree semantics and remains embedded in the Modal boundary profile. |
| Compile spike | `dense_torch_mcts_compile_spike` | Mode constant: `:168`. Compile helper gate: `:8202-8374`. Helper kernels: `:8450-8594`. Local tests: `tests/test_source_state_batched_observation_boundary_profile.py:3883-4010`. | Same dense Torch body, but tries `torch.compile(..., fullgraph=True)` for selection and backup helpers only when CUDA, all roots/actions legal, and root noise is zero. Falls back with telemetry. |
| Compact Torch | `compact_torch_search_service` / `compact_torch_device_tree_fixed_shape_v0` | Boundary mode: `source_state_batched_observation_boundary_profile.py:169`, `:7601-7819`. Standalone service: `src/curvyzero/training/compact_torch_search_service.py:1-6`, `:21-29`, `:79-238`, `:572-784`. | First standalone `CompactRootBatchV1 -> CompactSearchResultV1` Torch candidate. It owns one model/search pass and validates via the compact service contract. Current tree helpers are eager; compile eligibility is telemetry/precondition logic, not actual compiled execution. |

The compact contract itself is narrow and useful: `CompactSearchServiceV1.run()`
takes `CompactRootBatchV1` and returns `CompactSearchResultV1`
(`src/curvyzero/training/compact_search_service.py:22-30`). Array-to-result
validation is centralized at `compact_search_result_v1_from_arrays(...)`
(`src/curvyzero/training/compact_search_service.py:63-88`) and the result
validator rejects illegal actions, illegal visit mass, nonfinite values, and bad
row identity (`src/curvyzero/training/compact_policy_row_bridge.py:254-363`).

## Best Base

Use `CompactTorchSearchServiceV1` as the code base for the next stronger
fixed-shape backend, but not as the final backend algorithm. It is the best
surface because it is already outside the giant Modal profile file, consumes the
right `CompactRootBatchV1`, emits a validated `CompactSearchResultV1`, records
profile-only labels, and has focused tests around fresh observations, masks,
compile eligibility, and select/backup helpers
(`src/curvyzero/training/compact_torch_search_service.py:79-238`,
`tests/test_compact_torch_search_service.py:250-323`,
`tests/test_compact_torch_search_service.py:327-401`).

Keep `_LightZeroCollectForwardCompactSearchService` as the semantic control and
fallback oracle, because it adapts real direct CTree output into the same compact
service contract (`source_state_batched_observation_boundary_profile.py:4916-4941`).
Do not base the next implementation primarily on `_run_dense_torch_mcts`; that
code is useful donor logic, but it is embedded in the profile boundary and the
latest same-denominator evidence says it is not a promotion candidate.

Current performance evidence supports that split:

- The 2026-05-23 compact Torch smoke says the service boundary is real, but the
  wall is the eager Python/Torch tree plus recurrent loop, not input transfer
  (`docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/experiment_log.md:5-39`).
- Same-denominator rows show dense Torch beating direct at sim16 but regressing
  at sim32, while service-tax/mock still expose headroom
  (`experiment_log.md:10537-10579`).
- The later wave says eager compact Torch is worse than direct CTree on H100 for
  sim16 and sim32, so the next move is fixed-shape compiled search, array-native
  CTree, MCTX/JAX comparator, or a slab/search service with delayed replay
  payloads (`experiment_log.md:10698-10771`).

## Small Change Or New Backend?

Small meaningful patch: make compact Torch compile telemetry match execution. In
`CompactTorchSearchServiceV1.run()`, eligibility is computed
(`compact_torch_search_service.py:132-140`) and can report compile-enabled
preconditions (`:302-417`), but `_run_compact_torch_tree_search()` still creates
and calls eager helpers directly (`:669-670`) with no `torch.compile` call in the
service path. A bounded patch could pass compiled helpers into the tree body only
when eligibility is true, with cache key = root count, sim count, observation
shape, dtype, device, model identity, action count, and root-noise policy.

That patch would make a compile experiment meaningful, but it is not enough for
the next speed thesis. The stronger backend needs new ownership:

```text
CompactRootBatchV1
-> fixed/padded R roots, A=3
-> preallocated tree/search buffers
-> no per-sim D2H/list payloads unless a CPU CTree backend explicitly requires them
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

The likely first implementation should either be a fixed-shape compiled/fused
Torch search service behind the standalone module, or a fixed-A3/array-native
CTree compatibility service if semantic preservation is the priority. MCTX/JAX
remains a strong side comparator, not a drop-in LightZero trainer path.

## Risk And Test Plan

Main risks:

- Semantic drift from LightZero CTree: PUCT score, reward/discount backup, root
  value fallback, root noise, temperature, and single-legal-action behavior.
- Shape drift: active-root compaction, terminal/autoreset roots, partial masks,
  and compile-cache reuse can break the fixed-shape claim.
- Identity drift: `root_index`, `env_row`, `player`, `policy_env_id`, and selected
  action at record `k` must line up with the env transition at record `k+1`.
- Timing drift: CUDA enqueue timing, fallback paths, compile warmup, and replay
  proof cost can make a row look faster or slower than the actual search boundary.

Test gates:

1. Local contract tests: `CompactRootBatchV1` active-root validation and
   `CompactSearchResultV1` illegal action/visit-mass failures
   (`compact_policy_row_bridge.py:124-250`, `:254-363`).
2. Direct CTree oracle tests: no-noise, deterministic small cases for selected
   action, visit distribution, root value, reward/discount backup, partial masks,
   and root noise legality.
3. Closed-loop compact proof: selected actions drive the next env step, compact
   replay index rows materialize to trusted target rows, terminal final
   observations survive autoreset, RND latest-frame identity stays attached, and
   non-identity `policy_env_id` is preserved. Existing direct and compact Torch
   smokes are at `tests/test_source_state_batched_observation_boundary_profile.py:1766-2070`.
4. Compile/static proof: assert no fallback, no dynamic-shape recompile, cache key
   includes shape/sim/device/dtype/model/action count, and compile warmup is
   excluded from warm timing but reported.
5. Profile-only keep/kill grid: same H100 B512/A16 sim16 and sim32 denominator
   with direct CTree, service-tax, mock, and candidate. Keep only if the candidate
   beats direct after required readback and passes the identity/replay gates.

No Coach-facing recommendation until a later capped stock-vs-candidate trainer
smoke calls the real trainer entrypoint with no fallback and with replay/sample/RND
digests. For this report, the answer is bounded: use the compact service harness,
write a new backend body, and keep direct CTree plus service-tax/mock as the
oracle/ceiling rails.
