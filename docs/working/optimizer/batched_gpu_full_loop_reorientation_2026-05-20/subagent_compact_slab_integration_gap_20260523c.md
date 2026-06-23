# Compact Slab Integration Gap 2026-05-23c

## Verdict

`CompactRolloutSlab` is not actually wired into the Modal hybrid boundary runner
or the hybrid profile manifest builder. It is implemented and tested one layer
down, inside the local hybrid manager path, but no current Modal/local-entrypoint
flag constructs it and no manifest row can request it.

The currently wired end-to-end mode is `compact_service_replay_proof`, which is a
separate helper path over compact search arrays. It validates action feedback,
two-phase payloads, and index rows, but it bypasses `CompactRolloutSlab`.

## Current Wiring

- `CompactSearchServiceV1` exists as the small service contract:
  `src/curvyzero/training/compact_search_service.py:22-30`. Its array adapter
  helper validates `selected_action`, `visit_policy`, and `root_value` into a
  `CompactSearchResultV1`: `src/curvyzero/training/compact_search_service.py:63-88`.
- `CompactRolloutSlab` is implemented as a profile-only owner:
  `src/curvyzero/training/compact_rollout_slab.py:48-115`. It builds
  `CompactRootBatchV1`, calls `search_service.run(...)`, maps selected actions
  back to dense joint actions, and exposes slab telemetry. The delayed commit
  path validates that the next env step applied the staged actions:
  `src/curvyzero/training/compact_rollout_slab.py:117-144`.
- `HybridBatchedObservationProfileManager` can accept an injected slab:
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:590-607`.
  During `step(...)`, it builds `HybridCompactBatch` when either a batched probe
  or slab is present, then calls `self.compact_rollout_slab.step(...)`:
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:1004-1043`.
  The step object exposes `compact_rollout_slab_step` and telemetry:
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:1136-1154`.
- The high-level `run_hybrid_observation_profile(...)` is not wired for the
  slab. Its signature accepts `policy_search_probe` and `batched_stack_probe`,
  but no `compact_rollout_slab`: `src/curvyzero/training/source_state_hybrid_observation_profile.py:1360-1364`.
  It constructs the manager without passing a slab:
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:1386-1390`.
  Its only compact action-feedback loop is `compact_service_replay_proof`:
  `src/curvyzero/training/source_state_hybrid_observation_profile.py:1438-1455`
  and `src/curvyzero/training/source_state_hybrid_observation_profile.py:1513-1540`.
- The Modal profile module has CompactSearchService adapters:
  `_LightZeroCollectForwardCompactSearchService` at
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:4916-4941`
  and `_LightZeroArrayCeilingCompactSearchService` at
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:4944-4970`.
  The Modal hybrid implementation constructs only `batched_stack_probe`
  variants and passes only `batched_stack_probe` into
  `run_hybrid_observation_profile(...)`:
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2324-2436`.
- The Modal local entrypoint exposes `hybrid_compact_service_replay_proof`, but
  no `hybrid_compact_rollout_slab` argument:
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:9535-9562`.
  The config dict similarly carries replay proof and LightZero probe settings,
  but no slab switch:
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:9607-9629`.
- The manifest builder has `--compact-service-replay-proof`, but no compact slab
  flag: `scripts/build_curvytron_hybrid_observation_profile_grid.py:485-597`.
  Command emission adds replay proof and LightZero probe flags only:
  `scripts/build_curvytron_hybrid_observation_profile_grid.py:145-278`.
  Row metadata has `compact_service_replay_proof`, not slab metadata:
  `scripts/build_curvytron_hybrid_observation_profile_grid.py:390-463`.
- The durable manifest runner only validates profile-only commands and summarizes
  array/proof telemetry. It has no slab-specific preflight or summary fields:
  `scripts/run_curvytron_hybrid_observation_profile_manifest.py:92-125` and
  `scripts/run_curvytron_hybrid_observation_profile_manifest.py:148-319`.

## Exact Flags And Modes

Current Modal-facing hybrid/profile flags are hyphenated versions of the
`main(...)` parameters in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:9501-9571`.
The builder uses shorter names and emits the `--hybrid-*` Modal flags.

Search/proof flags:

- Builder: `--compact-service-replay-proof`; Modal: `--hybrid-compact-service-replay-proof`.
- Builder: `--lightzero-collect-forward-probe`; Modal: `--hybrid-lightzero-collect-forward-probe`.
- Builder: `--lightzero-initial-inference-probe`; Modal: `--hybrid-lightzero-initial-inference-probe`.
- Builder: `--lightzero-array-ceiling-probe`; Modal: `--hybrid-lightzero-array-ceiling-probe`.
- Builder: `--lightzero-mcts-arrays-boundary-probe`; Modal: `--hybrid-lightzero-mcts-arrays-boundary-probe`.
- Builder: `--lightzero-mock-service-materialize-public-output`; Modal: `--hybrid-lightzero-mock-service-materialize-public-output`.
- Builder and Modal also expose `lightzero-consumer-num-simulations`,
  `temperature`, `epsilon`, `root-noise-weight`, `use-cuda`, and
  `collect-with-pure-policy`: `scripts/build_curvytron_hybrid_observation_profile_grid.py:557-573`.

Array-ceiling modes are:

- `policy_arrays`
- `mock_search_service`
- `service_tax_probe`
- `recurrent_toy`
- `dense_torch_mcts`
- `dense_torch_mcts_compile_spike`
- `compact_torch_search_service`

References: `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:163-182`
and `scripts/build_curvytron_hybrid_observation_profile_grid.py:34-42`.

MCTS arrays-boundary impls are:

- `stock_facade`
- `direct_ctree_arrays`
- `direct_ctree_gpu_latent`
- `direct_ctree_gpu_latent_precomputed_recurrent`

References: `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:198-209`
and `scripts/build_curvytron_hybrid_observation_profile_grid.py:28-33`.

Input modes are shared by array-ceiling and MCTS-array probes:

- `host_uint8`
- `host_uint8_pinned`
- `host_float32`
- `resident_torch_reuse`

References: `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:183-192`
and `scripts/build_curvytron_hybrid_observation_profile_grid.py:523-547`.

Replay-proof eligibility is narrower than the full mode set. Builder validation
allows replay proof for direct MCTS arrays or array-ceiling modes
`mock_search_service`, `service_tax_probe`, `dense_torch_mcts`,
`dense_torch_mcts_compile_spike`, and `compact_torch_search_service`:
`scripts/build_curvytron_hybrid_observation_profile_grid.py:43-51` and
`scripts/build_curvytron_hybrid_observation_profile_grid.py:293-345`. Modal
runtime validation mirrors that and rejects stale `resident_torch_reuse` and
`stock_facade` for direct replay proof:
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2240-2275`.

There is no current `compact_rollout_slab` flag, row field, or Modal mode.

## Smallest Missing Implementation

The smallest profile-only end-to-end implementation is a new opt-in slab mode
that reuses the already implemented manager hook and existing search-service
adapters:

1. Add `compact_rollout_slab: Any | None = None` to
   `run_hybrid_observation_profile(...)`, pass it into
   `HybridBatchedObservationProfileManager`, and reject combining it with
   `compact_service_replay_proof` until there is a reason to support both.
2. In the high-level profile loop, mirror the replay-proof action-feedback
   pattern: keep `next_compact_rollout_slab_action`; use it as the next
   `actions`; after each step, set it from
   `step.compact_rollout_slab_step.next_joint_action`. The manager already times
   `compact_rollout_slab_sec`; the runner should also aggregate
   `compact_rollout_slab_calls`, committed index-row count, and last slab
   telemetry.
3. Add a Modal flag, for example `hybrid_compact_rollout_slab`, plus builder
   flag `--compact-rollout-slab`. Validate it requires a compact search source:
   either direct MCTS arrays-boundary with a direct impl, or an array-ceiling
   mode that emits compact search arrays. Keep the same fresh-input guard as
   replay proof.
4. In `_run_hybrid_observation_profile_impl(...)`, construct a `CompactRolloutSlab`
   after the selected probe is built:
   - direct MCTS arrays: use `_LightZeroCollectForwardCompactSearchService(batched_stack_probe)`;
   - array-ceiling compact-array modes: use `_LightZeroArrayCeilingCompactSearchService(batched_stack_probe)`;
   - `compact_torch_search_service`: either construct `CompactTorchSearchServiceV1`
     directly, or teach the array-ceiling adapter to call `run_compact_batch(...)`.
     Current `compact_torch_search_service` service execution lives behind
     `run_compact_batch(...)`, not plain `run(...)`:
     `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:7539-7639`.
5. Pass the slab into `run_hybrid_observation_profile(...)` and include slab
   config/result fields in the Modal JSON and durable runner compact summary.

This is deliberately smaller than a training integration: it stays profile-only,
does not call `train_muzero`, and does not change live trainer defaults.

## Tests

Existing tests that already cover pieces:

- Slab staging/commit and illegal-action rejection:
  `tests/test_compact_search_replay_contract.py:442-567`.
- Manager-level slab injection:
  `tests/test_source_state_hybrid_observation_profile.py:79-134`.
- Modal CompactSearchService adapters:
  `tests/test_source_state_batched_observation_boundary_profile.py:1668-1763`.
- Real direct CTree compact service closed-loop proof:
  `tests/test_source_state_batched_observation_boundary_profile.py:1766-1855`.
- Compact Torch service mode proof:
  `tests/test_source_state_batched_observation_boundary_profile.py:2073-2183`.
- Builder coverage for replay-proof and current mode validation:
  `tests/test_curvytron_hybrid_observation_profile_grid_builder.py:237-340`,
  `tests/test_curvytron_hybrid_observation_profile_grid_builder.py:415-430`,
  and `tests/test_curvytron_hybrid_observation_profile_grid_builder.py:541-563`.

Add tests before any Modal run:

- `test_run_hybrid_observation_profile_accepts_compact_rollout_slab_and_drives_next_action`:
  use a fake `CompactSearchServiceV1`, pass a slab into the high-level runner,
  assert step 1 applied step 0's staged action, and assert committed index rows
  plus runner-level slab telemetry.
- `test_hybrid_grid_can_emit_compact_rollout_slab_service_tax_rows`: builder emits
  `--compact-rollout-slab`, Modal command includes the `--hybrid-*` slab flag,
  and row metadata labels the slab mode separately from `compact_service_replay_proof`.
- `test_hybrid_grid_rejects_compact_rollout_slab_without_search_service`: reject no
  LightZero compact search source, `resident_torch_reuse`, and direct
  `stock_facade`.
- `test_modal_hybrid_constructs_compact_rollout_slab_for_array_ceiling`: monkeypatch
  the profile runner or use a tiny fake probe to assert `_run_hybrid_observation_profile_impl`
  passes a non-null slab when the flag is set.
- `test_array_ceiling_service_tax_compact_rollout_slab_drives_next_step_and_index_rows`
  and `test_compact_torch_search_service_compact_rollout_slab_drives_next_step_and_index_rows`:
  exercise real `CompactSearchServiceV1` backends behind the slab without live Modal.
- `test_hybrid_manifest_runner_compact_line_reports_compact_rollout_slab`: ensure
  the durable runner preserves slab fields in `summary`.

Suggested local bounded run after implementation:

```bash
uv run pytest -q \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py \
  tests/test_curvytron_hybrid_observation_profile_manifest_runner.py \
  -k "compact_rollout_slab or compact_service or compact_torch_search_service or compact_replay"
```

Do not use live Modal or trainer runs for the first pass; a builder dry-run and
local unit tests are enough to prove the profile-only integration.
