# Compact Slab Validation Audit - 2026-05-23d

Scope: compact slab/profile path only. I did not inspect or touch live Coach training runs, trainer defaults, or checkpoint jobs.

## Verdict

The core CompactRolloutSlab contract is mostly sound for profile-only runs:

- The selected action from search is staged and fed into the next environment step.
- The next step must prove it actually used that staged joint action before replay index rows are emitted.
- Root identity/order is guarded by the compact search result validator and replay-index builder.
- The LightZero compact service adapters preserve row/player/policy id identity and now carry small profile telemetry while dropping bulky debug arrays.

There is one important profiling-report caveat before larger runs: the manifest summary appears to prefer the last slab search telemetry value for `probe_total_sec` over the aggregate `timings["compact_rollout_slab_sec"]`. If that telemetry is per-call, `probe_roots_per_sec` can be inflated. Fix or explicitly audit this before trusting large-grid summary throughput numbers.

## Checks

### Selected Action Feedback

Implementation: `src/curvyzero/training/compact_rollout_slab.py:80`

`CompactRolloutSlab.step()` searches the current compact batch, converts active-root selected actions back to dense `[batch, player]` joint actions, and stores them in `_pending`.

The profile loop then reuses `next_compact_rollout_slab_action` as the next call's environment action in `src/curvyzero/training/source_state_hybrid_observation_profile.py:1456`.

The next slab call rejects any batch whose `joint_action` does not match the staged selected actions in `src/curvyzero/training/compact_rollout_slab.py:118`.

Test coverage:

- `tests/test_compact_search_replay_contract.py::test_compact_rollout_slab_stages_actions_and_commits_previous_index_rows`
- `tests/test_compact_search_replay_contract.py::test_compact_rollout_slab_rejects_next_batch_that_ignored_staged_actions`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_manager_can_use_profile_only_compact_rollout_slab`

### Committed Replay Index Rows

Implementation: `src/curvyzero/training/compact_rollout_slab.py:128`

Replay index rows are built only for the previous pending search, using the next batch's reward/done/final-row facts. The builder also verifies that `selected_action` matches the dense `next_joint_action`.

Test coverage:

- `tests/test_compact_search_replay_contract.py::test_compact_rollout_slab_stages_actions_and_commits_previous_index_rows`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_real_direct_ctree_compact_service_drives_next_step_and_matches_rows`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_compact_torch_search_service_drives_next_step_and_matches_rows`

### Root Identity And Order

Implementation: `src/curvyzero/training/compact_policy_row_bridge.py:276`

`validate_compact_search_result_v1()` constructs `root_index`, `env_row`, `player`, and `policy_env_id` from `np.flatnonzero(active_root_mask)`, so validated results are in compact active-root order.

Replay builders re-check that the search result still matches the root batch in `src/curvyzero/training/compact_policy_row_bridge.py:381` and `src/curvyzero/training/compact_policy_row_bridge.py:447`.

Adapter tests cover the LightZero service edges:

- `tests/test_source_state_batched_observation_boundary_profile.py::test_lightzero_compact_search_service_adapter_preserves_root_identity`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_lightzero_array_ceiling_compact_search_service_adapter_preserves_identity`

### Summary Telemetry

Implementation:

- Adapter telemetry trim: `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:4962`
- Slab flattening: `src/curvyzero/training/compact_rollout_slab.py:193`
- Manifest summary: `scripts/run_curvytron_hybrid_observation_profile_manifest.py:152`

I added a focused test:

- `tests/test_compact_search_replay_contract.py::test_compact_rollout_slab_flattens_search_profile_telemetry`

This proves direct CTree-style nested telemetry is flattened into the slab fields for total/model/search/H2D/byte counters.

Audit caveat: manifest summary should not treat one last-step `compact_rollout_slab_search_service_total_sec` as the denominator for all roots unless that value has already been aggregated across calls. Current code looks like it may do that. Before larger runs, add/fix a focused manifest-runner test for "slab summary uses aggregate timings for denominator, not last telemetry".

## Validation Run

Passed:

```bash
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py::test_compact_rollout_slab_stages_actions_and_commits_previous_index_rows \
  tests/test_compact_search_replay_contract.py::test_compact_rollout_slab_flattens_search_profile_telemetry \
  tests/test_compact_search_replay_contract.py::test_compact_rollout_slab_rejects_next_batch_that_ignored_staged_actions \
  tests/test_compact_search_replay_contract.py::test_selected_joint_action_from_search_result_rejects_illegal_actions
```

Passed:

```bash
uv run ruff check tests/test_compact_search_replay_contract.py
```

## Missing Gate Before Larger Profile-Only Runs

Fix or prove the manifest summary denominator first. Then run a small real direct CTree slab profile and check these fields are present and plausible:

- `compact_rollout_slab_enabled=true`
- `compact_rollout_slab_calls == measured steps`
- `compact_rollout_slab_committed_index_rows == (measured steps - 1-ish warmup effects) * active roots`
- `compact_rollout_slab_search_impl`
- aggregate `probe_total_sec`
- aggregate or clearly per-call `model_sec`, `search_sec`, `h2d_sec`

The current slab is ready for profile-only canaries, but the summary denominator caveat is important enough that I would not use large-grid roots/sec numbers as final evidence until it is fixed or explicitly explained.
