# Fixed Shape Validation Plan

Date: 2026-05-23

Purpose: validate a future `FixedShapeBatchedSearchOwnerV0` behind
`CompactSearchServiceV1` before we trust any speedup.

This is local/profile-only work. Do not touch live Modal runs. Do not wire this
into Coach training until the gates below pass.

## Core Rule

Speed does not count until correctness is boring.

The fixed-shape owner may change the search algorithm later, but it may not
change identity, legality, action feedback, terminal replay, or sampler-visible
rows by accident.

## Exact Local Tests To Have

Run the existing compact contract suite first:

```bash
uv run pytest \
  tests/test_compact_torch_search_service.py \
  tests/test_mctx_compact_search_service.py \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_hybrid_observation_profile.py
```

These tests already cover the compact root/search/replay contract, delayed
payload digest checks, non-prefix active roots, final observations before
autoreset, compact slab action feedback, and profile-only metadata.

Add a new focused file:

```text
tests/test_fixed_shape_batched_search_owner.py
```

Required tests:

- `test_fixed_shape_owner_returns_valid_compact_search_result_v1`
  - Build a tiny `CompactRootBatchV1`.
  - Run the owner through the `CompactSearchServiceV1` protocol.
  - Pass the result through `validate_compact_search_result_v1`.
  - Assert `root_index`, `env_row`, `player`, and `policy_env_id` match the active roots exactly.

- `test_fixed_shape_owner_preserves_non_prefix_active_root_order`
  - Use active roots `[False, True, False, True]`.
  - Use `policy_env_id=[101, 103, 107, 109]`.
  - Expected output roots are `[1, 3]`, not `[0, 1]`.
  - No inactive or done root may appear in `selected_action`, visits, values, or replay rows.

- `test_fixed_shape_owner_rejects_or_masks_no_legal_action_roots`
  - Active root with no legal action must fail before search.
  - Inactive root with no legal action may be padded internally, but must not appear in output.

- `test_fixed_shape_owner_never_selects_or_visits_illegal_actions`
  - Use mixed masks: all legal, two legal, and one legal.
  - `selected_action` must be legal.
  - `visit_policy` and `raw_visit_counts` must put zero mass on illegal actions.
  - Each `visit_policy` row must sum to 1 within `1e-6`.

- `test_fixed_shape_owner_selected_actions_drive_next_env_step`
  - Feed the search result through `selected_joint_action_from_search_result`.
  - Step the compact slab with that action.
  - `build_compact_replay_index_rows_v1_from_search_result` must accept the next batch.
  - A batch that ignores one selected active-seat action must fail.

- `test_fixed_shape_owner_two_phase_payload_blocks_stale_replay`
  - Split action-critical and replay-critical payloads.
  - Assert sample visibility is false before payload flush and true after.
  - Reordered identity, wrong handle, and same-identity changed visits/values must fail.

- `test_fixed_shape_owner_index_rows_materialize_same_as_immediate_rows`
  - Build compact index rows from the owner result.
  - Materialize at the sampler edge.
  - Compare to `build_compact_target_rows_from_search_arrays_v0`.
  - Include a terminal row and verify `final_observation` is used instead of autoreset state.

- `test_fixed_shape_owner_preserves_rnd_latest_frame_and_player_perspective`
  - Use the existing latest-frame sentinel pattern.
  - `extract_policy_gray64_latest_for_rnd_from_compact_observation` must point at the same compact roots as replay rows.
  - Player perspective must not swap across seats.

- `test_fixed_shape_owner_same_shape_calls_use_fresh_observations`
  - Run two same-shape batches through one owner instance.
  - Change only a sentinel pixel that the fake model reads.
  - The selected action must change with the new observation.
  - This catches stale resident buffers and cached root tensors.

- `test_fixed_shape_owner_reports_fail_closed_profile_counters`
  - Assert fixed-shape counters are present.
  - Assert forbidden hot-path counters are zero.
  - Assert real search/action/replay counters are nonzero.

If the local environment has LightZero installed, these stock boundary tests are
required for trust. A skip is fine for day-to-day local work, but not for a
promotion claim:

```bash
uv run pytest \
  tests/test_compact_search_replay_contract.py::test_compact_index_rows_materialized_stock_lightzero_target_hooks_match \
  tests/test_compact_search_replay_contract.py::test_compact_index_rows_materialized_stock_lightzero_public_sample_matches
```

## Tiny CTree Oracle Fixtures

Keep the oracle fixtures tiny. They should be easy to debug by eye.

Use zero root noise, deterministic seeds, `ACTION_COUNT=3`, fixed simulation
count, and no epsilon exploration.

### Fixture 1: One Root, One Simulation

Shape:

```text
root_count=1
active_root_mask=[True]
legal_mask=[[True, True, True]]
root_policy_logits favor action 1
reward/value zeros
simulations=1
```

Strict CTree-clone mode must match:

- selected action;
- raw visit counts;
- normalized visit policy;
- root value within `1e-6`;
- no illegal action count.

### Fixture 2: Four Row/Player Roots

Shape:

```text
batch_size=2
player_count=2
root_index=[0, 1, 2, 3]
env_row=[0, 0, 1, 1]
player=[0, 1, 0, 1]
policy_env_id=[10, 11, 12, 13]
mixed legal masks
```

Must match:

- sidecar identity exactly;
- active output order exactly;
- legal action behavior exactly;
- strict CTree-clone search outputs within tolerance.

### Fixture 3: Non-Prefix Active Roots

Shape:

```text
active_root_mask=[False, True, False, True]
done_root=[True, False, True, False]
policy_env_id=[101, 103, 107, 109]
```

Must match:

- output `root_index=[1, 3]`;
- output `policy_env_id=[103, 109]`;
- no output for roots 0 or 2;
- fixed-shape padding counters may show padded roots, but replay rows may not.

### Fixture 4: Ties

Shape:

```text
equal priors
equal values
all actions legal
zero noise
```

This fixture is not a promotion gate for exact action equality unless the CTree
oracle and owner use the same deterministic tie break.

Must match:

- identity;
- legality;
- visit mass only on legal actions;
- root value tolerance.

Can differ:

- which tied action wins, if tie handling is explicitly documented.

### Fixture 5: Terminal Replay

Shape:

```text
two records
record 1 has one terminal env row
final_observation differs from autoreset/latest observation
```

Must match:

- next action feedback from record 0 into record 1;
- final reward map;
- `next_final_observation_row`;
- materialized target rows against the immediate compact rows.

## CTree Oracle Commands

Use the existing stock-vs-direct helper as the reference check for the current
CTree path:

```bash
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seed-start 20260523 \
  --seeds 4 \
  --batch-rows 4 \
  --num-simulations 8 \
  --impls direct_ctree_gpu_latent \
  --root-noise-weight 0.0 \
  --epsilon 0.0 \
  --temperature 1.0 \
  --action-mask-scenario mixed_legal_cycle \
  --strict-exact
```

Add the fixed-shape equivalent before trusting the owner:

```text
scripts/compare_fixed_shape_owner_ctree.py
```

It should feed the same `CompactRootBatchV1` into:

- reference: LightZero CTree/direct CTree compact search;
- candidate: `FixedShapeBatchedSearchOwnerV0`;
- comparator: `CompactSearchComparatorServiceV1`.

Required output fields:

- `identity_match`;
- `action_match_fraction`;
- `visit_l1_mean`;
- `visit_l1_max`;
- `root_value_abs_diff_mean`;
- `root_value_abs_diff_max`;
- `illegal_action_count`;
- `fixed_shape_forbidden_hot_path_count`;
- `candidate_search_impl`;
- `reference_search_impl`;
- `root_noise_weight`;
- `actual_search_simulations`.

Optional CTree-list ABI price check:

```bash
uv run python scripts/benchmark_lightzero_ctree_no_model.py \
  --roots 4 \
  --simulations 8 \
  --iterations 3 \
  --warmup 1 \
  --legal-profiles mixed_2of3 \
  --backends ctree-list \
  --root-noise zero \
  --root-noise-weight 0.0
```

If the vendored flat A3 extension is built, also run it with
`--backends ctree-flat-a3 --flat-a3-parity-check`.

## What Must Match

Always exact:

- `root_index`;
- `env_row`;
- `player`;
- `policy_env_id`;
- active-root count and active-root order;
- action mask legality;
- selected action legality;
- zero visit/count mass on illegal actions;
- action feedback into the next env step;
- replay index rows;
- delayed replay payload handle and digest;
- sample visibility before/after payload flush;
- terminal `final_observation` and `final_reward_map`;
- RND latest-frame root mapping;
- metadata that says profile-only and not trainer-ready.

Strict CTree-clone mode must also match:

- selected actions, except explicitly tied roots;
- raw visit counts;
- normalized visits within `1e-6`;
- searched root values within `1e-6` on CPU and `1e-5` on GPU;
- predicted root value/logits if the same model outputs are used.

## What Can Differ

These may differ only when declared as approximate search or an algorithm
change:

- visit distribution shape;
- root values;
- selected actions on non-tied roots;
- raw visit counts, if the candidate search does not expose CTree-shaped counts;
- timing fields;
- compile status on a non-speed local smoke.

Approximate search is acceptable only if identity and replay are still exact
and the comparator thresholds are set before looking at speed.

Initial approximate thresholds:

```text
non_tie_action_match_fraction >= 0.99
mean_visit_l1 <= 0.03
max_visit_l1 <= 0.25
root_value_abs_diff_mean <= 0.02
root_value_abs_diff_max <= 0.10
illegal_action_count == 0
identity_match == true
```

If these fail, the result can still be a research row, but it is not a safe
optimizer speedup. Label it `algorithm_change`, not `ctree_replacement`.

## End-To-End Stock Boundary Checks

The fixed-shape owner must survive four boundaries.

1. Compact search boundary:
   - `CompactRootBatchV1 -> CompactSearchResultV1`.
   - Validate with `validate_compact_search_result_v1`.
   - Compare against CTree oracle with `CompactSearchComparatorServiceV1`.

2. Env action boundary:
   - `CompactSearchResultV1 -> selected_joint_action_from_search_result`.
   - The next compact env step must show the same selected active-seat actions.
   - Ignored or stale selected actions must fail.

3. Replay boundary:
   - `CompactSearchResultV1 -> CompactReplayIndexRowsV1`.
   - Index rows must materialize to the same target rows as immediate compact rows.
   - Delayed replay payloads must stay hidden until the payload digest checks.

4. Stock LightZero sampler boundary:
   - Materialized compact rows must feed native LightZero game segments.
   - MuZeroGameBuffer target hooks and public `sample` output must match rows.
   - These tests may import-skip locally, but they must run in a promotion env.

Only after those pass should we run matched profile rows against the original
stock loop denominator. A profile-only compact win is not a Coach-loop win.

## Profile Counters

Add fixed-shape owner counters under the search result `profile_telemetry`, then
let `CompactRolloutSlab` promote them into slab totals.

Forbidden counters that must be zero for a speed claim:

- `fixed_shape_owner_ctree_calls`;
- `fixed_shape_owner_python_root_lists_built`;
- `fixed_shape_owner_python_simulation_payloads_built`;
- `fixed_shape_owner_scalar_timestep_rows`;
- `fixed_shape_owner_python_rows_materialized`;
- `fixed_shape_owner_rnd_materialized_rows`;
- `fixed_shape_owner_illegal_action_count`;
- `fixed_shape_owner_identity_mismatch_count`;
- `fixed_shape_owner_stale_payload_count`;
- `fixed_shape_owner_replay_rows_visible_before_payload`;
- `fixed_shape_owner_compile_fallback_count` in the compiled fixed-shape row.

Expected nonzero counters:

- `fixed_shape_owner_calls`;
- `fixed_shape_owner_root_count`;
- `fixed_shape_owner_active_root_count`;
- `fixed_shape_owner_padded_root_count` for padded fixtures;
- `fixed_shape_owner_masked_inactive_root_count` for inactive-root fixtures;
- `fixed_shape_owner_requested_simulations`;
- `fixed_shape_owner_actual_search_simulations`;
- `fixed_shape_owner_search_sec`;
- `fixed_shape_owner_model_sec` when using a real model;
- `fixed_shape_owner_action_d2h_bytes`;
- `fixed_shape_owner_replay_payload_d2h_bytes`;
- `compact_rollout_slab_calls`;
- `compact_rollout_slab_total_roots`;
- `compact_rollout_slab_committed_index_row_count`;
- `compact_rollout_slab_stored_index_group_count`;
- `compact_service_replay_proof_calls` when replay proof is enabled;
- `compact_service_replay_proof_target_row_count`.

Existing compact counters that must stay clean:

- `calls_train_muzero == False`;
- `stock_lightzero_integrated == False`;
- `trainer_defaults_changed == False`;
- `touches_live_runs == False`;
- `materialize_scalar_timestep == False` for compact-owned collection tests;
- `compact_rollout_slab_profile_only == True`;
- `compact_rollout_slab_python_rows_materialized == 0`;
- `compact_rollout_slab_rnd_materialized_rows == 0`, unless the row is an
  explicit RND materialization test;
- `compact_rollout_slab_root_observation_copy_bytes == 0` after resident buffers
  are claimed.

Tail behavior must be explicit:

- if the profile closes without a real final next batch,
  `compact_rollout_slab_dropped_pending_search_count` should be nonzero;
- if a test supplies the final next batch, dropped pending searches should be
  zero and committed rows should increase once.

## Kill Criteria

Kill or demote the fixed-shape owner if any of these happen:

- identity mismatch in roots, env rows, players, or policy ids;
- selected action or visit mass on an illegal action;
- replay rows become sample-visible before the replay payload arrives;
- stale replay payload with same identity is accepted;
- terminal rows use autoreset/latest observation instead of `final_observation`;
- RND latest-frame rows point at the wrong compact root or player;
- stock LightZero target/public sample parity cannot be run in a promotion env;
- CTree/list/per-simulation host payloads sneak back into the fixed-shape hot path;
- scalar timesteps or Python policy rows are rebuilt in compact-owned collection;
- compile/fixed-shape fallback is common in the row used for speed claims;
- same-shape calls reuse stale observations;
- approximate search misses the declared thresholds and is still presented as a
  CTree replacement;
- the only speedup comes from not copying action or replay payloads that the env
  or learner still needs;
- matched compact denominator speedup over direct CTree is less than `25-30%`
  after counters are clean;
- the candidate touches live Modal/Coach runs before these local gates pass.

If the owner passes correctness but not speed, keep the tests and delete the
architecture bet. That is a good outcome.
