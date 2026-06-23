# Compact Service Next Slice, 2026-05-22

Status: read-only code exploration report. No runtime code changed. No live runs
touched.

## Files Read

- `src/curvyzero/training/compact_policy_row_bridge.py`
- `src/curvyzero/training/replay_chunk_v0.py`
- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `tests/test_compact_search_replay_contract.py`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/compact_search_replay_service_contract_20260522.md`

## Plain Read

The smallest useful next step is not a new trainer and not a new GPU MCTS
implementation.

The smallest useful next step is a profile-only compact service slice:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> existing direct_ctree_gpu_latent search wrapper
-> CompactSearchResultV1
-> compact target-row/replay proof
```

This keeps the current LightZero-compatible search hook as the control service
while proving that the data can stay in explicit compact arrays across the
search/replay boundary. If this slice is not clean, deeper GPU search work will
only hide bugs behind more moving parts.

## What Already Exists

`HybridCompactBatch` already carries the right pre-scalar facts:

- `[B,P,4,64,64]` observation;
- `[B,P,3]` action mask;
- reward, done, row/player ids;
- `target_reward`, `done_root`, `to_play`, `active_root_mask`;
- final-observation/autoreset/terminal sidecars;
- episode metadata and joint action.

`CompactPolicySearchArraysV0` already validates active-root search output:

- selected action legality;
- zero illegal visit mass;
- visit policy sums to one;
- row/player id alignment;
- fixed-opponent `to_play=-1`;
- active roots equal non-done roots with legal actions.

`build_compact_target_rows_from_search_arrays_v0()` already proves the compact
search output can match the existing object target-row builder for key cases:

- terminal/final observation before autoreset;
- non-prefix active roots;
- compact RND latest-frame extraction.

This is good scaffolding. It is not yet a compact service or compact replay
owner.

## Smallest Next Code Slice

Add one narrow module, likely:

```text
src/curvyzero/training/compact_search_replay_service.py
```

### Dataclasses To Add

`CompactRootBatchV1`

Purpose: checked input to a batched search service, derived from
`HybridCompactBatch`.

Fields should be boring and explicit:

- `observation`
- `legal_mask`
- `active_root_mask`
- `to_play`
- `env_row`
- `player`
- `policy_env_id`
- `target_reward`
- `done_root`
- terminal/final/autoreset sidecars
- metadata dict with schema id, renderer surface, search lane, RND mode, seed,
  and model/checkpoint identity when available.

`CompactSearchResultV1`

Purpose: checked output from active roots only.

Fields:

- `root_index`
- `env_row`
- `player`
- `policy_env_id`
- `selected_action`
- `visit_policy`
- `raw_visit_counts` if available, optional at first
- `root_value`
- metadata dict with search impl, simulations, temperature/noise settings,
  model eval counts, fallback count, illegal-action count.

`CompactReplayProofChunkV1`

Purpose: profile-only proof chunk, not a trainer default.

Fields:

- time-major observation/action-mask/reward/done/final-observation arrays;
- selected action, visit policy, root value for searched live seats;
- row/player ids and `to_play`;
- optional RND latest-frame/bonus arrays if RND is enabled in the profile.

This can stay in-memory first. Do not start with a storage format unless the
profile needs it.

### Functions To Add

`build_compact_root_batch_v1(batch: HybridCompactBatch, *, metadata: Mapping) -> CompactRootBatchV1`

This should be mostly validation and reshaping. It should fail closed on:

- non-binary masks;
- wrong row/player order;
- done roots marked active;
- active roots with no legal action;
- unsupported `to_play`;
- missing final observation sidecar when a terminal row needs it.

`run_compact_search_service_v1(root_batch: CompactRootBatchV1, service: Protocol) -> CompactSearchResultV1`

The first service should wrap the existing `direct_ctree_gpu_latent` path. That
keeps this slice about the boundary, not about inventing search.

`build_compact_replay_proof_chunk_v1(records: Sequence[...]) -> CompactReplayProofChunkV1`

This can begin with a two-record fixture:

- record `k` search consumes observation `k`;
- selected action at `k` must match joint action at `k+1`;
- reward/done/next observation come from `k+1`;
- terminal next observation uses `final_observation`, not the autoreset state.

`build_target_rows_from_compact_service_v1(chunk, search_result, *, record_index)`

This can delegate to the existing `build_compact_target_rows_from_search_arrays_v0`
at first. The point is to pin the contract shape before replacing the remaining
row-loop/dict internals.

## Tests To Add First

Extend `tests/test_compact_search_replay_contract.py`; do not create a huge new
test tree yet.

P0 tests:

- `test_compact_root_batch_v1_from_hybrid_batch_validates_active_roots`
- `test_compact_root_batch_v1_rejects_done_active_root`
- `test_compact_root_batch_v1_rejects_illegal_or_fractional_mask`
- `test_compact_search_result_v1_rejects_illegal_selected_action`
- `test_compact_search_result_v1_rejects_illegal_visit_mass`
- `test_compact_service_target_rows_match_object_bridge_two_record_terminal`
- `test_compact_service_target_rows_match_object_bridge_non_prefix_roots`
- `test_compact_service_preserves_policy_env_id_with_non_identity_ids`
- `test_compact_service_uses_final_observation_before_autoreset`
- `test_compact_service_rnd_latest_frame_matches_compact_observation`

One important extra: add a player-perspective sentinel. The current tests check
row/player mapping, but the future service also needs to prove it did not swap
player observations or masks.

## What Must Stay Profile-Only

These pieces should explicitly report `profile_only=True` and should not be
wired into Coach training defaults:

- `CompactRootBatchV1`
- `CompactSearchResultV1`
- `CompactReplayProofChunkV1`
- the direct CTree service wrapper;
- any precomputed recurrent-output mode;
- any compact replay sampler until target parity and full-loop profile gates
  pass.

This lane is a proof harness. It should not call `train_muzero`, mutate Coach
launcher defaults, change tournament defaults, or touch live Modal runs.

## Correctness Risks That Matter Most

1. Observation/action timing.

Search at record `k` must consume observation `k`; the selected action must be
the action that produced record `k+1`; reward/done/next observation must come
from record `k+1`.

2. Terminal/autoreset confusion.

If a row dies and autoresets, the learner target must see the final observation
from the dead row, not the fresh reset observation.

3. Player perspective swaps.

The compact service must preserve which player owns each observation, mask,
reward, and selected action. Shape checks will not catch a player swap.

4. Legal mask semantics.

Masks must be binary. Selected actions must be legal. Visit policy must assign
zero mass to illegal actions and sum to one over legal actions.

5. Fixed-opponent lane assumptions.

The current compact bridge requires `to_play=-1`. Do not silently generalize
this to true two-seat self-play until there is a separate contract.

6. Row id identity assumptions.

Tests should include non-identity `policy_env_id` values. The contract should
not accidentally depend on `policy_env_id == compact_root_index`.

7. RND coupling.

RND should read the same latest frame that the policy sees. It should remain an
independent sidecar, not baked into search correctness.

8. False speed wins.

Skipping `BaseEnvTimestep` or `PolicyRowRecordV0` in a profile is only useful if
the output still proves the same target rows. Every speed row should say
whether it is compact-only, target-row parity checked, or learner-compatible.

## Recommended Implementation Order

1. Add `CompactRootBatchV1`, `CompactSearchResultV1`, and validators.
2. Add root/result tests with synthetic fixtures only.
3. Add the service wrapper around current direct CTree search.
4. Add a two-record compact replay proof chunk.
5. Add target-row parity tests against the existing object path.
6. Only then run a profile row that compares:

```text
current direct_ctree_gpu_latent profile
vs
compact root/result/target-row proof profile
```

Kill this lane early if the closed compact proof cannot show a plausible
3x-class boundary improvement in a fair profile denominator. Keep it if it
proves correctness and clearly removes the scalar/object fanout that direct
CTree alone could not remove.
