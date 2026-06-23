# RND + Player-Perspective Promotion Gate

Date: 2026-05-23

Status: bounded code/doc audit. No source code, Modal state, live Coach runs, or
volumes touched.

## Verdict

Do not make the compact/fixed-shape search service trainer-facing yet.

The compact replay/search contracts are close, and the RND adapter has useful
unit coverage. The missing gate is the combined trainer-facing proof:

```text
closed compact batch -> real compact search arrays -> selected actions drive env
-> CompactReplayIndexRowsV1 -> RND input/reward-model path -> learner-facing row
identity
```

with player-perspective sentinels preserved across both players and terminal
final-observation-before-autoreset rows.

## Existing Coverage

- `CompactSearchServiceV1` is only a small protocol plus
  `compact_search_result_v1_from_arrays(...)`; it validates arrays but does not
  own trainer integration.
- `CompactRootBatchV1` validates row-major `env_row/player/policy_env_id`,
  `to_play=-1`, active roots, masks, target reward, terminal/final-observation
  sidecars.
- `CompactSearchResultV1` validates selected-action legality, visit-policy
  normalization/illegal mass, optional counts/logits/value payloads.
- `CompactReplayIndexRowsV1` proves the hot-path replay shape can avoid
  observation/next-observation materialization and can later materialize the
  same target rows as the object path.
- Existing compact tests cover two-record terminal rows, three-record
  `record_index=1`, non-prefix active roots `[1, 3]`, non-identity
  `policy_env_id`, action-feedback, and a compact RND latest-frame/order check.
- Existing hybrid tests prove compact batches can feed RND latest frames with
  `materialize_scalar_timestep=False`, and prove compact replay action feedback
  with index rows.
- Existing RND tests cover config normalization, reward-model entrypoint
  patches, latest-channel extraction, uint8 normalization, stale channel
  rejection, predictor training, frozen target, zero-weight target neutrality,
  positive-weight bounded mutation, seed/cadence metrics.
- Existing renderer/hybrid tests cover row-major player order and terminal
  final-observation scalarization.

## Remaining Gaps

- The RND proof is split from the compact replay proof. There is no single test
  where `CompactReplayIndexRowsV1` and RND latest-frame extraction/reward-model
  behavior are driven by the same closed compact search step.
- The real boundary tests still emphasize `CompactReplayChunkV1` materialized
  target rows or service-result validation. The trainer hot path needs an
  index-row proof from real compact search arrays, not only fake hybrid probes.
- Player-perspective rejection exists in target-row construction, but the
  service/index-row gate should explicitly reject swapped `search_result.player`
  and swapped compact observations on the index-row path.
- Terminal RND alignment is not proven in the combined path: RND must not read a
  post-autoreset frame when the learner/replay target for the same transition
  uses terminal `final_observation`.
- RND meter mode is safe-looking, but positive RND remains a separate learning
  gate. Trainer-facing promotion should either be no-RND/`rnd_meter_v0` only, or
  add a positive-RND normalization/objective proof.
- The compact lane is fixed-opponent only today: `to_play=-1`. Current-policy
  two-seat/self-play promotion needs a separate `to_play` contract.

## Smallest Tests To Add

1. `tests/test_compact_search_replay_contract.py`
   Add an index-row fail-closed test that mutates a valid
   `CompactSearchResultV1` by swapping `player` while leaving shapes plausible.
   Assert `build_compact_replay_index_rows_v1_from_search_result(...)` rejects
   it. In the same fixture, swap compact observations between players and assert
   the index/materialization path rejects or fails parity.

2. `tests/test_source_state_hybrid_observation_profile.py`
   Add one closed compact proof test with `materialize_scalar_timestep=False`,
   `stack_storage_dtype="uint8"`, terminal/autoreset present, and a probe that
   emits compact search arrays and extracts RND latest frames from the same
   `HybridCompactBatch`. Assert action feedback, index-row target count,
   `compact_root_row -> env_row/player`, and RND latest-frame sentinels all
   match.

3. `tests/test_source_state_batched_observation_boundary_profile.py`
   Add one real boundary test for direct/service-tax compact arrays that builds
   `CompactReplayIndexRowsV1` from the boundary output and replay chunk. Assert
   both players, non-identity `policy_env_id`, selected actions, rewards, done,
   and final-observation markers survive without materializing observations.

4. `tests/test_exploration_bonus.py` or the compact replay contract test
   Feed compact-derived RND latest frames/segments into `CurvyRNDRewardModel` in
   meter mode. Assert predictor changes, target hash stays fixed, target reward
   is unchanged, and the consumed latest-frame order matches
   `(env_row, player)`.

## Promotion Rule

Compact/fixed-shape search can become trainer-facing only after the above pass
on the same closed compact denominator as the speed rows. Until then, keep it
profile-only and keep Coach on stock LightZero/frozen opponent.
