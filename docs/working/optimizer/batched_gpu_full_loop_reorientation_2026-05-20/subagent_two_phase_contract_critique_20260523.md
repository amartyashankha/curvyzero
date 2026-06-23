# Two-Phase Compact Search Contract Critique - 2026-05-23

Scope: critique of the profile-only two-phase compact search contract after
reading:

- `src/curvyzero/training/compact_search_service.py`
- `tests/test_compact_search_replay_contract.py`
- validation ladder docs in this optimizer folder, especially
  `subagent_validation_ladder_wave2_20260523.md` and
  `subagent_optimizer_validation_gate_audit_20260523.md`

I did not touch live Coach training runs. This note is not launch advice.

## Short Read

The new `CompactSearchActionStepV1` / `CompactSearchReplayPayloadV1` split is a
good first contract slice. `validate_compact_search_two_phase_payload_v1(...)`
checks that delayed replay payloads still match the action half by handle,
`root_index`, `env_row`, `player`, and `policy_env_id`.
`CompactSearchPayloadGateV1` also gives a small profile-only visibility gate:
registered action steps are not sample-visible until a checked payload arrives.

What is still missing is the hard part of a real delayed replay/search service:
service-result reordering at row level, replay-owner/sample integration,
terminal/autoreset buffer immutability, and proof that the env-action path never
waits on replay-only fields.

## What Can Still Go Wrong

### 1. Out-of-order service results are only handle-gated

`CompactSearchPayloadGateV1.attach_replay_payload(...)` allows batch B's payload
to arrive before batch A's payload by handle. That is useful, but it does not
prove row-level out-of-order service behavior. Inside a payload,
`validate_compact_search_two_phase_payload_v1(...)` requires identical array
order. Separately, `build_compact_replay_index_rows_v1_from_search_result(...)`
requires `search_result.root_index == flatnonzero(root_batch.active_root_mask)`.

That means a future service has only two safe choices:

- canonicalize every result back to root-batch active order before building
  `CompactSearchResultV1`; or
- introduce an attach/reorder function that joins by stable identity and fails
  on duplicate/missing rows.

Right now that rule is implicit. A real service could batch roots internally,
return payload rows in service order, and pass the handle-level gate only to fail
late or, worse, tempt a caller to attach arrays by position elsewhere.

Missing tests:

- `tests/test_compact_search_replay_contract.py::test_two_phase_payload_reorders_or_rejects_shuffled_rows_before_visibility`
  - Build an `action_step` over active roots `[0, 1, 2, 3]`.
  - Build a `replay_payload` with the same handle but rows permuted.
  - Assert the contract either reorders by `(root_index, env_row, player, policy_env_id)` before visibility, or rejects before any sampler-visible state.
- `tests/test_compact_search_replay_contract.py::test_two_phase_payload_rejects_duplicate_or_missing_identity_rows`
  - Same handle, one duplicated `policy_env_id`, one missing root.
  - Must fail before `CompactSearchPayloadGateV1.complete_count` changes.
- `tests/test_source_state_batched_observation_boundary_profile.py::test_compact_service_two_outstanding_batches_b_returns_before_a_applies_correct_actions`
  - Two outstanding compact root batches.
  - Payload B arrives first.
  - Env actions for A and B are still applied from their own action steps, and replay rows attach to the right root ids.

### 2. Incomplete-row visibility is not wired to replay/sample materialization

`CompactSearchPayloadGateV1.is_sample_visible(...)` proves a local dictionary
gate, not a replay owner. `build_compact_replay_index_rows_v1_from_search_result(...)`
still takes a complete `CompactSearchResultV1` with visit policy and root value.
`materialize_compact_target_rows_from_index_rows_v1(...)` takes completed index
rows and a chunk; it does not know about payload handles or the gate.

So the current test `test_compact_search_payload_gate_hides_rows_until_payload_arrives`
does not yet prove that a learner sampler, LightZero buffer, or future compact
replay ring cannot see a partially written row. It only proves that callers who
voluntarily call `require_replay_payload(...)` get a fail-closed check.

Missing tests:

- `tests/test_compact_search_replay_contract.py::test_pending_action_step_cannot_materialize_index_rows_or_sample_batch`
  - Register an action step.
  - Attempt the future replay-owner/sample path before `attach_replay_payload`.
  - Assert no target row or `build_source_state_multiplayer_sample_batch_v0(...)`
    output can be produced.
- `tests/test_compact_search_replay_contract.py::test_replay_payload_gate_is_required_at_materialization_edge`
  - Attach payload through `CompactSearchPayloadGateV1`.
  - Only then allow `materialize_compact_target_rows_from_index_rows_v1(...)`
    or the future compact replay owner to expose rows.
- Future owner test in the file that introduces the owner:
  `CompactReplayWriter/CompactReplayRing::test_partial_rows_are_hidden_from_sampler_until_action_reward_done_policy_value_final_obs_and_rnd_are_complete`.

### 3. Terminal final observation and RND are still split proofs

The compact replay tests are strong for synthetic target rows:
`test_two_record_compact_rows_use_final_observation_before_autoreset_and_rnd_latest`,
`test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows`, and
`test_compact_search_service_index_rows_feed_rnd_model_and_terminal_final_obs`.

The remaining risk is a delayed service with a resident/persistent buffer. The
current tests prove terminal `next_observation` and current-root RND latest-frame
behavior, but they do not prove that a delayed replay/RND consumer cannot read a
post-autoreset buffer after the terminal frame has been overwritten. The real
direct CTree and compact Torch closed-loop tests also do not force a mixed
terminal/live batch where one row terminals/autoresets while another row remains
live.

Missing tests:

- `tests/test_source_state_batched_observation_boundary_profile.py::test_real_direct_ctree_compact_service_mixed_terminal_live_final_obs_and_rnd_latest`
  - Direct CTree compact service.
  - One env row terminal/autoreset, one env row live.
  - Terminal row is not searched as live; live row still is searched.
  - Materialized terminal `next_observation` equals `final_observation`, not reset state.
  - RND latest-frame proof is recorded for the same row/player ids.
- `tests/test_source_state_batched_observation_boundary_profile.py::test_compact_torch_search_service_mixed_terminal_live_final_obs_and_rnd_latest`
  - Same shape for `CompactTorchSearchServiceV1`.
- `tests/test_compact_search_replay_contract.py::test_terminal_final_observation_is_immutable_after_autoreset_buffer_mutation`
  - Build terminal index rows.
  - Mutate the source/current observation buffer to reset-looking sentinels.
  - `materialize_compact_target_rows_from_index_rows_v1(...)` must still produce the captured final frame.
- `tests/test_compact_search_replay_contract.py::test_inactive_terminal_root_poison_never_reaches_replay_rnd_or_sample`
  - Put absurd selected actions, visit policy, root value, and observation sentinels on inactive/terminal roots.
  - Assert replay rows, RND inputs, and sample rows contain only active roots.

### 4. Action-critical sync is separated in data shape, not yet in execution

The dataclasses separate action-critical fields from replay-critical fields:
`CompactSearchActionStepV1` does not carry `visit_policy`, `root_value`, raw
visits, predicted values, or logits. That is the right API direction.

But the constructors still split an already complete `CompactSearchResultV1`.
The current profile proof in `source_state_hybrid_observation_profile.py`
also uses `_latest_compact_search_arrays_from_probe(...)`, which copies
`selected_action`, `visit_policy`, and `root_value` together before
`_joint_action_from_compact_search_arrays(...)` stages env actions. So the
contract does not yet prove that selected actions can return first while replay
payloads flush later. It proves that once a full result exists, the full result
can be partitioned safely.

Missing tests:

- `tests/test_compact_search_replay_contract.py::test_action_step_creation_and_env_action_path_do_not_require_replay_arrays`
  - Build/stage env actions from `CompactSearchActionStepV1` only.
  - No `visit_policy`, `root_value`, `raw_visit_counts`, `predicted_value`, or logits may be read.
- `tests/test_source_state_batched_observation_boundary_profile.py::test_compact_service_replay_proof_uses_action_step_not_full_search_arrays_for_next_action`
  - Replace `_compact_service_replay_proof_next_state(...)` usage with an action-step path in the profile-only proof.
  - Assert the next env `joint_action` checksum equals the action step checksum while replay payload bytes are not required on that path.
- `tests/test_curvytron_hybrid_observation_profile_manifest_runner.py::test_compact_service_summary_splits_action_readback_bytes_from_replay_payload_bytes`
  - Summary must report selected-action readback separately from replay payload readback.
  - A speed row cannot call the critical path action-only if it synchronized visit policy/root value first.

### 5. Metadata is not promotion-grade yet

The gate keys only by `replay_payload_handle`, and metadata is optional. A real
service needs enough identity to reject stale payloads after handle reuse,
policy-version drift, root-noise/tie-mode mismatch, or perspective mismatch.

Missing metadata assertions:

- `root_batch_id` or `request_id`
- `record_index` and expected `next_record_index`
- service sequence number / producer id
- policy version or model digest
- perspective contract id and controlled-player digest
- seed, root noise, temperature, epsilon, and tie classification
- action checksum and applied joint-action checksum
- replay payload completeness state

Missing tests:

- `tests/test_compact_search_replay_contract.py::test_replay_payload_handle_reuse_requires_new_epoch_or_request_id`
- `tests/test_compact_search_replay_contract.py::test_two_phase_payload_rejects_policy_version_or_perspective_mismatch`
- `tests/test_lightzero_phase_profiler.py::test_candidate_summary_requires_action_replay_rnd_digests_and_zero_fallback`

## Minimum Next Gate

Before this supports a real delayed replay/search service, add one profile-only
owner-level test that runs the whole chain:

```text
record k compact roots
-> action step only drives env action for record k+1
-> replay payload arrives later and possibly out of order
-> incomplete rows stay sampler-invisible
-> terminal/live rows materialize with final observations
-> RND/sample rows match the trusted immediate path
```

Until that exists, keep this contract profile-only. It is a good identity and
visibility primitive, but it is not yet a delayed replay service.
