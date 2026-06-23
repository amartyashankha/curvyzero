# Validation Ladder Wave 2 - 2026-05-23

Scope: validation/test critique for future device-resident and search-service
optimizer work. This is not Coach launch advice. The point is to enable
aggressive speed work without silently training a different algorithm.

## Promotion Invariant

Every candidate must prove this exact chain:

```text
policy observation at record k
-> CompactRootBatchV1 with stable root ids
-> search result: selected_action, visit policy, root value
-> selected_action applied as env joint_action for record k+1
-> replay index row with the same ids
-> materialized target row and learner-visible sample
-> optional RND latest-frame/reward-model edge
```

The same `env_row`, `player`, `policy_env_id`, controlled-player perspective,
legal mask, terminal/final-observation rule, reward rule, root-noise state, and
sample visibility state must survive the whole chain.

## Short Read

Current coverage is stronger than the old fear case:

- Compact replay/index/RND tests are locally strong.
- Real direct LightZero CTree has a compact-service closed-loop proof where
  search-selected actions drive the next hybrid env step and replay/sample rows
  match the trusted immediate path.
- Compact Torch search service now has the same local action-to-env and replay
  row proof, but it is still profile-only and not LightZero CTree semantics.
- Service-tax and mock-search lanes store compact arrays and compact replay
  proofs, but they are ceiling/falsifier modes. They must stay labelled
  `profile_only` and `not_mcts` or `not_lightzero_ctree`.

The biggest missing pieces are now promotion-grade, not unit-contract-grade:
out-of-order service attachment, incomplete-row sample visibility, mixed
terminal/live resident buffers, full-loop RND metrics, noisy statistical parity,
and matched stock-vs-candidate full-loop smokes.

## Ladder

### L0. Mode Labels And Promotion Lock

What this protects:

- A profile-only service-tax/mock/MCTX/compact-Torch row getting summarized as
  train-facing LightZero MuZero.

Existing tests:

- `tests/test_compact_torch_search_service.py`
  - `test_compile_eligibility_reports_profile_only_not_lightzero_ctree_labels`
- `tests/test_mctx_synthetic_benchmark_legality.py`
  - `test_mctx_compact_visual_service_profile_row_is_labeled_profile_only`
- `tests/test_source_state_batched_observation_boundary_profile.py`
  - `test_validate_boundary_config_accepts_mock_search_service_ceiling_contract`
  - `test_validate_boundary_config_accepts_service_tax_compact_replay_contract`
  - `test_validate_boundary_config_accepts_compact_torch_service_compact_replay_contract`
  - `test_lightzero_array_ceiling_mock_search_service_is_named_ceiling`
  - `test_lightzero_array_ceiling_service_tax_probe_stores_compact_search_arrays`
  - `test_array_ceiling_compact_torch_search_service_mode_owns_compact_service_run`
  - `test_array_ceiling_compact_torch_search_service_rejects_mislabeled_input_modes`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py`
  - compact replay rows for `compact_torch_search_service`, `service_tax_probe`,
    and `mock_search_service`
  - rejection of mislabeled compact Torch input modes

Missing:

- A single `promotion_eligible=false` result gate for every profile-only row.
- A result summarizer hard fail if required fields are absent:
  backend kind, profile/training label, `not_lightzero_ctree`, fallback count,
  action-feedback proof, replay proof, RND mode, and root-noise metadata.

### L1. Root Identity, Player, Perspective, And Legality

What this protects:

- Search/replay attaching by array position after compaction or service
  batching.
- Player 0 observation paired with player 1 action/reward.
- Illegal selected actions or visit mass leaking through.

Existing tests:

- `tests/test_compact_search_replay_contract.py`
  - `test_compact_service_v1_round_trips_target_rows_and_identity_sidecars`
  - `test_compact_search_result_v1_rejects_identity_and_legality_errors`
  - `test_three_record_compact_rows_map_non_prefix_active_roots_to_compacted_policy_rows`
  - `test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows`
  - `test_compact_search_service_index_rows_preserve_rnd_and_player_perspective`
  - `test_compact_root_batch_can_keep_observation_view_for_profile_hot_path`
- `tests/test_compact_torch_search_service.py`
  - `test_compact_torch_search_service_preserves_active_root_order_and_legal_masks`
  - `test_fixed_shape_masks_reject_non_binary_masks`
  - `test_inactive_roots_do_not_make_all_actions_illegal_unless_forced_active`
- `tests/test_multiplayer_source_state_target_rows.py`
  - `test_compact_target_rows_reject_swapped_player_perspective_batch`
  - `test_compact_search_policy_records_keep_active_root_order_for_p4_and_mixed_done`
  - `test_compact_search_policy_records_reject_bad_sidecars_before_replay`
- `tests/test_source_state_batched_observation_profile_cpu.py`
  - controlled-row and both-player row-major perspective tests
  - palette tracks avatar color ids, not player indices
- `tests/test_source_state_batched_observation_boundary_profile.py`
  - `test_lightzero_compact_search_service_adapter_preserves_root_identity`
  - `test_lightzero_array_ceiling_compact_search_service_adapter_preserves_identity`
  - `test_lightzero_collect_forward_probe_rejects_fractional_action_masks`
  - `test_lightzero_mcts_arrays_boundary_real_policy_cpu_single_legal_action_exact`
  - `test_lightzero_mcts_arrays_boundary_real_policy_cpu_biased_logits_respect_masks`
- `tests/test_mctx_synthetic_benchmark_legality.py`
  - legality summary and row-major root-order guards

Missing:

- Out-of-order service-result test: shuffle/cohort roots internally, return
  results in different order, and prove replay attaches by stable ids.
- Inactive-root poison test: inactive roots contain absurd values/actions and
  never reach replay, RND, or samples.
- Promotion metadata must carry a perspective contract id and both-player
  digest through root batch, search result, replay rows, and sample rows.

### L2. Selected Action Applied To Env

What this protects:

- The policy learning from a transition caused by a different action than the
  one search selected.

Existing tests:

- `tests/test_policy_row_mapping.py`
  - `test_selected_policy_actions_map_back_to_joint_action_with_noop_padding`
  - `test_selected_policy_actions_can_be_compact_for_padded_mapping`
  - `test_selected_policy_actions_reject_illegal_active_action`
- `tests/test_source_state_batched_observation_mock_collector.py`
  - `test_scalar_action_bridge_commits_one_joint_action_from_scalar_actions`
  - `test_scalar_action_bridge_allows_complete_row_omission_but_rejects_partial_missing_extra_and_invalid_actions`
  - `test_profile_env_manager_surface_tracks_ready_obs_and_step_results`
- `tests/test_source_state_batched_observation_boundary_profile.py`
  - `test_real_direct_ctree_compact_service_drives_next_step_and_matches_rows`
  - `test_compact_torch_search_service_drives_next_step_and_matches_rows`
- `tests/test_compact_search_replay_contract.py`
  - `test_compact_replay_index_rows_skip_observation_materialization`
  - `test_deferred_search_payload_rows_match_immediate_rows_for_non_prefix_roots`
  - selected-action mismatch rejection against `next_joint_action`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`
  - `test_source_state_visual_survival_step_and_terminal_telemetry`
  - `test_source_state_visual_survival_action_override_is_explicit_and_seeded`
  - `test_source_state_visual_survival_action_repeat_is_one_policy_transition`

Missing:

- Mandatory remote-profile checksum:
  `selected_action_checksum == applied_joint_action_checksum` for every
  candidate row that claims closed-loop replay proof.
- Queue/stale-action test with two outstanding search batches where batch B
  returns before batch A.
- Service-tax/mock must not be allowed to pass this gate as trainer-facing
  semantics. They can pass it only as labelled ceiling modes.

### L3. Terminal Final Observation And Autoreset

What this protects:

- A terminal transition storing a reset frame as `next_observation`.
- Terminal roots being searched as if live.
- Resident buffers mutating final frames before replay/RND consumes them.

Existing tests:

- `tests/test_2p_product_path_fidelity.py`
  - `test_2p_vector_multiplayer_product_path_visual_bonus_terminal_and_replay`
- `tests/test_vector_env_replay_recorder.py`
  - `test_vector_env_replay_recorder_uses_batch_info_for_terminal_autoreset_rows`
  - `test_vector_env_replay_recorder_packs_same_frame_wall_draw_terminal_rows`
  - terminal-row close/identity metadata tests
- `tests/test_multiplayer_source_state_trainer_replay.py`
  - `test_records_reset_step_terminal_arrays_and_preserves_final_visual_observation`
  - `test_variable_policy_rows_are_stored_per_record_with_env_player_maps`
- `tests/test_multiplayer_source_state_target_rows.py`
  - `test_compact_search_arrays_use_terminal_final_observation_without_records`
  - `test_target_rows_use_terminal_final_observation_and_final_reward_map`
- `tests/test_compact_search_replay_contract.py`
  - `test_two_record_compact_rows_use_final_observation_before_autoreset_and_rnd_latest`
  - `test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows`
  - `test_compact_search_service_index_rows_feed_rnd_model_and_terminal_final_obs`
- `tests/test_source_state_batched_observation_mock_collector.py`
  - `test_profile_env_manager_keeps_terminal_timestep_before_autoreset`
  - `test_surface_policy_materializer_attaches_terminal_final_observation`
  - `test_surface_policy_materializer_requires_terminal_final_observation`
  - `test_scalar_materializer_attaches_terminal_final_observation`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py`
  - `test_source_state_visual_survival_terminal_snapshots_survive_manual_reset`
  - `test_source_state_visual_survival_step_and_terminal_telemetry`
- `tests/test_source_state_batched_observation_profile_cpu.py`
  - `test_terminal_step_captures_final_stack_before_any_reset`
  - `test_both_players_terminal_info_exposes_stacked_final_observation_contract`

Missing:

- Real compact-service mixed terminal/live test for both direct CTree and
  compact Torch: one env row terminal/autoreset, one row live; terminal row not
  searched; live row still searched; terminal next obs equals final obs.
- Resident-buffer final-observation copy test: mutate the live/persistent
  buffer after terminal capture and prove replay/RND final rows do not change.

### L4. RND Latest Frame And Reward Model

What this protects:

- RND reading a stale stack channel, wrong player, or reset frame.
- Meter-only RND changing target rewards.
- Positive RND being smuggled in as "same algorithm" validation.

Existing tests:

- `tests/test_exploration_bonus.py`
  - RND config fail-closed tests
  - `test_rnd_meter_spec_selects_reward_model_entrypoint_and_metadata`
  - `test_lightzero_rnd_meter_patch_is_atomic_and_weight_zero`
  - latest gray64 extractor tests for LightZero and compact shapes
  - `test_compact_policy_gray64_latest_adapter_ignores_stale_stack_channels`
  - `test_curvy_rnd_reward_model_trains_predictor_freezes_target_and_preserves_zero_weight_reward`
  - seed/cadence/metrics tests
- `tests/test_compact_search_replay_contract.py`
  - `test_compact_search_service_index_rows_preserve_rnd_and_player_perspective`
  - `test_compact_search_service_index_rows_feed_rnd_model_and_terminal_final_obs`
- `tests/test_multiplayer_source_state_target_rows.py`
  - `test_compact_target_rows_preserve_rnd_latest_frame_order_without_records`
- `tests/test_source_state_batched_observation_profile_cpu.py`
  - `test_batched_controlled_rows_stack_feeds_rnd_latest_gray64_input`
  - `test_batched_both_player_stacks_can_materialize_rnd_inputs_per_player_view`
- `tests/test_source_state_batched_observation_mock_collector.py`
  - `test_mock_collector_profile_runs_rnd_latest_frame_meter`
  - `test_mock_collector_profile_keeps_rnd_reward_meter_non_mutating`

Missing:

- Full-loop `rnd_meter_v0` smoke with `require_rnd_metrics=true`: must prove
  reward-model entrypoint, `collect_data`, `train_with_data`, `estimate`,
  `train_cnt_rnd > 0`, predictor hash changed, target hash unchanged, and
  target reward unchanged.
- Same RND latest-frame proof on a real compact-service terminal/autoreset row.
- Positive `rnd_replay_target_v0` needs an explicit "objective changed" label
  in any speed/training summary.

### L5. Replay Row And Sample Visibility

What this protects:

- Learner sampling a row before action, reward, done, policy target, root value,
  final-observation masks, or RND sidecars are complete.
- Compact/index rows materializing to different learner targets than the
  trusted builder.

Existing tests:

- `tests/test_compact_search_replay_contract.py`
  - `test_compact_replay_index_rows_skip_observation_materialization`
  - `test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows`
  - `test_compact_index_rows_materialized_sample_batch_matches_immediate_rows`
  - `test_compact_index_rows_materialized_stock_lightzero_target_hooks_match`
  - `test_compact_index_rows_materialized_stock_lightzero_public_sample_matches`
  - `test_compact_search_service_v1_protocol_runs_fake_service_to_index_rows`
- `tests/test_source_state_batched_observation_boundary_profile.py`
  - `test_real_direct_ctree_compact_service_drives_next_step_and_matches_rows`
  - `test_compact_torch_search_service_drives_next_step_and_matches_rows`
  - `test_direct_ctree_compact_output_can_feed_checked_target_rows`
- `tests/test_multiplayer_source_state_target_rows.py`
  - `test_sample_batch_is_deterministic_for_same_seed_and_tracks_row_ids`
  - `test_sample_batch_copies_arrays_instead_of_aliasing_target_rows`
  - `test_sample_batch_rejects_invalid_batch_size`
- `tests/test_multiplayer_source_state_native_bridge.py`
  - rows grouped by `(env_row, player)` and LightZero-style segment specs
- `tests/test_multiplayer_source_state_lightzero_native_bridge.py`
  - fake and optional real LightZero `GameSegment`/buffer push canaries
- `tests/test_multiplayer_source_state_trainer_replay.py`
  - replay arrays copied, variable policy rows stored per record, bad live-mask
    rows rejected

Missing:

- Incomplete-row visibility test: replay writer inserts an index row without
  visit policy/root value/final-observation/RND completion; sampler must hide or
  reject it.
- Out-of-order replay payload flush test: action is used immediately, payload
  arrives later, and attach is by stable ids rather than current compact order.
- Compact Torch closed-loop sample-batch parity should match the direct CTree
  test, not only materialized row parity.
- A remote artifact digest for root ids, replay rows, materialized samples, and
  LightZero sample rows under one fixed seed.

### L6. No-Noise Exact Parity And Noisy Statistical Parity

What this protects:

- Treating stochastic or tie-heavy differences as exact parity.
- Hidden root-noise or seed drift making two algorithms look comparable.
- Candidate search semantics diverging under legal masks, support transforms, or
  backup math.

Existing tests:

- `tests/test_source_state_batched_observation_boundary_profile.py`
  - `test_hybrid_profile_grid_can_override_lightzero_root_noise_weight`
  - `test_lightzero_mcts_arrays_boundary_real_policy_cpu_matches_stock_values_and_masks`
  - `test_lightzero_mcts_arrays_boundary_real_policy_cpu_biased_logits_match_top_actions`
  - `test_lightzero_mcts_arrays_boundary_real_policy_cpu_single_legal_action_exact`
  - `test_lightzero_mcts_arrays_boundary_precomputed_recurrent_respects_mixed_masks`
  - `test_lightzero_mcts_arrays_boundary_real_policy_cpu_biased_logits_respect_masks`
  - `test_direct_ctree_collect_search_hook_preserves_masked_raw_visit_contract`
- `tests/test_compact_torch_search_service.py`
  - `test_fixed_shape_forced_masks_and_noise_preconditions_are_deterministic`
  - `test_select_and_backup_helpers_operate_on_tiny_tensors_without_lightzero`
  - `test_compact_torch_search_service_uses_fresh_observations_for_same_shape_calls`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py`
  - root-noise override emitted into fixed-denominator profile command

Missing:

- Candidate-vs-direct no-noise parity fixture for the full
  `CompactSearchServiceV1` output: selected actions, legal visit mass, raw
  counts when available, and root values.
- Seeded noisy statistical gate: shared seed list, root noise over legal action
  slots only, confidence bands, and explicit non-exact label.
- Metadata gate for every parity claim:
  `root_noise_weight`, `root_dirichlet_alpha`, temperature, epsilon, seed,
  tie classification, and whether the fixture is exact or statistical.

### L7. Train-Facing Full-Loop Smoke

What this protects:

- A profile loop passing all local contracts but failing when used by the real
  LightZero trainer, replay buffer, learner, or RND path.

Existing tests:

- `tests/test_lightzero_phase_profiler.py`
  - train call cap and phase profiler install tests
  - direct CTree collect hook schema/fallback tests
  - compact output summary records backend, fallback counts, timers, and
    readback byte counts
- `tests/test_lightzero_config_builder.py`
  - config surface tests, including RND/exploration bonus patches indirectly
- `tests/test_compact_search_replay_contract.py`
  - optional local LightZero target hooks and public sample parity when `lzero`
    is installed

Missing:

- Matched stock-vs-candidate tiny full-loop smoke:
  same seed, same config, sidecars disabled or matched, sparse checkpoints,
  same RND mode, same death mode, and no hidden fallback.
- The smoke must emit:
  action-feedback checksum, root-id digest, perspective digest, replay digest,
  sample digest, terminal/final-observation digest, RND metrics digest when
  enabled, fallback count, illegal action count, and bytes/timers for
  observation/action/replay payload movement.
- Hard fail if a candidate claims train-facing promotion while fallback count is
  nonzero or RND metrics are absent when RND is enabled.

## Requested Boundary Matrix

| Boundary | Already Covered | Still Missing |
| --- | --- | --- |
| Selected action applied to env | `tests/test_source_state_batched_observation_boundary_profile.py` direct CTree and compact Torch closed-loop tests; `tests/test_policy_row_mapping.py`; scalar bridge tests in `tests/test_source_state_batched_observation_mock_collector.py`; mismatch rejection in `tests/test_compact_search_replay_contract.py`. | Remote action-feedback checksum; out-of-order service results; service-tax/mock must stay profile-only. |
| Root ids/player/perspective | Compact identity and non-prefix tests in `tests/test_compact_search_replay_contract.py`; perspective tests in `tests/test_source_state_batched_observation_profile_cpu.py`; adapter identity tests in `tests/test_source_state_batched_observation_boundary_profile.py`. | Attach-by-id under shuffled service returns; inactive-root poison; perspective contract digest through sample/training artifacts. |
| Terminal final observation | Product path, vector replay recorder, trainer replay, compact replay, source-state materializer, and source-state LightZero env terminal tests listed above. | Real compact-service mixed terminal/live autorestart gate; resident final-frame immutability after buffer mutation. |
| RND latest frame/reward model | `tests/test_exploration_bonus.py`; compact replay RND tests; target-row RND order tests; mock collector RND meter tests. | Full-loop RND metrics smoke; terminal/autoreset RND latest on real compact service; positive RND objective-change labeling. |
| Replay row/sample visibility | Compact index/materialization/sample tests; direct CTree real closed-loop sample parity; optional LightZero target/sample tests. | Incomplete-row hidden/rejected; out-of-order payload flush; compact Torch sample parity; remote digest over replay and samples. |
| Noisy vs no-noise parity | No-noise CPU direct/stock mask/value tests; compact Torch deterministic mask/noise preconditions; root-noise command override tests. | Full service output parity vs direct CTree; seeded noisy statistical gate; mandatory noise/seed/tie metadata. |

## Minimal New Tests To Add Next

1. `tests/test_source_state_batched_observation_boundary_profile.py`
   - `test_real_compact_service_mixed_terminal_live_final_obs_and_rnd_latest`
   - Do this for direct CTree first, then compact Torch.

2. `tests/test_compact_search_replay_contract.py`
   - `test_compact_replay_sampler_hides_incomplete_index_rows`
   - `test_compact_replay_payload_attach_by_stable_ids_after_out_of_order_flush`
   - `test_inactive_root_poison_never_reaches_replay_rnd_or_sample`

3. `tests/test_compact_torch_search_service.py`
   - `test_compact_torch_search_service_no_noise_parity_with_direct_ctree_toy`
   - Exact fixture only: single legal and clear-preference roots, no ties.

4. `tests/test_lightzero_phase_profiler.py` or a new promotion test file
   - `test_candidate_summary_requires_action_replay_rnd_digests_and_zero_fallback`

5. Remote profile artifact validator
   - Validate one JSON result has the full promotion field set before any
     speed summary can call it train-facing.

## Minimal Local Gate Command

```sh
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_compact_torch_search_service.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_profile_cpu.py \
  tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_multiplayer_source_state_target_rows.py \
  tests/test_multiplayer_source_state_trainer_replay.py \
  tests/test_exploration_bonus.py \
  tests/test_lightzero_phase_profiler.py
```

This is not sufficient for train-facing promotion. It is the local contract
ladder before remote full-loop smokes.

## Kill Conditions

Stop promotion and fix validation if any of these is true:

- selected search action is not proven to be the next env `joint_action`;
- `root_index`, `env_row`, `player`, or `policy_env_id` changes across root,
  search, replay, or sample;
- selected action is illegal, or visit/raw count mass lands on illegal actions;
- terminal next observation uses an autoreset frame;
- terminal row is searched as live;
- controlled-player perspective is chosen by optimizer code;
- RND meter changes target rewards;
- RND-enabled row lacks reward-model entrypoint and metrics proof;
- a learner can sample a partial compact row;
- root-noise state or tie classification is missing from a parity claim;
- fallback count is nonzero in a promoted row;
- service-tax/mock/MCTX/compact-Torch profile-only rows are summarized as Coach
  training semantics.

