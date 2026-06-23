# Subagent Test Coverage Map - 2026-05-21

Scope: map existing tests against the fast-wrong risks in the batched GPU
optimizer lane. I reviewed docs, tests, and source only. I did not touch live
training runs, launch Modal jobs, or edit production code.

## Plain Read

The current local coverage is strong for profile-only shape contracts:
row/player order, stack FIFO, terminal final-observation materialization,
fixed-opponent `to_play=-1`, zero-mask filtering in the LightZero
collect-forward probe, RND latest-frame extraction, profile-only labels, and
basic timing/byte labels.

The highest-risk missing proof is still not another renderer microbenchmark.
It is the stock-entrypoint context:

- mixed terminal and live rows through the stock-profile `train_muzero` manager
  adapter;
- `rnd_meter_v0` through the stock-profile reward-model entrypoint;
- compact speed rows refusing to summarize unless semantic identity fields are
  present.

Until those gates pass, current speed rows are useful optimizer evidence, not a
safe training-default recommendation.

## Source Anchors Reviewed

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
  owns `_view_major_to_row_major_frames`, `_row_major_render_rows`,
  `_row_major_render_players`, `_push_row_major_frames_into_stack`,
  `_latest_uint8_frames_from_stack`, `_persistent_delta_state`,
  `_LightZeroCollectForwardStackProbe`, `_LightZeroInitialInferenceStackProbe`,
  `_policy_output_row_from_plain`, `_extract_eval_action_from_plain`, and
  `_compact_hybrid_observation_profile_result`.
- `src/curvyzero/training/source_state_batched_observation_mock_collector.py`
  owns `BatchedLightZeroScalarActionBridge`,
  `BatchedLightZeroProfileEnvManager`,
  `BatchedLightZeroStockEnvManagerAdapter`,
  `materialize_lightzero_scalar_timestep`, and
  `materialize_trainer_surface_policy_timestep`.
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py` owns
  `SourceStateMultiplayerTrainerSurface` and the renderer-backed profile stack.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py` owns the
  pre-scalar hybrid manager and optional batched-stack consumer handoff.
- `src/curvyzero/training/exploration_bonus.py` owns `rnd_meter_v0`,
  `policy_gray64_latest/v0`, `CurvyRNDRewardModel`, predictor/target hashes,
  and meter-mode reward neutrality.
- `src/curvyzero/training/lightzero_config_builder.py` owns source-state
  LightZero config surfaces and RND bundle patching.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  owns stock `train_muzero`/`train_muzero_with_reward_model` launch plumbing and
  the phase profiler.

## Existing Coverage By Risk

### Row/Player Order And Perspective

Already covered:

- `tests/test_source_state_batched_observation_boundary_profile.py::test_view_major_to_row_major_frames_interleaves_player_views_by_row`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_row_major_render_index_helpers_match_boundary_order`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_dynamic_renderer_accepts_requested_player_order`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_compact_hybrid_observation_profile_result_keeps_mapping_edges`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_both_players_view_updates_stacks_in_row_major_render_order`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_both_players_cpu_oracle_matches_direct_per_player_renders`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_facade_accepts_explicit_profile_gpu_candidate_renderer_without_cpu_fallback`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_reset_emits_source_state_visual_surface_for_supported_player_counts`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_step_maps_live_policy_rows_for_multiplayer_player_counts`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_step_preserves_player_major_joint_action_mask_and_survival_bonus_reward`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_renderer_backed_gpu_candidate_preserves_row_player_order`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_manager_exposes_row_player_scalar_ids`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_renderer_backed_mode_preserves_row_major_player_order`
- `tests/test_source_state_batched_observation_mock_collector.py::test_batched_surface_profile_loop_materializes_policy_rows_in_surface_order`
- `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_action_bridge_exposes_lightzero_env_ids_after_reset`
- `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_action_bridge_commits_one_joint_action_from_scalar_actions`
- `tests/test_policy_row_mapping.py::test_policy_row_mapping_compacts_live_rows_with_legal_actions_and_ids`
- `tests/test_policy_row_mapping.py::test_selected_policy_actions_map_back_to_joint_action_with_noop_padding`
- `tests/test_policy_row_mapping.py::test_empty_policy_mapping_round_trips_to_all_noop_joint_action`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_both_view_config_preserves_two_controlled_view_cpu_oracle_order`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_adversarial_fixture_cpu_oracle_renders_both_controlled_views`
- `tests/test_vector_multiplayer_observation.py::test_4p_public_state_packs_present_alive_ego_rows_with_masks_and_ids`
- `tests/test_vector_multiplayer_observation.py::test_3p_absent_dead_and_terminal_slots_are_padded_without_legal_actions`
- `tests/test_source_state_visual_survival_learner_seat_regression.py::test_fixed_learner_seat_1_resets_and_legacy_ego_config_rejects`
- `tests/test_source_state_visual_survival_learner_seat_regression.py::test_fixed_learner_seats_match_selected_controlled_player_renderer`
- `tests/test_source_state_visual_survival_learner_seat_regression.py::test_fixed_learner_seat_1_routes_learner_action_to_player_1_and_reports_control`
- `tests/test_lightzero_config_builder.py::test_visual_survival_surface_pins_policy_observation_perspective_contract`

What this proves:

- Row-major policy order is pinned as `(row0,player0), (row0,player1), ...`.
- Renderer requests, profile timesteps, scalar env ids, and policy row metadata
  agree locally.
- Player perspective is not just metadata; several sentinel tests assert the
  actual policy stack row matches the requested player view.

Residual risk:

- No stock `train_muzero` batched-profile test currently proves that row/player
  mapping survives a mixed terminal/live collection batch where terminal roots
  are filtered but terminal timesteps still need delivery.

### GPU/CPU Observation Parity And Surface Identity

Already covered:

- `tests/test_multiplayer_source_state_trainer_surface.py::test_renderer_backed_cpu_oracle_matches_dirty_cache_surface`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_renderer_backed_profile_requires_explicit_renderer`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_renderer_backed_profile_can_require_exact_renderer_backend`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_fast_gray64_direct_is_rejected_for_trainer_surface`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_body_circles_fast_is_rejected_by_current_trainer_surface`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_accepts_direct_gray64_surface_canary`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_accepts_direct_gray64_hybrid_canary`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_accepts_persistent_gpu_profile_backend`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_rejects_persistent_gpu_profile_without_canary`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_rejects_persistent_gpu_profile_block_surface`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_rejects_direct_gray64_without_surface_canary`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_rejects_direct_gray64_cpu_dirty_surface`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_tolerant_parity_accepts_tiny_uint8_and_stack_drift`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_tolerant_parity_rejects_large_uint8_drift`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_exact_parity_rejects_tiny_drift`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_production_to_benchmark_masks_visual_trail_active_slots_past_cursor`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_cpu_reference_palette_respects_non_identity_avatar_color`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_production_to_benchmark_preserves_avatar_color_for_gpu_palette`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_cpu_previous_owner_trail_slots_connects_interleaved_same_owner_slots`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_owner_ordered_compact_adversarial_fixture_matches_cpu_oracle_without_gpu`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_adversarial_fixture_jax_parity_when_jax_is_available`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_adversarial_fixture_jax_two_view_parity_when_jax_is_available`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_jax_two_view_parity_when_jax_is_available`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_simple_symbols_keep_all_twelve_bonus_identities`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_simple_symbol_bonus_overwrites_underlying_trail`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_heads_overwrite_bonus_symbols`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_simple_symbols_jax_parity_when_jax_is_available`
- `tests/test_vector_visual_observation.py::test_cpu_oracle_browser_lines_connect_interleaved_same_owner_trails_and_pin_crossing_order`
- `tests/test_vector_visual_observation.py::test_cpu_oracle_visual_trail_write_cursor_ignores_active_stale_tail_slots`
- `tests/test_vector_visual_observation.py::test_cpu_oracle_live_heads_draw_over_bonus_symbols`
- `tests/test_vector_visual_observation.py::test_direct_fast_simple_bonus_symbols_are_distinct_and_not_remapped`
- `tests/test_vector_visual_observation.py::test_direct_fast_simple_bonus_symbols_stay_distinct_across_offsets_and_radii`
- `tests/test_vector_visual_observation.py::test_direct_fast_simple_bonus_symbols_overwrite_underlying_trails`
- `tests/test_vector_visual_observation.py::test_cpu_oracle_simple_bonus_symbols_overlay_browser_line_trails`

What this proves:

- Renderer-backed CPU oracle can exactly match the dirty-cache surface.
- `direct_gray64` has focused CPU-direct/JAX parity on adversarial two-view and
  simple-symbol fixtures.
- Config tests prevent accidentally calling the approximate direct surface a
  trainer default or a CPU-oracle replacement.

Residual risk:

- `direct_gray64` is still policy-space approximation, not browser/canvas or
  production CPU-oracle exact parity.
- Persistent framebuffer has cold-start and append tests, but lacks local
  cursor-regression/row-selective reset/palette-mutation cache-invalidation
  tests and a same-surface persistent-vs-stateless exact GPU parity smoke.

### Stack, Reset, Autoreset, And Final Observation

Already covered:

- `tests/test_source_state_batched_observation_boundary_profile.py::test_push_row_major_frames_into_stack_shifts_fifo_and_normalizes_last_channel`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_push_row_major_frames_into_stack_rejects_bad_dtype`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_latest_uint8_frames_from_stack_rounds_latest_channel`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_push_row_major_frames_into_stack_resets_selected_terminal_rows_only`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_step_shifts_fifo_stack_and_respects_per_row_controlled_player_actions`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_both_players_step_shifts_fifo_stack_for_all_player_views`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_terminal_step_captures_final_stack_before_any_reset`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_both_players_terminal_info_exposes_stacked_final_observation_contract`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_terminal_final_observation_is_visual_stack_not_metadata_observation`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_terminal_final_observation_matches_direct_render_after_dirty_cache_warmup`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_p4_terminal_final_observation_is_visual_stack_after_three_wall_deaths`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_renderer_backed_gpu_candidate_stack_fifo_newest_frame_last`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_autoreset_terminal_rows_are_counted`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_renderer_backed_terminal_final_observation_uses_terminal_frame`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_renderer_backed_uint8_stack_stores_bytes_but_scalarizes_float32`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_renderer_backed_uint8_terminal_final_observation_scalarizes_float32`
- `tests/test_source_state_batched_observation_mock_collector.py::test_batched_surface_profile_loop_partial_reset_keeps_neighboring_rows`
- `tests/test_source_state_batched_observation_mock_collector.py::test_profile_env_manager_keeps_terminal_timestep_before_autoreset`
- `tests/test_source_state_batched_observation_mock_collector.py::test_surface_policy_materializer_attaches_terminal_final_observation`
- `tests/test_source_state_batched_observation_mock_collector.py::test_surface_policy_materializer_requires_terminal_final_observation`
- `tests/test_source_state_batched_observation_mock_collector.py::test_surface_policy_materializer_rejects_malformed_terminal_final_observation`
- `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_materializer_attaches_terminal_final_observation`
- `tests/test_vector_reset.py::test_reset_arrays_snapshots_terminal_rows_before_reset_and_returns_copies`
- `tests/test_vector_autoreset.py::test_plan_autoreset_rows_defaults_to_done_rows_and_stages_copied_terminal_data`
- `tests/test_vector_autoreset.py::test_apply_autoreset_rows_snapshots_terminal_rows_before_reset_and_clears_flags`
- `tests/test_vector_autoreset.py::test_plan_autoreset_rows_keeps_done_terminated_and_truncated_distinct`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_terminal_snapshots_survive_manual_reset`
- `tests/test_curvyzero_lightzero_smoke.py::test_terminal_step_returns_final_observation_reward_map_and_blocks_autoreset`
- `tests/test_curvyzero_lightzero_smoke.py::test_truncation_uses_done_timeout_metadata_and_zero_terminal_mask`

What this proves:

- Newest frame is last channel; reset/step FIFO order is pinned.
- Terminal final observations are visual stacks, not public debug metadata.
- Autoreset staging captures terminal data before reset in local env/runtime
  paths.

Residual risk:

- Most profile-manager terminal tests terminate all rows together. The missing
  high-value case is mixed terminal/live rows in the same batch, with terminal
  roots filtered and live rows still row-major under their original physical row
  ids.

### LightZero `to_play`, Action Masks, And Decode

Already covered:

- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_rejects_lightzero_collect_forward_without_persistent_direct64`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_accepts_lightzero_collect_forward_contract`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_accepts_lightzero_initial_inference_contract`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_policy_output_row_from_plain_handles_batched_mapping_outputs`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_lightzero_collect_forward_stack_probe_flattens_roots_and_decodes`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_lightzero_collect_forward_stack_probe_filters_zero_mask_roots`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_lightzero_initial_inference_stack_probe_calls_model_only`
- `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_action_bridge_exposes_lightzero_env_ids_after_reset`
- `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_action_bridge_allows_complete_row_omission_but_rejects_partial_missing_extra_and_invalid_actions`
- `tests/test_source_state_batched_observation_mock_collector.py::test_stock_env_manager_adapter_returns_env_id_timestep_mapping`
- `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_materializer_preserves_batch_action_mask_order`
- `tests/test_policy_row_mapping.py::test_selected_policy_actions_reject_illegal_active_action`
- `tests/test_curvyzero_lightzero_smoke.py::test_reset_returns_pinned_lightzero_observation_and_metadata`
- `tests/test_curvyzero_lightzero_env.py::test_registered_lightzero_env_reuses_local_smoke_semantics`
- `tests/test_lightzero_source_state_wrapper_product_fidelity.py::test_lightzero_source_state_joint_action_wrapper_routes_controls_visuals_and_terminal`

What this proves:

- The current fixed-opponent path uses `to_play=-1`.
- Flattened masks are `[N,3]`, zero-mask roots are filtered in the
  collect-forward probe, and compact `ready_env_id` is dense.
- The initial-inference split is covered by a local fake model and does not
  include tree/search/action decode.

Residual risk:

- The collect-forward probe does not yet have an illegal decoded-action
  fail-closed test.
- Decode variants are under-covered: string ready-id dicts, list outputs,
  nested root outputs, and `selected_actions`/visit-distribution variants.
- There is no action-feedback loop test that takes decoded real consumer
  actions back through the scalar action bridge for one step.

### RND, Death Mode, And Profile Semantics

Already covered:

- `tests/test_exploration_bonus.py::test_exploration_bonus_spec_fails_closed`
- `tests/test_exploration_bonus.py::test_rnd_meter_spec_selects_reward_model_entrypoint_and_metadata`
- `tests/test_exploration_bonus.py::test_lightzero_rnd_meter_patch_is_atomic_and_weight_zero`
- `tests/test_exploration_bonus.py::test_latest_gray64_adapter_extracts_from_batched_unroll`
- `tests/test_exploration_bonus.py::test_latest_gray64_adapter_extracts_from_flat_replay_batch`
- `tests/test_exploration_bonus.py::test_latest_gray64_adapter_extracts_from_lightzero_channel_unroll_batch`
- `tests/test_exploration_bonus.py::test_latest_gray64_adapter_rejects_unnormalized_values`
- `tests/test_exploration_bonus.py::test_curvy_rnd_reward_model_trains_predictor_freezes_target_and_preserves_zero_weight_reward`
- `tests/test_exploration_bonus.py::test_curvy_rnd_reward_model_seed_controls_init_and_sampling`
- `tests/test_exploration_bonus.py::test_curvy_rnd_reward_model_update_cadence_and_small_buffer_metrics`
- `tests/test_exploration_bonus.py::test_curvy_rnd_reward_model_reports_disable_cudnn_flag`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_batched_controlled_rows_stack_feeds_rnd_latest_gray64_input`
- `tests/test_source_state_batched_observation_profile_cpu.py::test_batched_both_player_stacks_can_materialize_rnd_inputs_per_player_view`
- `tests/test_source_state_batched_observation_mock_collector.py::test_mock_collector_profile_runs_rnd_latest_frame_meter`
- `tests/test_source_state_batched_observation_mock_collector.py::test_mock_collector_profile_keeps_rnd_reward_meter_non_mutating`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_accepts_mock_collector_payload_flags`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_validate_boundary_config_rejects_rnd_without_payload_profile`
- `tests/test_multiplayer_source_state_trainer_surface.py::test_profile_no_death_mode_is_preserved_and_labeled_not_source_fidelity`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_profile_no_death_exercises_natural_bonus_timer`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_profile_no_death_handles_long_natural_bonus_stack`
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_profile_no_death_handles_crowded_bonus_positions`

What this proves:

- `rnd_meter_v0` is metric-only, weight zero, and selects
  `train_muzero_with_reward_model`.
- RND local adapters extract latest channel `[1,64,64]` from policy stacks.
- Predictor changes, target stays frozen, and meter mode preserves target
  rewards in the local reward-model test.
- `profile_no_death` is labeled as profile-only and not source-fidelity death
  behavior.

Residual risk:

- No stock batched-manager RND gate proves `train_muzero_with_reward_model`
  calls, predictor/target hashes, latest-frame alignment, target reward
  neutrality, and normal terminal/autoreset behavior all together.
- Existing profile RND tests are local mock/profile evidence, not production
  replay evidence.

### Profile Labels, Denominators, And Speed-Row Identity

Already covered:

- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_metadata_stays_profile_only_and_does_not_touch_trainer_defaults`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_reports_timing_and_byte_fields`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_runs_batched_stack_probe_before_scalar_materialization`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_can_skip_scalar_materialization_for_batched_stack_probe`
- `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_rejects_skipping_scalar_materialization_without_consumer`
- `tests/test_source_state_batched_observation_mock_collector.py::test_mock_collector_profile_materializes_lightzero_shaped_rows`
- `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_action_bridge_profiles_materialized_timestep_bytes_and_counts`
- `tests/test_source_state_batched_observation_boundary_profile.py::test_compact_hybrid_observation_profile_result_keeps_mapping_edges`
- `tests/test_lightzero_phase_profiler.py::test_lightzero_phase_profiler_can_stop_after_learner_train_calls`
- `tests/test_lightzero_phase_profiler.py::test_curvytron_train_call_cap_installs_phase_profile_outside_profile_mode`
- `tests/test_lightzero_phase_profiler.py::test_curvytron_phase_profiler_records_mcts_model_device_and_batch`
- `tests/test_lightzero_phase_profiler.py::test_curvytron_compact_output_uses_mcts_root_fallback_for_profile_steps`
- `tests/test_lightzero_phase_profiler.py::test_curvytron_compact_output_does_not_use_mcts_root_fallback_when_eval_ran`
- `tests/test_lightzero_config_builder.py::test_extract_visual_survival_surface_records_public_config_contract`
- `tests/test_lightzero_config_builder.py::test_public_visual_survival_builder_builds_source_state_contract`
- `tests/test_lightzero_config_builder.py::test_public_visual_survival_builder_can_add_rnd_meter_bundle`
- `tests/test_lightzero_config_builder.py::test_visual_survival_surface_pins_policy_observation_perspective_contract`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py::test_hybrid_profile_grid_uses_boundary_compute_names`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py::test_hybrid_profile_grid_can_emit_resident_probe_rows_with_scalar_edge`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py::test_hybrid_profile_grid_can_emit_lightzero_collect_forward_rows`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py::test_hybrid_profile_grid_can_emit_lightzero_initial_inference_rows`
- `tests/test_curvytron_optimizer_profile_manifest_runner.py::test_profile_manifest_preflight_requires_compact_output_detail`

What this proves:

- Profile-only paths report `profile_only`, `calls_train_muzero`,
  `stock_lightzero_integrated`, `trainer_defaults_changed`, and
  `touches_live_runs`.
- Compact results preserve row/player head/tail samples.
- Phase-profile compact output labels the denominator source when it falls back
  to MCTS root count.
- Hybrid manifest rows carry LightZero collect-forward and initial-inference
  labels.

Residual risk:

- There is no single row-summary acceptance test that rejects a speed row when
  required semantic identity fields are absent or contradictory.
- Some timing fields are generic and can overclaim if read without the
  LightZero-specific labels; docs already warn that collect-forward timing
  includes CPU tree/search work.

## Missing Highest-Risk Gates

### P0. Stock Batched-Manager Mixed Terminal/Live Gate

Smallest next test/run:

- Add a tiny local or profile-only stock-entrypoint test named like
  `tests/test_source_state_batched_observation_mock_collector.py::test_stock_profile_manager_mixed_terminal_live_rows_preserve_terminal_timestep_and_live_roots`
  if it can be done with the adapter locally.
- If it needs stock LightZero, run a bounded profile-only canary that calls
  stock `train_muzero`, uses `env_manager_type=curvyzero_batched_profile`,
  disables eval/GIF/checkpoint clutter, and stops after the first learner call.

Pass criteria:

- `called_train_muzero=true`.
- `env_manager_type=curvyzero_batched_profile`.
- `death_mode=normal`, not `profile_no_death`.
- One batch has at least one terminal physical row and one live physical row.
- Terminal scalar timesteps have `done=true`,
  `final_observation_present=true`, nonzero `final_observation_nbytes`, and no
  collect-ready roots.
- Live rows remain ready and keep original `policy_env_row`/`policy_player`
  values; they are not compacted into physical row zero.
- Zero terminal masks are counted as filtered/skipped, not forwarded to
  collect/search.

Why this is first: it pins row/player order, masks, reset/autoreset, terminal
final observation, and stock adapter semantics in one gate.

### P0. Stock Batched-Manager `rnd_meter_v0` Gate

Smallest next test/run:

- Add a bounded profile-only stock row with `rnd_meter_v0` and a matched no-RND
  anchor. This must call `train_muzero_with_reward_model`, not only the mock
  collector.

Pass criteria:

- `exploration_bonus.mode=rnd_meter_v0`.
- `trainer_entrypoint=lzero.entry.train_muzero_with_reward_model`.
- `feature_source=policy_gray64_latest/v0`.
- `rnd_metrics.input_shape=[1,64,64]`.
- `rnd_metrics.source_observation_shape=[4,64,64]`.
- `collect_data_calls > 0`, `train_with_data_calls > 0`, and
  `estimate_calls > 0`.
- Predictor hash changes after training; target hash stays unchanged.
- `last_target_reward_changed=false` and reward target deltas are zero.
- A sampled checksum or marker proves RND latest frames match the latest channel
  of the manager policy stack for the same env ids/players.
- Terminal rows do not let RND read post-reset frames as terminal final frames.

Why this is first-tier: RND rows can look like speed regressions or gains while
silently testing a different input source, cadence, or reward target.

### P0. Speed-Row Semantic Attestation Test

Smallest next test:

- Add a parser/summary test near
  `tests/test_lightzero_phase_profiler.py` or
  `tests/test_curvytron_optimizer_profile_manifest_runner.py`, named like
  `test_profile_speed_summary_rejects_rows_missing_semantic_identity`.

Reject rows missing these fields or equivalent compact evidence:

- profile identity: `profile_only`, `called_train_muzero`,
  `stock_lightzero_integrated`, `touches_live_runs`;
- backend identity: manager type, renderer backend, render surface, stack dtype,
  no-hidden-fallback flag;
- denominator identity: env steps, MCTS roots, simulation count, learner calls,
  replay sample calls, denominator source;
- mapping identity: `policy_env_id`, `policy_env_row`, `policy_player`, head
  and tail samples;
- stack identity: observation shape, latest-frame dtype/stack dtype,
  materialized scalar timestep count;
- terminal identity: death mode, terminal row count, autoreset row count,
  final-observation presence/count/bytes;
- LightZero consumer identity when present: `to_play` mode, total roots,
  consumed roots, filtered zero-mask roots, illegal action count, CPU-tree label;
- RND identity when present: mode, feature source, train/estimate counters,
  predictor/target hashes, target reward unchanged.

Why this is first-tier: it blocks the fast-wrong table failure mode, where a row
looks faster because it changed backend, death mode, denominator, or consumer
path.

## Smallest Local Tests To Add Next

1. `tests/test_source_state_batched_observation_mock_collector.py::test_scalar_materializer_mixed_terminal_live_zero_mask_keeps_physical_row_ids`

   Use direct `materialize_lightzero_scalar_timestep(...)` inputs:
   `B=2`, `P=2`, row 0 terminal, row 1 live, terminal final-observation sentinel
   only on row 0, terminal action masks all false, live action masks true.
   Assert final observations attach only to row 0 players, live row ids remain
   row 1/player 0 and row 1/player 1, and masks flatten row-major.

2. `tests/test_source_state_batched_observation_boundary_profile.py::test_lightzero_collect_forward_stack_probe_rejects_decoded_illegal_action`

   Fake policy returns an action not allowed by the passed mask. Assert
   `_LightZeroCollectForwardStackProbe.run(...)` raises `ValueError` and does
   not return success telemetry.

3. `tests/test_source_state_batched_observation_boundary_profile.py::test_policy_output_row_from_plain_handles_ready_id_and_list_variants`

   Cover dict keyed by string ready ids, list output, nested root output, and
   dict-of-arrays with `selected_actions`, `visit_count_distributions`, and
   `predicted_value`.

4. `tests/test_source_state_batched_observation_boundary_profile.py::test_persistent_delta_state_cursor_regression_resets_only_regressed_rows`

   Use two rows: row 0 previous cursor `5` and current cursor `2`; row 1
   previous cursor `2` and current cursor `4`. Assert reset mask is `[1,0]`,
   row 0 does not connect to stale previous owner position, and row 1 appends
   only current delta slots.

5. `tests/test_source_state_batched_observation_boundary_profile.py::test_persistent_delta_state_palette_mutation_invalidates_row_cache`

   If invalidation is exposed locally, change only row 0 `avatar_color` between
   frames. Assert row 0 rebuilds/resets and row 1 remains incremental.

6. `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_fake_consumer_sees_row_player_step_sentinels`

   Extend the existing sentinel renderer pattern so the fake consumer checks
   reset and step sentinels in the pre-scalar `[B,2,4,64,64]` stack:
   latest channel equals current step, previous channel equals reset/previous
   step, and flattened consumer order is row-major.

7. `tests/test_source_state_hybrid_observation_profile.py::test_hybrid_lightzero_probe_result_attests_profile_only_and_scalar_off`

   Use a fake batched-stack probe with LightZero-shaped telemetry and
   `materialize_scalar_timestep=False`. Assert profile-only/no-live-run labels,
   `materialized_timestep_count=0`, backend/semantics labels, and LightZero
   root counters are preserved.

8. `tests/test_lightzero_phase_profiler.py::test_compact_profile_output_requires_backend_mapping_terminal_and_consumer_identity`

   Build one complete compact speed row and one missing-fields row. Assert the
   missing-fields row is rejected before it can enter a comparison table.

## Trust Matrix

| Area | Trust now | Do not trust yet |
| --- | --- | --- |
| Row/player order | Local renderer-backed, hybrid, materializer, scalar bridge, and policy-row mapping tests. | Mixed terminal/live stock `train_muzero` batch. |
| GPU/CPU observation | CPU-oracle renderer-backed exact parity; direct CPU/JAX policy-space parity on focused fixtures. | `direct_gray64` as production CPU-oracle/browser-pixel parity; persistent exact same-surface GPU parity through reset/cache mutation. |
| Stack/reset/final obs | FIFO, selected reset, terminal final stack, autoreset staging, uint8 scalarization. | Mixed terminal/live rows through stock profile adapter and LightZero root filtering. |
| LightZero masks/to_play | Fixed-opponent `to_play=-1`, `[N,3]` masks, zero-mask filtering, model-only initial inference. | Illegal decoded action fail-closed; broader output decode variants; action-feedback loop proof. |
| RND | Local config, latest-frame extraction, meter neutrality, predictor/target behavior. | Stock batched-manager RND meter row with real reward-model entrypoint and terminal/autoreset alignment. |
| Profile labels | Profile-only/no-live labels, compact row mapping, phase denominator fallback labels. | A single strict speed-row attestation gate used by summary tooling. |

Bottom line: the next parallel work should be two or three cheap local tests
plus one bounded stock-entrypoint gate, not more render-only proof.
