# Compact Speed-Row Floor Bundle 2026-05-30

Status: OPT-062 complete; OPT-064 follow-up complete; OPT-065 first fast-path
cut complete; OPT-065 timing split plus final-sync CUDA-event probe says the
old initial sync was real queued model execution, not a removable phase sync.
The safe `model_compile_mode=default` probe now proves model-forward compile can
reduce the service/model bucket without the CUDAGraph overwrite. Fresh
pre-guard paired H100 repeats favored compile, and the dispatch split showed
the dispatch residual itself is tiny. OPT-066 supersedes the old action-drift
blocker: fixed-root parity and post-guard action/trajectory checks now pass,
but the formal decision parks compile because end-to-end wall speed did not
repeatably improve. OPT-073b validates the latest compact Torch action-path
telemetry on H100 but does not produce a wall-speed win. OPT-074 binds OPT-073b
plus a current CUDA-event repeat into a decision packet and selects initial
model forward/compile-safe service work next. OPT-075 implements the smallest
safe cleanup in that lane and measures it on H100; it narrows the fixed-floor
gap but does not prove an initial-model-forward speed win. Remaining work is to
decompose the compact Torch search dispatch/service envelope under
same-denominator gates. OPT-076 has now completed that decomposition on H100
and emitted a refreshed service-bucket decision. Dispatch residual/readback/tree
residual are small; replay-index/commit/sample-gate work is the next selected
target. OPT-077/077b has now reduced that bucket on H100: packed scalar exposed
metadata as the child, and metadata counts/checksums reduced replay-index build
to `0.145s` and commit to `0.185s`. It is not a top-level speed win. The
refreshed decision selects `initial_model_forward_compile_safe_path`. OPT-078
then removed recurrent nested `no_grad`, exposed guard timing, and hardened the
service-bucket decision guard checks. H100 row/floor/decision passed, but it is
not a speed win: compact `10740.8` steps/sec, fixed-floor wall ratio `1.281`,
guard total `0.208s`. The refreshed decision still selects
`initial_model_forward_compile_safe_path`.
This is an optimizer engineering decomposition, not promotion evidence.

Latest compact Torch timing and decision reports:

```text
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-timing-eager-sim1-canonical-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-timing-eager-sim1-canonical-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-host-phase-cudaevent-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-timing-host-phase-cudaevent-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-cudaevent-finalsync-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-timing-cudaevent-finalsync-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-modelcompile-default-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-modelcompile-default-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-modelcompile-default-canonical-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-modelcompile-default-canonical-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-pair-eager-canonical-sim1-r1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-pair-modelcompile-default-canonical-sim1-r1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-pair-eager-canonical-sim1-r2-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-pair-modelcompile-default-canonical-sim1-r2-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-pair-eager-canonical-sim1-r1-dispatchsplit-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-pair-modelcompile-default-canonical-sim1-r1-dispatchsplit-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-pair-eager-canonical-sim1-r2-dispatchsplit-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt065-compact-speed-row-floor-bundle-pair-modelcompile-default-canonical-sim1-r2-dispatchsplit-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_compile_eager_speed_pair_results/opt065-compile-eager-speed-pair-review-r1r2-20260531/compile_eager_speed_pair_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt065-h100-timing-modelcompile-sim1-warm-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-fastpath1sim-20260530/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-fastpath1sim-promoted-20260530/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt073-h100-eager-sim1-canonical-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt073-h100-eager-sim1-canonical-20260531b/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt073b-compact-speed-row-floor-bundle-h100-b1024a16-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt074-h100-host-phase-cudaevent-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt074-compact-speed-row-floor-bundle-host-phase-cudaevent-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_torch_service_bucket_decision_results/opt074-compact-torch-service-bucket-decision-20260531/compact_torch_service_bucket_decision_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt075-h100-guardonly-host-phase-cudaevent-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt075-compact-speed-row-floor-bundle-guardonly-host-phase-cudaevent-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt076-h100-dispatch-envelope-host-phase-cudaevent-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt076-compact-speed-row-floor-bundle-dispatch-envelope-host-phase-cudaevent-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_torch_service_bucket_decision_results/opt076-compact-torch-service-bucket-decision-dispatch-envelope-host-phase-cudaevent-sim1-20260531/compact_torch_service_bucket_decision_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt077-h100-replay-index-packed-scalar-host-phase-cudaevent-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt077-compact-speed-row-floor-bundle-replay-index-packed-scalar-host-phase-cudaevent-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt077b-h100-replay-index-metadata-counts-host-phase-cudaevent-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt077b-compact-speed-row-floor-bundle-replay-index-metadata-counts-host-phase-cudaevent-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_torch_service_bucket_decision_results/opt077b-compact-torch-service-bucket-decision-replay-index-metadata-counts-host-phase-cudaevent-sim1-20260531/compact_torch_service_bucket_decision_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt078-h100-recurrent-guard-timing-host-phase-cudaevent-sim1-20260531/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/opt078-compact-speed-row-floor-bundle-recurrent-guard-timing-host-phase-cudaevent-sim1-20260531/compact_speed_row_floor_bundle_report.json
artifacts/local/curvytron_compact_torch_service_bucket_decision_results/opt078-compact-torch-service-bucket-decision-recurrent-guard-timing-host-phase-cudaevent-sim1-20260531/compact_torch_service_bucket_decision_report.json
```

Latest read:

```text
timing_split_steps_per_sec=12335.7
timing_split_wall_sec=14.942
timing_split_service_total_sec=2.808
timing_split_model_sec=2.249
timing_split_tree_search_sec=0.506
timing_split_initial_enqueue_sec=0.344
timing_split_initial_sync_sec=1.905
timing_split_recurrent_enqueue_sec=0.186
timing_split_tree_policy_build_sec=0.276
timing_split_tree_sync_sec=0.0048
timing_split_bundle_read=compact_rollout_slab_search_service_dominant
timing_split_bundle_ratio=0.989
host_phase_cuda_event_steps_per_sec=9876.1
host_phase_cuda_event_initial_sync_enabled=true
host_phase_cuda_event_initial_cuda_event_sec=2.232
host_phase_cuda_event_initial_host_sync_sec=1.751
host_phase_cuda_event_recurrent_cuda_event_sec=0.471
host_phase_cuda_event_tree_cuda_event_sec=0.536
host_phase_cuda_event_tree_host_sync_sec=0.0049
final_sync_steps_per_sec=9041.2
final_sync_wall_sec=20.387
final_sync_initial_sync_enabled=false
final_sync_initial_enqueue_sec=0.683
final_sync_initial_host_sync_sec=0.0
final_sync_initial_cuda_event_sec=2.249
final_sync_recurrent_cuda_event_sec=0.475
final_sync_tree_cuda_event_sec=0.543
final_sync_action_wall_sec=2.946
final_sync_replay_d2h_bytes=0
final_sync_action_d2h_bytes=737280
final_sync_fast_path_recurrent_calls=180/180
final_sync_bundle_ratio=1.350
promoted_counter_fast_path_count=180
promoted_counter_recurrent_calls=180
model_compile_probe_ok=false
model_compile_failure=Torch CUDAGraph overwritten-output error
model_compile_default_event_ok=true
model_compile_default_event_steps_per_sec=10772.9
model_compile_default_event_wall_sec=17.110
model_compile_default_event_initial_cuda_event_sec=1.391
model_compile_default_event_initial_host_sync_sec=0.696
model_compile_default_event_recurrent_cuda_event_sec=0.333
model_compile_default_event_service_total_sec=1.916
model_compile_default_event_bundle_read=compact_rollout_slab_search_dispatch_wall_dominant
model_compile_default_canonical_ok=true
model_compile_default_canonical_steps_per_sec=10258.6
model_compile_default_canonical_wall_sec=17.967
model_compile_default_canonical_service_total_sec=1.806
model_compile_default_canonical_model_sec=1.324
model_compile_default_canonical_search_sec=0.424
model_compile_default_canonical_bundle_read=compact_rollout_slab_search_dispatch_wall_dominant
pair_r1_eager_steps_per_sec=9840.1
pair_r1_compile_steps_per_sec=12450.3
pair_r2_eager_steps_per_sec=11520.7
pair_r2_compile_steps_per_sec=12563.9
dispatchsplit_residual_delta_sec_range=0.0032..0.0038
dispatchsplit_max_residual_abs_share_of_gap=0.0030
compile_eager_pair_review_decision=not_approved_action_trajectory_mismatch
compile_eager_pair_review_speed_claim_allowed=false
post_guard_compile_eager_review_decision=not_approved_compile_not_faster
compile_default_status=parked_optional_off_by_default
opt073_first_support_steps_per_sec=12428.4
opt073_first_support_wall_sec=14.831
opt073b_canonical_telemetry_steps_per_sec=9394.0
opt073b_canonical_telemetry_wall_sec=19.621
opt073b_fast_path_count=180
opt073b_root_prior_softmax_skipped_count=180
opt073b_selection_mode_last=masked_logits_argmax
opt073b_action_d2h_bytes=737280
opt073b_replay_d2h_bytes=0
opt073b_committed_replay_d2h_bytes=0
opt073b_fixed_floor_steps_per_sec=12203.9
opt073b_compact_torch_vs_floor_wall_ratio=1.299
opt073b_read=telemetry_validated_not_speed_win
opt074_current_event_steps_per_sec=9392.4
opt074_current_event_wall_sec=19.624
opt074_current_event_service_total_sec=2.236
opt074_current_event_model_sec=1.548
opt074_current_event_search_sec=0.423
opt074_current_event_initial_cuda_event_sec=1.536
opt074_current_event_recurrent_cuda_event_sec=0.333
opt074_current_event_tree_cuda_event_sec=0.402
opt074_decision=select_initial_model_forward_path
opt074_selected_next_target=initial_model_forward_compile_safe_path
opt074_speed_claim_allowed=false
opt075_guardonly_steps_per_sec=12563.1
opt075_guardonly_wall_sec=14.672
opt075_guardonly_service_total_sec=2.164
opt075_guardonly_action_wall_sec=2.217
opt075_guardonly_model_sec=1.571
opt075_guardonly_search_sec=0.381
opt075_guardonly_initial_cuda_event_sec=1.562
opt075_guardonly_recurrent_cuda_event_sec=0.321
opt075_guardonly_tree_cuda_event_sec=0.363
opt075_guardonly_action_d2h_bytes=737280
opt075_guardonly_replay_d2h_bytes=0
opt075_guardonly_committed_replay_d2h_bytes=0
opt075_guardonly_fast_path_softmax_skip=180/180
opt075_guardonly_fixed_floor_steps_per_sec=13759.0
opt075_guardonly_compact_torch_vs_floor_wall_ratio=1.095
opt075_guardonly_read=safe_cleanup_not_initial_forward_speed_claim
opt076_local_instrumentation=service/slab/profile/bundle_fields_landed
opt076_local_validation=ruff,focused_propagation_7,compact_torch_floor_43,replay_contract_6
opt076_steps_per_sec=12846.8
opt076_wall_sec=14.348
opt076_fixed_floor_steps_per_sec=13759.0
opt076_compact_torch_vs_floor_wall_ratio=1.071
opt076_dispatch_residual_delta_sec=0.005
opt076_action_residual_sec=0.004
opt076_action_readback_sec=0.011
opt076_core_residual_sec=0.152
opt076_initial_cuda_event_sec=1.510
opt076_tree_total_sec=0.368
opt076_replay_index_rows_build_delta_sec=2.457
opt076_commit_previous_delta_sec=2.491
opt076_decision=select_replay_index_or_sample_gate
opt077_packed_scalar_steps_per_sec=10781.5
opt077_packed_scalar_wall_sec=17.096
opt077_packed_scalar_replay_index_rows_build_sec=2.259
opt077_packed_scalar_metadata_sec=1.651
opt077_packed_scalar_scalar_device_transfer_sec=0.067
opt077b_metadata_counts_steps_per_sec=11735.2
opt077b_metadata_counts_wall_sec=15.707
opt077b_metadata_counts_fixed_floor_ratio=1.172
opt077b_replay_index_rows_build_sec=0.145
opt077b_commit_previous_sec=0.185
opt077b_metadata_sec=0.013
opt077b_decision=select_initial_model_forward_path
opt078_recurrent_guard_steps_per_sec=10740.8
opt078_recurrent_guard_wall_sec=17.161
opt078_recurrent_guard_fixed_floor_ratio=1.281
opt078_initial_cuda_event_sec=1.546
opt078_recurrent_cuda_event_sec=0.336
opt078_inference_guard_total_sec=0.208
opt078_learner_sample_delta_sec=1.610
opt078_actor_wall_delta_sec=1.360
opt078_decision=select_initial_model_forward_path
next_target=initial_model_forward_compile_safe_path
```

Durable report:

```text
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-20260530/compact_speed_row_floor_bundle_report.json
```

Manifest:

```text
artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-20260530/manifest.json
```

What landed:

- `curvyzero_compact_speed_row_floor_bundle/v1`
- `scripts/build_compact_speed_row_floor_bundle.py`
- `tests/test_compact_speed_row_floor_bundle.py`
- `--search-service-kind` backend selection in the compact Coach speed-row
  smoke and Modal producer
- `FixedShapeBatchedSearchOwnerV0.flush_device_replay_payload(...)` so the
  no-search floor can run under resident-observation compact trainer rows
  without host replay fallback

Same-denominator H100 rows:

| Role | Run | Speed |
| --- | --- | --- |
| accepted device-target Coach row | `optimizer-compact-coach-speed-row-h100-b1024a16-threshold-20260530` | `10018.5` compact trainer env steps/sec |
| real compact Torch sibling | `optimizer-speed-row-compact-torch-h100-b1024a16-threshold-20260530` | `5005.0` compact trainer env steps/sec |
| fixed-shape no-search floor | `optimizer-speed-row-fixed-shape-h100-b1024a16-threshold-20260530` | `13759.0` compact trainer env steps/sec |

Bound denominator:

```text
candidate=optimizer-compact-unified-lifecycle-smoke-20260530
hardware=H100
batch_size=1024
actor_count=16
steps=180
warmup_steps=45
sample_batch_size=512
sample_interval=8
replay_pair_capacity=4096
learner_device=cuda
learner_train_steps=1
num_simulations=1
model_identity_scope=candidate_loaded_checkpoint
speed_currency=compact_trainer_env_steps_per_sec
```

Read:

```text
status=same_denominator_speed_row_floor_bundle_complete
engineering_read=compact_rollout_slab_non_service_dominant
decision_confidence=limited_shape_matched_trajectory_different
compact_torch_vs_floor_wall_delta_sec=23.430787325
compact_rollout_slab_delta_sec=22.589985107
search_delta_sec=3.027254810
compact_rollout_slab_non_service_delta_sec=19.562730297
search_delta_abs_share_of_measured_gap=0.1292
compact_rollout_slab_non_service_delta_abs_share_of_measured_gap=0.8349
search_dominance_claim=false
next_target=decompose_compact_rollout_slab_commit_flush_materialization
trajectory_checksum_match=false
promotion_claim=false
training_speedup_claim=false
calls_train_muzero=false
touches_live_runs=false
```

Historical OPT-062/063 interpretation:

The fixed-shape no-search floor was materially faster than the accepted
device-target trainer row, while the real compact Torch search-service sibling
was slower than both. The residual-aware read initially said the compact Torch
slab envelope was the wall and that raw `search_service_total_sec` explained
only about `13%` of the compact-vs-floor measured gap. OPT-064 superseded this
as a current target: commit/flush/replay-index materialization were not the
dominant wall.

OPT-064 follow-up:

```text
slab-timer bundle=artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-slabtimers-20260530/compact_speed_row_floor_bundle_report.json
pre-cache compact Torch row=artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-speed-row-compact-torch-h100-b1024a16-threshold-dispatchenv-20260530/compact_coach_speed_row_modal_report.json
cached compact Torch row=artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-speed-row-compact-torch-h100-b1024a16-threshold-dispatchenv-cache-20260530/compact_coach_speed_row_modal_report.json
cached bundle=artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-dispatchenv-cache-20260530/compact_speed_row_floor_bundle_report.json
```

The H100 slab-timer rerun showed the old residual target was too broad:
commit/flush/replay-index build were not the dominant wall. The large delta was
`compact_rollout_slab_search_dispatch_wall_sec`. Adding compact Torch service
envelope timers showed that dispatch wall was inside
`CompactTorchSearchServiceV1.run_action_step`, and the pre-cache row spent
`9.769s` in `metadata_build_sec` while `service_total_sec` was only `3.086s`.
The cause was repeated policy-refresh model-state digest computation for every
action step. Caching the digest/metadata at `refresh_model_state` produced the
cached row: compact Torch speed `9608.9`, wall `19.182s`,
`metadata_build_sec=0.0034`, `service_action_wall_sec=3.148`, and
`service_total_sec=3.080`.

The current cached bundle reads:

```text
engineering_read=compact_rollout_slab_search_service_dominant
compact_torch_vs_floor_wall_ratio=1.270
compact_torch_vs_floor_wall_delta_sec=4.079
compact_rollout_slab_delta_sec=4.085
search_delta_sec=3.042
compact_rollout_slab_non_service_delta_sec=1.043
search_dominance_claim=true
next_target=optimize_compact_torch_search_service
```

So the next speed work is compact Torch search-service/model/search wall, not
game mechanics and not slab commit/flush.

OPT-065 then made the first narrow cut by adding the sim1 fast path. That
reduced tree-search wall (`0.776s -> 0.529s`) and service total
(`3.080s -> 2.825s`) in the clean H100 row, while model inference stayed flat
(`2.243s -> 2.237s`).

The later timing-split row sharpened the target without changing behavior:
service total `2.808s`, model `2.249s`, tree `0.506s`, initial enqueue
`0.344s`, initial sync `1.905s`, recurrent enqueue `0.186s`, tree
policy/count build `0.276s`, and final tree sync `0.0048s`. That made initial
model sync/forward execution the next target. The first warm model-compile
probe failed closed with a Torch CUDAGraph overwritten-output error; the later
`model_compile_mode=default` slice avoided that failure, but its pair review now
blocks approval on action/trajectory mismatch.

The host-phase CUDA-event row adds device elapsed timing while preserving the
old host syncs. It is slower (`9876.1` steps/sec) and should not be used as the
speed headline. Its useful numbers are initial CUDA event `2.232s`, initial
host sync `1.751s`, recurrent CUDA event `0.471s`, tree CUDA event `0.536s`,
and final tree host sync `0.0049s`. The later final-sync row proved phase-sync
deletion is not a wall win, and the later `model_compile_mode=default` rows
proved compile can reduce the model bucket while leaving end-to-end wall status
open.

The bundle does not prove a gameplay-quality trajectory, because service
policies choose different actions and the action/trajectory checksums differ.
It also does not compare against stock `train_muzero`, publish promotion, touch
live runs, or claim rating/leaderboard quality.

Validation:

```text
uv run ruff check src/curvyzero/training/fixed_shape_batched_search_owner.py src/curvyzero/training/compact_speed_row_floor_bundle.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py scripts/build_compact_speed_row_floor_bundle.py src/curvyzero/infra/modal/compact_coach_speed_row.py tests/test_fixed_shape_batched_search_owner.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_speed_row_floor_bundle.py
uv run pytest tests/test_fixed_shape_batched_search_owner.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_speed_row_floor_bundle.py tests/test_compact_coach_speed_row.py -q
uv run ruff check src/curvyzero/training/compact_speed_row_floor_bundle.py tests/test_compact_speed_row_floor_bundle.py scripts/build_compact_speed_row_floor_bundle.py
uv run pytest tests/test_compact_speed_row_floor_bundle.py -q
uv run ruff check src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_speed_row_floor_bundle.py tests/test_compact_search_replay_contract.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_speed_row_floor_bundle.py
uv run pytest tests/test_compact_search_replay_contract.py -q
uv run ruff check src/curvyzero/training/compact_torch_search_service.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/infra/modal/compact_coach_speed_row.py src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py scripts/build_curvytron_hybrid_observation_profile_grid.py tests/test_compact_torch_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py
uv run pytest tests/test_compact_torch_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py -q
```

Result:

```text
ruff passed
31 passed, 2 warnings
real bundle validated
residual-aware refresh: ruff passed; 31 passed, 2 warnings
OPT-064 local timers: ruff passed; compact search replay contract 43 passed, 2 warnings
OPT-064 cached dispatch-envelope fix: ruff passed; targeted suite 148 passed, 2 warnings
OPT-065 timing split: ruff passed; targeted suite 289 passed, 2 warnings
OPT-065 timing modes/CUDA events: ruff passed; targeted suite 249 passed, 2 warnings; CLI help passed; host-phase CUDA-event H100 row and bundle validated
OPT-065 model compile mode: ruff passed; targeted suite 227 passed, 2 warnings; CLI help passed; event and canonical H100 rows plus bundles validated
```
