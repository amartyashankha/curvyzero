# Shared Training Reporting Contract - 2026-05-09

Purpose: one compact reporting shape for LightZero lanes and repo-native PPO
lanes. Use this for smoke reports, run manifests, scorecards, and handoff
summaries. Missing fields should be written as `null`, `unknown`, or
`not_applicable`, not silently omitted.

## Required Top-Level Shape

```text
report_version: shared_training_reporting_contract/v0
run_profile: ...
lane_contract: ...
timing_profile: ...
throughput: ...
latency: ...
checkpoint_refs: ...
seed_reset: ...
schemas: ...
scorecard: ...
artifacts: ...
non_claims: ...
```

## Run Profile

```text
run_id: stable local run id or Modal app id
lane: lightzero_official_atari | lightzero_custom_dummy_pong | repo_native_ppo
framework: LightZero MuZero | repo-native PPO | other
task: cartpole | official_atari_pong | dummy_pong | curvytron_1v1 | other
variant: tabular_ego | raster_flat | raster_stack4_ego | visual_atari | rays_v0
mode: train | eval | scorecard | actor_dry_run | learner_smoke | probe
quality_label: no_quality_smoke | infrastructure_pass | signal_fail | candidate_learning
host: local | modal
hardware: cpu | gpu_l4 | other
started_at_utc: ISO-8601 or null
duration_wall_sec: float or null
code_ref: commit, branch, script path, or null
config_ref: config path plus important overrides
artifact_root: path or volume prefix
```

## Lane Contract

```text
players: int
batch_shape: [B] or [B,P]
time_shape: [T,B,P] when rollout data exists
control_shape: single_ego | simultaneous_joint_action
policy_rows: compact_live_rows | framework_internal | not_applicable
action_application: action_id | joint_action[B,P]
opponent_type: scripted | random | frozen_checkpoint | shared_current_policy | none
train_reward_source: sparse_env_reward | shaped_reward | framework_default | none
telemetry_reward_only: survival | loss_delay | clearance | contact | none
terminal_contract: done | terminated/truncated | final_observation | final_reward_map
autoreset_policy: explicit_after_staging | framework_hidden | none
masked_action_policy: required | not_applicable
```

`simultaneous_joint_action` and `joint_action[B,P]` are trainer/reporting shapes. For
CurvyTron source-fidelity runs, they must describe the wrapper decision/control snapshot
that is converted into held source controls over elapsed-ms frames.

## Timing Buckets

All timing values are seconds over the run window unless a report explicitly
states otherwise.

```text
reset_autoreset_sec
observation_packing_sec
row_compaction_sec
policy_forward_sec
search_sec
action_scatter_sec
env_step_sec
rollout_staging_sec
replay_write_sec
target_construction_sec
learner_update_sec
checkpoint_write_sec
eval_scorecard_sec
artifact_write_sec
actor_idle_sec
learner_idle_sec
framework_overhead_sec
unbucketed_sec
```

Rules:

- LightZero may report framework-internal buckets as `unknown`; still report
  total train/eval wall time and any wrapper-measured buckets.
- Repo-native PPO should report every bucket it owns directly.
- `search_sec` is required for MuZero/MCTS eval or collection; use
  `not_applicable` for PPO without search.

## Throughput

```text
env_steps
env_steps_per_sec
agent_rows
agent_rows_per_sec
completed_games
completed_games_per_min
train_iterations
learner_updates
samples_per_update
replay_or_rollout_bytes
checkpoint_count
scorecard_episodes
timeout_count
```

For simultaneous lanes, `agent_rows` means live player rows, not only env rows.

## Latency

```text
action_latency_ms_p50
action_latency_ms_p95
action_latency_ms_p99
env_step_ms_p50
env_step_ms_p95
policy_forward_ms_p50
policy_forward_ms_p95
search_ms_p50
search_ms_p95
learner_update_ms_p50
learner_update_ms_p95
policy_staleness_updates_p50
policy_staleness_updates_p95
```

Use `null` when the run only has aggregate timing. Do not convert aggregate
wall time into fake percentile latency.

## Checkpoint References

```text
checkpoint_id: logical id, iteration, or step
checkpoint_path: exact artifact path
checkpoint_kind: lightzero_pth_tar | torch_pt | numpy_npz | none
checkpoint_source: trained_this_run | loaded_for_eval | frozen_opponent | none
strict_load: true | false | not_applicable
loaded_checkpoint_id: id or null
loaded_checkpoint_path: path or null
opponent_checkpoint_id: id or null
opponent_checkpoint_path: path or null
best_checkpoint_id: id or null
latest_checkpoint_id: id or null
```

## Seed And Reset Details

```text
global_seed
env_seed
reset_seed_shape: scalar | [B] | [T,B] | unknown
reset_profile: default | contact_pressure | official_env | custom
reset_pressure_agent: ego | opponent | both | not_applicable
randomization: none | start_state | opponent | env | action_exploration
episode_cap_steps
collector_env_count
eval_env_count
autoreset_count
terminal_count
truncation_count
seed_reuse_policy: fixed | per_env | per_episode | unknown
```

Reports must state whether reset differences are part of the experiment or
only implementation plumbing.

## Observation Schema

```text
schema_id: curvyzero_egocentric_rays/v0 | lightzero_atari_stack | dummy_pong_tabular | other
raw_observation_shape
policy_observation_shape
stored_observation_shape
dtype
frame_history
perspective: ego | global | framework_default
normalization
final_observation_recorded: true | false | framework_hidden
```

## Action Schema

```text
action_space_type: discrete
action_count
action_ids: ordered labels
legal_action_mask_shape
mask_required: true | false
masked_action_violations
executed_action_shape
policy_target_shape
action_histogram
joint_action_shape: [B,P] | not_applicable
```

For LightZero/MuZero, separate `executed_action` from MCTS/root-visit policy
targets whenever target telemetry exists.

## Reward And Target Schema

```text
env_reward_shape
env_reward_values
train_reward_target: sparse_env | shaped | framework_default | none
value_target: mcts_bootstrap | discounted_return | gae_return | none
policy_target: mcts_root_visits | sampled_action_logprob | action_weights | none
discount
td_steps
support_scale: value/reward support details or null
final_reward_map_shape
telemetry_metrics: survival_steps, shaped_loss_delay, score, entropy, etc.
```

Do not mix telemetry rewards into training-reward claims unless the run actually
trained on them.

Pong-specific rule: report survival length as a first-class scorecard signal.
For custom dummy Pong, default `train_reward_target` is sparse score
`+1/-1/0`; `survival_steps` and `shaped_loss_delay_return` are telemetry unless
the run is explicitly labeled as a shaped-objective ablation. For official
Atari Pong eval, keep `steps_survived` next to manual/stock return and reward
counts. Never summarize a Pong row as only wins/losses.

## Scorecard

```text
eval_method: LightZero MuZeroEvaluator | manual_policy_eval | repo_native_scorecard | none
eval_strict_load: true | false | not_applicable
baselines: random_uniform, scripted, lagged, frozen_checkpoint, track_ball, none
episodes_per_row
mean_return
mean_survival_steps
win_loss_draw
action_entropy
root_visit_summary
failure_modes: action_collapse, timeout, masked_action_violation, load_fallback, none
```

## Artifacts

```text
run_config
contract_manifest
profile_report
rollout_buffer
replay_or_target_sidecar
metrics_jsonl
scorecard_json
checkpoint_files
sample_terminal_trace
logs
```

Use exact paths, not prose descriptions, whenever possible.

## Non-Claims

Every report must include explicit non-claims. Choose all that apply and add
plain text when needed.

```text
not_a_learning_result
not_policy_quality
not_curvytron_solved
not_visual_atari_parity
not_lightzero_replacement
not_repo_native_replacement
not_full_self_play
not_source_fidelity_proof
not_speed_claim
not_checkpoint_improvement
```

Required lane-specific defaults:

- LightZero official Atari smokes: not CurvyTron, not custom dummy Pong, not a
  policy-quality claim unless held-out eval improves.
- LightZero custom dummy Pong: not official Atari parity, not full simultaneous
  multiplayer self-play, not CurvyTron solved.
- Repo-native PPO dry runs and learner smokes: not a learning result, not a
  LightZero replacement, not a final framework decision.

## Minimal Plain Summary

Each human-facing summary should end with this compact block:

```text
lane:
run_id:
checkpoint:
contract:
throughput:
latency:
scorecard:
non_claims:
next_gate:
```
