# Subagent RND + Batched Manager Gate Critique

Date: 2026-05-20

Scope: critique only. Do not edit trainer code, change defaults, or touch live
Coach training runs.

## Plain Read

Do not interpret new C512 RND profile rows as learning evidence or renderer
evidence until they pass the RND and batched-manager gates below.

A meaningful C512 `rnd_meter_v0` row must prove four things at once:

1. The batched manager really supplied the trainer-visible policy frames.
2. RND consumed the intended latest gray64 policy frame, not a stale/reset/source
   shortcut.
3. RND trained and estimated at the intended cadence while leaving reward
   targets unchanged in meter mode.
4. Throughput used a trustworthy step denominator, not the compact
   reward-model-path zero-envstep trap.

The row can be useful if it is only a meter/overhead row. It is not positive-RND
evidence, because positive reward is still blocked on intrinsic normalization and
support/resume semantics.

## Failure Modes And Required Proof

### 1. RND Input Source Is Wrong

What can go wrong:

- The row claims `policy_gray64_latest/v0`, but RND receives the whole stack,
  flattened source state, terminal zero fill, or the wrong player perspective.
- The no-RND and RND rows silently use different observation backends.
- The config JSON still says `policy_observation_backend=cpu_oracle` because of
  a scalar wrapper field, and that gets mistaken for the actual batched profile
  backend.

Required proof:

- `exploration_bonus.mode=rnd_meter_v0`.
- `feature_source=policy_gray64_latest/v0`.
- `input_shape=[1,64,64]`.
- `source_observation_shape=[4,64,64]`.
- `trainer_entrypoint=lzero.entry.train_muzero_with_reward_model`.
- `env_manager_type=curvyzero_batched_profile`.
- Batched-manager metadata shows `trainer_observation_stack_backend=renderer_backed_profile`,
  `renderer_backed_stack_profile=true`, an explicit renderer backend name, and
  `trainer_observation_no_hidden_fallback=true`.
- The row summary should preserve strict backend language: the command may still
  carry a wrapper `policy_observation_backend=cpu_oracle`, but the meaningful
  batched row identity is the manager/surface backend above.

### 2. Latest-Frame Extraction Is Not Proven Through The Batched Manager

What can go wrong:

- Local RND adapter tests pass, but the stock batched-manager path feeds replay
  observations in a different shape.
- `collect_data()` trains on latest frames from collected `GameSegment.obs_segment`,
  while `estimate()` extracts latest frames from replay `train_data`; those are
  two separate shape paths.
- Terminal rows/autoreset can make the newest frame after reset appear in
  `ready_obs` while the timestep should still carry the terminal final
  observation.
- Live-policy row filtering can disturb row/player order or drop rows, so the
  RND batch no longer aligns with rewards and env ids.

Required proof:

- `rnd_metrics.input_shape=[1,64,64]` and `source_observation_shape=[4,64,64]`.
- A semantic gate or row artifact explicitly says
  `rnd_latest_frame_matches_policy_stack`.
- Batched manager artifacts show `policy_env_row`, `policy_player`, and scalar
  env-id mappings are row-major and stable for the measured no-death row.
- For normal-death rows, require `final_observation_present=true` on terminal
  timesteps before autoreset and matching `final_observation_row_mask`; otherwise
  do not use the row to bless RND manager semantics.
- `collect_data_calls > 0`, `estimate_calls > 0`, and `buffer_count` is large
  enough to train. A row with only construction but no valid buffered frames is
  not meaningful.

### 3. Train/Estimate Cadence Is Misread

What can go wrong:

- RND trains once per huge collection wave but estimates several learner batches,
  making novelty stale.
- Small buffer skips hide the fact that the row did not train.
- `rnd_update_per_collect=1` is accidentally treated as the current serious
  default, even though the planning/default scale is now `100`.
- C512 changes the collect/replay ratio enough that update10/update100 rows are
  not comparable to previous small smokes.

Required proof:

- `rnd_batch_size` and `rnd_update_per_collect` appear in the row label and
  config.
- `collect_data_calls`, `train_with_data_calls`, `estimate_calls`,
  `train_cnt_rnd`, and `estimate_cnt_rnd` are present.
- `train_cnt_per_estimate` is present and plausible for the configured cadence.
- `train_with_data_skipped_small_buffer_count=0` after warmup for throughput
  rows. If nonzero, report it and treat cadence/overhead as partially invalid.
- For the planned C512 grid, compare no-RND, update10, and update100 as separate
  cadence rows. Do not collapse them into a single "RND overhead" number.

### 4. Predictor Did Not Actually Change

What can go wrong:

- The reward-model entrypoint is selected, but no optimizer step happens.
- The predictor hashes are absent, stale, or equal because training skipped.
- The row times RND estimate only and then labels it as RND training overhead.

Required proof:

- `last_predictor_hash_before_train` and `last_predictor_hash_after_train` are
  both present.
- The before/after predictor hashes differ after a train call.
- `last_train_loss` is finite.
- Raw MSE metrics are present: mean, std, p50, and p95.
- RND method timers are split at least into `collect_data`, `train_with_data`,
  `estimate`, metrics snapshot/write, and state hash. A single aggregate RND
  bucket is not enough to diagnose C512 overhead.

### 5. Target Network Is Not Frozen

What can go wrong:

- Target parameters accidentally remain trainable or get overwritten by
  checkpoint/resume/state loading.
- Momentum/target-representation settings in LightZero config create an
  unintended target update path.
- The target hash changes because the metric hashes a different object or a
  newly constructed model rather than the same target module.

Required proof:

- `last_target_hash_before_train` and `last_target_hash_after_train` are both
  present and equal.
- Current `target_hash` is present.
- `target_update_*` policy config values are recorded, but the Curvy RND target
  module hash must still be frozen for meter-mode interpretation.
- If target hash changes, reject the row as an RND correctness row even if
  throughput looks good.

### 6. Meter Mode Mutates Reward Targets

What can go wrong:

- `weight=0.0` is set, but the reward-model adapter still writes changed target
  rewards through dtype/shape/copy behavior.
- `rnd_replay_target_v0` or nonzero weight slips into a row labeled as meter.
- Batch min/max normalized intrinsic reward is logged and mistaken for actual
  reward influence.

Required proof:

- `mode=rnd_meter_v0`.
- `intrinsic_reward_weight=0.0`.
- `training_effect=reward_target_unchanged`.
- `target_reward_effect=unchanged`.
- `last_target_reward_changed=false`.
- `last_target_reward_delta_abs_max=0.0`.
- `last_target_reward_delta_abs_mean=0.0`.
- If any reward delta is nonzero, the row is not comparable with no-RND
  throughput or training behavior.

### 7. GPU Contention Hides The Real Bottleneck

What can go wrong:

- Batched render and CUDA RND share the same H100 and serialize in ways that make
  a renderer row look slow or an RND row look cheap depending on timing overlap.
- Learner/search/model forward also uses GPU, so RND timings cannot be read as
  isolated side cost.
- State-hash computation or metrics writing pulls tensors back to CPU and adds
  synchronization.
- A CUDA RND row is compared to a CPU RND row or a no-RND row without reporting
  GPU utilization/memory.

Required proof:

- Record RND device (`cpu` or `cuda`) and deterministic/cuDNN flags.
- Record H100 max memory and max utilization for the row.
- Preserve phase timers for `collector_collect`, manager step, policy forward,
  MCTS search, learner train, and RND method buckets.
- For batched manager rows, record renderer/stack timing from
  `trainer_surface_profile_timing`: env step, stack update, reward, package, and
  renderer telemetry where available.
- Interpret CUDA RND as a whole-loop contention experiment, not as pure RND
  kernel timing. The clean comparison is matched no-RND versus RND on the same
  manager/topology, with the same C512/sim/death/eval settings.

### 8. Compact Env-Step Counter Lies

What can go wrong:

- The LightZero reward-model path reports `env_steps_collected=0` in compact
  telemetry even though MCTS roots prove collection occurred.
- Throughput becomes `0 steps/s` or uses a fallback denominator without being
  labeled.
- No-RND rows use collector delta while RND rows use MCTS-root fallback; that is
  acceptable only if the workload counts match and the source is explicit.

Required proof:

- Keep both `env_steps_collected_raw` and `env_steps_collected_source`.
- If raw collector envstep is zero, only accept
  `mcts_search_root_sum_profile_fallback` when:
  `lightzero_eval_freq=0`, profile stock eval skipping is enabled,
  evaluator calls/skips are both zero, MCTS root count is positive, and full
  phase counts are retained.
- Match no-RND and RND rows on MCTS roots, simulation budget, replay samples,
  learner calls, collector count, episode count, source max steps, and death
  mode before comparing wall time.
- Report the denominator source in the table. A C512 row without a denominator
  source is not meaningful.

## C512 RND Row Acceptance Checklist

Before reading a C512 RND row as meaningful, require:

- Same topology as its no-RND anchor: C512, same sim count, same source max
  steps, same death/eval/GIF/checkpoint settings, same manager type.
- `called_train_muzero=true` and the actual entrypoint is
  `train_muzero_with_reward_model` for RND rows.
- `replay_sample_calls` and `learner_train_calls` match the planned workload.
- `mcts_search_root_sum` or fallback step count matches the expected env-step
  denominator.
- RND metrics: constructed, input shape, collect/train/estimate counts, skip
  count, predictor hash change, target hash unchanged, raw MSE percentiles, and
  target reward unchanged.
- Timers: collector collect, manager step/reset if available, stack/render,
  policy forward, MCTS, learner train, RND collect/train/estimate, metrics
  write/hash, wall clock.
- GPU telemetry: max memory, max utilization, RND device, and renderer backend.

## Interpretation Rules

- A passed `rnd_meter_v0` row proves overhead and plumbing, not learning.
- A failed meter neutrality proof blocks positive-RND interpretation.
- A row with missing RND hashes/counters can still be a throughput smoke, but it
  is not an RND correctness row.
- A row with fallback step counts can still be useful, but only if the fallback
  source is explicit and workload counts match the no-RND anchor.
- Do not mix RND cadence conclusions into renderer claims. If update100 is slow,
  that is an RND cadence cost unless the matched no-RND row also regressed in
  renderer/manager timing.

