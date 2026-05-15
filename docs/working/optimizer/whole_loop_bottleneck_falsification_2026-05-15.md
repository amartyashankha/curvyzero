# Whole-Loop Bottleneck Falsification Grid

Date: 2026-05-15

Purpose: propose the next profile matrix after the render work improved. The current
production path is still CPU oracle observations; the GPU renderer remains lab-gated.
This note is intentionally skeptical that render is the best next target.

## Updated Hypothesis

The next whole-loop bottleneck is more likely to be at the collector boundary than in
raw rendering alone: LightZero search batching, env-manager IPC, large per-step Python
`info` payloads, reset/final-observation churn, replay object churn, and artifact work
can all hide outside the obvious env render buckets.

Render remains a risk for future longer policies, but render-only wins are not enough
unless the full-loop profile also moves. Short bad-policy trajectories can understate
future trail/render costs and overstate reset churn; no-death profiles can do the
opposite by hiding terminal/final-observation and reset costs.

## Important Profile Gap

Current env telemetry has useful buckets for observation, vector runtime, opponent
action, reward, and `step_total_before_info`, but it does not isolate the cost of
building and transporting the full LightZero timestep payload.

The suspected missing buckets are:

- `_base_info` and `_step_info` Python object construction.
- `BaseEnvTimestep` payload size and pickle/IPC cost in subprocess env manager.
- `final_observation` copying at terminal steps.
- reset-time construction of a fresh `VectorMultiplayerEnv`.
- replay push/sample/update-priority object churn over longer learner horizons.
- CUDA synchronization relocation when profile timers are async.
- checkpoint, mirror, summary, and volume-commit artifact tax outside clean profiles.

Telemetry stride does not remove `_step_info` construction or IPC payload cost; it only
reduces sampled JSON writes after the full info object already exists.

## Common Profile Controls

Use these unless an experiment overrides them:

- policy observation backend: CPU oracle.
- policy observation: `browser_lines + simple_symbols`.
- manager: subprocess for whole-loop rows; base manager only for attribution rows.
- hardware: L4/T4 first, H100 only where the test is explicitly about GPU/search
  synchronization.
- source-state size: 512 for current production surface; include 256 only as a
  control.
- eval, GIF, tournament, and checkpoint writes: off for clean throughput rows.
- telemetry stride: 128 for normal profile rows; 1 only for narrow attribution rows.
- stop condition: learner-train-call count, not wall-clock time.

## Prioritized Experiments

### 1. Info/IPC Payload Ablation

Question: is the largest hidden tax the per-step Python metadata payload rather than
render or MCTS?

Knobs:

- `collector_env_num`: 64, 128.
- `num_simulations`: 8.
- `batch_size`: 32.
- `n_episode`: equal to `collector_env_num`.
- `source_state_size`: 512.
- `disable_death_for_profile`: true for the first pass.
- `opponent_runtime_mode`: `blank_canvas_noop`.
- `profile_env_timing_enabled`: true.
- `profile_telemetry_stride`: 128.
- Proposed profile-only knob: `profile_info_payload_mode=full|minimal`.
- Proposed counters: `step_info_sec`, `base_info_sec`, `info_pickle_sec`,
  `info_pickle_bytes`, `obs_pickle_bytes`, `final_observation_copy_sec`.

Expected bottleneck signal:

- Wall or collector time drops by at least 15% in `minimal` mode.
- Or worker-side `step_info + pickle` is at least 20% of collector CPU time.
- Named render/search buckets do not explain the delta.

Kill criteria:

- Wall changes by less than 5% at both C64 and C128.
- `step_info + pickle` is less than 5% of collector time.
- Payload bytes per step are small enough that IPC cannot plausibly dominate.

Interpretation:

- If this fires, render is not the next best target. The next target is profile-mode
  payload slimming, static metadata caching, and moving large debug-only fields out of
  hot timesteps.

### 2. CUDA Sync Relocation Sentinel

Question: are async GPU timings making MCTS, model inference, or learner work look
cheaper than they are?

Knobs:

- hardware: L4/T4 and H100.
- `collector_env_num`: 64, 128.
- `num_simulations`: 8, 16.
- `batch_size`: 32.
- `n_episode`: equal to `collector_env_num`.
- `source_state_size`: 512.
- `disable_death_for_profile`: true.
- paired rows: `profile_cuda_sync_enabled=false|true`.
- `stop_after_learner_train_calls`: 12.
- `profile_telemetry_stride`: 128.

Expected bottleneck signal:

- Sync-on rows move material time into `mcts_search`, model initial/recurrent
  inference, or learner train.
- H100 changes relative ranking only after sync is enabled.
- Wall grows modestly but bucket attribution shifts by more than 10 percentage points.

Kill criteria:

- Sync-on wall is less than 8% slower.
- Bucket shares move less than 10 percentage points.
- H100/L4 ordering and C64/C128 ordering are unchanged.

Interpretation:

- If this fires, do not optimize based on async bucket percentages. Use sync-on
  attribution rows for target selection even if sync-off remains the throughput number.

### 3. Reset/Terminal Trajectory Bias Ladder

Question: are current short bad-policy rows and no-death rows hiding opposite future
bottlenecks?

Knobs:

- `collector_env_num`: 32, 64.
- `num_simulations`: 8.
- `batch_size`: 32.
- `n_episode`: equal to `collector_env_num`.
- `source_state_size`: 512.
- trajectory regimes:
  - normal death.
  - no-death 512-step cap.
  - proposed profile-only forced terminal every 32 steps.
  - proposed profile-only forced terminal every 128 steps.
  - proposed warm-start high-trail state before collection.
- counters: `reset_sec`, `reset_count`, `done_count`,
  `final_observation_copy_sec`, cold-cache observation time, episode length histogram.

Expected bottleneck signal:

- Normal or forced-terminal rows show reset/final-observation/cold-cache cost above
  10% of collector wall.
- No-death rows show observation/render/trail cost growing with horizon while terminal
  costs vanish.
- Warm-start rows are slower than fresh no-death rows at the same step count.

Kill criteria:

- Reset plus terminal-copy plus cold-cache costs remain below 10% in every trajectory
  regime.
- Long warm-start rows are within 5% of ordinary no-death rows.
- Episode length distribution does not change the top two buckets.

Interpretation:

- If this fires, short normal training profiles are not enough for capacity planning.
  Keep a horizon ladder in the standard profile suite.

### 4. Collector/Search Boundary Fragmentation

Question: is search throughput limited by root batching and LightZero collector
boundaries rather than by raw MCTS or render speed?

Knobs:

- hardware: L4/T4 first; H100 after L4 shape is known.
- `collector_env_num`: 32, 64, 128.
- `n_episode`: `collector_env_num / 2`, `collector_env_num`, `collector_env_num * 2`
  where LightZero accepts it.
- `num_simulations`: 4, 8, 16.
- `batch_size`: 32.
- `source_state_size`: 512.
- trajectory regimes: normal death and no-death.
- counters: `mcts_search_calls`, `root_batch_mean`, `root_batch_p50/p95`,
  `model_initial_batch_mean`, `model_recurrent_batch_mean`, roots/sec, sims/sec,
  collector wall/sec.

Expected bottleneck signal:

- Throughput improves when `n_episode` increases at fixed `collector_env_num`.
- H100 only wins when root batches are wide enough.
- C128 fails to improve because root batches fragment or collector control overhead
  grows.

Kill criteria:

- Roots/sec scales with `collector_env_num` and `num_simulations`.
- `root_batch_mean` remains stable or improves with width.
- `n_episode` changes wall by less than 5%.

Interpretation:

- If this fires, prefer collector/search batching work over renderer work. The fastest
  env step does not matter if the search boundary cannot feed wide roots.

### 5. Replay And Artifact Tax Sentinel

Question: do clean profile rows hide late-loop replay cost and production artifact
work?

Knobs:

- `collector_env_num`: 64, 128.
- `num_simulations`: 8.
- `batch_size`: 32.
- `source_state_size`: 512.
- `disable_death_for_profile`: false and true, separate rows.
- `stop_after_learner_train_calls`: 12, 48, 96.
- clean row: eval/GIF/checkpoint/volume commit off.
- live-tax row: production checkpoint cadence, `commit_on_checkpoint=true`,
  production summary/artifact behavior, eval/GIF only if normally enabled.
- counters: replay push/sample/update-priority/remove-oldest, learner train,
  checkpoint save, volume commit, post-run summary scan, file count, inode delta,
  run directory bytes.

Expected bottleneck signal:

- Replay buckets grow after the buffer fills or after longer learner horizons.
- Live-tax row loses at least 15% wall versus clean row.
- Checkpoint or volume commit causes visible collector or learner stalls.
- Artifact file count grows fast enough to worsen the existing `/runs` inode pressure.

Kill criteria:

- 48-call and 96-call rows keep replay below 10% of wall.
- Live-tax row is less than 5% slower than clean row.
- File count and inode deltas are negligible for one profile unit.

Interpretation:

- If this fires, clean throughput is not the deploy bottleneck. Treat artifact cadence,
  replay memory behavior, and run-directory pressure as first-class optimizer work.

## Decision Order

Run experiment 1 first. It is the cheapest way to falsify the current profiler view
because it targets a bucket the existing tables mostly omit.

Then run experiment 4 at the current best width to decide whether search batching or
collector width is the next scaling lever. Run experiment 2 before trusting GPU/H100
comparisons. Keep experiment 3 in the suite before declaring a renderer win durable,
because short policies and no-death policies bias the bottleneck in opposite
directions. Run experiment 5 before any production queue recommendation.

## Stop Rules For Render Work

Pause renderer optimization if any of these are true:

- Info/IPC payload ablation improves wall by at least 15%.
- Root batch fragmentation explains more than half of the C64 to C128 scaling loss.
- Sync-on attribution moves more than 10 percentage points into search/model/learner.
- Reset/final-observation costs exceed 10% under normal or forced-terminal profiles.
- Live artifact tax exceeds 15% versus clean profile throughput.

Continue renderer optimization only if these falsifiers fail and render/observation
still accounts for at least 25% of full-loop wall on the production CPU oracle path.
