# Continuous Optimization Loop

Date: 2026-05-12

Status: active Optimizer operating memory.

## Operating Rule

Keep going. There is no natural stopping condition for the optimizer lane. The
loop is:

1. Reorient from current docs, code, and live run evidence.
2. State the current Amdahl picture in plain language.
3. Pick the highest-leverage unknowns.
4. Test pieces in isolation before touching the live training path.
5. Integrate only when an isolated experiment predicts a real whole-loop win.
6. Reprofile the whole loop after integration.
7. Update docs before moving on.

This is not permission to make random live-path changes. It is a rule to keep
an active queue of measurements, hypotheses, and small experiments.

## Identity

You are the Optimizer. Your job is speed, setup, measurement, Amdahl
prioritization, and throughput. Keep recommendations simple and label the
audience: Coach, Environment, or Optimizer. Do not turn this into taxonomy.

## Current North Star

Current reset, 2026-05-12: the optimizer goal is faster CurvyTron throughput on
the trusted stock LightZero `train_muzero` path first:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--opponent-use-cuda=false
```

The custom `--mode two-seat-selfplay` path remains useful as postmortem/profile
evidence, but it is not the trusted learning lane until replay/target semantics
are fixed. Coach owns learning claims. Environment owns source fidelity.
Optimizer owns measurement, setup, throughput, Amdahl prioritization, and speed
changes.

## Current Main Recommendation

Do not start by scaling the old two-seat matrix. First profile stock LightZero
`train_muzero` with `env_variant=source_state_fixed_opponent`, a frozen
checkpoint opponent, and the opponent on CPU by default. The small optimizer
patch now makes that split explicit:

- `policy.cuda=true` for the learner on GPU runs.
- `opponent_use_cuda=false` by default for env-owned frozen checkpoint
  opponents.
- Try `env_manager_type=base` as the matched canary.
- Then try `env_manager_type=subprocess` with one and then several collector
  envs.
- Keep LightZero in-loop eval, background eval, GIF, and frequent checkpoints
  off for profiles unless measuring those costs.

See
[stock frozen optimizer pivot](stock_frozen_optimizer_pivot_2026-05-12.md).

The older fast-direct/browser-lines recommendations below are historical
custom-adapter notes unless explicitly relabeled.

## Historical Custom-Adapter Fidelity Tradeoff

This section is historical custom-adapter context, not current Coach guidance.
`fast_gray64_direct` was the speed surface for the custom two-seat path, not a
browser-fidelity truth claim. It kept the main semantic visual facts:
trail/head positions, self/other contrast, bonus presence, and bonus type luma.
It also dropped real visual detail: connected anti-aliased browser lines, sprite
texture, exact downsample coverage, and browser pixel parity.

Optimizer stance:

- Historical recommendation only: the old custom adapter trained mostly on
  `fast_gray64_direct`.
- Keep one or two small `browser_lines` sentinels to detect obvious
  approximation damage.
- Keep semantic render tests and mask-level comparisons alive.
- Do not claim `fast_gray64_direct` is source faithful.
- If fast-direct learning looks good and browser-lines sentinel looks bad, ask
  whether the approximation is helping by simplifying the task or breaking the
  real task.
- If fast-direct learning looks bad, do not automatically blame the
  approximation; profile search, reward, replay, learner, stochasticity, and
  eval setup first.

## Current Amdahl Picture

Fresh isolated Amdahl matrix, 2026-05-12:

```text
fast-main, L4/T4, B64, sim8, fast_direct, learner on:
  wall=58.69s, collect=36.13s
  search=14.87s, policy_forward=13.40s
  visual=4.16s
  obs+replay noise=9.52s
  env+autoreset=4.38s
  learner=2.79s

fast-no-noise, same shape with observation noise 0:
  wall=45.61s, collect=28.24s
  search=15.46s, visual=4.24s
  obs+replay noise near 0

h100-b128, H100, B128, sim8, fast_direct:
  wall=62.49s, collect=42.11s
  search=12.68s, visual=3.62s
  obs+replay noise=14.94s

browser-nodeath, L4/T4, B8, sim2, browser_lines, collect only:
  wall=58.24s, visual=31.37s, search=9.01s

fast-nodeath, matched fast_direct:
  wall=33.31s, visual=2.59s, search=9.80s
```

Plain read:

- For current `fast_gray64_direct` main training, rendering is not the main
  bottleneck. Search/collect forward and CPU-side observation noise are the
  first targets.
- Disabling observation noise is not an optimizer recommendation for training;
  it is an Amdahl bound. With default noise on, noise is a major CPU term.
- H100/B128 improves search rows per second, but larger batch also exposes more
  CPU noise/reset/replay work. H100 is a scale probe, not automatically the
  default.
- For `browser_lines` long-survival/no-death, rendering is still the bottleneck
  and should be optimized only if that rich renderer becomes the selected
  training surface.

Detailed table:
[fresh Amdahl re-exploration](fresh_amdahl_reexploration_2026-05-12.md).

Wave 2 collect-only scaling read, 2026-05-12:

```text
B64/L4/sim8/default noise:
  wall=217.39s, replay_rows=98,304, about 452 replay rows/s
  search=75.44s, visual=19.68s, obs+replay noise=52.20s

B64/L4/sim8/no noise:
  wall=130.49s, replay_rows=98,304, about 753 replay rows/s
  search=59.77s, visual=15.92s, obs+replay noise near 0

B128/H100/sim8/default noise:
  wall=182.58s, replay_rows=131,072, about 718 replay rows/s
  search=48.22s, visual=14.63s, obs+replay noise=58.11s
```

Plain read: the next fast-direct target is CPU observation noise. H100 helps
search, but once search gets faster the CPU noise term becomes the limiter.
This is an implementation target, not an instruction to remove augmentation from
Coach training. After optimizing noise, reprofile B64/B128 on L4 and H100.

Fresh stock-vs-custom control read, 2026-05-12:

```text
stock source_state_fixed_opponent:
  train_muzero wall=21.689s, roots=818, learner_updates=4

stock source_state_joint_action:
  train_muzero wall=19.261s, roots=929, learner_updates=4

custom two-seat matched wait run:
  elapsed=19.674s, policy/search rows=1024, learner_updates=4
```

Plain read: stock `train_muzero` works and should remain a serious control
lane, but the tiny matched profile does not prove stock is faster than custom
two-seat. The custom path's risk is not obvious speed collapse; it is replay and
target correctness. Also do not compare raw `steps/s` across these paths:
custom `steps` are physical CurvyTron ticks, while its real work count here is
`64 envs * 2 seats * 8 ticks = 1024` policy/search rows.

See
[stock train-MuZero vs two-seat profile plan](train_muzero_stock_vs_two_seat_profile_plan_2026-05-12.md).

Known speed signal:

```text
B64/L4/sim8 browser_lines:        about 768s wall
B64/L4/sim8 fast_gray64_direct:   about 203s wall
B128/L4/sim8 fast_gray64_direct:  about 726s wall, worse per replay row
B128/H100/sim8 fast_gray64_direct about 429s wall, useful scale probe
```

Plain read:

- Rich browser-lines rendering was the old wall-clock limiter.
- Fast direct rendering changes the bottleneck mix. Fresh read-only live-run
  evidence now points to policy/search as the largest named bucket in the
  running fast-direct rows.
- Do not overgeneralize that to every future setting. If Coach returns to
  `browser_lines`, or if trained agents survive much longer and accumulate long
  trails, rendering can become the limiter again. The right question is always:
  which render surface and which survival length are being timed?
- Larger self-play batches do more work, but only matter if learner/eval quality
  per wall clock improves. Optimizer measures throughput; Coach reads learning.
- H100 helps some larger fast-direct jobs, but L4/T4 remains the default until
  larger batches/search prove the GPU is the limiter.
- Timing caveat: `policy_search_sec` is the old wrapper bucket and includes the
  newer `policy_tensor_prepare_sec`, `policy_collect_forward_sec`, and
  `policy_output_decode_sec` pieces. Do not add those sub-buckets to
  `policy_search_sec` as if they were independent wall-clock time.

Read-only live evidence, 2026-05-12:

```text
overnight40a row 01, old-prefix, L4/T4, B64, sim8, fast_direct, iteration 360:
  total_replay_rows_collected=2.949M over 10850s, about 272 replay rows/s
  policy_search_sec=16.41, visual_stack_update_sec=1.39,
  observation_noise_sec+replay_observation_noise_sec=3.49,
  loop_autoreset_sec=2.55, env_step_sec=0.45

overnight40a row 08, old-prefix, L4/T4, B32, sim8, fast_direct, iteration 690:
  total_replay_rows_collected=2.826M over 10641s, about 266 replay rows/s
  policy_search_sec=6.54, visual_stack_update_sec=0.73,
  observation_noise_sec+replay_observation_noise_sec=1.81,
  loop_autoreset_sec=1.36, env_step_sec=0.28

overnight40a row 09, old-prefix, L4/T4, B128, sim8, fast_direct, iteration 90:
  total_replay_rows_collected=1.475M over 10081s, about 146 replay rows/s
  policy_search_sec=79.57, visual_stack_update_sec=3.29,
  observation_noise_sec+replay_observation_noise_sec=9.40,
  loop_autoreset_sec=6.40, env_step_sec=1.14

overnight40a row 33, H100, B128, sim8, fast_direct, iteration 120:
  policy_search_sec=54.9, visual_stack_update_sec=2.25,
  observation_noise_sec+replay_observation_noise_sec=8.87,
  loop_autoreset_sec=4.63, env_step_sec=1.18

overnight40a row 37, H100, B256, sim8, fast_direct, iteration 20:
  policy_search_sec=323.5, visual_stack_update_sec=5.95,
  observation_noise_sec+replay_observation_noise_sec=19.74,
  loop_autoreset_sec=12.26, env_step_sec=1.42

mixpast row 01, L4/T4, B64, sim8, fast_direct, obs noise 0, iteration 40:
  policy_search_sec=16.44, visual_stack_update_sec=1.44,
  observation_noise_sec+replay_observation_noise_sec near 0,
  loop_autoreset_sec=2.51, env_step_sec=0.43
```

Plain read from the live rows: render is now small for `fast_gray64_direct`.
The next optimizer target is search/MCTS batching and search scaling, with
observation noise and autoreset/env pressure as secondary CPU terms. Do not
interrupt or mutate these overnight runs; only read their artifacts.

Batch-size read: compare replay rows per second, not iteration count. B32 and
B64 are close in current L4/T4 evidence; B128 on L4/T4 is worse. B128/H100 is a
scale probe, not the default.

Long-survival caveat: no-death and trained-policy profiles still need to be
kept separate from current short-episode training rows. A fast-direct short-run
profile can say "search dominates this run"; it cannot prove that rich
browser-lines rendering will stay cheap once agents survive for many more
steps.

Fresh isolated profiles, 2026-05-12:

```text
opt-stacknocopy-nonoise-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct, observation_noise_std=0.0
  elapsed=118.4s
  policy_search_sec sum=54.6s
  loop_autoreset_sec sum=13.4s
  visual_stack_update_sec sum=15.9s

opt-stacknocopy-noise-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct, observation_noise_std=0.10
  elapsed=154.4s
  policy_search_sec sum=55.0s
  observation_noise_sec + replay_observation_noise_sec sum=36.1s
  loop_autoreset_sec sum=13.5s
  visual_stack_update_sec sum=16.6s

opt-browser-nodeath-sentinel-20260512
  B8, sim2, 8 iterations, collect only, profile_no_death, browser_lines
  elapsed=53.6s
  visual_stack_update_sec sum=31.2s
  policy_search_sec sum=9.0s

opt-fastdirect-nodeath-sentinel-20260512
  B8, sim2, 8 iterations, collect only, profile_no_death, fast_gray64_direct
  elapsed=25.5s
  visual_stack_update_sec sum=2.4s
  policy_search_sec sum=9.1s
```

Plain read: the no-copy stack hot path is a modest real win. The no-death
render A/B is the sharper signal: same workload, same seed, same search settings,
and `fast_gray64_direct` cuts visual time by about `13x` and wall time by about
`2.1x` versus `browser_lines`. So rendering is still the main bottleneck in the
rich long-survival regime, while search/noise/reset share the fast-direct regime.

Fresh full-loop profile, 2026-05-12:

```text
opt-fullloop-fast-b64-sim8-20260512
  B64, sim8, 24 iterations, 64 collect steps/iteration
  4 learner updates/iteration, normal death, fast_gray64_direct, L4/T4
  elapsed=401.5s, about 16.7s/iteration
  completed episodes stay short: mean iteration mean steps about 14.4

collect timing mean/iteration:
  policy_search_sec=4.80
    policy_collect_forward_sec=4.35
    policy_output_decode_sec=0.30
    policy_tensor_prepare_sec=0.07
  observation_noise_sec + replay_observation_noise_sec=2.92
  visual_stack_update_sec=1.40
  loop_autoreset_sec=1.21
  replay_row_build_sec=0.47
  env_step_sec=0.43
```

Plain read: fast-direct normal-death training is no longer render-bound.
The biggest named collect bucket is LightZero collect forward/search. The old
profile also had a large hidden gap between collect timers and wall time. A
read-only critique found the missing terms: replay sampling, learner update,
checkpoint save, progress writes/commits, and especially the per-update model
hash check that copied model tensors back to CPU.

Optimizer patch landed after this profile: real learner updates no longer hash
the model before and after every update by default. Use
`--two-seat-verify-model-update-hash` only when debugging correctness. The
two-seat path now reports `learner_timing_summary` and
`iteration_timing_summary` so the next full-loop run can say where the hidden
wall time went.

Follow-up no-hash full-loop profile, 2026-05-12:

```text
opt-nohash-fullloop-fast-b64-sim8-20260512
  B64, sim8, 12 iterations, 64 collect steps/iteration
  4 learner updates/iteration, normal death, fast_gray64_direct, L4/T4
  verify_model_update_hash=false
  elapsed=289.4s, about 24.1s/iteration

iteration timing mean:
  collect_total_sec=18.32
  iteration_wall_before_progress_sec=22.80
  iteration_wall_sec=23.34

learner timing mean:
  learner_sample_sec=1.26
  learner_batch_build_sec=1.68
  learner_forward_sec=0.36
  learner_update_total_sec=2.04

collect timing mean:
  policy_search_sec=6.94
    policy_collect_forward_sec=6.33
    policy_output_decode_sec=0.34
    policy_tensor_prepare_sec=0.10
  observation_noise_sec + replay_observation_noise_sec=4.99
  visual_stack_update_sec=1.99
  loop_autoreset_sec=1.85
  replay_row_build_sec=0.67
  env_step_sec=0.73
```

Plain read: model hashing was a suspicious telemetry tax, but it was not the
whole missing wall-clock story. With hashes off, actual learner forward is small.
At that point the learner-side Python problem was replay sampling plus learner
batch and target construction. Search/collect forward was still the largest
single bucket, and observation noise was material. The latest run was slower
than the earlier 24-iteration profile, so do not claim a hash-off speedup;
treat it as a better attribution profile on a moving codebase. The replay-cache
profile below supersedes the replay/sample target read.

Replay-cache full-loop profile, 2026-05-12:

```text
opt-replaycache-fullloop-fast-b64-sim8-20260512
  B64, sim8, 12 iterations, 64 collect steps/iteration
  4 learner updates/iteration, normal death, fast_gray64_direct, L4/T4
  verify_model_update_hash=false
  elapsed=234.9s, about 19.6s/iteration

iteration timing mean:
  collect_total_sec=16.18
  iteration_wall_before_progress_sec=18.52
  iteration_wall_sec=18.90

learner timing mean:
  learner_context_build_sec=0.31
  learner_sample_sec=0.06
  learner_batch_build_sec=0.14
  learner_forward_sec=0.51
  learner_update_total_sec=0.65

collect timing mean:
  policy_search_sec=6.35
    policy_collect_forward_sec=5.68
    policy_output_decode_sec=0.39
    policy_tensor_prepare_sec=0.13
  observation_noise_sec + replay_observation_noise_sec=4.31
  visual_stack_update_sec=1.67
  loop_autoreset_sec=1.57
  replay_row_build_sec=0.61
  env_step_sec=0.63
```

Plain read: the replay return-context cache was the right low-risk patch. It
kept target semantics and moved learner-side replay/sample/target work from
roughly `2.9s/iteration` in the no-hash attribution run to roughly
`0.5s/iteration` including the new shared context build. Whole-loop time before
progress dropped from about `22.8s/iteration` to about `18.5s/iteration` on the
same broad B64/L4/sim8 shape. Do not oversell this as a universal speedup,
because Modal variance and code drift are real; do keep it because it removes
repeated Python work without changing MuZero search or targets.

Updated Amdahl read after replay-cache: for `fast_gray64_direct` normal-death
training, learner replay/sample is no longer the main named bottleneck. The
largest named term is now LightZero collect/search/model work, followed by
observation noise, autoreset/env, and visual stack. For `browser_lines`
long-survival profiles, rendering remains a separate dominant bottleneck.

Tooling cleanup: `scripts/summarize_curvytron_lightzero_profiles.py` now
surfaces the two-seat learner and iteration timing fields directly:
`iter_preprog`, `collect`, `learn_sample`, `learn_ctx`, `learn_batch`,
`learn_fwd`, `learn_update`, `ckpt`, and `commit`. Use that before opening full
JSON by hand.

Post-dirty browser-lines long-survival profile, 2026-05-12:

```text
opt-render-dirty-browser-b16-sim4-20260512
  B16, sim4, 8 iterations, 64 collect steps/iteration
  collect only, profile_no_death, browser_lines, L4/T4
  elapsed=92.7s
  visual_stack_update_sec sum=61.3s
  policy_search_sec sum=14.4s
  dirty_render hit_rate=0.9965, fallbacks=0
```

Plain read: dirty rendering works and avoids full trail redraw most of the
time, but browser-lines long-survival is still render-heavy. That does not
contradict the fast-direct profile; it means we must keep saying which visual
surface and death mode each speed number used.

Local render microbench, 2026-05-12:

```text
dirty/reuse full_stack_update:
  B16/P2/L1024 bonus4: 53.4ms/update
  B16/P2/L4096 bonus4: 56.9ms/update

independent gray64 render only:
  B16/P2/L1024 bonus4: 29.0ms per player-view call
  B16/P2/L4096 bonus4: 107.8ms per player-view call
```

Plain read: the dirty/reuse path fixed the old trail-length scaling. The
remaining browser-lines cost is fixed per-row composition/downsample/overlay
work, not a simple redraw-all-trail-points problem.

## Do Not Touch Live Path Until

Before another production-path optimization lands, get one or more of:

- Fine-grained profile that shows the next dominant component after
  `fast_gray64_direct`.
- Isolated microbench that predicts a real whole-loop improvement.
- Evidence that the live Coach matrix is blocked by a setup/performance bug.

Allowed now:

- Read-only code/doc review.
- Modal run monitoring. For overnight Coach runs this means read-only status,
  progress, logs, checkpoints, and summaries only; no restarts, no forced evals,
  no GIF triggers, no checkpoint edits, and no Volume mutations.
- Isolated local or Modal profiling experiments.
- Docs updates.
- Small validation-only changes if existing code is broken.

## Next Unknowns

1. How much of `policy_collect_forward_sec` is LightZero tree work, model
   inference, Python adapter overhead, tensor movement, and CUDA sync?
2. How much of the remaining observation-noise bucket can be removed by
   implementation cleanup without changing Coach's noise setting?
3. Which self-play width actually improves replay rows/sec and checkpoint
   learning signal per wall clock?
4. Does H100 become worth it only at B128/B256 or higher search?
5. Does observation noise or action-repeat/no-op stochasticity add material CPU
   overhead?
6. Are checkpoint eval/GIF artifacts safely amortized at the recommended
   cadence?
7. Are current telemetry timers fine-grained enough, or are they hiding the next
   real Amdahl term?

New timer splits for future two-seat runs, 2026-05-12:

- `policy_tensor_prepare_sec`: NumPy coercion, action-mask prep, `to_play`, and
  tensor move to the policy model device.
- `policy_collect_forward_sec`: the LightZero `MuZeroPolicy.collect_mode.forward`
  call. This is the main MCTS/search/model bucket.
- `policy_output_decode_sec`: per-row output extraction, action extraction, and
  compact MCTS metadata.
- `policy_batch_fallback_sec`: serial fallback if batched collect fails.
- `learner_sample_sec`: replay sampling and NumPy batch assembly for one or
  more learner updates in an iteration.
- `learner_update_total_sec`: wall time around the learner update call.
- `learner_batch_build_sec`: adapter work to build LightZero learner batches.
- `learner_forward_sec`: `MuZeroPolicy.learn_mode.forward`.
- `model_hash_sec`: only present when explicit model-hash verification is on.
- `checkpoint_save_sec`, `progress_write_sec`, and `progress_commit_sec`: outer
  loop artifact costs.

Keep `policy_search_sec` as the total old bucket. Use these sub-buckets only to
decide what isolated optimizer experiment comes next.

Immediate optimizer plan after replay-cache:

1. Run a tight search/collect isolation sweep, not another broad training
   matrix: B32/B64/B128 at sim8, plus B64 at sim16 and sim32, eval/GIF off,
   enough warm-up to ignore first iterations.
2. Run one matched observation-noise pair after replay-cache:
   `observation_noise_std=0.10` versus `0.0`, same B64/sim8 settings, to price
   the remaining CPU noise term. Do not silently turn noise off for Coach.
3. Run one browser-lines no-death sentinel only if Environment changes rendering
   again or Coach needs a fidelity stress read. Keep it separate from the
   fast-direct training Amdahl story.
4. Do not pursue multi-GPU, GPU renderer, or distributed actor rewrites until
   these tighter profiles show a bottleneck large enough to pay for the
   complexity.

Validation smoke, 2026-05-12:

```text
opt-searchsplit-smoke-20260512 / searchsplit-smoke-20260512
B4, sim2, fast_direct, 2 iterations, no checkpoint, background eval/GIF off
ok=true, lightzero_policy_model_device=cuda:0

first iteration policy_collect_forward_sec=1.059181
last iteration  policy_collect_forward_sec=0.040710
last iteration  policy_search_sec=0.046144
last iteration  policy_tensor_prepare_sec=0.000641
last iteration  policy_output_decode_sec=0.004397
```

Plain read: GPU/model path is active for this smoke, and warm-up is real. Do not
use first-iteration timing for steady-state decisions.

Matched decode profiles, 2026-05-12:

```text
Before decode fix:
  opt-searchsplit-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct
  elapsed=332.7s
  policy_search_sec sum=195.4s
  policy_collect_forward_sec sum=50.8s
  policy_output_decode_sec sum=142.7s
  visual_stack_update_sec sum=16.7s

After first decode fix:
  opt-decodefix-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct
  elapsed=291.8s
  policy_search_sec sum=85.6s
  policy_collect_forward_sec sum=73.0s
  policy_output_decode_sec sum=9.6s
  visual_stack_update_sec sum=24.6s
```

Plain read: the worst part of `policy_output_decode_sec` was repeated whole-batch
conversion of LightZero output into plain Python. The first fix converts once per
batch and reuses the plain object per row. This is not an algorithm change.

Follow-up code now also avoids storing full diagnostic `compact_output` in the
batched hot path. It preserves the selected action, visit-count policy target,
and root value directly on the search record. Focused local validation passed:

```text
uv run ruff check src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  tests/test_curvytron_two_seat_render_mode.py
uv run pytest tests/test_curvytron_two_seat_render_mode.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py -q

51 passed, 1 skipped
```

Lean decode verification:

```text
opt-decodelean-b32-sim8-20260512
  B32, sim8, 12 iterations, collect only, fast_direct
  elapsed=169.3s
  policy_search_sec sum=52.8s
  policy_collect_forward_sec sum=48.8s
  policy_output_decode_sec sum=2.3s
  observation_noise_sec + replay_observation_noise_sec sum=36.1s
  loop_autoreset_sec sum=32.6s
  visual_stack_update_sec sum=12.1s

opt-decodelean-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct
  elapsed=264.4s
  policy_search_sec sum=77.0s
  policy_collect_forward_sec sum=69.5s
  policy_output_decode_sec sum=4.1s
  observation_noise_sec + replay_observation_noise_sec sum=65.2s
  loop_autoreset_sec sum=48.3s
  visual_stack_update_sec sum=21.8s
```

Plain read: decode is no longer a major Amdahl term for the fast-direct profile.
The next large non-algorithmic term is default observation noise generation.
That knob is currently `0.10`, so it is not a bogus zero-noise copy. A follow-up
patch now generates float32 noise directly and clips in-place. Open profiles:
`opt-noisef32-b64-sim8-20260512` for default noise and
`opt-nonoise-b64-sim8-20260512` as the no-noise Amdahl bound.

Noise profile results:

```text
opt-noisef32-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct, observation_noise_std=0.10
  elapsed=170.3s
  observation_noise_sec + replay_observation_noise_sec sum=35.2s
  policy_search_sec sum=52.9s
  loop_autoreset_sec sum=33.2s
  visual_stack_update_sec sum=16.6s

opt-nonoise-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct, observation_noise_std=0.0
  elapsed=129.5s
  observation_noise_sec + replay_observation_noise_sec sum=0.005s
  policy_search_sec sum=52.2s
  loop_autoreset_sec sum=31.1s
  visual_stack_update_sec sum=16.0s
```

Plain read: the float32 noise patch is a real speedup and keeps the default
noise setting. Turning noise off is much faster, but that is a training-quality
decision for Coach, not a silent optimizer default.

Reset-row patch, 2026-05-12: `_refresh_reset_rows_in_visual_stack` no longer
creates a new full-batch stack and renders all rows just to refresh reset rows.
`SourceStateGray64Stack4.reset_rows(...)` now clears and rerenders only masked
rows, resets row-local browser-line/dirty-render caches, and leaves live rows'
FIFO stack intact. Focused validation passed:

```text
uv run ruff check src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  tests/test_curvytron_two_seat_render_mode.py
uv run pytest tests/test_curvytron_two_seat_render_mode.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py -q

53 passed, 1 skipped
```

Open verification: `opt-resetrow-b64-sim8-20260512` profiles lean decode,
float32 noise, and row-only reset refresh together.

Reset-row profile first read:

```text
opt-resetrow-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct, observation_noise_std=0.10
  elapsed=217.6s
  loop_autoreset_sec sum=20.4s
  observation_noise_sec + replay_observation_noise_sec sum=52.3s
  policy_search_sec sum=75.0s
  visual_stack_update_sec sum=21.1s
```

Plain read: the intended reset bucket improved versus the float32-noise baseline
(`33.2s` -> `20.4s`), but total wall clock got worse because unrelated policy
and noise buckets were slower in that Modal run. Do not sell this as a global
speedup yet. Open isolation profile: `opt-resetrow-nonoise-b64-sim8-20260512`.

Reset-row no-noise isolation:

```text
opt-resetrow-nonoise-b64-sim8-20260512
  B64, sim8, 12 iterations, collect only, fast_direct, observation_noise_std=0.0
  elapsed=125.1s
  loop_autoreset_sec sum=14.4s
  policy_search_sec sum=55.7s
  visual_stack_update_sec sum=17.0s
```

Plain read: the reset refresh change is a real targeted improvement. Against
the no-noise bound, it cuts `loop_autoreset_sec` from about `31.1s` to `14.4s`
and moves wall time from about `129.5s` to `125.1s`. The wall-clock gain is
modest because search and render still remain.

Correctness review and extra tests:

```text
uv run ruff check src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  tests/test_curvytron_two_seat_render_mode.py
uv run pytest tests/test_curvytron_two_seat_render_mode.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py -q

54 passed, 2 skipped
```

Known caveat: float32 noise is distribution-equivalent but not bit-identical to
the old float64-normal-then-cast stream for fixed seeds. Treat old/new fixed-seed
trajectory comparisons accordingly.

Stack no-copy hot path, 2026-05-12: `SourceStateGray64Stack4.update(...)` and
`reset_rows(...)` keep their copy-safe public default, but the two-seat trainer
now calls them with `copy=False`. The trainer already copies active policy rows
into the replay row, so this should remove a full `[B,P,4,64,64]` stack copy
from each hot collection step without changing pixels. Local validation remains:
`54 passed, 2 skipped`.

No-copy profiles:

```text
opt-stacknocopy-nonoise-b64-sim8-20260512
  elapsed=118.4s
  prior no-noise reset-row baseline=125.1s

opt-stacknocopy-noise-b64-sim8-20260512
  elapsed=154.4s
  prior float32-noise/reset-row best comparable signal=170.3s, with Modal
  variance caveat because the reset-row default-noise run was noisy
```

Plain read: this is worth keeping, but it is not the next big Amdahl target.
For `fast_gray64_direct`, next large terms are LightZero collect/search and
default observation noise. For `browser_lines` long-survival, render remains the
large term.

## Active Experiment Queue

These should be piecewise before live-path edits:

- Stock frozen CPU/base profile as denominator.
- Stock frozen L4/base profile matching the passed canary.
- Stock frozen L4/subprocess profile with `opponent_use_cuda=false`.
- Stock frozen wider subprocess collector profiles, raising `collector_env_num`
  and `n_episode` together.
- Long-survival `profile_no_death` stock frozen profile to see whether
  browser-lines render/env terms dominate once episodes stop ending quickly.
- Larger batch/search probes only after the stock subprocess path is stable.

Safe monitoring rule for live Coach runs:

- Build run and attempt lists from launch logs because early overnight rows used
  an older run-id shape.
- Use `lightzero_curvytron_run_status` and direct reads of `progress_latest.json`
  for status. These are read-only.
- Avoid running launch scripts, training entrypoints, eval/GIF/subscriber
  backfills, cleanup scripts, or any Modal Volume mutation against live runs.

## Documentation Rule

Keep this file short. Put detailed evidence in focused docs:

- Current status: `current_status_2026-05-09.md`
- Historical Coach handoff tombstone:
  `coach_next_training_run_recommendations_2026-05-12.md` was the old custom
  `--mode two-seat-selfplay` matrix, not current trusted guidance.
- Render details: `render_optimization_research_2026-05-12.md`
- Open questions: `questions.md`
- Small task list: `backlog.md`

When facts change, update the highest-level file first, then the detailed file.
