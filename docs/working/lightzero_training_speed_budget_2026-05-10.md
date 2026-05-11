# LightZero Pong Training Speed Budget - 2026-05-10

Purpose: one practical place for current LightZero official/control Pong timing
data. These are rough operating numbers for polling, eval scheduling, and next
experiment size. They are not a stable benchmark.

## Claim

Known Modal runs say official/control LightZero Atari Pong is slow enough that
checkpoint/eval work should be planned in minutes and training waves in hours.
Single H100 is faster than L4/T4, but current profiles still look dominated by
LightZero evaluator/collector/env/MCTS wall time, not sustained GPU learning
throughput.

## Non-claim

This does not prove H100 is cost-effective for Pong, that 4x H100 would help,
or that these speeds transfer to CurvyTron. It also does not prove quality;
stock return and same-run eval still decide whether a checkpoint improved.

## Sources Checked

- `docs/working/lightzero_compute_scaling_2026-05-10.md`
- `docs/working/lightzero_parallel_run_matrix_2026-05-10.md`
- `docs/working/training_coach_active_board_2026-05-10.md`
- `docs/working/lightzero_modal_background_launch_patterns_2026-05-10.md`
- Modal Volume snapshots fetched locally under
  `artifacts/local/lightzero-training-timing/volume-snapshots/`
- Local eval manifests under `artifacts/local/lightzero-eval-manifests/`

## Short Profiler

Synchronous profiler runs stopped after 5 `BaseLearner.train` calls:

| Compute | Actual GPU | Remote elapsed | `train_muzero` wall | Eval | Collect | 5 learner calls | Max GPU util |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpu-l4-t4` | NVIDIA L4 | 266.1s | 251.6s | 101.4s | 140.8s | 1.62s | 16% |
| `gpu-h100` | NVIDIA H100 80GB HBM3 | 224.0s | 205.5s | 87.4s | 98.6s | 1.24s | 29% |
| `gpu-l4-t4-cpu16` | NVIDIA L4 | 244.5s | 237.0s | 92.4s | profile run | 1.53s | not summarized |
| `gpu-h100-cpu16` | NVIDIA H100 80GB HBM3 | 163.9s | 148.6s | 57.7s | profile run | 1.24s | not summarized |

Read: H100 was about 16% faster on remote elapsed and about 18% faster inside
`train_muzero` for this tiny profiled slice. The learner train bucket itself was
only about 1-2 seconds. The big buckets were evaluator and collector time.
The first L4/T4 CPU16 profile was about 8% faster than the earlier L4/T4
profile on remote elapsed (`266.1s -> 244.5s`). That is helpful but not a
breakthrough. It says more CPU helps some, but independent parallel runs and
parallel eval calls are still the larger practical lever.

The first H100 CPU16 profile was much faster than the earlier H100 profile
(`224.0s -> 163.9s` remote elapsed). Plain read: for this short slice,
extra CPU plus H100 helped the evaluator/collector-heavy path. This still
does not prove long-run quality or multi-GPU usefulness; it only says
`gpu-h100-cpu16` is a good candidate for faster stock/control Pong sweeps.

Useful caveat: the phase timers are inclusive wrappers. They split
collect/eval/replay/learner/checkpoint wall time, but do not yet split MCTS tree
bookkeeping from model inference inside collection/eval.

## CPU Parallelism Knob Inventory

Stock/control exact wrapper:

| Knob | Current stock/control value | Where |
| --- | ---: | --- |
| Modal train CPU count | `8.0` for `gpu-l4-t4`, `gpu-h100`, `gpu-h100x4` | `lightzero_pong_exact_reproduction.py` Modal decorators |
| Modal eval CPU count | `2.0` for `gpu-l4-t4` eval | `lightzero_pong_eval_smoke.py` Modal decorator |
| Env manager | `subprocess` | installed LightZero Atari config surface |
| Collector envs | `8` | `env.collector_env_num`, `policy.collector_env_num`, `policy.n_episode` |
| Evaluator envs | `3` | `env.evaluator_env_num`, `policy.evaluator_env_num`, `env.n_evaluator_episode` |
| MCTS sims | `50` | `policy.num_simulations` |
| Learner batch size | `256` | `policy.batch_size` |
| Game segment length | `400` | `policy.game_segment_length` |
| Update per collect | `None` | stock/control train path |
| Replay ratio | `0.25` | stock/control train path |
| Eval frequency | `2000` | stock/control train path |

Tiny/debug wrappers are intentionally different and should not be treated as
stock/control: dry defaults are `collector_env_num=1`,
`evaluator_env_num=1`, `num_simulations=2`, `batch_size=4`, and
`update_per_collect=1`; tiny train defaults are capped at 4 collectors, 1
evaluator, 25 sims, and batch 64.

Eval wrapper caveat: the strict eval helper rebuilds config for checkpoint
loading and stock/manual rollout telemetry. Its default CLI values come from
the tiny helper unless callers pass the stock/control values; live queue
commands already pass `collector_env_num=8`, `evaluator_env_num=3`,
`num_simulations=50`, `batch_size=256`, and should use
`--update-per-collect -1` for stock-like config drift checks when relevant.

## CPU Throughput Lane Status

Historical CPU throughput notes below are still useful for reading old
artifacts, but current eval waves should default to `gpu-l4-t4-cpu40`.
Modal rejected CPU64 in this workspace. CPU8/CPU16 eval guidance is stale
unless it is explicitly an old-artifact comparison.

Code exposes resource-only Modal lanes that do not mutate LightZero
collector/evaluator/search/batch config:

- Train: `--compute gpu-l4-t4-cpu16` uses L4/T4 with `cpu=16.0`.
- Train: `--compute gpu-h100-cpu16` uses H100 with `cpu=16.0`.
- Historical eval: `--compute gpu-l4-t4-cpu8` uses L4/T4 eval with `cpu=8.0`.
- Current eval default: `--compute gpu-l4-t4-cpu40`.

These are throughput/resource variants, not new stock/control quality lanes.
Run ids and attempt ids should include `cpu16`, `cpu8`, or `cpu40` so timing
cannot be mixed with earlier train/eval numbers.

First safe profiling commands:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train --compute gpu-l4-t4-cpu16 --seed 0 \
  --run-id lz-visual-pong-cpu-throughput-profile-s0 \
  --attempt-id profile-l4-t4-cpu16-traincalls5-sync-20260510 \
  --progress-interval-sec 15 \
  --profile-phases --gpu-sample-interval-sec 2 \
  --profile-stop-after-learner-train-calls 5 \
  --wait-for-train
```

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train --compute gpu-h100-cpu16 --seed 0 \
  --run-id lz-visual-pong-cpu-throughput-profile-s0 \
  --attempt-id profile-h100-cpu16-traincalls5-sync-20260510 \
  --progress-interval-sec 15 \
  --profile-phases --gpu-sample-interval-sec 2 \
  --profile-stop-after-learner-train-calls 5 \
  --wait-for-train
```

Eval CPU probe shape:

```text
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run-id> --attempt-id <attempt-id> \
  --eval-id cpu40-eval-throughput-probe-stock2048-seed<seed> \
  --compute gpu-l4-t4-cpu40 --eval-pass custom --seed <seed> \
  --max-eval-steps 2048 --max-episode-steps 2048 \
  --max-env-step <train-env-step-cap> \
  --selected-iterations 0,10000 \
  --group-size 1 --max-parallel-launches 2 --execute
```

Use lower-CPU eval lanes only as an A/B against the same checkpoints and same
eval cap already measured with `gpu-l4-t4-cpu40`; the quality rows should match
and only elapsed time should move.

## Safe First Changes

1. Compare Modal CPU allocation before changing LightZero config. This is the
   cleanest throughput test because it leaves env count, MCTS sims, batch size,
   replay ratio, and objective untouched.
2. Use `gpu-l4-t4-cpu16` or `gpu-h100-cpu16` only for a short profiler capped
   after 5 learner calls, then one 32768-env-step timing run if the profiler
   improves collector/evaluator wall time.
3. Use `gpu-l4-t4-cpu40` for current eval throughput. Use `gpu-l4-t4-cpu8`
   only when reading or reproducing old CPU-throughput comparisons, so timing
   changes can be checked without changing the comparison target.
4. Do not raise `collector_env_num` or `evaluator_env_num` inside the
   stock/control lane yet. Changing 8/3 changes collection/eval cadence and
   may alter training dynamics; if tested, label it a separate throughput
   ablation such as `collector16-evaluator6`.
5. Do not reduce `num_simulations` from 50 for official/control timing. That
   changes MCTS strength and experiment meaning. Lower sims are fine for debug
   probes only.

## Completed Training Runs

These are completed spawned background train attempts with Volume
`train/summary.json` and/or `train/progress/latest.json` fetched.

| Run | Compute | Max env step | Ckpt cadence | Remote elapsed | Train elapsed | Latest checkpoint | Checkpoint files | Rough sec / 1k learner iter |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| seed 0 repeat A | `gpu-l4-t4` | 32768 | 1000 | 3133.8s / 52.2m | 3121.7s / 52.0m | `iteration_9559` | 12 | 327s |
| seed 1 | `gpu-l4-t4` | 65536 | 1000 | 6416.1s / 1.78h | 6403.8s / 1.78h | `iteration_18459` | 21 | 347s |
| seed 2 | `gpu-l4-t4` | 65536 | 1000 | 5345.3s / 1.48h | 5333.2s / 1.48h | `iteration_16829` | 19 | 317s |
| seed 3 | `gpu-l4-t4` | 65536 | 1000 | 7776.2s / 2.16h | 7761.7s / 2.16h | `iteration_17010` | 20 | 456s |
| seed 1 H100 | `gpu-h100` | 65536 | 1000 | 4363.9s / 1.21h | 4347.8s / 1.21h | `iteration_17504` | 20 | 248s |

Rough read:

- L4/T4 65536-env-step runs have landed around 1.5-2.2 hours, with large
  seed/run variance.
- The same seed 1 H100 run landed around 1.2 hours, about 32% less wall time
  than seed 1 L4/T4. This is stronger than the short profiler delta, but still
  one paired-seed diagnostic.
- A 32768-env-step L4/T4 repeat landed around 52 minutes and stopped near
  `iteration_9559`.
- Checkpoints are large: roughly 96 MB each. A 1000-cadence 65536 run wrote
  about 1.8-1.9 GB of checkpoint files.

## Checkpoint Cadence Observed

Observed pattern with `--save-ckpt-after-iter-override 1000`:

- `iteration_0.pth.tar` appears early.
- Normal checkpoints appear every 1000 learner iterations.
- A final/latest non-round checkpoint appears when training stops, for example
  `iteration_9559`, `iteration_16829`, `iteration_17010`, `iteration_17504`,
  or `iteration_18459`.
- `ckpt_best.pth.tar` may appear separately and should not be treated as final
  without same-run eval.

For the active long H100 stock wave with `save_ckpt_after_iter=5000`, expect
coarser but cheaper checkpoint accounting: first useful checkpoint at
`iteration_5000`, then `10000`, `15000`, etc., plus final/latest.

## Eval Timing

All eval timings below are from `gpu-l4-t4` eval manifests.

512-step cap:

- Typical per-checkpoint remote elapsed: about 145-152s.
- Episode bucket: about 72-76s.
- Two-checkpoint 512 evals therefore feel like a few minutes when launched in
  parallel.

2048-step cap:

- Flat weak rows surviving about 761-882 steps usually took about 210-232s per
  checkpoint.
- Longer-survival rows took longer: `iteration_17010` at `1236/2048` took
  about 289s; `iteration_16000` at `1605/2048` took about 390s.
- Six-checkpoint compact parallel evals started all workers together. The seed
  2 compact batch finished in about 4 minutes; the seed 3 compact batch, with a
  longer-survival row, finished in about 6.5 minutes.

Read: use 2048-step eval for meaningful survival checks, but keep compact
checkpoint sets. It is fast enough to run in parallel, expensive enough that
evaluating every 1000-step checkpoint is mostly noise unless a run is under
active diagnosis.

## Polling Budget

For 1000-cadence L4/T4 runs:

- Polling more often than every 3-5 minutes usually gives little new signal.
- The wrapper progress interval has been 120s or 300s. Match that unless a
  checkpoint is expected imminently.
- Once `iteration_1000` appears, evaluate `iteration_0` versus `iteration_1000`
  if the run is a new lane/config. After that, prefer compact curve points:
  `5000`, `10000`, `16000`, latest/final, and any obvious quality pivot.

For 5000-cadence long H100 runs:

- Poll every 5-10 minutes.
- First useful eval trigger is `iteration_5000`.
- Then evaluate `10000`, `20000`, latest/final, and any later checkpoint near a
  visible stock-return improvement.

For eval polling:

- A 2048 compact parallel eval should usually settle within 5-8 minutes.
- If a row survives much longer than prior rows, allow extra minutes before
  assuming the eval is stuck.

## Next Experiment Size

Practical recommendation:

- Keep L4/T4 for cheap control waves and debugging. A 32768 run is roughly a
  one-hour question; a 65536 run is roughly a two-hour question.
- Use single H100 only when wall-clock matters, especially for 200k stock
  undertraining tests. Current rough extrapolation from the seed 1 H100 run is
  about 3.5-4 hours for a 200k-env-step run if scaling is close to linear.
- Do not use `gpu-h100x4` yet. Current code is a single trainer process and the
  profiler does not show enough GPU pressure to justify multi-GPU spend.
- Prefer fewer, longer stock/control runs over many tiny variants until the
  stock eval path has clearer return movement. Good next stock scale is 1-2
  seeds at 200k with checkpoint cadence 5000, not a broad 8-seed sweep.
- Keep shaped-reward ablations separate and smaller. For shaped diagnostics,
  32768 with 1000-cadence is enough to decide whether the shaping changes
  survival/return behavior before spending on long runs.

## Unknowns

- The long H100 stock runs listed on the active board did not have visible
  `train/progress/latest.json` or attempt directories at the checked refs when
  this doc was written. Treat their timing as unknown until Volume progress
  appears.
- We do not yet have a deep split of MCTS tree work versus model inference.
- We do not know whether 200k-env-step runs keep the same seconds-per-learner
  iteration as the 32768/65536 runs.
- Eval still has manual-vs-stock divergence on many rows. Stock return remains
  primary official quality; manual survival is a labeled diagnostic until
  stock-path survival is instrumented.
