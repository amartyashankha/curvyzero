# LightZero Official Pong Compute Scaling

Date: 2026-05-10

## Claim

The first direct L4 vs H100 profile says official/control LightZero Atari Pong is
not mainly GPU-bound at this scale.

In a run stopped after 5 `BaseLearner.train` calls, most time went into
evaluator and collector work. The GPU learner section was small:

| Compute | Actual GPU | Remote elapsed | `train_muzero` wall | Eval | Collect | 5 learner calls | Max GPU util |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpu-l4-t4` | NVIDIA L4 | 266.1s | 251.6s | 101.4s | 140.8s | 1.62s | 16% |
| `gpu-h100` | NVIDIA H100 80GB HBM3 | 224.0s | 205.5s | 87.4s | 98.6s | 1.24s | 29% |

H100 was faster, especially in collection/eval wall time, but the profile still
looks dominated by LightZero evaluator/collector/env/MCTS loop work rather than
bulk neural training throughput. Spending on H100 for long official Pong runs is
not justified yet unless the next matrix shows this 15-20% wall-clock gain is
worth the price.

## Non-claim

- This is not a full 200k-env-step training result.
- This is not a learning-quality claim.
- This is not proof that all MuZero workloads are CPU-bound. It is only this
  official LightZero Pong wrapper and config.
- The phase profiler is an inclusive wrapper. It splits evaluator, collector,
  replay sample, learner train, checkpoint, and GPU samples, but it does not yet
  split MCTS tree bookkeeping from model inference inside collection.
- No 4x H100 profile was launched. Current LightZero code is a single trainer
  process and does not obviously use multi-GPU data parallelism, so 4x H100 would
  likely waste money until proven otherwise.

## GPU Options Added

`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py` now supports:

- `cpu`: unchanged default for dry mode; train mode remains blocked.
- `gpu-l4-t4`: unchanged default GPU path, Modal resource `["L4", "T4"]`.
- `gpu-h100`: new single-H100 path, Modal resource `"H100"`.
- `gpu-h100x4`: new 4x-H100 path, Modal resource `"H100:4"`.

Local Modal notes already document that GPU counts use `:n`, with examples like
`H100:8`, so `H100:4` is not guessed from scratch. Still, do not run
`gpu-h100x4` for this workload until a single-GPU run proves real GPU pressure.

## Commands

Synchronous short profile command. `--wait-for-train` is for profiling only; the
old background-spawn behavior remains the default for `--mode train`.

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train --compute gpu-l4-t4 --seed 0 \
  --run-id lz-visual-pong-compute-scaling-profile-s0 \
  --attempt-id profile-l4-t4-traincalls5-sync-20260510 \
  --progress-interval-sec 15 \
  --profile-phases --gpu-sample-interval-sec 2 \
  --profile-stop-after-learner-train-calls 5 \
  --wait-for-train
```

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train --compute gpu-h100 --seed 0 \
  --run-id lz-visual-pong-compute-scaling-profile-s0 \
  --attempt-id profile-h100-traincalls5-sync-20260510 \
  --progress-interval-sec 15 \
  --profile-phases --gpu-sample-interval-sec 2 \
  --profile-stop-after-learner-train-calls 5 \
  --wait-for-train
```

Volume summaries:

- `training/lightzero-official-visual-pong/lz-visual-pong-compute-scaling-profile-s0/attempts/profile-l4-t4-traincalls5-sync-20260510/train/summary.json`
- `training/lightzero-official-visual-pong/lz-visual-pong-compute-scaling-profile-s0/attempts/profile-h100-traincalls5-sync-20260510/train/summary.json`

## Recommendation

Next run matrix:

1. Use `gpu-l4-t4` for the next real official/control Pong run.
2. Add one matching `gpu-h100` run only if wall-clock time is the main blocker
   and cost is acceptable.
3. Do not run `gpu-h100x4` yet. First add deeper profiler hooks around
   `MuZeroPolicy._forward_collect`, MCTS `search`, and env stepping, or build a
   direct one-collect/sample/train harness that skips evaluator startup.
4. If H100 is retested, keep the same seed and same profiler cap so the
   comparison remains interpretable.

## H100 Quality Read

The H100 seed-1 official/control run completed to latest/final
`iteration_17504` and now has a compact 2048-cap strict no-fallback stock
evaluator curve:
`docs/working/lightzero_parallel_run_matrix_2026-05-10.md#seed-1-h100-compact-latest-curve-eval`.

Claim: H100 produced a real but non-durable quality signal in this run. The best
compact checkpoint, `iteration_10000`, survived `1731/2048` (`+968` steps
versus same-run `iteration_0`), improved manual return from `-21` to `-19`,
improved stock return from `-21` to `-16`, and recorded `2` positive rewards.

Non-claim: this does not overturn the compute-scaling recommendation by itself.
Latest `iteration_17504` fell back near baseline at `782/2048`, manual and
stock return `-21`, with `0` positive rewards. H100 made the run finish sooner,
but this compact curve does not show durable solved-Pong quality or justify
multi-H100 spending.
