# 2026-05-09 Dummy Survival Checkpoint Sweep Smoke

## Question

Can dummy survival save periodic checkpoints and use existing learned-checkpoint
eval loading to select a better mid-run artifact than the final checkpoint?

## Setup

- Trainer: `scripts/run_dummy_survival_train.py`
- Sweep: `scripts/run_dummy_survival_checkpoint_sweep.py`
- Training: 8 iterations, 20 episodes per iteration, seed 0
- Periodic checkpoint cadence: every 2 completed iterations
- Eval baselines: `random_uniform`, `one_step_safe`
- Learned-checkpoint sweep: 10 episodes per policy, seed 123

## Command

```sh
uv run python scripts/run_dummy_survival_train.py \
  --iterations 8 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2 \
  --checkpoint-every-iterations 2

uv run python scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/checkpoints \
  --episodes 10 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2_eval_seed123_e10
```

## Results

- Smoke completed successfully.
- Periodic checkpoints were written for completed iterations 2, 4, 6, and 8.
- Each periodic checkpoint metadata includes both zero-based `iteration` and
  one-based `completed_iterations`.

| Policy | Crash rate | Survival rate | Mean steps | Max steps |
| --- | ---: | ---: | ---: | ---: |
| `random_uniform` | 1.0 | 0.0 | 8.8 | 13 |
| `one_step_safe` | 0.0 | 1.0 | 80.0 | 80 |
| `learned:iteration-0004` | 1.0 | 0.0 | 20.8 | 22 |
| `learned:iteration-0006` | 1.0 | 0.0 | 15.0 | 16 |
| `learned:iteration-0008` | 1.0 | 0.0 | 13.0 | 13 |
| `learned:iteration-0002` | 1.0 | 0.0 | 11.2 | 14 |

Best checkpoint by the sweep rank was `iteration-0004.npz`, with mean steps
20.8 on the fixed seed-123 eval. The final periodic checkpoint,
`iteration-0008.npz`, scored 13.0 mean steps on the same eval.

## Interpretation

The checkpoint-sweep lane works and captures artifact evidence that the final
checkpoint is not necessarily the best artifact for this dummy learner. This
does not make the learned policy good yet: every learned checkpoint still
crashes every episode and remains far below `one_step_safe`.

## Artifacts

- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/summary.json`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/checkpoint.npz`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/iteration_metrics.jsonl`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/checkpoints/iteration-0002.npz`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/checkpoints/iteration-0004.npz`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/checkpoints/iteration-0006.npz`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2/checkpoints/iteration-0008.npz`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2_eval_seed123_e10/summary.json`
- `artifacts/local/dummy_survival_checkpoint_sweep_i8_e20_seed0_c2_eval_seed123_e10/checkpoint_eval.jsonl`

## Follow-ups

- Use a larger eval episode count before treating checkpoint selection as
  stable.
- Keep this as a local artifact-evidence lane until the main training thread
  decides whether the sweep command should become canonical.
