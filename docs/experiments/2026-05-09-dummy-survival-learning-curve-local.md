# 2026-05-09 Dummy Survival Learning Curve Local

## Question

Does more local training data improve learned-checkpoint eval for the existing
dummy survival learner?

## Setup

- Trainer: `scripts/run_dummy_survival_train.py`
- Evaluator: `scripts/run_dummy_survival_eval.py`
- Training runs: 20 iterations, 50 episodes per iteration
- Training seeds: 0 and 1
- Eval baselines: `random_uniform`, `one_step_safe`
- Learned-checkpoint eval: 10 episodes, seed 123, matching the checkpoint smoke
  comparison seed
- Prior checkpoint smoke reference: learned crash rate 1.0 and mean steps 25.5
  on 10 episodes with seed 123

## Command

```sh
uv run python scripts/run_dummy_survival_train.py \
  --iterations 20 \
  --episodes-per-iter 50 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_learning_curve_i20_e50_seed0

uv run python scripts/run_dummy_survival_eval.py \
  --episodes 10 \
  --seed 123 \
  --policies random_uniform one_step_safe \
  --checkpoint-policy learned:artifacts/local/dummy_survival_learning_curve_i20_e50_seed0/checkpoint.npz \
  --output-dir artifacts/local/dummy_survival_learning_curve_i20_e50_seed0_eval_seed123_e10

uv run python scripts/run_dummy_survival_train.py \
  --iterations 20 \
  --episodes-per-iter 50 \
  --seed 1 \
  --output-dir artifacts/local/dummy_survival_learning_curve_i20_e50_seed1

uv run python scripts/run_dummy_survival_eval.py \
  --episodes 10 \
  --seed 123 \
  --policies random_uniform one_step_safe \
  --checkpoint-policy learned:artifacts/local/dummy_survival_learning_curve_i20_e50_seed1/checkpoint.npz \
  --output-dir artifacts/local/dummy_survival_learning_curve_i20_e50_seed1_eval_seed123_e10
```

## Results

Training summary:

| Run | States | Dynamics edges | Final train eval crash rate | Final train eval mean steps |
| --- | ---: | ---: | ---: | ---: |
| seed 0, 20x50 | 1094 | 3360 | 1.0 | 12.4 |
| seed 1, 20x50 | 1096 | 3460 | 1.0 | 26.2 |

Fixed seed-123 learned-checkpoint eval:

| Policy | Crash rate | Survival rate | Mean steps | Max steps |
| --- | ---: | ---: | ---: | ---: |
| `random_uniform` | 1.0 | 0.0 | 8.8 | 13 |
| `one_step_safe` | 0.0 | 1.0 | 80.0 | 80 |
| `learned:dummy_survival_learning_curve_i20_e50_seed0/checkpoint` | 1.0 | 0.0 | 10.0 | 10 |
| `learned:dummy_survival_learning_curve_i20_e50_seed1/checkpoint` | 1.0 | 0.0 | 25.0 | 25 |

The baseline rows are identical across both eval runs because the same eval seed
and baseline policies were used.

## Interpretation

More local training does not appear to move the dummy learner in a useful
direction yet. Both larger checkpoints still crash every eval episode. Seed 0
regressed versus the prior 5x20 checkpoint smoke, dropping from 25.5 to 10.0
mean learned steps on the same 10-episode seed-123 eval. Seed 1 roughly matched
the prior smoke at 25.0 mean steps, but still did not improve crash rate,
survival rate, or approach `one_step_safe`.

The per-iteration training evals are noisy and sometimes reach higher mean step
counts mid-run, but the saved final checkpoints do not show a monotonic
learning-curve improvement. This suggests the next useful experiment is not just
"more episodes" with the current dummy learner; it is either checkpoint
selection over periodic checkpoints or a small trainer/planner fix after a
focused inspection.

## Artifacts

- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed0/summary.json`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed0/checkpoint.npz`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed0/iteration_metrics.jsonl`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed0_eval_seed123_e10/summary.json`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed0_eval_seed123_e10/episodes.jsonl`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed1/summary.json`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed1/checkpoint.npz`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed1/iteration_metrics.jsonl`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed1_eval_seed123_e10/summary.json`
- `artifacts/local/dummy_survival_learning_curve_i20_e50_seed1_eval_seed123_e10/episodes.jsonl`

Artifact sizes were small: each training run directory was about 84K and each
eval directory was about 16K.

## Follow-ups

- If this lane continues, save/evaluate periodic checkpoints instead of only
  the final checkpoint so mid-run variance is visible in the learned-checkpoint
  table.
- Inspect why the learner still records no survivals under more data before
  increasing local training size again.
