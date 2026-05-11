# 2026-05-09 Dummy Survival Unknown Next Value

## Question

If unknown next states are treated as bad instead of neutral, do later survival
checkpoints stop getting worse?

## Setup

- Code change: `scripts/run_dummy_survival_train.py` now accepts
  `--planner-unknown-next-value`.
- Test value: `-1.0`.
- Training: 8 iterations, 20 episodes per iteration, seed 0.
- Eval sweep: 20 episodes, seed 123.

## Command

```sh
PYTHONPATH=src python3 -m py_compile \
  src/curvyzero/training/dummy_survival.py \
  scripts/run_dummy_survival_train.py

PYTHONPATH=src python3 scripts/run_dummy_survival_train.py \
  --iterations 8 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_unknown_pessimism_i8_e20_seed0_c2 \
  --checkpoint-every-iterations 2 \
  --planner-unknown-next-value -1.0

PYTHONPATH=src python3 scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_unknown_pessimism_i8_e20_seed0_c2/checkpoints \
  --episodes 20 \
  --seed 123 \
  --split-id dummy_survival_monitor_v0 \
  --output-dir artifacts/local/dummy_survival_unknown_pessimism_sweep_seed123_e20
```

## Results

- Final training eval survival: 40%.
- Best checkpoint: `iteration-0002`, survival 55%.
- Latest checkpoint: `iteration-0008`, survival 45%.
- `untrained_model_same_planner`: survival 100%.
- `one_step_safe`: survival 100%.

## Interpretation

This did not fix survival learning. The learned checkpoints still lose to the
empty planner on the same eval seeds. Keep this flag as a diagnostic option,
not as a main training path.

This supports the current plan: pause survival as a core target and move the
main training work toward Pong.

## Artifacts

- `artifacts/local/dummy_survival_unknown_pessimism_i8_e20_seed0_c2/summary.json`
- `artifacts/local/dummy_survival_unknown_pessimism_i8_e20_seed0_c2/checkpoints/`
- `artifacts/local/dummy_survival_unknown_pessimism_sweep_seed123_e20/summary.json`
- `artifacts/local/dummy_survival_unknown_pessimism_sweep_seed123_e20/checkpoint_eval.jsonl`

## Follow-ups

- Do not spend more main-thread time on survival unless it directly helps Pong,
  Modal training runs, or the eval system.
- If survival is revisited, compare every result to
  `untrained_model_same_planner`.
