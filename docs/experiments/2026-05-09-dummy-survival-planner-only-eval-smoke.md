# 2026-05-09 Dummy Survival Planner-Only Eval Smoke

## Question

Does the survival evaluator separate learned-checkpoint behavior from the
hand-coded `DummyPlanner` safety prior?

## Setup

- Code change: `run_dummy_survival_eval.py` now includes
  `untrained_model_same_planner` in the default policy set.
- Policy behavior: instantiate a fresh empty `DummyMuZeroModel`, use the same
  `DummyPlanner` as learned checkpoints, and evaluate with `epsilon=0.0` and
  `explore_unknown=False`.
- Eval: 5 episodes, seed 0.

## Command

```sh
python3 -m py_compile \
  src/curvyzero/training/dummy_survival_eval.py \
  scripts/run_dummy_survival_eval.py \
  scripts/run_dummy_survival_checkpoint_sweep.py

PYTHONPATH=src python3 scripts/run_dummy_survival_eval.py \
  --episodes 5 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_eval_planner_only_smoke
```

## Results

- `random_uniform`: survival rate 0.0, crash rate 1.0, mean steps 8.4.
- `one_step_safe`: survival rate 1.0, crash rate 0.0, mean steps 80.0.
- `untrained_model_same_planner`: survival rate 1.0, crash rate 0.0, mean
  steps 80.0.

## Interpretation

This is the clearest eval correction so far: the safety-aware planner alone can
solve the tiny survival monitor split. Early learned checkpoints matching
`one_step_safe` should not be described as learned policy progress unless they
also beat or meaningfully differ from `untrained_model_same_planner` on a
separated selection/heldout split.

Checkpoint sweeps are still useful, but they now need to report planner-only
floors so degradation and apparent wins are not hidden.

## Artifacts

- `artifacts/local/dummy_survival_eval_planner_only_smoke/summary.json`
- `artifacts/local/dummy_survival_eval_planner_only_smoke/episodes.jsonl`

## Follow-ups

- Add policy execution metadata to every eval artifact so future readers can
  see planner config, epsilon, tie-breaks, and safety priors.
- Treat the old fixed seed-123 sweep as monitor/debug, not heldout evidence.
- Before larger runs, require checkpoint selection on one split and one
  confirmation pass on a heldout split.
