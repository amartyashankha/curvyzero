# 2026-05-09 Dummy Survival Safety Planner Smoke

## Question

Does a tiny planner safety mask fix the dummy survival learner enough to create
a real learned-checkpoint eval signal?

## Setup

- Code change: `DummyPlanner` now penalizes actions with zero immediate
  clearance and breaks ties by score, clearance, straight preference, then
  action id.
- Cleanup: removed unused/dead TD-target code from `DummyUpdater`.
- Trainer: `scripts/run_dummy_survival_train.py`
- Sweep: `scripts/run_dummy_survival_checkpoint_sweep.py`
- Training: 8 iterations, 20 episodes per iteration, seed 0
- Periodic checkpoint cadence: every 2 completed iterations
- Eval: 10 episodes, seed 123

## Command

```sh
python3 -m py_compile \
  src/curvyzero/training/dummy_survival.py \
  scripts/run_dummy_survival_train.py \
  scripts/run_dummy_survival_eval.py \
  scripts/run_dummy_survival_checkpoint_sweep.py

uv run python scripts/run_dummy_survival_train.py \
  --iterations 8 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2 \
  --checkpoint-every-iterations 2

uv run python scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 10 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2_eval_seed123_e10

uv run python scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 5 \
  --seed 124 \
  --output-dir artifacts/local/dummy_survival_safety_planner_manifest_smoke
```

## Results

- Smoke completed successfully.
- Fixed baselines remained stable:
  - `random_uniform`: crash rate 1.0, survival rate 0.0, mean steps 8.8.
  - `one_step_safe`: crash rate 0.0, survival rate 1.0, mean steps 80.0.
- Best learned checkpoint: `iteration-0002.npz`.
- `iteration-0002`: crash rate 0.0, survival rate 1.0, mean steps 80.0.
- `iteration-0004`: crash rate 0.0, survival rate 1.0, mean steps 80.0.
- `iteration-0006`: crash rate 0.9, survival rate 0.1, mean steps 46.7.
- `iteration-0008`: crash rate 0.9, survival rate 0.1, mean steps 66.5.

Training-run internal eval also showed 100% survival for iterations 1 through
4, then degradation later. Training rollouts still crashed because epsilon
exploration was active during collection.

## Interpretation

This was the first positive dummy survival checkpoint signal, but a later
planner-only eval corrected the interpretation: an untrained model with the
same `DummyPlanner` safety prior also matches `one_step_safe` on the tiny
monitor split. The early checkpoints are therefore checkpoint-plus-planner
signals, not clean evidence of learned policy progress.

The result also proves checkpoint selection matters. Later checkpoints degraded
after more crash-heavy replay and updates, so the final checkpoint should not
be assumed best.

Follow-up sweep with the default `untrained_model_same_planner` baseline:

```sh
PYTHONPATH=src python3 scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 10 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2_eval_seed123_e10_with_planner_only
```

That sweep showed `untrained_model_same_planner` at 100% survival on the same
10 seeds, while checkpoints 2 and 4 also reached 100% and checkpoints 6 and 8
degraded to 10%. This keeps the degradation finding, but downgrades the
"learned" claim.

## Artifacts

- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/summary.json`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoint.npz`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/iteration_metrics.jsonl`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints/iteration-0002.npz`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints/iteration-0004.npz`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints/iteration-0006.npz`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints/iteration-0008.npz`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2_eval_seed123_e10/summary.json`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2_eval_seed123_e10/checkpoint_eval.jsonl`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2_eval_seed123_e10_with_planner_only/summary.json`
- `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2_eval_seed123_e10_with_planner_only/checkpoint_eval.jsonl`
- `artifacts/local/dummy_survival_safety_planner_manifest_smoke/best_checkpoint.json`
- `artifacts/local/dummy_survival_safety_planner_manifest_smoke/best_checkpoint_path.txt`

## Follow-ups

- Add a best-checkpoint selection convention before running larger local or
  Modal dummy survival jobs.
- Investigate why later updates degrade the learned policy before scaling the
  dummy learner.
- Port the same minimal safety/tie-break principle to Tiny Line Duel only if
  its evals show the same unknown/tie pathology.
