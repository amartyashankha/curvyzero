# 2026-05-09 Dummy Survival Eval Protocol Fields Smoke

## Question

Can the dummy survival evaluator and checkpoint sweep emit the v0 eval protocol
fields needed before larger local or Modal runs?

## Setup

- Eval summary now records `eval_split` metadata: split id, split role, seed
  generation, base seed, seed count, seed-list hash, and paired-seat flag.
- Checkpoint sweep summary now records `selected_checkpoint`,
  `latest_checkpoint`, `selection_metric`, `heldout_required`, and
  `selection_record`.
- Checkpoint sweep now writes `selection_record.json`.

## Command

```sh
python3 -m py_compile \
  src/curvyzero/training/dummy_survival.py \
  src/curvyzero/training/dummy_survival_eval.py \
  scripts/run_dummy_survival_train.py \
  scripts/run_dummy_survival_eval.py \
  scripts/run_dummy_survival_checkpoint_sweep.py

PYTHONPATH=src python3 scripts/run_dummy_survival_eval.py \
  --episodes 3 \
  --seed 0 \
  --split-id dummy_survival_debug_v0 \
  --split-role debug \
  --output-dir artifacts/local/dummy_survival_eval_split_metadata_smoke

PYTHONPATH=src python3 scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 3 \
  --seed 123 \
  --split-id dummy_survival_selection_smoke_v0 \
  --split-role selection \
  --output-dir artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke
```

## Results

- Eval split metadata was written with a stable seed-list hash.
- Sweep artifacts now include `selection_record.json`.
- Selection smoke marked `heldout_required: true` because the split role was
  `selection`.
- Selected checkpoint: `iteration-0002`, survival rate 1.0 on the 3-episode
  selection smoke.
- Latest checkpoint: `iteration-0008`, survival rate 0.0 on the same 3-episode
  selection smoke.
- Required baselines included `random_uniform`, `one_step_safe`, and
  `untrained_model_same_planner`.

## Interpretation

The artifact shape now exposes the important eval truth: best-checkpoint
selection can be useful, but the degraded latest checkpoint stays visible and
selection-role sweeps demand heldout confirmation. This is still a tiny smoke,
not a quality claim.

## Artifacts

- `artifacts/local/dummy_survival_eval_split_metadata_smoke/summary.json`
- `artifacts/local/dummy_survival_eval_split_metadata_smoke/episodes.jsonl`
- `artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/summary.json`
- `artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/checkpoint_eval.jsonl`
- `artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/best_checkpoint.json`
- `artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/best_checkpoint_path.txt`
- `artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/selection_record.json`

## Follow-ups

- Add heldout confirmation command shape that evaluates selected, latest, and
  required baselines on a separate split.
- Mirror these fields in Modal Volume run-management jobs once local layout is
  boring.
