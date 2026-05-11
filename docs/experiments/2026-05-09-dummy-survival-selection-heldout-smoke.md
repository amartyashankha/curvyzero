# 2026-05-09 Dummy Survival Selection Heldout Smoke

## Question

Can a preselected dummy survival checkpoint be confirmed on a heldout split
without hiding the latest checkpoint or planner-only baseline?

## Setup

- New script: `scripts/run_dummy_survival_selection_holdout.py`.
- Input selection record:
  `artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/selection_record.json`.
- Heldout eval: 3 episodes, seed 456, split id
  `dummy_survival_heldout_smoke_v0`.

## Command

```sh
python3 -m py_compile \
  scripts/run_dummy_survival_selection_holdout.py \
  src/curvyzero/training/dummy_survival_eval.py

PYTHONPATH=src python3 scripts/run_dummy_survival_selection_holdout.py \
  --selection-record artifacts/local/dummy_survival_checkpoint_sweep_selection_record_smoke/selection_record.json \
  --episodes 3 \
  --seed 456 \
  --split-id dummy_survival_heldout_smoke_v0 \
  --output-dir artifacts/local/dummy_survival_selection_holdout_smoke
```

## Results

- Selected checkpoint: `iteration-0002`, survival rate 1.0.
- Latest checkpoint: `iteration-0008`, survival rate 0.0.
- Planner-only baseline: `untrained_model_same_planner`, survival rate 1.0.
- `selected_vs_latest`: `better`.
- `selected_vs_planner_only`: `tied`.
- `claim_status`: `inconclusive`.

## Interpretation

The confirmation path works and preserves the important distinction. The
selected checkpoint is better than the degraded latest checkpoint on this tiny
heldout smoke, but it does not beat the untrained planner-only baseline, so it
is not learning evidence.

This is only a 3-episode plumbing smoke. A real selection/heldout run should
use the episode counts in `docs/design/training_eval_protocol.md`.

## Artifacts

- `artifacts/local/dummy_survival_selection_holdout_smoke/summary.json`
- `artifacts/local/dummy_survival_selection_holdout_smoke/episodes.jsonl`
- `artifacts/local/dummy_survival_selection_holdout_smoke/holdout_confirmation.json`

## Follow-ups

- Use this heldout confirmation shape after larger selection sweeps.
- Mirror the same selected/latest/planner-only confirmation in Modal eval jobs.
