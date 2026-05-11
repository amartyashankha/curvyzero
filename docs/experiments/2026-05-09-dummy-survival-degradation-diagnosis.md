# 2026-05-09 Dummy Survival Degradation Diagnosis

## Question

Why do later dummy survival checkpoints degrade after early checkpoints and the
planner-only baseline survive?

## Setup

- New script: `scripts/analyze_dummy_survival_checkpoints.py`.
- Checkpoints:
  `artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints`.
- Eval/diagnosis split: 20 episodes, seed 123.

## Command

```sh
PYTHONPATH=src python3 -m py_compile scripts/analyze_dummy_survival_checkpoints.py

PYTHONPATH=src python3 scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 20 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_degradation_probe_eval_seed123_e20

PYTHONPATH=src python3 scripts/analyze_dummy_survival_checkpoints.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 20 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20
```

## Results

- `untrained_model_same_planner`: survival 1.0, lower-clearance overrides 0.
- `iteration-0002`: survival 1.0, lower-clearance overrides 45.
- `iteration-0004`: survival 1.0, lower-clearance overrides 76.
- `iteration-0006`: survival 0.2, lower-clearance overrides 184.
- `iteration-0008`: survival 0.2, lower-clearance overrides 244.
- All checkpoint Q ranges were non-positive, roughly `[-0.999, 0.000]`.

Concrete failure: on env seed `33158374`, `iteration-0008` picked `straight`
with clearance 4 over the safety action `right` with clearance 6 because the
learned next value for `straight` was less negative. It later crashed at step
65; planner-only and `iteration-0002` survived.

## Interpretation

Later checkpoints degrade because learned Q/dynamics increasingly override the
planner-only safety prior with a crash-only, non-positive value landscape.
Crash-heavy replay makes known routes negative, while unknown or under-updated
next states remain at `0.0` and can look attractive. The planner-only baseline
survives because all model scores tie at zero and the safety/clearance
tie-break controls behavior.

This explains the current failure mode without pretending the dummy trainer is
fixed.

## Artifacts

- `artifacts/local/dummy_survival_degradation_probe_eval_seed123_e20/summary.json`
- `artifacts/local/dummy_survival_degradation_probe_eval_seed123_e20/checkpoint_eval.jsonl`
- `artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20/summary.json`
- `artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20/episodes.jsonl`
- `artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20/overrides.jsonl`

## Follow-ups

- Do not scale this tabular dummy learner blindly.
- Next fix candidates should target the confound directly: planner score
  calibration, unknown-state pessimism, collection quality, or separating
  safety-prior eval from learned-value eval.
- Keep `untrained_model_same_planner` in every survival learned-checkpoint eval.
