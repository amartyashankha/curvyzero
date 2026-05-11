# 2026-05-08 Dummy Survival Smoke

## Question

Can the first dummy single-player training loop run end-to-end and write summary plus checkpoint artifacts?

## Setup

- Package: `curvyzero.training.dummy_survival`.
- Task: solo turning-survival grid toy with left/straight/right actions, wall/trail crashes, and sparse terminal reward.
- Learner: tabular NumPy dummy with MuZero-shaped representation, dynamics, prediction, planning, replay, and update seams.
- Important caveat: this is infrastructure scaffolding, not real MuZero.

## Command

```sh
uv run python scripts/run_dummy_survival_train.py --iterations 5 --episodes-per-iter 20 --seed 0 --output-dir artifacts/local/dummy_survival_smoke
```

## Results

- Smoke completed successfully.
- Final eval: 20 episodes, mean steps 22.5, max steps 27.
- Final eval crash rate: 1.0.
- Final eval survival rate: 0.0.
- Model summary: 268 tabular states, 521 learned dynamics edges.

## Artifacts

- `artifacts/local/dummy_survival_smoke/summary.json`
- `artifacts/local/dummy_survival_smoke/checkpoint.npz`
- `artifacts/local/dummy_survival_smoke/iteration_metrics.jsonl`
