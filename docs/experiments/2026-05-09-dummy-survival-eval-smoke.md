# 2026-05-09 Dummy Survival Eval Smoke

## Question

Can the first fixed-baseline EVAL1 table for dummy survival run end-to-end and
write summary plus per-episode artifacts?

## Setup

- Package: `curvyzero.training.dummy_survival_eval`.
- Task: solo turning-survival grid toy with left/straight/right actions,
  wall/trail crashes, and sparse terminal reward.
- Policies: `random_uniform` and `one_step_safe`.
- Important caveat: this is a signal/debugging floor, not a leaderboard and
  not a checkpoint evaluation.

## Command

```sh
uv run python scripts/run_dummy_survival_eval.py \
  --episodes 50 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_eval_smoke
```

## Results

- Smoke completed successfully.
- `random_uniform`: 50 episodes, mean terminal reward -1.0, crash rate 1.0,
  survival rate 0.0, mean steps 9.86, max steps 19.
- `one_step_safe`: 50 episodes, mean terminal reward 1.0, crash rate 0.0,
  survival rate 1.0, mean steps 80.0, max steps 80.
- Action histograms:
  - `random_uniform`: left 174, straight 155, right 164.
  - `one_step_safe`: left 800, straight 3200, right 0.

## Interpretation

The evaluator produces the intended boring fixed-seed contrast. The clearance
heuristic is strong enough on this toy to provide a useful scripted ceiling,
while random remains a crash-heavy floor.

## Artifacts

- `artifacts/local/dummy_survival_eval_smoke/summary.json`
- `artifacts/local/dummy_survival_eval_smoke/episodes.jsonl`

## Follow-ups

- Learned checkpoint loading now exists; rerun the checkpoint eval smoke after
  trainer changes.
- Keep the dummy survival table shape aligned with Tiny Line Duel eval output
  as both gain learned checkpoint opponents.
