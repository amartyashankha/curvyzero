# 2026-05-08 Dummy Line Duel Smoke

## Question

Can the dummy two-player line-duel scaffold run end-to-end with simultaneous
actions, ego-perspective replay rows, and JSON/NPZ artifacts?

## Command

```sh
uv run python scripts/run_dummy_line_duel_train.py \
  --iterations 3 \
  --episodes-per-iter 10 \
  --seed 0 \
  --output-dir artifacts/local/dummy_line_duel_smoke
```

## Results

- Smoke completed successfully.
- In-module canaries covered same-cell draw, wall-death win/loss reward, and cross-swap draw.
- Replay rows written: 278 ego-perspective rows across 30 training episodes.
- Final eval: 20 episodes, 20 draws, 0 truncations, mean steps 6.0.
- Model summary: 104 tabular states, 155 learned dynamics edges.
- Post-review fix: model observation bucketing now uses `LineDuelConfig.width`
  and `height` instead of hardcoded `11 x 11`.
- Post-review verification: `python3 -m py_compile` passed and a 1-iteration
  local smoke with seed `7` wrote artifacts under
  `/private/tmp/curvy_line_duel_postfix_smoke`.

## Interpretation

This is a training-interface smoke, not a quality claim. The all-draw final
eval is useful because it exposes a deterministic scaffold behavior we can now
evaluate against random/scripted baselines.

## Artifacts

- `artifacts/local/dummy_line_duel_smoke/summary.json`
- `artifacts/local/dummy_line_duel_smoke/checkpoint.npz`
- `artifacts/local/dummy_line_duel_smoke/iteration_metrics.jsonl`
- `artifacts/local/dummy_line_duel_smoke/replay_rows.jsonl`

## Follow-ups

- EVAL2 baseline matrix and explicit learned checkpoint loading now exist; use
  those eval smokes after trainer changes.
- Add paired-seat eval once reset variety or seat-swap setup is available.
