# 2026-05-09 Dummy Line Duel Checkpoint Eval Smoke

## Question

Can EVAL2 load a Tiny Line Duel `checkpoint.npz` and compare the learned
checkpoint policy against fixed random/sticky/scripted baselines without adding
league or rating machinery?

## Setup

- Evaluator: `scripts/run_dummy_line_duel_eval.py`
- Checkpoint: `artifacts/local/dummy_line_duel_smoke/checkpoint.npz`
- Baselines: `random_uniform`, `random_sticky`, `one_step_safe`
- Episodes: 5 per seat-specific matchup
- Seed: 123

## Command

```sh
uv run python scripts/run_dummy_line_duel_eval.py \
  --episodes 5 \
  --seed 123 \
  --checkpoint-policy learned:artifacts/local/dummy_line_duel_smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy_line_duel_checkpoint_eval_smoke
```

## Results

- Smoke completed successfully.
- Checkpoint loaded as `learned_dummy_line_duel_smoke`.
- Loaded checkpoint metadata: 3 iterations, 10 episodes per iteration, seed 0.
- Loaded model summary: 104 states, 155 learned dynamics edges.
- Matrix covered the fixed baseline rows plus learned-vs-baseline rows, 75
  total episodes.
- Versus `random_uniform`: learned lost both seatings, with 0 wins, 8 random
  wins, and 2 draws across 10 episodes.
- Versus `random_sticky`: learned was mixed, with 3 learned wins, 3 sticky wins,
  and 4 draws across 10 episodes.
- Versus `one_step_safe`: learned lost both seatings, 0 wins to 10 losses.
- Learned policy death cause was dominated by `occupied_cell` collisions.

## Interpretation

Checkpoint loading works for Tiny Line Duel, and the evaluator now gives an
honest early signal: the dummy learned checkpoint is weak and collision-prone.
That is useful because future trainer changes can be judged against random,
sticky, and one-step-safe without inventing a league.

## Artifacts

- `artifacts/local/dummy_line_duel_checkpoint_eval_smoke/summary.json`
- `artifacts/local/dummy_line_duel_checkpoint_eval_smoke/episodes.jsonl`

## Follow-ups

- Keep checkpoint opponents as explicit eval inputs, not a hidden policy pool.
- Add reset variety later so seat-paired comparisons are more meaningful.
