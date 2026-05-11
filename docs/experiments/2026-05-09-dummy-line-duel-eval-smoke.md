# 2026-05-09 Dummy Line Duel Eval Smoke

## Question

Can EVAL2 produce a deterministic Tiny Line Duel baseline matrix with fixed
random/scripted policies, paired seats for mixed matchups, and JSON artifacts?

## Setup

- Evaluator: `scripts/run_dummy_line_duel_eval.py`
- Policies: `random_uniform`, `random_sticky`, `one_step_safe`
- Episodes: 20 per match
- Seed: 0
- No trained checkpoint loading, league, or ratings

## Command

```sh
uv run python scripts/run_dummy_line_duel_eval.py \
  --episodes 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_line_duel_eval_smoke
```

## Results

- Smoke completed successfully.
- Matrix covered 9 seat-specific matchups and 180 total episodes.
- `one_step_safe` beat `random_uniform` in both seats: 20-0 as player 1, and
  19-0 with 1 draw as player 0.
- `one_step_safe` beat `random_sticky` in both seats: 20-0 as player 1, and
  18-0 with 2 draws as player 0.
- No truncations were observed.
- Mirror checks exposed useful baselines: random-vs-random had 4 draws,
  sticky-vs-sticky had 10 draws, and one-step-safe-vs-one-step-safe produced
  20 simultaneous out-of-bounds draws.

## Interpretation

This is the first concrete EVAL2 signal/debugging table. The scripted baseline
is strong enough to serve as a sanity floor against random policies, while the
mirror rows expose deterministic draw and seat-bias behavior to watch before
adding learned checkpoint opponents.

## Artifacts

- `artifacts/local/dummy_line_duel_eval_smoke/summary.json`
- `artifacts/local/dummy_line_duel_eval_smoke/episodes.jsonl`

## Follow-ups

- Learned checkpoint opponents now exist; rerun the checkpoint eval smoke after
  trainer changes.
- Add richer reset variety later so paired-seat comparisons cover more than the
  current deterministic spawn geometry.
