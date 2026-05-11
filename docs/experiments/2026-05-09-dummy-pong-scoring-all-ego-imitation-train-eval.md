# 2026-05-09 dummy pong scoring all-ego imitation train eval

## Question

Does training the tiny supervised raster policy from all-ego scoring replay
improve Pong eval behavior?

This still copies action labels. It is not reward optimization, MuZero, or
self-play.

## Setup

- Source replay:
  `artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09`
- Row policy: `all`
- Rows: 392
- Behavior policies in rows: `random_uniform`, `track_ball`
- Reward values: `-1.0`, `0.0`, `1.0`
- Nonzero reward rows: 16
- Positive reward rows: 8
- Negative reward rows: 8

## Commands

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-scoring-all-ego-imitation-train-smoke-2026-05-09 \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 1.0 \
  --validation-fraction 0.2
```

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 32 \
  --seed 0 \
  --checkpoint-policy learned:artifacts/local/dummy-pong-scoring-all-ego-imitation-train-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-scoring-all-ego-imitation-eval-e32-seed0-2026-05-09
```

## Train Results

- Train rows: 314
- Validation rows: 78
- Train accuracy: 0.602
- Validation accuracy: 0.487
- All-row accuracy: 0.579

The action-copy target is noisy because the replay mixes a good scripted policy
with random actions.

## Eval Results

32 episodes per seating, 64 episodes per learned-vs-baseline pair group:

- `track_ball` versus `random_uniform`: `track_ball` won 64/64.
- all-ego trained checkpoint versus `random_uniform`: learned won 41/64.
- all-ego trained checkpoint versus `track_ball`: learned won 0/64,
  `track_ball` won 50/64, and 14/64 games truncated.

For comparison:

- mirror-replay imitation checkpoint versus random: 43/64.
- positive-only scoring-replay checkpoint versus random: 44/64.

## Interpretation

All-ego scoring replay is better for value targets because it includes wins and
losses. It is worse for plain supervised action copying because half the rows
copy `random_uniform`.

This result says not to keep scaling the current action-clone learner. The next
useful step is either cleaner expert-only policy data against random opponents,
or a value/reward target that can use both winning and losing ego rows.

## Artifacts

- `artifacts/local/dummy-pong-scoring-all-ego-imitation-train-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-all-ego-imitation-train-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-scoring-all-ego-imitation-eval-e32-seed0-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-all-ego-imitation-eval-e32-seed0-2026-05-09/episodes.jsonl`

## Follow-ups

- Do not train policy cloning directly on random-action rows unless the goal is
  to clone a mixed behavior policy.
- Use all-ego rows for value/reward-target experiments.
- For the next policy clone, prefer expert-only `track_ball` rows collected
  against random opponents.
