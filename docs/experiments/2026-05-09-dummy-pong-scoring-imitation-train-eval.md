# 2026-05-09 dummy pong scoring imitation train eval

## Question

Can the existing tiny supervised raster learner train from score-bearing Pong
scoring replay rows and improve eval behavior?

This is still supervised copying of `target_action_id` / `target_action_label`.
It is not MuZero, reward optimization, planning, or self-play improvement.

## Setup

- Source replay:
  `artifacts/local/dummy-pong-scoring-replay-smoke-2026-05-09`
- Source row schema: `dummy_pong_scoring_replay_row_v0`
- Rows: 196
- Target/behavior policy IDs in emitted rows: `track_ball`
- Reward values in emitted rows: `0.0`, `1.0`
- Nonzero reward rows: 8
- Negative reward rows: 0
- Important limitation: this scoring smoke emits only `track_ball` ego rows, and
  all emitted score rows are positive winning rows. It does not provide losing
  ego examples or negative value targets.

## Code Change

`src/curvyzero/training/dummy_pong_imitation_train.py` now accepts both:

- `dummy_pong_imitation_replay_row_v0`
- `dummy_pong_scoring_replay_row_v0`

The loader still validates the supervised contract: raster schema/shape,
`ego_agent`, `target_action_id`, `target_action_label`, joint ego action,
reward field, next raster shape, and the shared Pong action labels.

## Commands

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  scripts/train_dummy_pong_imitation.py
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-scoring-replay-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-scoring-imitation-train-smoke-2026-05-09 \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 1.0 \
  --validation-fraction 0.2
```

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 32 \
  --seed 0 \
  --checkpoint-policy learned:artifacts/local/dummy-pong-scoring-imitation-train-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-scoring-imitation-eval-e32-seed0-2026-05-09
```

## Train Results

- Train rows: 157
- Validation rows: 39
- Train accuracy: 1.000
- Validation accuracy: 0.667
- All-row accuracy: 0.934

This overfits the tiny scoring smoke, as expected.

## Eval Results

32 episodes per seating, 64 episodes per learned-vs-baseline pair group:

- `track_ball` versus `random_uniform`: `track_ball` won 64/64.
- scoring-trained learned checkpoint versus `random_uniform`: learned won
  44/64, with no truncations.
- scoring-trained learned checkpoint versus `track_ball`: learned won 0/64,
  `track_ball` won 43/64, and 21/64 games truncated.
- `track_ball` versus `track_ball`: 32/32 truncations.

Compared with the earlier imitation checkpoint eval:

- Learned versus random moved from 43/64 to 44/64, a tiny improvement.
- Learned versus `track_ball` did not improve. It stayed at 0 learned wins and
  became more beatable by `track_ball` on this eval seed set.

## Interpretation

The tiny raster learner can train from score-bearing replay rows once they
carry the same supervised action fields. This answers the loader/training smoke
question positively.

The behavior result is modest: the checkpoint is still just copying winning
`track_ball` rows from a small, positive-only scoring smoke. It did not learn
from reward and did not close the gap to scripted `track_ball`.

## Artifacts

- `artifacts/local/dummy-pong-scoring-imitation-train-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-imitation-train-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-scoring-imitation-eval-e32-seed0-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-imitation-eval-e32-seed0-2026-05-09/episodes.jsonl`
