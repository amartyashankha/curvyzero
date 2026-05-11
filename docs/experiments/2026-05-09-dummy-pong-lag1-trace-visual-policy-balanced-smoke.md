# 2026-05-09 dummy pong lag-1 trace visual-policy balanced smoke

## Question

Does class-weighted supervised loss fix the lag-1 visual-policy lane's
near-always-`up` action skew and improve scoreboard pressure against
`lagged_track_ball_1`?

## Setup

- Source replay: `artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09`.
- Replay size: 1,332 raster rows from 40 exact winning traces.
- Trainer: `scripts/train_dummy_pong_imitation.py`.
- New option: `--class-weighting balanced`, which computes inverse target-action
  frequency weights from the training split per ego-agent head.
- Eval: existing checkpoint scoreboard with 8 episodes per seating, paired
  seats, seed `8050914`.

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  scripts/train_dummy_pong_imitation.py
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09 \
  --epochs 300 \
  --learning-rate 0.05 \
  --validation-fraction 0.2 \
  --seed 7050913 \
  --class-weighting balanced \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_trace_visual_policy_balanced \
  --split-role smoke \
  --checkpoint lag1_trace_visual_balanced_lr005=artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-scoreboard-smoke-2026-05-09
```

A diagnostic `--learning-rate 1.0` run was also attempted at
`artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-smoke-2026-05-09`;
it over-corrected toward `down` and was not promoted for scoring.

## Results

Compile passed. No pytest was run.

Class weights from the training split:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 0.3608 | 179.6667 | 4.4917 |
| `player_1` | 0.3585 | 87.8333 | 5.0190 |

Training, balanced loss, learning rate `0.05`:

| Split | Rows | Accuracy | Weighted loss | Predicted up | Predicted stay | Predicted down | Up acc | Stay acc | Down acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all rows | 1,332 | 0.8318 | 0.6737 | 1,063 | 55 | 214 | 0.8384 | 0.7500 | 0.7381 |
| validation | 266 | 0.8496 | 1.7524 | 221 | 11 | 34 | 0.8555 | 0.0000 | 0.7778 |

Scoreboard smoke:

| Row | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_lag1_trace_visual_balanced_lr005_vs_lagged_track_ball_1` | 16 | 6 | 10 | 0 | 12.1250 | -0.2500 | -0.2177 |
| `learned_lag1_trace_visual_balanced_lr005_vs_random_uniform` | 16 | 10 | 6 | 0 | 17.6250 | 0.2500 | 0.2654 |
| `learned_lag1_trace_visual_balanced_lr005_vs_track_ball` | 16 | 0 | 14 | 2 | 29.5625 | -0.8750 | -0.8143 |

Previous unweighted smoke, same replay and scoreboard size:

| Row | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_lag1_trace_visual_vs_lagged_track_ball_1` | 16 | 5 | 7 | 4 | 38.0625 | -0.1250 | -0.1047 |
| `learned_lag1_trace_visual_vs_random_uniform` | 16 | 10 | 6 | 0 | 13.5000 | 0.2500 | 0.2711 |
| `learned_lag1_trace_visual_vs_track_ball` | 16 | 0 | 11 | 5 | 47.1250 | -0.6875 | -0.6474 |

## Interpretation

Class weighting fixed the most obvious supervised-label symptom: the checkpoint
no longer predicts almost everything as `up`, and it recovers useful `down`
class accuracy. It did not fix the lane.

The lag-1 scoreboard moved only from 5/16 to 6/16 learned wins, while
`track_ball` survival got worse: fewer truncations, shorter games, and a worse
loss-delay proxy. The balanced run remains far below the CEM-v2 positive
baseline of 53/64 wins versus `lagged_track_ball_1`.

## Artifacts

- `artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-scoreboard-smoke-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-smoke-2026-05-09/summary.json`

## Follow-ups

- Do not scale class weighting alone. It is useful as a trainer option and
  diagnostic, but this replay still needs better coverage or augmentation.
- Keep future visual-policy claims compared to CEM-v2's 53/64 lag-1 scoreboard
  row, and continue reporting action histograms plus per-class accuracy.
