# 2026-05-09 dummy pong lag-1 trace raster-only policy smoke

## Question

Can the existing lag-1 exact-trace imitation lane train and score a truthful
visual-only baseline where the policy logits use one-hot raster cells plus the
per-ego policy head only, with no decoded geometry suffix?

## Setup

- Source replay: `artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09`.
- Replay size: 1,332 raster rows from 40 exact winning traces.
- Trainer: `scripts/train_dummy_pong_imitation.py`.
- Feature mode: `--feature-mode raster_only`.
- Loss: `--class-weighting balanced`, matching the prior lag-1 balanced smoke.
- Eval: existing checkpoint scoreboard with 8 episodes per seating, paired
  seats, seed `8050914`.

## Commands

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/train_dummy_pong_imitation.py \
  scripts/run_dummy_pong_checkpoint_scoreboard.py
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09 \
  --epochs 300 \
  --learning-rate 0.05 \
  --validation-fraction 0.2 \
  --seed 7050913 \
  --class-weighting balanced \
  --feature-mode raster_only \
  --output-dir artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_trace_raster_only_policy_balanced \
  --split-role smoke \
  --checkpoint lag1_trace_raster_only_balanced_lr005=artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-scoreboard-smoke-2026-05-09
```

```sh
uv run python -c 'from pathlib import Path; from curvyzero.training.dummy_pong_imitation_train import DummyPongImitationPolicy; paths=[Path("artifacts/local/dummy-pong-lag1-trace-visual-policy-balanced-lr005-smoke-2026-05-09/checkpoint.npz"), Path("artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-smoke-2026-05-09/checkpoint.npz")];
for path in paths:
    policy=DummyPongImitationPolicy.load_checkpoint(path)
    print(path, policy.feature_mode, policy.weights.shape)'
```

## Results

Compile passed. No pytest was run.

Backward-compatible checkpoint load check:

| Checkpoint | Feature mode | Weights shape |
| --- | --- | --- |
| prior balanced lag-1 trace | `raster_plus_geometry` | `(2, 681, 3)` |
| new raster-only lag-1 trace | `raster_only` | `(2, 675, 3)` |

Training, balanced loss, learning rate `0.05`:

| Split | Rows | Accuracy | Weighted loss | Predicted up | Predicted stay | Predicted down | Up acc | Stay acc | Down acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,066 | 0.8180 | 0.4276 | 848 | 43 | 175 | 0.8320 | 1.0000 | 0.6267 |
| validation | 266 | 0.8459 | 1.7215 | 220 | 11 | 35 | 0.8516 | 0.0000 | 0.7778 |
| all rows | 1,332 | 0.8236 | 0.6860 | 1,068 | 54 | 210 | 0.8360 | 0.7500 | 0.6429 |

Scoreboard smoke:

| Row | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_lag1_trace_raster_only_balanced_lr005_vs_lagged_track_ball_1` | 16 | 6 | 10 | 0 | 12.1250 | -0.2500 |
| `learned_lag1_trace_raster_only_balanced_lr005_vs_random_uniform` | 16 | 10 | 6 | 0 | 12.1250 | 0.2500 |
| `learned_lag1_trace_raster_only_balanced_lr005_vs_track_ball` | 16 | 0 | 14 | 2 | 29.5625 | -0.8750 |

Baseline sanity stayed in the expected shape:

| Row | Episodes | Result |
| --- | ---: | --- |
| `track_ball_vs_track_ball` | 8 | 8/8 truncations, 120.0 mean steps |
| `random_uniform_vs_track_ball` | 16 | `track_ball` won 16/16 |

## Interpretation

The feature ablation is now real: the checkpoint records
`feature_mode=raster_only`, `feature_encoding_id=dummy_pong_raster_one_hot_v0`,
and a 675-wide feature axis, exactly the raster one-hot size for 9x15x5 cells.
Eval reloads and uses that mode.

As a policy result, raster-only is not viable yet. It ties the previous
balanced geometry-augmented lag-1 result on the headline lag-1 row at 6/16, and
it beats random 10/16, but it remains far below CEM-v2's 53/64 lag-1 row and
does not survive well against default `track_ball`.

## Artifacts

- `artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-policy-balanced-lr005-scoreboard-smoke-2026-05-09/episodes.jsonl`

## Decision

Keep `raster_only` as the honest visual-only ablation baseline. Do not call it
solved: it needs better data coverage or a different learner before scaling.
