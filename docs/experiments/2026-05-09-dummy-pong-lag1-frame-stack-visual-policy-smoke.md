# 2026-05-09 dummy pong lag-1 frame-stack visual-policy smoke

## Question

Can the lag-1 exact-trace raster imitation lane benefit from a two-frame raster
stack? The hypothesis is that single-frame raster observations hide velocity, so
some exact-trace labels may be partially unlearnable from one frame.

## Setup

- Builder: `scripts/build_dummy_pong_lag1_trace_replay.py`
- Trainer: `scripts/train_dummy_pong_imitation.py`
- Eval: `scripts/run_dummy_pong_eval.py`
- Replay transform: vertical mirror plus action oversampling.
- Policy feature mode: `raster_plus_geometry`
- Train schedule: 400 epochs, seed 0, learning rate 0.5.
- Eval: 16 episodes per seating, seed 17.
- No pytest was run.

## Commands

```sh
uv run python -m py_compile \
  scripts/build_dummy_pong_lag1_trace_replay.py \
  scripts/train_dummy_pong_imitation.py \
  scripts/run_dummy_pong_eval.py \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  src/curvyzero/training/dummy_pong_eval.py
```

```sh
uv run python scripts/build_dummy_pong_lag1_trace_replay.py \
  --max-steps 120 \
  --repeats 1 \
  --include-vertical-mirror \
  --balance-actions oversample \
  --balance-seed 0 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-replay-stack1-smoke-2026-05-09
```

```sh
uv run python scripts/build_dummy_pong_lag1_trace_replay.py \
  --max-steps 120 \
  --repeats 1 \
  --frame-stack 2 \
  --include-vertical-mirror \
  --balance-actions oversample \
  --balance-seed 0 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-stack1-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-policy-stack1-e400-smoke-2026-05-09 \
  --seed 0 \
  --epochs 400 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --feature-mode raster_plus_geometry \
  --frame-stack 1
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-policy-stack2-e400-smoke-2026-05-09 \
  --seed 0 \
  --epochs 400 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --feature-mode raster_plus_geometry \
  --frame-stack 2
```

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 16 \
  --seed 17 \
  --checkpoint-policy learned:stack1=artifacts/local/dummy-pong-lag1-trace-policy-stack1-e400-smoke-2026-05-09/checkpoint.npz \
  --checkpoint-policy learned:stack2=artifacts/local/dummy-pong-lag1-trace-policy-stack2-e400-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-frame-stack-eval-e16-seed17-2026-05-09
```

## Results

Frame stack is implemented end to end for the lag-1 trace replay, imitation
trainer, checkpoint metadata, and eval loader. Both `stack1` and `stack2`
checkpoints were loadable through `learned:<name>=.../checkpoint.npz` eval.

Both replays emitted 3,984 rows after mirroring and oversampling, with balanced
per-agent targets: 664 `up`, 664 `stay`, and 664 `down`.

| Policy | Feature shape | Train acc | Validation acc | All-row predicted actions |
| --- | ---: | ---: | ---: | --- |
| stack 1 | 681 | 0.9448 | 0.9398 | up 1291, stay 1351, down 1342 |
| stack 2 | 1356 | 0.9294 | 0.9297 | up 1223, stay 1362, down 1399 |

Closed-loop eval summary:

| Pair group | Learned wins | Opponent wins | Truncations | Mean steps |
| --- | ---: | ---: | ---: | ---: |
| stack1 vs `lagged_track_ball_1` | 11/32 | 20/32 | 1 | 14.25 |
| stack2 vs `lagged_track_ball_1` | 13/32 | 15/32 | 4 | 25.09 |
| stack1 vs `random_uniform` | 12/32 | 20/32 | 0 | 10.75 |
| stack2 vs `random_uniform` | 15/32 | 17/32 | 0 | 12.47 |
| stack1 vs `track_ball` | 0/32 | 28/32 | 4 | 29.56 |
| stack2 vs `track_ball` | 0/32 | 26/32 | 6 | 36.22 |

Sanity baselines in the same eval:

- `track_ball` beat `random_uniform` 32/32.
- `track_ball` versus `track_ball` survived to 16/16 truncations at the 120-step cap.

## Interpretation

The two-frame stack did not improve supervised label accuracy in this small
linear-policy smoke. It did, however, slightly improve the visual lane in
closed-loop behavior: more wins versus `lagged_track_ball_1`, more wins versus
random, and longer survival against default `track_ball`. This is a weak
positive signal for frame history as an input feature, not a solved lane.

The learned policies still lose the main lag-1 row, fail the >50% lag-1 gate,
and remain far below the CEM-v2 lag-1 monitor result. Stack-2 improved lag-1
wins to 13/32 and default-`track_ball` truncations to 6/32, but it still failed
the random sanity row by losing 17/32 against `random_uniform`. The stack-2
action histograms are also very seat-biased in eval, so the next useful check
is not more data scale; it is a small model/optimization comparison that can
use the extra temporal feature without collapsing into an absolute-direction
policy.

Next hypothesis: the current linear policy class is too weak for this aliased,
piecewise label map. Before moving to on-policy RL, try a tiny MLP or small CNN
on stack-2 `raster_only` inputs and compare directly against CEM-v2 as the
positive lag-1 score-pressure baseline.

## Follow-up

The next smallest nonlinear check is complete:
`2026-05-09-dummy-pong-lag1-raster-only-mlp-policy-smoke.md`. A one-hidden-layer
NumPy MLP with `frame_stack=2` and `feature_mode=raster_only` passed the cheap
lag-1 gate, scoring 26/32 wins versus `lagged_track_ball_1`, 19/32 versus
`random_uniform`, and 10/32 truncations with 61.91 mean steps versus default
`track_ball`.

## Artifacts

- `artifacts/local/dummy-pong-lag1-trace-replay-stack1-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-replay-stack1-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-policy-stack1-e400-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-policy-stack1-e400-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-policy-stack2-e400-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-policy-stack2-e400-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-frame-stack-eval-e16-seed17-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-frame-stack-eval-e16-seed17-2026-05-09/episodes.jsonl`
