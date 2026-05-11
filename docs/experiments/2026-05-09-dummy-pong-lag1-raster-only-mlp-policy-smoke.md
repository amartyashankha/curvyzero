# 2026-05-09 dummy pong lag-1 raster-only MLP policy smoke

## Question

Can the smallest nonlinear raster policy fix the lag-1 Pong failure after the
two-frame linear policy improved but stayed below the win gate?

## Setup

- Replay: existing
  `artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09`
- Trainer: `scripts/train_dummy_pong_imitation.py`
- Eval: `scripts/run_dummy_pong_eval.py`
- Policy: one-hidden-layer NumPy MLP, per ego agent.
- Feature mode: `raster_only`
- Frame stack: 2
- Main schedule: hidden dim 128, 800 epochs, Adam full batch, learning rate
  0.005, balanced class weighting, seed 0.
- Eval: 16 episodes per seating, seed 17.
- No pytest was run.

## Code Changes

- Added `dummy_pong_imitation_mlp_policy_checkpoint_v0` checkpoints with
  `hidden_weights`, `hidden_bias`, `output_weights`, and `output_bias`.
- Added `--model-type mlp` and `--hidden-dim` to
  `scripts/train_dummy_pong_imitation.py`; the default remains the old linear
  policy.
- Added eval loading support through the existing `learned:<checkpoint.npz>`
  path. Linear checkpoints still load through the old schema.

## Commands

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/train_dummy_pong_imitation.py \
  scripts/run_dummy_pong_eval.py \
  scripts/run_dummy_pong_checkpoint_scoreboard.py
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-seed0-smoke-2026-05-09 \
  --seed 0 \
  --epochs 800 \
  --learning-rate 0.005 \
  --validation-fraction 0.2 \
  --class-weighting balanced \
  --feature-mode raster_only \
  --frame-stack 2 \
  --model-type mlp \
  --hidden-dim 128
```

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 16 \
  --seed 17 \
  --checkpoint-policy learned:mlp_stack2_h128=artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-seed0-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-eval-e16-seed17-2026-05-09
```

## Results

Training summary:

| Metric | Value |
| --- | ---: |
| Rows | 3,984 |
| Feature width | 1,350 |
| Hidden dim | 128 |
| Train accuracy | 0.9881 |
| Validation accuracy | 0.9548 |
| All-row accuracy | 0.9814 |
| All-row predicted actions | up 1305, stay 1329, down 1350 |

Closed-loop eval summary:

| Pair group | Learned wins | Opponent wins | Truncations | Mean steps |
| --- | ---: | ---: | ---: | ---: |
| MLP vs `lagged_track_ball_1` | 26/32 | 1/32 | 5 | 35.47 |
| MLP vs `random_uniform` | 19/32 | 13/32 | 0 | 14.53 |
| MLP vs default `track_ball` | 0/32 | 22/32 | 10 | 61.91 |

Sanity baselines in the same eval:

- `track_ball` beat `random_uniform` 32/32.
- `track_ball` versus `track_ball` survived to 16/16 truncations at the
  120-step cap.

Small sweep, same replay and eval shape:

| Policy | Lag-1 wins | Random wins | Track-ball truncations | Track-ball mean steps |
| --- | ---: | ---: | ---: | ---: |
| h32 seed0 | 24/32 | 18/32 | 7/32 | 49.34 |
| h64 seed0 | 22/32 | 17/32 | 8/32 | 50.78 |
| h64 seed1 | 23/32 | 18/32 | 4/32 | 36.78 |
| h64 seed2 | 24/32 | 17/32 | 4/32 | 33.00 |
| h128 seed0 | 24/32 | 19/32 | 12/32 | 64.44 |

The single-checkpoint h128 rerun used the normal learned-policy pair seed and
improved to 26/32 lag-1 wins.

## Heldout Scoreboard

Ran a larger local checkpoint scoreboard on a different seed from the original
seed-17 smoke. This used `--episodes 32`, which gives 64 paired-seat rows for
each learned-vs-baseline pair.

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 29 \
  --checkpoint mlp_stack2_h128=artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-seed0-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-heldout-scoreboard-e32-seed29-2026-05-09 \
  --split-id heldout-seed29 \
  --split-role heldout
```

Heldout learned-vs-baseline rows:

| Pair group | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP vs `lagged_track_ball_1` | 64 | 43 | 9 | 12 | 36.0469 | 0.5313 | 0.5359 |
| MLP vs `random_uniform` | 64 | 36 | 28 | 0 | 15.2188 | 0.1250 | 0.1539 |
| MLP vs default `track_ball` | 64 | 0 | 41 | 23 | 62.1719 | -0.6406 | -0.5613 |

Baseline sanity rows:

- `track_ball` beat `random_uniform` 64/64.
- `track_ball` versus `track_ball` reached 32/32 truncations at 120.0 mean
  steps.
- `lagged_track_ball_1` was roughly even with `random_uniform`: 33/64 versus
  31/64.

The shaped proxy uses the current Pong diagnostic rule: win `+1.0`, loss
`-1.0 + 0.5 * steps / 120`, truncation `0.0`.

## Interpretation

This passes the immediate stronger-raster policy gate: a small raster-only
two-frame MLP wins more than 50% against `lagged_track_ball_1` and also beats
random in the cheap local eval. It remains far behind the CEM-v2 geometry
baseline on lag-1 scoreboard scale, and it still cannot beat default
`track_ball`, which is expected to be a survival/tie floor in the default
geometry.

The important change is not just supervised accuracy. The MLP converts the
same exact-trace stack-2 replay into a closed-loop score-pressure policy,
where the linear stack-2 policy reached only 13/32 lag-1 wins in the prior
smoke.

The heldout seed-29 scoreboard confirms the local smoke direction: 43/64
learned wins versus lag-1 still clears the >50% gate, and 36/64 versus random
keeps random sanity positive. It does not promote the checkpoint above CEM-v2:
CEM-v2 remains the stronger lag-1 scoreboard baseline at 53/64, and its
default-`track_ball` row was a clean 64/64 survival tie rather than this MLP's
23/64 truncations and 41/64 losses.

## Artifacts

- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-seed0-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-seed0-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-eval-e16-seed17-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-eval-e16-seed17-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-sweep-eval-e16-seed17-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-sweep-eval-e16-seed17-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-heldout-scoreboard-e32-seed29-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-raster-only-mlp-stack2-h128-e800-lr005-heldout-scoreboard-e32-seed29-2026-05-09/episodes.jsonl`
