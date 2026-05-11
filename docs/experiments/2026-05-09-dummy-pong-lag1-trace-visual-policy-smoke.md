# 2026-05-09 dummy pong lag-1 trace visual-policy smoke

## Question

Can the smallest post-CEM learned policy use visual/raster input to learn score
pressure against `lagged_track_ball_1`?

## Setup

- Environment: `PongConfig(width=15,height=9,paddle_height=3,max_steps=120)`.
- Target opponent: `lagged_track_ball_1`.
- Replay labels: exact target-ladder DP winning traces, converted to
  learner-ready raster rows.
- Learner: existing supervised raster softmax checkpoint path,
  `scripts/train_dummy_pong_imitation.py`.
- Eval: existing checkpoint scoreboard, including wins vs
  `lagged_track_ball_1`, random sanity, and survival/tie diagnostics vs default
  `track_ball`.

## Command

```sh
uv run python -m py_compile scripts/build_dummy_pong_lag1_trace_replay.py
```

```sh
uv run python scripts/build_dummy_pong_lag1_trace_replay.py \
  --max-steps 120 \
  --repeats 1 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09 \
  --epochs 300 \
  --learning-rate 1.0 \
  --validation-fraction 0.2 \
  --seed 7050913 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_trace_visual_policy \
  --split-role smoke \
  --checkpoint lag1_trace_visual=artifacts/local/dummy-pong-lag1-trace-visual-policy-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-smoke-2026-05-09
```

## Results

Compile passed.

Replay:

| Rows | Exact traces | Positive terminal rows | Truncated rows |
| ---: | ---: | ---: | ---: |
| 1,332 | 40 | 40 | 0 |

Replay target labels were highly imbalanced:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 622 | 2 | 42 |
| `player_1` | 622 | 2 | 42 |

Training:

| Split | Rows | Accuracy | Loss | Predicted up | Predicted stay | Predicted down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all rows | 1,332 | 0.9407 | 0.3832 | 1,311 | 0 | 21 |
| validation | 266 | 0.9699 | 0.1889 | 262 | 0 | 4 |

Scoreboard smoke:

| Row | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_lag1_trace_visual_vs_lagged_track_ball_1` | 16 | 5 | 7 | 4 | 38.0625 | -0.125 | -0.1047 |
| `learned_lag1_trace_visual_vs_random_uniform` | 16 | 10 | 6 | 0 | 13.5 | 0.25 | 0.2711 |
| `learned_lag1_trace_visual_vs_track_ball` | 16 | 0 | 11 | 5 | 47.125 | -0.6875 | -0.6474 |

The shaped proxy uses the current Pong diagnostic rule: win `+1.0`, loss
`-1.0 + 0.5 * steps / 120`, truncation `0.0`.

## Interpretation

This starts the true visual-policy lane: the policy checkpoint acts from
`raster_grid` through the normal learned-checkpoint eval path, not from CEM
geometry search. It also gets nonzero score pressure against lag-1 on the
scoreboard.

It is not a pass. CEM-v2 remains the positive baseline at 53/64 wins versus
`lagged_track_ball_1`. This trace-smoke checkpoint reached only 5/16 versus
lag-1 and learned an almost-always-`up` policy because the exact trace dataset
is very imbalanced. Do not scale this exact replay blindly.

Decision rule: CEM-v2 is the score-pressure baseline to beat or imitate. Exact
trace visual behavioral cloning is a valid visual lane, but this first setting
is weak because label imbalance makes the checkpoint collapse toward `up`. The
next visual comparison is class-weighted training versus balanced/augmented
replay, with the same scoreboard gate: >50% wins versus
`lagged_track_ball_1`, random sanity, and reported default-`track_ball`
survival/tie diagnostics.

## Artifacts

- `artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-smoke-2026-05-09/episodes.jsonl`

## Follow-ups

- Recommended next experiment: compare class-weighted training against
  balanced/augmented replay before scaling.
- Compare any promoted visual checkpoint against the CEM-v2 scoreboard result,
  not just against random.
