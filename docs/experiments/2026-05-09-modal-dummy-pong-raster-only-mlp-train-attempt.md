# 2026-05-09 Modal dummy Pong raster-only MLP train attempt

## Question

Can the stack-2 `raster_only` MLP lane run through the new Modal
train-attempt wrapper, write durable `curvyzero-runs` artifacts, and score the
returned checkpoint with the Modal scoreboard?

## Setup

- Modal train app: `curvyzero-dummy-pong-imitation-train-attempt`.
- Modal scoreboard app: `curvyzero-dummy-pong-scoreboard-attempt`.
- Replay source:
  `artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09/replay_rows.jsonl`.
- Replay Volume ref:
  `training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl`.
- Model: one-hidden-layer NumPy MLP, hidden dim 128.
- Features: `raster_only`, `frame_stack=2`.
- Training: seed 0, 800 epochs, learning rate 0.005, validation fraction 0.2,
  balanced class weighting.
- Scoreboard: 32 episodes per match, seed 9050913, split id
  `dummy_pong_lag1_raster_only_mlp_modal`, monitor role.

## Commands

```sh
modal volume put curvyzero-runs \
  artifacts/local/dummy-pong-lag1-trace-replay-stack2-smoke-2026-05-09/replay_rows.jsonl \
  training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_imitation_train_attempt \
  --replay-path ref:training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl \
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
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints mlp_stack2=ref:training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/attempts/attempt-20260509T055813Z-944cfc7b4c22/train/checkpoint.npz \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_lag1_raster_only_mlp_modal \
  --split-role monitor
```

## Result

- Train Modal app run: `ap-D7PXnC4IvX4gezXH7uhwRm`.
- Train run id: `pong-imitation-20260509T055813Z-4506d7a50304`.
- Train attempt id: `attempt-20260509T055813Z-944cfc7b4c22`.
- Train remote elapsed: `53.120402s`; client elapsed: `56.053112s`.
- Scoreboard Modal app run: `ap-nQPH5XM4aXszCOtWUDi7bX`.
- Scoreboard run id: `pong-scoreboard-20260509T055921Z-402ab3dba50c`.
- Scoreboard attempt id: `attempt-20260509T055921Z-3bf5d8d94220`.
- Scoreboard remote elapsed: `2.291173s`; client elapsed: `4.979699s`.
- Checkpoint ref:
  `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/attempts/attempt-20260509T055813Z-944cfc7b4c22/train/checkpoint.npz`.
- Canonical checkpoint ref:
  `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/checkpoints/iteration-000800/checkpoint.npz`.
- Checkpoint sha256:
  `848cacb5e5d8d327dc4399f029943789365d54efe24be1cfacc8d4ee73b661c6`.

Training metrics:

- Rows: 3,984 total, 3,187 train, 797 validation.
- All rows: 0.981425702811245 accuracy, 0.0670735065802018 loss.
- Validation: 0.9548306148055207 accuracy, 0.23976493817193373 loss.
- Predicted action histogram, all rows: down 1,350, stay 1,329, up 1,305.

Scoreboard metrics:

| Pair group | Episodes | Learned wins | Baseline wins | Truncations | Mean steps | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_mlp_stack2_vs_lagged_track_ball_1` | 64 | 49 | 4 | 11 | 33.265625 | 0.703125 |
| `learned_mlp_stack2_vs_random_uniform` | 64 | 34 | 30 | 0 | 14.875 | 0.0625 |
| `learned_mlp_stack2_vs_track_ball` | 64 | 0 | 39 | 25 | 66.1875 | -0.609375 |

Baseline sanity rows:

- `lagged_track_ball_1_vs_lagged_track_ball_1`: 26 wins, 6 truncations,
  31.75 mean steps over 32 episodes.
- `lagged_track_ball_1_vs_track_ball`: lag-1 0/64, `track_ball` 48/64,
  16 truncations, 42.1875 mean steps.
- `random_uniform_vs_lagged_track_ball_1`: random 36/64, lag-1 28/64,
  14.53125 mean steps.
- `random_uniform_vs_random_uniform`: 32/32 random wins, 10.75 mean steps.
- `random_uniform_vs_track_ball`: random 0/64, `track_ball` 64/64,
  27.59375 mean steps.
- `track_ball_vs_track_ball`: 32/32 truncations, 120.0 mean steps.

## Read

The new wrapper path is real: the replay was uploaded to the Volume, the
train-attempt copied it into the attempt directory, wrote manifests,
`summary.json`, `checkpoint.npz`, and `checkpoints/latest.json`, and the
scoreboard loaded the returned checkpoint by `ref:`.

This run improves the raster-only MLP lag-1 row relative to the earlier local
heldout seed-29 scoreboard, moving from 43/64 to 49/64 wins versus
`lagged_track_ball_1`. It still does not overtake the CEM-v2 geometry baseline
on the same lag-1 comparison, where CEM-v2 scored 53/64, and it still does not
win against default `track_ball`. The useful signal is visual-only score
pressure plus better survival against default `track_ball`: 25/64 truncations
and 66.1875 mean steps.

## Artifacts

- `training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/run.json`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/latest_attempt.json`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/checkpoints/latest.json`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/attempts/attempt-20260509T055813Z-944cfc7b4c22/replay/replay_rows.jsonl`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/attempts/attempt-20260509T055813Z-944cfc7b4c22/train/summary.json`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/attempts/attempt-20260509T055813Z-944cfc7b4c22/train/checkpoint.npz`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/checkpoints/iteration-000800/checkpoint.npz`
- `training/dummy-pong/pong-imitation-20260509T055813Z-4506d7a50304/checkpoints/iteration-000800/metadata.json`
- `training/dummy-pong/pong-scoreboard-20260509T055921Z-402ab3dba50c/run.json`
- `training/dummy-pong/pong-scoreboard-20260509T055921Z-402ab3dba50c/latest_attempt.json`
- `training/dummy-pong/pong-scoreboard-20260509T055921Z-402ab3dba50c/attempts/attempt-20260509T055921Z-3bf5d8d94220/attempt.json`
- `training/dummy-pong/pong-scoreboard-20260509T055921Z-402ab3dba50c/attempts/attempt-20260509T055921Z-3bf5d8d94220/eval/checkpoint-scoreboard/summary.json`
- `training/dummy-pong/pong-scoreboard-20260509T055921Z-402ab3dba50c/attempts/attempt-20260509T055921Z-3bf5d8d94220/eval/checkpoint-scoreboard/episodes.jsonl`

## Next Decision

Treat the stack-2 raster-only MLP as Modal-backed positive visual-only
score-pressure evidence, not as the best Pong checkpoint. CEM-v2 remains the
stronger lag-1 baseline, and default `track_ball` remains a survival/tie
comparison rather than a win gate.
