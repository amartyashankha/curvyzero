# 2026-05-09 Modal dummy Pong CEM-v2 train attempt

## Question

Can the working local CEM-v2 lag-1 Pong lane run as a real Modal
train-attempt path using the simple whole-job plus `curvyzero-runs` Volume
pattern?

## Setup

- Modal app: `curvyzero-dummy-pong-cem-train-attempt`.
- Wrapper:
  `src/curvyzero/infra/modal/dummy_pong_cem_train_attempt.py`.
- Trainer: `curvyzero.training.dummy_pong_cem_train`.
- Geometry: `PongConfig(width=15,height=9,paddle_height=3,max_steps=120)`.
- Opponent weights: `lagged_track_ball_1=1.0`, `random_uniform=0.10`,
  `track_ball=0.10`.
- Target opponent: `lagged_track_ball_1`.
- Storage: `curvyzero-runs` Volume, run/attempt layout under
  `training/dummy-pong/`.

The wrapper is one CPU Modal Function. It calls the NumPy CEM trainer directly,
writes `summary.json`, `checkpoint.npz`, and `cem_rows.jsonl`, mirrors the
checkpoint to the canonical `checkpoints/latest.json` pointer, commits the
Volume, and returns refs. It does not call Modal from environment steps or CEM
candidate rollouts.

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_cem_train_attempt \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --generations 8 \
  --population-size 32 \
  --elite-count 8 \
  --eval-games 16 \
  --seed 8050913 \
  --opponent-weights lagged_track_ball_1=1.0,random_uniform=0.10,track_ball=0.10 \
  --target-opponent-id lagged_track_ball_1 \
  --loss-delay-weight 0.5 \
  --truncation-value 0.0
```

Then score the produced checkpoint with the existing Modal scoreboard:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints cem_v2=ref:training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/attempts/attempt-20260509T045950Z-f16d342d760b/train/checkpoint.npz \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_cem_v2_modal_lagged_track_ball_1 \
  --split-role monitor
```

## Results

Passed.

- Modal train app run: `ap-SzIu3KSSe7NRAq2Iqn33Yu`
- Train run id: `pong-cem-20260509T045950Z-e8b06974a402`
- Train attempt id: `attempt-20260509T045950Z-f16d342d760b`
- Remote train elapsed: `126.404224s`
- Rows artifact: `14` JSONL rows in `cem_rows.jsonl`
- Modal scoreboard app run: `ap-nulgA7l3s4pfcMZUZhOyuO`
- Scoreboard run id: `pong-scoreboard-20260509T050220Z-84b0c61e5ab9`
- Scoreboard attempt id: `attempt-20260509T050220Z-b0a25fb91c80`
- Remote scoreboard elapsed: `3.840436s`

Training final eval:

| Eval slice | Episodes | Learner wins | Opponent wins/losses | Truncations | Mean steps | Mean score return | Mean shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final eval vs `lagged_track_ball_1` | 32 | 25 | 0 losses | 7 | 38.6875 | 0.78125 | 0.78125 |
| Final eval vs `random_uniform` | 32 | 30 | 2 | 0 | 27.9375 | 0.875 | 0.8770833333 |
| Final eval vs `track_ball` | 32 | 0 | 0 | 32 | 120.0 | 0.0 | 0.0 |

Modal checkpoint scoreboard:

| Row | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_cem_v2_vs_lagged_track_ball_1` | 64 | 53 | 1 | 10 | 31.34375 | 0.8125 |
| `learned_cem_v2_vs_random_uniform` | 64 | 60 | 4 | 0 | 22.4375 | 0.875 |
| `learned_cem_v2_vs_track_ball` | 64 | 0 | 0 | 64 | 120.0 | 0.0 |

## Interpretation

CEM-v2 is now a Modal-backed lane. The documented monitor config ran remotely,
wrote durable Volume artifacts, produced the same positive lag-1 score pressure
as the local monitor, and the existing Modal scoreboard loaded the checkpoint
by Volume ref.

For the visual-policy lane, this Modal-backed checkpoint is the score-pressure
baseline to beat or imitate. The next visual comparison should stay focused on
whether class-weighted exact-trace BC or balanced/augmented replay fixes the
current label-imbalance failure.

This is still a compact CPU NumPy learner, not MuZero and not a GPU path. The
default `track_ball` row remains a survival/tie diagnostic: full-length
truncations are expected there, not a hard win gate.

## Artifacts

- `training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/run.json`
- `training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/latest_attempt.json`
- `training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/attempts/attempt-20260509T045950Z-f16d342d760b/train/summary.json`
- `training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/attempts/attempt-20260509T045950Z-f16d342d760b/train/checkpoint.npz`
- `training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/attempts/attempt-20260509T045950Z-f16d342d760b/train/cem_rows.jsonl`
- `training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/checkpoints/latest.json`
- `training/dummy-pong/pong-scoreboard-20260509T050220Z-84b0c61e5ab9/attempts/attempt-20260509T050220Z-b0a25fb91c80/eval/checkpoint-scoreboard/summary.json`
- `training/dummy-pong/pong-scoreboard-20260509T050220Z-84b0c61e5ab9/attempts/attempt-20260509T050220Z-b0a25fb91c80/eval/checkpoint-scoreboard/episodes.jsonl`

## Follow-ups

- Treat `dummy_pong_cem_train_attempt` as the Modal reproduction path for the
  lag-1 CEM-v2 baseline.
- Keep future CEM-v2 claims paired with the Modal scoreboard, preferably with a
  heldout split before claiming robustness beyond this monitor.
- Keep visual-policy claims paired with the same gate: >50% wins versus
  `lagged_track_ball_1`, random sanity, and default-`track_ball` survival/tie
  reporting.
