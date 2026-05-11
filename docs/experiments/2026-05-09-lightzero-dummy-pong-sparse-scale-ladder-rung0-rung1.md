# 2026-05-09 LightZero MuZero Dummy Pong Sparse Scale Ladder Rung 0/Rung 1

## Question

Does the sparse H120 dummy Pong `tabular_ego` lane improve from one more
reproducibility seed or from a pure 2x budget increase?

This is **LightZero MuZero** on the custom dummy Pong env. It is not visual
training and it is not a raster claim. Feature mode stayed explicitly
`tabular_ego` for every train and scorecard below. No pytest was run.

User follow-up folded into this run: do not treat higher eval simulations as
the fix. A separate MCTS diagnostic found that higher simulations removed
8-sim visit ties but collapsed the first-N debug action histogram to down
`[0, 0, 24]`. This ladder therefore prioritized learning-signal/config-volume:
rung 0 seed reproducibility and rung 1 pure 2x budget.

## Rung 0: Seed 10 Repro

Train command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 1024 --pong-episode-max-steps 120 --max-train-iter 16 --num-simulations 8 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10 --attempt-id train-1024x16-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-r6ji1adcFQ4KWECZAhmQ9l`

Result refs:

```text
run_id: lz-dpong-sparse-h120-lag1-s10
attempt_id: train-1024x16-sparse-h120
summary_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/attempts/train-1024x16-sparse-h120/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/attempts/train-1024x16-sparse-h120/train/episodes.jsonl
training_signals_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/attempts/train-1024x16-sparse-h120/train/lightzero_training_signals.json
```

Checkpoint refs:

```text
ckpt_best: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: d461b8be6378cc8c16d7d6b3f827a3337bcecc7d810a31befb5a9b7695bde01b

iteration_0: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/checkpoints/lightzero/iteration_0.pth.tar
  sha256: d9aa2fd6ab46e0dfb21a384081f1f9dbc112d0d4c9dcea25294b56797e4a7795

iteration_16: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/checkpoints/lightzero/iteration_16.pth.tar
  sha256: c0ea2709daa745d22848370fd5a639f04a59174f7aecb75ddf3728273a0239eb
```

Trainer-side telemetry:

| Metric | Value |
| --- | ---: |
| Episodes | 14 |
| Wins / losses / timeouts | 7 / 6 / 1 |
| Survival mean / median / p90 | 19.1429 / 8.0 / 19.0 |
| Survival std / max | 28.3973 / 120.0 |
| Score return mean / std | 0.0714 / 0.9610 |
| Shaped loss-delay mean / std | 0.0923 / 0.9379 |
| Truncation rate | 0.0714 |
| Train action counts, player_0 `[up, stay, down]` | `[220, 33, 15]` |
| Unique seeds | 10 |
| Seed dominance warning | `false` |

## Rung 0 Independent MCTS Scorecard

The first scorecard app hit a Modal build race because a local script changed
during image build:
`https://modal.com/apps/modal-labs/shankha-dev/ap-omnbUc2OzRucMEE0o6pESz`.
It produced no eval artifacts.

Successful scorecard command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter0-s10=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/checkpoints/lightzero/iteration_0.pth.tar,lightzero:iter16-s10=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/checkpoints/lightzero/iteration_16.pth.tar,lightzero:best-s10=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/checkpoints/lightzero/ckpt_best.pth.tar --episodes 8 --seed 1701 --split-id dummy_pong_sparse_h120_lag1_ladder_v0 --eval-id mcts-scoreboard-rung0-s10-iter0-iter16-best-s1701 --max-env-step 120 --num-simulations 8 --run-id lz-dpong-sparse-h120-lag1-s10 --attempt-id train-1024x16-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-mzH17rndwFsyEzqoCWeFuH`

Artifacts:

```text
eval_dir: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/attempts/train-1024x16-sparse-h120/eval/mcts-scoreboard-rung0-s10-iter0-iter16-best-s1701
summary_json: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/attempts/train-1024x16-sparse-h120/eval/mcts-scoreboard-rung0-s10-iter0-iter16-best-s1701/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10/attempts/train-1024x16-sparse-h120/eval/mcts-scoreboard-rung0-s10-iter0-iter16-best-s1701/episodes.jsonl
```

Scorecard config:

| Field | Value |
| --- | --- |
| Feature mode | `tabular_ego` |
| Feature schema | `dummy_pong_lightzero_tabular_ego_v0` |
| Episodes per match | `8` |
| Paired seats | `true` |
| Eval horizon | `120` |
| MCTS simulations | `8` |
| Strict load | `ok=true`, `missing_keys=[]`, `unexpected_keys=[]` |

Learned checkpoint rows:

| Checkpoint row | Episodes | LZ wins / opp wins / truncs | Survival mean / median / p90 | LZ shaped | LZ score | LZ actions `[up,stay,down]` |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `iter0-s10` vs `lagged_track_ball_1` | 16 | 8 / 6 / 2 | 26.8125 / 8.0 / 80.5 | 0.14323 | 0.125 | `[373,56,0]` |
| `iter0-s10` vs `random_uniform` | 16 | 5 / 11 / 0 | 11.4375 / 8.0 / 19.0 | -0.34063 | -0.375 | `[165,18,0]` |
| `iter0-s10` vs `track_ball` | 16 | 0 / 14 / 2 | 32.3125 / 19.0 / 80.5 | -0.80286 | -0.875 | `[465,52,0]` |
| `iter16-s10` vs `lagged_track_ball_1` | 16 | 5 / 9 / 2 | 26.125 / 13.5 / 69.5 | -0.21979 | -0.25 | `[418,0,0]` |
| `iter16-s10` vs `random_uniform` | 16 | 6 / 10 / 0 | 12.125 / 8.0 / 19.0 | -0.21484 | -0.25 | `[194,0,0]` |
| `iter16-s10` vs `track_ball` | 16 | 0 / 13 / 3 | 37.25 / 19.0 / 120.0 | -0.75104 | -0.8125 | `[596,0,0]` |
| `best-s10` vs `lagged_track_ball_1` | 16 | 6 / 10 / 0 | 12.125 / 8.0 / 19.0 | -0.21771 | -0.25 | `[194,0,0]` |
| `best-s10` vs `random_uniform` | 16 | 8 / 8 / 0 | 9.375 / 8.0 / 13.5 | 0.01953 | 0.0 | `[150,0,0]` |
| `best-s10` vs `track_ball` | 16 | 0 / 14 / 2 | 28.875 / 19.0 / 80.5 | -0.81719 | -0.875 | `[462,0,0]` |

Action entropy read: `iteration_16` and `ckpt_best` are fully collapsed to
all-up in every learned-vs-baseline row, so normalized action entropy is `0.0`
for the learned policy on those rows.

## Rung 1: Pure 2x Budget

Train command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 2048 --pong-episode-max-steps 120 --max-train-iter 32 --num-simulations 8 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10-2x --attempt-id train-2048x32-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-WweYWudOYhJ3GLSR5BvH74`

Result refs:

```text
run_id: lz-dpong-sparse-h120-lag1-s10-2x
attempt_id: train-2048x32-sparse-h120
summary_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/train/episodes.jsonl
training_signals_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/train/lightzero_training_signals.json
```

Checkpoint refs:

```text
ckpt_best: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: d461b8be6378cc8c16d7d6b3f827a3337bcecc7d810a31befb5a9b7695bde01b

iteration_0: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/checkpoints/lightzero/iteration_0.pth.tar
  sha256: d9aa2fd6ab46e0dfb21a384081f1f9dbc112d0d4c9dcea25294b56797e4a7795

iteration_32: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/checkpoints/lightzero/iteration_32.pth.tar
  sha256: 4cf4cf9bfba0ecfb31018a1088abc22093cb87fc7a00eebfc153a21d9a3369a1
```

Trainer-side telemetry:

| Metric | Value |
| --- | ---: |
| Episodes | 26 |
| Wins / losses / timeouts | 11 / 13 / 2 |
| Survival mean / median / p90 | 21.2692 / 8.0 / 46.5 |
| Survival std / max | 30.6990 / 120.0 |
| Score return mean / std | -0.0769 / 0.9577 |
| Shaped loss-delay mean / std | -0.0479 / 0.9308 |
| Truncation rate | 0.0769 |
| Train action counts, player_0 `[up, stay, down]` | `[436, 65, 52]` |
| Unique seeds | 18 |
| Seed dominance warning | `false` |

## Rung 1 Independent MCTS Scorecard

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter0-s10-2x=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/checkpoints/lightzero/iteration_0.pth.tar,lightzero:iter32-s10-2x=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/checkpoints/lightzero/iteration_32.pth.tar,lightzero:best-s10-2x=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/checkpoints/lightzero/ckpt_best.pth.tar --episodes 8 --seed 1701 --split-id dummy_pong_sparse_h120_lag1_ladder_v0 --eval-id mcts-scoreboard-rung1-s10-2x-iter0-iter32-best-s1701 --max-env-step 120 --num-simulations 8 --run-id lz-dpong-sparse-h120-lag1-s10-2x --attempt-id train-2048x32-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-pdwuY2b7RpY5MAQjNUC3qR`

Artifacts:

```text
eval_dir: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/eval/mcts-scoreboard-rung1-s10-2x-iter0-iter32-best-s1701
summary_json: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/eval/mcts-scoreboard-rung1-s10-2x-iter0-iter32-best-s1701/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/eval/mcts-scoreboard-rung1-s10-2x-iter0-iter32-best-s1701/episodes.jsonl
```

Scorecard config:

| Field | Value |
| --- | --- |
| Feature mode | `tabular_ego` |
| Feature schema | `dummy_pong_lightzero_tabular_ego_v0` |
| Episodes per match | `8` |
| Paired seats | `true` |
| Eval horizon | `120` |
| MCTS simulations | `8` |
| Strict load | `ok=true`, `missing_keys=[]`, `unexpected_keys=[]` |

Learned checkpoint rows:

| Checkpoint row | Episodes | LZ wins / opp wins / truncs | Survival mean / median / p90 | LZ shaped | LZ score | LZ actions `[up,stay,down]` |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `iter0-s10-2x` vs `lagged_track_ball_1` | 16 | 8 / 6 / 2 | 25.4375 / 8.0 / 69.5 | 0.14323 | 0.125 | `[365,42,0]` |
| `iter0-s10-2x` vs `random_uniform` | 16 | 5 / 11 / 0 | 11.4375 / 8.0 / 19.0 | -0.34063 | -0.375 | `[159,24,0]` |
| `iter0-s10-2x` vs `track_ball` | 16 | 0 / 14 / 2 | 32.3125 / 19.0 / 80.5 | -0.80286 | -0.875 | `[459,58,0]` |
| `iter32-s10-2x` vs `lagged_track_ball_1` | 16 | 5 / 9 / 2 | 26.125 / 13.5 / 69.5 | -0.21979 | -0.25 | `[418,0,0]` |
| `iter32-s10-2x` vs `random_uniform` | 16 | 6 / 10 / 0 | 12.125 / 8.0 / 19.0 | -0.21484 | -0.25 | `[194,0,0]` |
| `iter32-s10-2x` vs `track_ball` | 16 | 0 / 13 / 3 | 37.25 / 19.0 / 120.0 | -0.75104 | -0.8125 | `[596,0,0]` |
| `best-s10-2x` vs `lagged_track_ball_1` | 16 | 6 / 10 / 0 | 12.125 / 8.0 / 19.0 | -0.21771 | -0.25 | `[194,0,0]` |
| `best-s10-2x` vs `random_uniform` | 16 | 8 / 8 / 0 | 9.375 / 8.0 / 13.5 | 0.01953 | 0.0 | `[150,0,0]` |
| `best-s10-2x` vs `track_ball` | 16 | 0 / 14 / 2 | 28.875 / 19.0 / 80.5 | -0.81719 | -0.875 | `[462,0,0]` |

Action entropy read: `iteration_32` and `ckpt_best` are fully collapsed to
all-up in every learned-vs-baseline row, so normalized action entropy is `0.0`
for the learned policy on those rows.

## Raster Bridge Note

The `raster_flat` bridge is separate from this ladder. A `raster_flat`
config/import path has passed on Modal, and a tiny raster smoke has its own
experiment log. Old `tabular_ego` checkpoints are not compatible with the
raster model input shape. Future raster scorecards must pass an explicit
`feature_mode=raster_flat` and must only score raster-trained checkpoints; do
not silently load tabular checkpoints through a raster scorecard or vice versa.

## Plain Read

Rung 0 seed 10 is mechanically clean but not a quality win. Compared with the
seed-9 sparse probe, the held-out final checkpoint is worse against
`random_uniform` and `lagged_track_ball_1` on shaped/score return, and it
collapses to all-up actions. Some `track_ball` survival rows are longer because
of truncation tail cases, but the learned policy still loses badly and the
action histogram is degenerate.

Rung 1 pure 2x does not rescue the sparse lane. Trainer-side p90 survival
increased, but independent MCTS scorecards are effectively unchanged from rung
0: final and best checkpoints still choose all-up, shaped return is not better,
score return is not better, and action entropy is collapsed. This is evidence
against "same config, just train longer" as the next fix.

Clear decision: do not proceed to higher eval-sim rungs as a fix. The active
question should move to learning signal/config volume, such as higher
update/replay pressure or a different objective/curriculum, while keeping
scorecards survival/shaped/action-entropy aware.
