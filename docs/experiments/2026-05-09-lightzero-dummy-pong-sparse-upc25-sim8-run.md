# 2026-05-09 LightZero Dummy Pong Sparse UPC25 Sim8 Run

## Question

After sparse ladder rung 0 and rung 1 showed that pure same-config length did
not improve heldout survival, shaped return, raw score, or action entropy, does
a small higher update/replay configuration produce a cleaner learning signal at
the same fixed 120-step sparse Pong horizon?

This is a config-volume test, not a longer same-config run. It keeps
train-time MCTS at 8 simulations to avoid mixing in the known 16+ simulation
eval-collapse effect, and changes update/replay pressure:

- `batch_size=64`
- `update_per_collect=25`
- `n_episode=4`
- `game_segment_length=120`

No pytest was run.

## Train Command

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 2048 --pong-episode-max-steps 120 --max-train-iter 32 --num-simulations 8 --batch-size 64 --update-per-collect 25 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 4 --game-segment-length 120 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10-upc25-sim8 --attempt-id train-2048x32-sim8-upc25-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-3cwlhH1XUZkwZuIFQHFqLA`

Result:

```text
ok: true
run_id: lz-dpong-sparse-h120-lag1-s10-upc25-sim8
attempt_id: train-2048x32-sim8-upc25-sparse-h120
summary_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/train/episodes.jsonl
training_signals_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/train/lightzero_training_signals.json
lightzero_artifacts_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/train/lightzero_artifacts_manifest.json
```

## Config Surface

| Field | Value |
| --- | --- |
| `feature_mode` | `tabular_ego` |
| `opponent_policy` | `lagged_track_ball_1` |
| `max_env_step` | `2048` training budget |
| `pong_episode_max_steps` | `120` explicit horizon |
| `max_train_iter` | `32` requested |
| `num_simulations` | `8` |
| `batch_size` | `64` |
| `update_per_collect` | `25` |
| `n_episode` | `4` |
| `game_segment_length` | `120` |
| `td_steps` / `num_unroll_steps` | `120` / `5` |
| `discount_factor` | `1.0` |
| reward/value support | `[-5, 6, 1]` |

Checkpoint refs:

```text
iteration_0: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/checkpoints/lightzero/iteration_0.pth.tar
  sha256: ec91d58bd35f702d4a22ec6be9c0e52af5a7f51c4010fb05a4747d3fab56a967

iteration_50: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/checkpoints/lightzero/iteration_50.pth.tar
  sha256: 4a85aa61c7412207fb0024f900654ba43118cd6b44dcacceabfe405b0ddad110

ckpt_best: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: 9d570812494b9f0351746cb4ee11530e0bd8ef14e602bdd7a3d11ee07ab166c1
```

`iteration_50` is the mirrored final checkpoint name produced by LightZero's
update counter under this `update_per_collect=25` run.

## Trainer-Side Telemetry

Trainer-side rows are diagnostic only; the heldout MCTS scorecard below is the
quality gate.

| Metric | Value |
| --- | ---: |
| Episodes | 20 |
| Wins / losses / timeouts | 10 / 9 / 1 |
| Survival mean / median / p90 | 19.1 / 8.0 / 21.2 |
| Shaped loss-delay mean | 0.07875 |
| Raw score mean | 0.05 |
| Unique seeds / top seed count | 12 / 2 |
| Seed dominance warning | `false` |

Train action counts for `player_0` LightZero ego:
`[259, 73, 50]` for `[up, stay, down]`.

## Heldout MCTS Scorecard

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints "lightzero:iter0-upc25=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/checkpoints/lightzero/iteration_0.pth.tar,lightzero:iter50-upc25=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/checkpoints/lightzero/iteration_50.pth.tar,lightzero:best-upc25=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/checkpoints/lightzero/ckpt_best.pth.tar" --episodes 8 --seed 1701 --split-id dummy_pong_sparse_h120_lag1_upc25_heldout_v0 --split-role heldout --eval-id mcts-scoreboard-upc25-sim8-iter0-iter50-best-e8 --max-env-step 120 --num-simulations 8 --feature-mode tabular_ego --run-id lz-dpong-sparse-h120-lag1-s10-upc25-sim8 --attempt-id train-2048x32-sim8-upc25-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-jIpZP07OHI5Y3dYjpWcW6C`

Artifacts:

```text
summary_json: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/eval/mcts-scoreboard-upc25-sim8-iter0-iter50-best-e8/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/eval/mcts-scoreboard-upc25-sim8-iter0-iter50-best-e8/episodes.jsonl
```

Scorecard config:

| Field | Value |
| --- | --- |
| Episodes per match | `8` |
| Total episodes | `264` |
| Paired seats | `true` |
| Eval horizon | `120` |
| MCTS simulations | `8` |
| Feature mode | `tabular_ego` |
| Strict load | all three checkpoints `ok=true`, no missing/unexpected keys |

### Iteration 50 Rows

| Opponent | Survival mean / median / p90 | Shaped mean | Raw score mean | Wins | Truncs | Action hist `[up, stay, down]` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 26.125 / 13.5 / 69.5 | -0.21979 | -0.25 | 5-9 | 2 | `[389, 0, 29]` |
| `random_uniform` | 12.125 / 8.0 / 19.0 | -0.21484 | -0.25 | 6-10 | 0 | `[175, 0, 19]` |
| `track_ball` | 38.625 / 19.0 / 120.0 | -0.74531 | -0.8125 | 0-13 | 3 | `[592, 0, 26]` |

Aggregate `iteration_50` learned action histogram across baseline opponents:
`[1156, 0, 74]`. The checkpoint uses a small amount of `down`, no `stay`, and
still has strongly collapsed eval behavior.

### Best Checkpoint Rows

| Opponent | Survival mean / median / p90 | Shaped mean | Raw score mean | Wins | Truncs | Action hist `[up, stay, down]` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 12.125 / 8.0 / 19.0 | -0.21771 | -0.25 | 6-10 | 0 | `[194, 0, 0]` |
| `random_uniform` | 9.375 / 8.0 / 13.5 | 0.01953 | 0.0 | 8-8 | 0 | `[150, 0, 0]` |
| `track_ball` | 28.875 / 19.0 / 80.5 | -0.81719 | -0.875 | 0-14 | 2 | `[462, 0, 0]` |

Aggregate `ckpt_best` learned action histogram across baseline opponents:
`[806, 0, 0]`.

### Initialization Control

| Opponent | Survival mean / median / p90 | Shaped mean | Raw score mean | Wins | Truncs | Action hist `[up, stay, down]` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 24.75 / 8.0 / 69.5 | 0.02031 | 0.0 | 7-7 | 2 | `[200, 196, 0]` |
| `random_uniform` | 13.5 / 8.0 / 19.0 | -0.21198 | -0.25 | 6-10 | 0 | `[104, 112, 0]` |
| `track_ball` | 32.3125 / 19.0 / 80.5 | -0.80286 | -0.875 | 0-14 | 2 | `[250, 267, 0]` |

The final checkpoint did not beat initialization cleanly. It improved survival
versus `track_ball` only, while shaped and raw return stayed bad and action
entropy worsened.

## Read

Higher update/replay at this small fixed-horizon sparse setting did not reveal
a useful learning signal. The trainer-side scorecard looked mildly alive, but
heldout MCTS remained weak:

- `iteration_50` lost to `lagged_track_ball_1` and `random_uniform` on raw
  score and shaped return.
- `ckpt_best` was exactly all-up across the three learned-vs-baseline rows.
- `iteration_50` was nearly all-up, with no `stay` and only 74 `down` actions
  out of 1230 learned actions.
- The heldout curve is not monotone: final does not clearly beat
  `iteration_0`, and best is more collapsed than final.

This closes "higher update/replay alone" as the next easy explanation for the
current sparse Pong failure. Do not propose longer same-config runs from here.

## Next Go/Stop

Stop:

- Same sparse objective plus more length or more update/replay.
- More eval simulations as a standalone fix; the existing sweep already showed
  tie removal with action collapse.

Go:

- One small objective/curriculum change that puts survival/loss-delay into the
  training target while still scoring on raw heldout Pong.
- Or one exploration/data-distribution probe that must report the same
  survival mean/median/p90, shaped score, raw score, and action histograms
  before any quality claim.
