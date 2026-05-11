# 2026-05-09 LightZero MuZero Dummy Pong Raster Flat Smoke

## Question

Can the Modal **LightZero MuZero** dummy Pong lane train from the
`raster_flat` visual bridge, mirror raster-compatible checkpoints, and run an
independent MCTS scorecard against the matching raster checkpoint?

This is a mechanical smoke only. It is separate from official Atari Pong and
from the tabular sparse ladder. No pytest was run.

Update after the sparse ladder: keep the current sparse ladder explicitly
`tabular_ego`. The `raster_flat` config/import bridge passed on Modal, but old
`tabular_ego` checkpoints are not compatible with the raster model input
shape. Future raster scorecards must pass `--feature-mode raster_flat`
explicitly and must score raster-trained checkpoints only, so a silent
tabular/raster mismatch cannot masquerade as a policy result.

## Train Command

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode raster_flat --opponent-policy lagged_track_ball_1 --max-env-step 512 --pong-episode-max-steps 120 --max-train-iter 8 --num-simulations 8 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-raster-flat-h120-lag1-s10 --attempt-id train-512x8-raster-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-8gdzlxRSLTrE5MN4gbUZbE`

Result:

```text
ok: true
run_id: lz-dpong-raster-flat-h120-lag1-s10
attempt_id: train-512x8-raster-h120
summary_ref: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/train/episodes.jsonl
training_signals_ref: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/train/lightzero_training_signals.json
lightzero_artifacts_ref: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/train/lightzero_artifacts_manifest.json
problems: []
```

## Config Surface

| Field | Value |
| --- | --- |
| `algorithm` | `LightZero MuZero` |
| `env` | `dummy_pong_lag1` |
| `feature_mode` | `raster_flat` |
| `feature_schema_id` | `dummy_pong_lightzero_raster_flat_v0` |
| `observation_shape` | `135` |
| `model_type` | `mlp` |
| `opponent_policy` | `lagged_track_ball_1` |
| `max_env_step` | `512` |
| `max_env_step_role` | `lightzero_training_budget` |
| `pong_episode_max_steps` | `120` explicit |
| `max_train_iter` | `8` |
| `num_simulations` | `8` |
| `batch_size` | `32` |
| `update_per_collect` | `8` |
| `n_episode` / `n_evaluator_episode` | `2` / `4` |
| `game_segment_length` | `50` |
| `td_steps` / `num_unroll_steps` | `120` / `5` |
| `discount_factor` | `1.0` |
| `reward_support_range` | `[-5.0, 6.0, 1.0]` |
| `value_support_range` | `[-5.0, 6.0, 1.0]` |

## Train Telemetry

Trainer-side scorecard:

| Metric | Value |
| --- | ---: |
| Episodes | 8 |
| Wins / losses / timeouts | 3 / 5 / 0 |
| Survival mean / median / p90 | 9.375 / 8.0 / 11.3 |
| Survival std / max | 3.6379 / 19.0 |
| Score return mean / std | -0.25 / 0.9682 |
| Shaped loss-delay mean / std | -0.22344 / 0.9478 |
| Truncation rate | 0.0 |
| Unique seeds | 5 |
| Top seed frequency | 2 / 8 |
| Seed dominance warning | `false` |

Train action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` LightZero ego | 35 | 27 | 13 |
| `player_1` lagged opponent | 21 | 19 | 35 |

Checkpoint refs:

```text
ckpt_best: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: 89a3c9cb10ecea8566b7b0e535f1979913e6a495eb4a051370ef94fb77135486

iteration_0: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/checkpoints/lightzero/iteration_0.pth.tar
  sha256: 6ae3ba976418f1d0f7385886b7aa5f671a11bc93be912f8ec863dc2ca00ddc5a

iteration_8: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/checkpoints/lightzero/iteration_8.pth.tar
  sha256: cf2109616d641ce1a926ddc766e6bb08786f05130235380c6ad746a74e08c5cd
```

## Independent Raster MCTS Scorecard

The scorecard used the raster-trained `iteration_8` checkpoint above. It did
not evaluate any old tabular checkpoint.

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter8-raster-h120=ref:training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/checkpoints/lightzero/iteration_8.pth.tar --episodes 4 --seed 1701 --split-id dummy_pong_raster_flat_h120_lag1_smoke_v0 --eval-id mcts-scoreboard-iter8-raster-h120-s1701-small --max-env-step 120 --num-simulations 8 --feature-mode raster_flat --run-id lz-dpong-raster-flat-h120-lag1-s10 --attempt-id train-512x8-raster-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-vzu2WJIrreEKJHZcWOOTM1`

Artifacts:

```text
eval_dir: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/eval/mcts-scoreboard-iter8-raster-h120-s1701-small
summary_json: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/eval/mcts-scoreboard-iter8-raster-h120-s1701-small/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/eval/mcts-scoreboard-iter8-raster-h120-s1701-small/episodes.jsonl
```

Scorecard config:

| Field | Value |
| --- | --- |
| Episodes per match | `4` |
| Total episodes | `60` |
| Paired seats | `true` |
| Eval horizon field | `max_env_step` |
| Eval horizon value | `120` |
| MCTS simulations | `8` |
| Strict load | `ok=true`, `missing_keys=[]`, `unexpected_keys=[]` |
| Feature mode | `raster_flat` |
| Feature schema | `dummy_pong_lightzero_raster_flat_v0` |

Learned checkpoint rows:

| Opponent | Episodes | LZ wins | Opp wins | LZ score mean | LZ shaped mean | Survival mean / median / p90 | Truncs | LZ actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `lagged_track_ball_1` | 8 | 4 | 2 | 0.25 | 0.25833 | 37.375 / 8.0 / 120.0 | 2 | `[181,118,0]` |
| `random_uniform` | 8 | 1 | 7 | -0.75 | -0.70365 | 12.125 / 8.0 / 19.0 | 0 | `[87,10,0]` |
| `track_ball` | 8 | 0 | 8 | -1.0 | -0.90938 | 21.75 / 19.0 / 30.0 | 0 | `[156,18,0]` |

Aggregate learned eval action histogram:
`[424,146,0]` for `[up, stay, down]`.

Baseline sanity context:

| Row | Result | Survival mean / p90 | Truncs | Actions |
| --- | --- | --- | ---: | --- |
| `lagged_track_ball_1` vs `lagged_track_ball_1` | 3 wins for lagged row | 36.0 / 86.4 | 1 | `[22,232,34]` |
| `random_uniform` vs `lagged_track_ball_1` | `4-4` | 10.75 / 19.0 | 0 | random `[27,33,26]`, lagged `[32,17,37]` |
| `random_uniform` vs `random_uniform` | 4 random wins in self row | 24.5 / 41.0 | 0 | `[60,53,83]` |
| `track_ball` vs `track_ball` | no scorer, all timeouts | 120.0 / 120.0 | 4 | `[162,666,132]` |

## Plain Read

The raster training path is mechanically working: Modal called LightZero
MuZero training with `feature_mode=raster_flat`, wrote raster-compatible
checkpoints, and the independent MCTS scorecard strict-loaded the matching
`iteration_8` checkpoint with the raster feature schema.

This is not a policy-quality win. The tiny scorecard beat the lagged target on
this paired row, but lost badly to `random_uniform`, lost all rows to
`track_ball`, and the learned MCTS policy selected zero `down` actions across
all learned rows. Treat this as a visual bridge smoke pass only, not a reason
to scale.
