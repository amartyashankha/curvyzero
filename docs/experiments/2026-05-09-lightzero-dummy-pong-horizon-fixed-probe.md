# 2026-05-09 LightZero Dummy Pong Horizon-Fixed Probe

## Question

With the `pong_episode_max_steps` patch in place, can a small lagged-opponent
run use `max_env_step` as the LightZero train budget while keeping the dummy
Pong episode horizon fixed at `120`?

This is a small Modal probe, not a scale run. No pytest was run.

## Train Command

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 1024 --pong-episode-max-steps 120 --max-train-iter 16 --num-simulations 8 --batch-size 32 --update-per-collect 4 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --seed 8 --run-id lz-dpong-hfixed120-lag1-s8 --attempt-id train-1024x16-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-grNtbuxOIvll10JFxPK9yJ`

Result:

```text
ok: true
run_id: lz-dpong-hfixed120-lag1-s8
attempt_id: train-1024x16-h120
summary_ref: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/attempts/train-1024x16-h120/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/attempts/train-1024x16-h120/train/episodes.jsonl
training_signals_ref: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/attempts/train-1024x16-h120/train/lightzero_training_signals.json
lightzero_artifacts_ref: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/attempts/train-1024x16-h120/train/lightzero_artifacts_manifest.json
```

Config fields persisted in the train summary:

| Field | Value |
| --- | --- |
| `max_env_step` | `1024` |
| `max_env_step_role` | `lightzero_training_budget` |
| `requested_pong_episode_max_steps` | `120` |
| `pong_episode_max_steps` | `120` |
| `effective_pong_episode_max_steps` | `120` |
| `pong_episode_max_steps_source` | `explicit` |
| `game_segment_length` | `50` |
| `n_episode` | `2` |

## Train Telemetry

Trainer-side scorecard:

| Metric | Value |
| --- | ---: |
| Episodes | 24 |
| Wins / losses / timeouts | 9 / 13 / 2 |
| Survival steps mean | 21.0 |
| Survival steps median | 8.0 |
| Survival steps p90 | 34.4 |
| Survival steps max | 120.0 |
| Score return mean | -0.1667 |
| Shaped loss-delay return mean | -0.1372 |
| Truncation rate | 0.0833 |
| Unique seeds | 18 |
| Top seed frequency | 2 / 24 |

Train action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` LightZero ego | 385 | 86 | 33 |
| `player_1` lagged opponent | 93 | 301 | 110 |

Checkpoint refs:

```text
ckpt_best: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: c97b84ac2ef902ca7f4c72d59ea0aa79d13a7a78f137fbb9238fa2ac06082c5e

iteration_0: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/checkpoints/lightzero/iteration_0.pth.tar
  sha256: d28b17e1affc14cc664c5719075b9ad963876d3f130cb34c42204cb407615bfa

iteration_16: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/checkpoints/lightzero/iteration_16.pth.tar
  sha256: cb771abd95b87b0337530c719fdbec6e3bf10a730edcd59c70c3dc40f0516ec5
```

Read: the horizon split worked. The training budget remained `1024`, while the
episode cap was explicitly fixed at `120`; trainer-side survival max hit exactly
`120`.

## Tiny Independent MCTS Scorecard

The MCTS scorecard wrapper does not currently expose a separate
`pong_episode_max_steps` flag. In this codepath the eval horizon is still named
`--max-env-step`, so this probe used `--max-env-step 120` for evaluation.

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter16-hfixed120=ref:training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/checkpoints/lightzero/iteration_16.pth.tar --episodes 4 --seed 1701 --split-id dummy_pong_hfixed120_lag1_probe_v0 --eval-id mcts-scoreboard-iter16-hfixed120-s1701-small --max-env-step 120 --num-simulations 8 --run-id lz-dpong-hfixed120-lag1-s8 --attempt-id train-1024x16-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-HqvSYhvAwmr1JgwH2i7QJZ`

Artifacts:

```text
eval_dir: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/attempts/train-1024x16-h120/eval/mcts-scoreboard-iter16-hfixed120-s1701-small
summary_json: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/attempts/train-1024x16-h120/eval/mcts-scoreboard-iter16-hfixed120-s1701-small/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-hfixed120-lag1-s8/attempts/train-1024x16-h120/eval/mcts-scoreboard-iter16-hfixed120-s1701-small/episodes.jsonl
```

Eval config:

| Field | Value |
| --- | --- |
| Episodes per match | 4 |
| Total episodes | 60 |
| Paired seats | `true` |
| Eval horizon field | `max_env_step` |
| Eval horizon value | `120` |
| `config.max_steps` | `120` |
| `independent_eval_max_steps` | `120` |
| MCTS simulations | 8 |
| Strict load | `ok=true`, `missing_keys=[]`, `unexpected_keys=[]` |

Learned checkpoint rows:

| Row | Episodes | Wins / losses / truncs | Survival mean / median / p90 | Shaped | Score | LZ actions |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `iter16-hfixed120` vs `lagged_track_ball_1` | 8 | 4 / 2 / 2 | 37.38 / 8 / 120 | 0.2583 | 0.25 | `[299, 0, 0]` |
| `iter16-hfixed120` vs `random_uniform` | 8 | 1 / 7 / 0 | 12.12 / 8 / 19 | -0.7036 | -0.75 | `[97, 0, 0]` |
| `iter16-hfixed120` vs `track_ball` | 8 | 0 / 8 / 0 | 21.75 / 19 / 30 | -0.9094 | -1.0 | `[174, 0, 0]` |

Aggregate learned eval action histogram:
`[570, 0, 0]` for `[up, stay, down]`.

Baseline sanity context:

| Row | Episodes | Wins / losses / truncs | Survival mean / median / p90 | Actions |
| --- | ---: | ---: | --- | --- |
| `random_uniform` vs `lagged_track_ball_1` | 8 | 4 / 4 / 0 | 10.75 / 8 / 19 | random `[27, 33, 26]`, lagged `[32, 17, 37]` |
| `lagged_track_ball_1` vs `lagged_track_ball_1` | 4 | 3 / 0 / 1 | 36 / 8 / 86.4 | lagged `[22, 232, 34]` |
| `track_ball` vs `track_ball` | 4 | 0 / 0 / 4 | 120 / 120 / 120 | track `[162, 666, 132]` |

## Interpretation

The fixed-horizon patch behaves as intended in this small lane: train
`max_env_step` is reported as a training budget, and the effective environment
horizon is fixed at `120`.

The learned checkpoint's tiny paired MCTS scorecard is mixed. It beats the
lagged row on this small sample, but loses badly to `random_uniform` and
`track_ball`. More importantly, eval-mode MCTS collapsed to a single action:
all learned actions were `up`. Treat this as a successful horizon-contract
probe, not evidence of robust policy quality.
