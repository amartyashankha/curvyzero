# 2026-05-09 LightZero Dummy Pong Sparse Settings Probe

## Question

Do the board-game-style sparse terminal settings run on dummy Pong without
scaling huge, and do they produce any useful fixed-horizon checkpoint signal?

This is intentionally a tiny Modal probe: `1024` LightZero env steps,
`16` train iterations, one collector env, one evaluator env, and 8 MCTS
simulations. No pytest was run.

## Train Command

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 1024 --pong-episode-max-steps 120 --max-train-iter 16 --num-simulations 8 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 9 --run-id lz-dpong-sparse-h120-lag1-s9 --attempt-id train-1024x16-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-5CMO2337y7AB9aWDGUIagx`

Result:

```text
ok: true
run_id: lz-dpong-sparse-h120-lag1-s9
attempt_id: train-1024x16-sparse-h120
summary_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/attempts/train-1024x16-sparse-h120/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/attempts/train-1024x16-sparse-h120/train/episodes.jsonl
training_signals_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/attempts/train-1024x16-sparse-h120/train/lightzero_training_signals.json
lightzero_artifacts_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/attempts/train-1024x16-sparse-h120/train/lightzero_artifacts_manifest.json
problems: []
```

`td_steps=120` completed cleanly with `game_segment_length=50`, so no
`td_steps=50` fallback was needed.

## Config Surface

| Field | Value |
| --- | --- |
| `env` | `dummy_pong_lag1` |
| `feature_mode` | `tabular_ego` |
| `opponent_policy` | `lagged_track_ball_1` |
| `max_env_step` | `1024` |
| `max_env_step_role` | `lightzero_training_budget` |
| `requested_pong_episode_max_steps` | `120` |
| `effective_pong_episode_max_steps` | `120` |
| `env_max_steps` | `120` |
| `max_train_iter` | `16` |
| `collector_env_num` / `evaluator_env_num` | `1` / `1` |
| `n_episode` / `n_evaluator_episode` | `2` / `4` |
| `num_simulations` | `8` |
| `batch_size` | `32` |
| `update_per_collect` | `8` |
| `game_segment_length` | `50` |
| `td_steps` | `120` |
| `num_unroll_steps` | `5` |
| `discount_factor` | `1.0` |
| `reward_support_range` | `[-5.0, 6.0, 1.0]` |
| `value_support_range` | `[-5.0, 6.0, 1.0]` |
| `dynamic_seed` | `true` |

## Train Telemetry

Trainer-side scorecard:

| Metric | Value |
| --- | ---: |
| Episodes | 12 |
| Wins / losses / timeouts | 4 / 7 / 1 |
| Survival steps mean | 22.8333 |
| Survival steps median | 13.5 |
| Survival steps p90 | 28.9 |
| Survival steps max | 120.0 |
| Score return mean | -0.25 |
| Shaped loss-delay return mean | -0.2191 |
| Truncation rate | 0.0833 |
| Unique seeds | 9 |
| Top seed frequency | 2 / 12 |
| Seed dominance warning | `false` |

Train action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` LightZero ego | 126 | 130 | 18 |
| `player_1` lagged opponent | 63 | 156 | 55 |

Checkpoint refs:

```text
ckpt_best: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: 5685915f6152815079eda6d31f6d38888a70eff910512837a45f6f617bb5d162

iteration_0: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/checkpoints/lightzero/iteration_0.pth.tar
  sha256: b7984f5db440798d104260d0a3a7bdc6f7e606fcbda377012292b05ff632cd1b

iteration_16: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/checkpoints/lightzero/iteration_16.pth.tar
  sha256: df463af5728cd78672f69870c56860d55ffdb3b9cc6e34d09ac628f48e7a2283
```

## Tiny Independent MCTS Scorecard

The independent scorecard wrapper still names the evaluation horizon
`--max-env-step`, so this uses `--max-env-step 120` as the eval horizon.

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter16-sparse-h120=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/checkpoints/lightzero/iteration_16.pth.tar --episodes 4 --seed 1701 --split-id dummy_pong_sparse_h120_lag1_probe_v0 --eval-id mcts-scoreboard-iter16-sparse-h120-s1701-small --max-env-step 120 --num-simulations 8 --run-id lz-dpong-sparse-h120-lag1-s9 --attempt-id train-1024x16-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-hWskc74M6dVB0oP941S2sC`

Artifacts:

```text
eval_dir: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/attempts/train-1024x16-sparse-h120/eval/mcts-scoreboard-iter16-sparse-h120-s1701-small
summary_json: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/attempts/train-1024x16-sparse-h120/eval/mcts-scoreboard-iter16-sparse-h120-s1701-small/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/attempts/train-1024x16-sparse-h120/eval/mcts-scoreboard-iter16-sparse-h120-s1701-small/episodes.jsonl
```

Eval config:

| Field | Value |
| --- | --- |
| Episodes per match | `4` |
| Total episodes | `60` |
| Paired seats | `true` |
| Eval horizon field | `max_env_step` |
| Eval horizon value | `120` |
| MCTS simulations | `8` |
| Strict load | `ok=true`, `missing_keys=[]`, `unexpected_keys=[]` |

Learned checkpoint rows:

| Row | Episodes | Wins / losses / truncs | Survival mean / median / p90 | Shaped | Score | LZ actions |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `iter16-sparse-h120` vs `lagged_track_ball_1` | 8 | 3 / 4 / 1 | 24.75 / 8 / 57 | -0.09688 | -0.125 | `[35, 163, 0]` |
| `iter16-sparse-h120` vs `random_uniform` | 8 | 4 / 4 / 0 | 16.25 / 19 / 19 | 0.03385 | 0 | `[36, 94, 0]` |
| `iter16-sparse-h120` vs `track_ball` | 8 | 0 / 8 / 0 | 16.25 / 19 / 22.3 | -0.9323 | -1 | `[32, 98, 0]` |

Aggregate learned eval action histogram:
`[103, 355, 0]` for `[up, stay, down]`.

Baseline sanity context:

| Row | Episodes | Wins / losses / truncs | Survival mean / median / p90 | Actions |
| --- | ---: | ---: | --- | --- |
| `lagged_track_ball_1` vs `lagged_track_ball_1` | 4 | 3 / 0 / 1 | 36 / 8 / 86.4 | `[22, 232, 34]` |
| `random_uniform` vs `lagged_track_ball_1` | 8 | 4 / 4 / 0 | 10.75 / 8 / 19 | random `[27, 33, 26]`, lagged `[32, 17, 37]` |
| `random_uniform` vs `random_uniform` | 4 | 4 / 0 / 0 | 24.5 / 24.5 / 41 | `[60, 53, 83]` |
| `track_ball` vs `track_ball` | 4 | 0 / 0 / 4 | 120 / 120 / 120 | `[162, 666, 132]` |

## First-N Debug MCTS Rows

The scorecard did not collapse to one action, but it did collapse away from
`down`, so I ran the tiny first-N debug slice against `random_uniform`.

Checkpoint fetch:

```text
modal volume get curvyzero-runs training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/checkpoints/lightzero/iteration_16.pth.tar /private/tmp/curvy-lz-sparse-h120-iter16.pth.tar --force
```

Debug command:

```text
PYTHONPATH=src uv run --with LightZero==0.2.0 python scripts/summarize_lightzero_pong_scorecards.py debug-mcts --checkpoint lightzero:iter16-sparse-h120=/private/tmp/curvy-lz-sparse-h120-iter16.pth.tar --rows 12 --seed 1701 --opponent-policy random_uniform --max-env-step 120 --num-simulations 8 --format md
```

Rows:

| row | seed | obs | logits | visits | action |
| --- | --- | --- | --- | --- | --- |
| 0 | 1701 | `step=0 dx=6 dy=2 vx=-1 vy=-1` | `[-0.01606,0.009217,0.02623]` | `[3,2,3]` | `0:up` |
| 1 | 1701 | `step=1 dx=5 dy=2 vx=-1 vy=-1` | `[-0.01561,0.008743,0.02633]` | `[2,3,3]` | `1:stay` |
| 2 | 1701 | `step=2 dx=4 dy=1 vx=-1 vy=-1` | `[-0.01589,0.009175,0.02587]` | `[2,3,3]` | `1:stay` |
| 3 | 1701 | `step=3 dx=3 dy=0 vx=-1 vy=-1` | `[-0.016,0.00969,0.02507]` | `[2,3,3]` | `1:stay` |
| 4 | 1701 | `step=4 dx=2 dy=-1 vx=-1 vy=-1` | `[-0.01639,0.01035,0.02447]` | `[2,3,3]` | `1:stay` |
| 5 | 1701 | `step=5 dx=1 dy=-2 vx=-1 vy=-1` | `[-0.01688,0.01122,0.02372]` | `[2,3,3]` | `1:stay` |
| 6 | 1701 | `step=6 dx=0 dy=-3 vx=-1 vy=-1` | `[-0.01705,0.01178,0.02302]` | `[2,3,3]` | `1:stay` |
| 7 | 1701 | `step=7 dx=-1 dy=-2 vx=-1 vy=1` | `[-0.02206,0.00612,0.03909]` | `[2,3,3]` | `1:stay` |
| 8 | 1702 | `step=0 dx=6 dy=-1 vx=1 vy=-1` | `[-0.02982,0.02276,0.0219]` | `[2,3,3]` | `1:stay` |
| 9 | 1702 | `step=1 dx=7 dy=-2 vx=1 vy=-1` | `[-0.0301,0.02314,0.02163]` | `[2,3,3]` | `1:stay` |
| 10 | 1702 | `step=2 dx=8 dy=-3 vx=1 vy=-1` | `[-0.03036,0.0237,0.02105]` | `[2,3,3]` | `1:stay` |
| 11 | 1702 | `step=3 dx=9 dy=-4 vx=1 vy=-1` | `[-0.03069,0.02438,0.02032]` | `[2,3,3]` | `1:stay` |

Read: the first debug row has a visit-count tie between `up` and `down`, then
the next 11 rows tie `stay` and `down` at `[2,3,3]`. The adapter's deterministic
selection chooses the first maximum, so these rows explain why `down` can vanish
from the scorecard even when its logits and visits are not obviously dead.

## Plain Read

The sparse terminal settings are now exposed and runnable in the small fixed
horizon lane. `td_steps=120` did not break despite `game_segment_length=50`, and
the config surface persisted the intended board-game-style knobs:
`discount_factor=1.0`, `td_steps=120`, `num_unroll_steps=5`, and integer
reward/value supports over `[-5, 6]`.

The checkpoint is not a useful controller. Trainer-side actions are mixed, but
independent eval-mode MCTS loses to `track_ball`, is slightly below the lagged
opponent, ties random on this tiny sample, and chooses no `down` actions. The
debug rows point at weak/tied 8-simulation search plus deterministic tie
selection, not a robust sparse-setting improvement.
