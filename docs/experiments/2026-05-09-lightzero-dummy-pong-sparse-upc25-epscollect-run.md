# 2026-05-09 LightZero Dummy Pong Sparse UPC25 Epsilon-Collect Run

## Question

After sparse rung0/rung1 and UPC25 showed no heldout learning under the true
sparse Pong reward, does a smallest data-distribution change help: a short
random warmup plus collect-time epsilon-greedy exploration?

This run preserves the environment reward and MuZero training target:

```text
env.step reward = +1 ego scores, -1 ego loses, 0 otherwise
```

Survival/loss-delay is telemetry only. No shaped reward target was used.

No pytest was run.

## Decision

Chosen change: keep the UPC25 sim8 sparse setup and add only collection
exploration:

- `random_collect_episode_num=8`
- `eps_greedy_exploration_in_collect=true`
- `eps_start=0.75`
- `eps_end=0.25`
- `eps_decay=2048`
- `fixed_temperature_value=0.5`

Why: previous runs collapsed to nearly/all `up` in heldout MCTS. This is the
smallest current-knob probe that can diversify replay without changing the
reward objective or adding wrapper code.

## Train Command

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 2048 --pong-episode-max-steps 120 --max-train-iter 32 --num-simulations 8 --batch-size 64 --update-per-collect 25 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 4 --game-segment-length 120 --random-collect-episode-num 8 --eps-greedy-exploration-in-collect --eps-start 0.75 --eps-end 0.25 --eps-decay 2048 --fixed-temperature-value 0.5 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-sparse-h120-lag1-s10-upc25-epscollect --attempt-id train-2048x32-sim8-upc25-epscollect-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-MYxTxehyWrDFkygrQTGXEk`

Result:

```text
ok: true
run_id: lz-dpong-sparse-h120-lag1-s10-upc25-epscollect
attempt_id: train-2048x32-sim8-upc25-epscollect-sparse-h120
summary_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/attempts/train-2048x32-sim8-upc25-epscollect-sparse-h120/train/summary.json
episodes_ref: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/attempts/train-2048x32-sim8-upc25-epscollect-sparse-h120/train/episodes.jsonl
```

Checkpoint refs:

```text
iteration_0: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/checkpoints/lightzero/iteration_0.pth.tar
  sha256: ec91d58bd35f702d4a22ec6be9c0e52af5a7f51c4010fb05a4747d3fab56a967

iteration_50: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/checkpoints/lightzero/iteration_50.pth.tar
  sha256: 40db7d30bc73c249c3715a0ed9671d47087a630bf0a6c50a1cee182ff10fcf51

ckpt_best: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/checkpoints/lightzero/ckpt_best.pth.tar
  sha256: 9d570812494b9f0351746cb4ee11530e0bd8ef14e602bdd7a3d11ee07ab166c1
```

## Trainer-Side Telemetry

Trainer-side rows are diagnostic only; the heldout MCTS scorecard below is the
quality gate.

| Metric | Value |
| --- | ---: |
| Episodes | 20 |
| Wins / losses / timeouts | 10 / 9 / 1 |
| Survival mean / median / p90 | 21.3 / 8.0 / 43.2 |
| Shaped loss-delay mean | 0.07875 |
| Raw score mean | 0.05 |
| Unique seeds / top seed count | 12 / 2 |
| Seed dominance warning | `false` |

Train action counts for `player_0` LightZero ego:
`[288, 74, 64]` for `[up, stay, down]`.

The exploration knobs did diversify trainer-side actions relative to UPC25
without changing the target reward.

## Heldout MCTS Scorecard

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints "lightzero:iter0-epscollect=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/checkpoints/lightzero/iteration_0.pth.tar,lightzero:iter50-epscollect=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/checkpoints/lightzero/iteration_50.pth.tar,lightzero:best-epscollect=ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/checkpoints/lightzero/ckpt_best.pth.tar" --episodes 8 --seed 1701 --split-id dummy_pong_sparse_h120_lag1_upc25_epscollect_heldout_v0 --split-role heldout --eval-id mcts-scoreboard-upc25-epscollect-sim8-iter0-iter50-best-e8 --max-env-step 120 --num-simulations 8 --feature-mode tabular_ego --run-id lz-dpong-sparse-h120-lag1-s10-upc25-epscollect --attempt-id train-2048x32-sim8-upc25-epscollect-sparse-h120
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-jnREG1t3V0jyw8dsOqtbUB`

Artifacts:

```text
summary_json: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/attempts/train-2048x32-sim8-upc25-epscollect-sparse-h120/eval/mcts-scoreboard-upc25-epscollect-sim8-iter0-iter50-best-e8/summary.json
episodes_jsonl: training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/attempts/train-2048x32-sim8-upc25-epscollect-sparse-h120/eval/mcts-scoreboard-upc25-epscollect-sim8-iter0-iter50-best-e8/episodes.jsonl
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
| `lagged_track_ball_1` | 26.125 / 13.5 / 69.5 | -0.21979 | -0.25 | 5-9 | 2 | `[400, 0, 18]` |
| `random_uniform` | 12.125 / 8.0 / 19.0 | -0.21484 | -0.25 | 6-10 | 0 | `[185, 0, 9]` |
| `track_ball` | 37.25 / 19.0 / 120.0 | -0.75104 | -0.8125 | 0-13 | 3 | `[573, 0, 23]` |

Aggregate `iteration_50` learned action histogram across baseline opponents:
`[1158, 0, 50]`.

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
| `lagged_track_ball_1` | 31.625 / 19.0 / 80.5 | 0.27760 | 0.25 | 9-5 | 2 | `[170, 336, 0]` |
| `random_uniform` | 11.4375 / 8.0 / 19.0 | -0.34063 | -0.375 | 5-11 | 0 | `[68, 115, 0]` |
| `track_ball` | 30.9375 / 19.0 / 80.5 | -0.80859 | -0.875 | 0-14 | 2 | `[175, 320, 0]` |

Aggregate `iteration_0` learned action histogram across baseline opponents:
`[413, 771, 0]`.

## Read

The knobs worked as a collection-distribution intervention but did not produce
a better heldout controller:

- Trainer-side action diversity improved to `[288, 74, 64]`.
- Heldout `iteration_50` still had no `stay` actions and stayed more than 95%
  `up` across learned-vs-baseline rows.
- Raw score did not improve versus UPC25 on `lagged_track_ball_1` or
  `random_uniform` (`-0.25` for both).
- `ckpt_best` was again exactly all-up across baseline opponents.
- The initialization control remains competitive or better on the main
  lagged opponent, so the final learned checkpoint is not a promotion
  candidate.

This closes "simple random warmup / epsilon collect under the same sparse
target" as the next smallest fix. It improved replay diversity, but the
learned eval policy still collapsed.

## Next Go/Stop

Stop:

- More same-objective sparse scaling with only collection exploration changed.
- Treating trainer-side diversity as progress without heldout MCTS action
  diversity and raw score.

Go:

- A scoreable contact/angle curriculum that changes the reset/opponent
  distribution toward states where correct paddle contact and scoring pressure
  occur, while preserving sparse env reward.
- Or a explicitly labeled temporary auxiliary target path, only if LightZero
  supports it cleanly without silently changing `env.step()` reward or the
  promoted evaluation metric.
