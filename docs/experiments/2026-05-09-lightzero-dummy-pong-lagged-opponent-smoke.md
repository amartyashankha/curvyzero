# 2026-05-09 LightZero MuZero Dummy Pong Lagged-Opponent Smoke

## Question

Can the next Modal whole-job **LightZero MuZero** dummy Pong lane train against
the scoreable `lagged_track_ball_1` opponent instead of `random_uniform`, and
does the final checkpoint improve under the independent MCTS scorecard?

## Timed-Out Scaled Attempt

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 2048 --max-train-iter 32 --num-simulations 8 --batch-size 32 --update-per-collect 1 --n-evaluator-episode 8 --collector-env-num 1 --evaluator-env-num 1 --seed 5
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-NTmqu91z38AXj4fUXceQDb`

Result: timed out at the wrapper's 1200 second limit before emitting a final
JSON result. No final checkpoint was scored from this run. This is a useful
cost/opacity blocker for `2048/32` against the lagged opponent, not a Pong
learning result.

## Completed Tiny Train

Fallback command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy lagged_track_ball_1 --max-env-step 512 --max-train-iter 8 --num-simulations 8 --batch-size 32 --update-per-collect 1 --n-evaluator-episode 8 --collector-env-num 1 --evaluator-env-num 1 --seed 6
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-LAPulWZ28eI17bOwZthc09`

Run:
`lz-dpong-20260509T161735Z-be3728357aad`

Attempt:
`attempt-20260509T161735Z-c74c7ccecc74`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/train/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/train/episodes.jsonl
training_signals: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/train/lightzero_training_signals.json
lightzero_artifacts: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/train/lightzero_artifacts_manifest.json
iteration_8: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/checkpoints/lightzero/iteration_8.pth.tar
```

`iteration_8.pth.tar` sha256:
`e3b7d2da5a32ddcecb5b09cb793981cf76711f74f6bd0aba8976f5b0d9d64e8e`.

Trainer-side scorecard:

| Metric | Value |
| --- | ---: |
| Episodes | 73 |
| Wins / losses / timeouts | 38 / 27 / 8 |
| Mean survival steps | 67.6027 |
| Median / p90 survival steps | 8 / 415.6 |
| Mean score return | 0.1507 |
| Mean shaped loss-delay return | 0.1548 |
| Truncation rate | 0.1096 |

Trainer action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 806 | 4083 | 46 |
| `player_1` | 354 | 4202 | 379 |

Seed gate:

| Field | Value |
| --- | --- |
| `seed_dominance_warning` | `false` |
| `seed_unique_count` | 65 |
| `seed_most_common` | 7 |
| `seed_most_common_count` | 2 / 73 |
| `seed_most_common_fraction` | 0.0274 |
| `seed_top5` | `7:2`, `8:2`, `9:2`, `10:2`, `11:2` |

Read: the tiny train passes the seed-diversity gate. The trainer-side win
count is positive against the lagged scripted opponent, but action counts
already show a strong `stay` bias.

## Paired Independent MCTS Scorecard

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter8-lagged-smoke=ref:training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/checkpoints/lightzero/iteration_8.pth.tar --episodes 16 --seed 1701 --split-id dummy_pong_lagged_opponent_smoke_heldout_v0 --eval-id mcts-scoreboard-lagged-opponent-512x8-iter8 --max-env-step 512 --num-simulations 8 --run-id lz-dpong-20260509T161735Z-be3728357aad --attempt-id attempt-20260509T161735Z-c74c7ccecc74
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-8I8UyjgpxB0kZK3F6AwmK6`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/eval/mcts-scoreboard-lagged-opponent-512x8-iter8/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/eval/mcts-scoreboard-lagged-opponent-512x8-iter8/episodes.jsonl
```

Config:

| Field | Value |
| --- | --- |
| Episodes per match | 16 |
| Total episodes | 240 |
| Eval seed | 1701 |
| Paired seats | `true` |
| Max env step | 512 |
| MCTS simulations | 8 |
| Strict load | `ok=true`, `missing_keys=[]`, `unexpected_keys=[]` |

Learned rows:

| Opponent | Episodes | LZ wins | Opp wins | LZ mean reward | LZ shaped | Mean steps | Truncs | LZ actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 32 | 18 | 9 | 0.28125 | 0.28378 | 91.5625 | 5 | `[0,2930,0]` |
| `random_uniform` | 32 | 21 | 11 | 0.31250 | 0.31754 | 14.5313 | 0 | `[0,465,0]` |
| `track_ball` | 32 | 0 | 19 | -0.59375 | -0.58710 | 214.8125 | 13 | `[0,6874,0]` |

Aggregate learned paired action histogram:
`[0,10269,0]`.

Baseline sanity rows looked coherent for the small horizon:
`random_uniform_vs_lagged_track_ball_1` was near even (`17-15` random over
lagged), `random_uniform_vs_track_ball` was `0-32`, and
`track_ball_vs_track_ball` truncated all 16 games at 512 steps.

## Player0-Only MCTS Control

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter8-lagged-smoke=ref:training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/checkpoints/lightzero/iteration_8.pth.tar --episodes 8 --seed 1701 --split-id dummy_pong_lagged_opponent_smoke_player0_only_v0 --eval-id mcts-scoreboard-lagged-opponent-512x8-iter8-player0-only --max-env-step 512 --num-simulations 8 --run-id lz-dpong-20260509T161735Z-be3728357aad --attempt-id attempt-20260509T161735Z-c74c7ccecc74 --no-paired-seats
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-I2iB3zf5K92dbYBIkemSPG`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/eval/mcts-scoreboard-lagged-opponent-512x8-iter8-player0-only/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/attempts/attempt-20260509T161735Z-c74c7ccecc74/eval/mcts-scoreboard-lagged-opponent-512x8-iter8-player0-only/episodes.jsonl
```

Learned rows:

| Opponent | Episodes | LZ wins | Opp wins | LZ mean reward | LZ shaped | Mean steps | Truncs | LZ actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 8 | 5 | 2 | 0.375 | 0.37695 | 75.125 | 1 | `[0,601,0]` |
| `random_uniform` | 8 | 6 | 2 | 0.500 | 0.50330 | 13.5 | 0 | `[0,108,0]` |
| `track_ball` | 8 | 0 | 5 | -0.625 | -0.61877 | 198.375 | 3 | `[0,1587,0]` |

Aggregate learned player0-only action histogram:
`[0,2296,0]`.

## Interpretation

This is better than the post-deep-seed-fix random-opponent run on small
scoreboard win counts against `random_uniform` and `lagged_track_ball_1`.
The prior paired random-opponent run lost to random (`27-37`) and lagged
(`27-33`) at `1024/16`; this tiny lagged-opponent run wins those rows at
`512/8` (`21-11` and `18-9`). The player0-only control also wins its small
random and lagged rows.

It is not a clean policy-quality improvement. The comparison is smaller and
shorter, the scaled `2048/32` lane timed out before artifacts, and the
independent MCTS policy collapsed to all `stay` actions in both paired and
player0-only eval. `track_ball` still beats it decisively. Treat this as honest
evidence that lagged-opponent training can complete cheaply and produce better
small-horizon scorecard counts, while also preserving the main caveat: the
policy remains degenerate.

No pytest.
