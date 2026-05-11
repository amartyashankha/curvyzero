# 2026-05-09 LightZero Dummy Pong Post-Seed-Fix Run

## Question

After the `DummyPongLightZeroEnv` dynamic reset seed fix, does a cheap
1024-step / 16-iteration CPU LightZero MuZero run show trustworthy seed
diversity and any independent MCTS scorecard improvement?

Correction note: this is a staged learner-ego versus `random_uniform` run, not
final multiplayer self-play. Judge it by honest checkpoint curves: wins,
survival steps, shaped loss-delay score, actions, seeds, and artifact refs.

## Train

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy random_uniform --max-env-step 1024 --max-train-iter 16 --num-simulations 8 --batch-size 32 --update-per-collect 1 --n-evaluator-episode 8 --collector-env-num 1 --evaluator-env-num 1 --seed 3
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-BmPV6roYBHlz2urjPbVxWb`

Run:
`lz-dpong-20260509T153355Z-0ea60caea3e3`

Attempt:
`attempt-20260509T153355Z-f981d8701b03`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/train/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/train/episodes.jsonl
training_signals: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/train/lightzero_training_signals.json
lightzero_artifacts: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/train/lightzero_artifacts_manifest.json
iteration_16: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/checkpoints/lightzero/iteration_16.pth.tar
```

`iteration_16.pth.tar` sha256:
`75b6cccef4002f5a4b11dd548c6c5ab11f99d5ff4a7d9bb7eea6d80764c0e8c5`.

Trainer-side scorecard:

| Metric | Value |
| --- | ---: |
| Episodes | 148 |
| Wins / losses / timeouts | 18 / 130 / 0 |
| Mean survival steps | 9.3378 |
| Median / p90 survival steps | 8 / 8 |
| Mean score return | -0.7568 |
| Mean shaped loss-delay return | -0.7530 |

Trainer action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 186 | 1077 | 119 |
| `player_1` | 512 | 772 | 98 |

Seed histogram:

| Field | Value |
| --- | --- |
| `seed_dominance_warning` | `true` |
| `seed_dominance_warning_reason` | `one episode_seed accounts for at least half of rows` |
| `seed_unique_count` | 20 |
| `seed_most_common` | 3 |
| `seed_most_common_count` | 129 / 148 |
| `seed_most_common_fraction` | 0.8716 |
| `seed_top5` | `3:129`, `4:1`, `5:1`, `6:1`, `7:1` |

Read: the planned dynamic-seed trust check did not pass on this trainer
summary. The run still reports a dominant seed `3`, so the repeated-seed
telemetry problem is not cleared by this experiment. Dynamic seeds remain the
right immediate trust check; broader reset randomization should wait until
after this is understood. If/when added, mild paddle-y jitter is the first
small reset-randomization extension to try.

## Independent MCTS Scorecard

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter16=ref:training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/checkpoints/lightzero/iteration_16.pth.tar --episodes 32 --seed 1701 --split-id dummy_pong_post_seed_fix_heldout_v0 --eval-id mcts-scoreboard-post-seed-fix-1024x16-iter16 --max-env-step 1024 --num-simulations 8 --run-id lz-dpong-20260509T153355Z-0ea60caea3e3 --attempt-id attempt-20260509T153355Z-f981d8701b03
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-N2YXIo4RMYr8XXbhbw7s3L`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/eval/mcts-scoreboard-post-seed-fix-1024x16-iter16/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/eval/mcts-scoreboard-post-seed-fix-1024x16-iter16/episodes.jsonl
```

Config:

| Field | Value |
| --- | --- |
| Episodes per match | 32 |
| Total episodes | 480 |
| Eval seed | 1701 |
| Split id | `dummy_pong_post_seed_fix_heldout_v0` |
| Split role | `monitor` |
| Paired seat | `true` |
| Max env step | 1024 |
| MCTS simulations | 8 |
| Action selection | `MuZeroPolicy.eval_mode.forward` |

Baseline rows:

| Row | Episodes | Wins | Mean steps | Truncs | Actions |
| --- | ---: | --- | ---: | ---: | --- |
| `lagged_track_ball_1_vs_lagged_track_ball_1` | 32 | `lagged_track_ball_1:26` | 199.875 | 6 | `lagged_track_ball_1:[274,12276,242]` |
| `lagged_track_ball_1_vs_track_ball` | 64 | `lagged_track_ball_1:0`, `track_ball:52` | 204.688 | 12 | `lagged_track_ball_1:[354,12329,417]`, `track_ball:[356,12333,411]` |
| `random_uniform_vs_lagged_track_ball_1` | 64 | `random_uniform:30`, `lagged_track_ball_1:34` | 12.125 | 0 | `random_uniform:[251,260,265]`, `lagged_track_ball_1:[266,207,303]` |
| `random_uniform_vs_random_uniform` | 32 | `random_uniform:32` | 15.906 | 0 | `random_uniform:[317,329,372]` |
| `random_uniform_vs_track_ball` | 64 | `random_uniform:0`, `track_ball:64` | 25.531 | 0 | `random_uniform:[540,558,536]`, `track_ball:[604,409,621]` |
| `track_ball_vs_track_ball` | 32 | `track_ball:0` | 1024.000 | 32 | `track_ball:[1172,63222,1142]` |

Learned MCTS rows:

| Row | Episodes | LZ wins | Opp wins | LZ mean reward | LZ shaped | Mean steps | Truncs | LZ actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lightzero_iter16_vs_lagged_track_ball_1` | 64 | 27 | 37 | -0.15625 | -0.15315 | 10.578 | 0 | `[0,134,543]` |
| `lightzero_iter16_vs_random_uniform` | 64 | 31 | 33 | -0.03125 | -0.02764 | 13.672 | 0 | `[0,196,679]` |
| `lightzero_iter16_vs_track_ball` | 64 | 0 | 57 | -0.890625 | -0.88329 | 127.031 | 7 | `[0,1347,6783]` |

Aggregate learned MCTS action histogram across the three learned rows:
`[0,1677,8005]`. Following the existing scorecard convention, this means zero
action-index-0 choices and heavy action-index-2 collapse. That is different
from the earlier up-only collapse, but it is still a degenerate policy shape,
not a useful controller.

## Interpretation

There is no reliable improvement claim here. The paired heldout MCTS scorecard
has `iteration_16` slightly below random (`31-33`), below
`lagged_track_ball_1` (`27-37`), and far below `track_ball` (`0-57`, with seven
paired-match truncations). Baseline sanity rows look broadly consistent with
prior scorecards: `track_ball` dominates random, `track_ball_vs_track_ball`
truncates at the horizon, and random-vs-random is balanced by construction.

The train-side seed histogram is still dominated by one seed, so the central
post-seed-fix trust check failed for this run. Treat the scorecard as useful
evidence that the final checkpoint does not currently improve, but not as proof
that the seed fix fully worked in the training telemetry path.

## Player_0-Only MCTS Control

This control scores the same post-seed-fix `iteration_16` checkpoint only in
the current training seat: checkpoint as `player_0`, baseline as `player_1`.

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-BM3gHko0cO0SbaU2rQwbLl`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/eval/mcts-scoreboard-post-seed-fix-1024x16-iter16-player0-only/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T153355Z-0ea60caea3e3/attempts/attempt-20260509T153355Z-f981d8701b03/eval/mcts-scoreboard-post-seed-fix-1024x16-iter16-player0-only/episodes.jsonl
```

Rows:

| Opponent | LZ wins | Opp wins | Mean steps | Truncs | LZ actions |
| --- | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 16 | 16 | 10.406 | 0 | `[0,60,273]` |
| `random_uniform` | 15 | 17 | 12.812 | 0 | `[0,97,313]` |
| `track_ball` | 0 | 28 | 142.219 | 4 | `[0,547,4004]` |

Read: player_0-only does not rescue the checkpoint. Seat pairing was not
hiding a good training-seat policy. Action collapse remains: no action index 0,
heavy action index 2.

No pytest.
