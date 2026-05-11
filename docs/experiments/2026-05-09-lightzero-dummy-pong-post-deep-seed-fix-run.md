# 2026-05-09 LightZero Dummy Pong Post-Deep-Seed-Fix Run

## Question

After making config `dynamic_seed=True` authoritative over env-manager
`seed(..., dynamic_seed=False)`, does the modest 1024-step / 16-iteration
LightZero MuZero trust check show real train seed diversity, and does the
resulting checkpoint improve under independent MCTS scoring?

Correction note: this is a staged learner-ego versus `random_uniform` run, not
final multiplayer self-play. MuZero data comes from repeated environment
interaction, so more actors/episodes/steps are available later, but this result
must be judged by honest checkpoint curves: wins, survival steps, shaped
loss-delay score, actions, seeds, and artifact refs.

## Train

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode tabular_ego --opponent-policy random_uniform --max-env-step 1024 --max-train-iter 16 --num-simulations 8 --batch-size 32 --update-per-collect 1 --n-evaluator-episode 8 --collector-env-num 1 --evaluator-env-num 1 --seed 4
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-zYhiKPXQKKijsOXbN29aiK`

Run:
`lz-dpong-20260509T154530Z-b049f29edb64`

Attempt:
`attempt-20260509T154530Z-ca60e4962603`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/train/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/train/episodes.jsonl
training_signals: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/train/lightzero_training_signals.json
lightzero_artifacts: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/train/lightzero_artifacts_manifest.json
iteration_16: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar
```

`iteration_16.pth.tar` sha256:
`e2dd80f8a08b15d750f5c8c643051b8e11e63eb5ce44a5ab71fee9fecaf88ee8`.

Trainer-side scorecard:

| Metric | Value |
| --- | ---: |
| Episodes | 148 |
| Wins / losses / timeouts | 84 / 64 / 0 |
| Mean survival steps | 13.1284 |
| Median / p90 survival steps | 8 / 30 |
| Mean score return | 0.1351 |
| Mean shaped loss-delay return | 0.1382 |

Trainer action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 999 | 853 | 91 |
| `player_1` | 687 | 649 | 607 |

Seed histogram:

| Field | Value |
| --- | --- |
| `seed_dominance_warning` | `false` |
| `seed_unique_count` | 131 |
| `seed_most_common` | 5 |
| `seed_most_common_count` | 2 / 148 |
| `seed_most_common_fraction` | 0.0135 |
| `seed_top5` | `5:2`, `6:2`, `7:2`, `8:2`, `9:2` |

Read: the seed-diversity gate passes. No one seed dominates, and the old
post-seed-fix failure mode of one seed owning most rows is gone in this run.

## Paired MCTS Scorecard

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter16=ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar --episodes 32 --seed 1701 --split-id dummy_pong_post_deep_seed_fix_heldout_v0 --eval-id mcts-scoreboard-post-deep-seed-fix-1024x16-iter16 --max-env-step 1024 --num-simulations 8 --run-id lz-dpong-20260509T154530Z-b049f29edb64 --attempt-id attempt-20260509T154530Z-ca60e4962603
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-GdYm0zsa8ifnsch1rUU9Vh`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/eval/mcts-scoreboard-post-deep-seed-fix-1024x16-iter16/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/eval/mcts-scoreboard-post-deep-seed-fix-1024x16-iter16/episodes.jsonl
```

Config and strict load:

| Field | Value |
| --- | --- |
| Episodes per match | 32 |
| Total episodes | 480 |
| Eval seed | 1701 |
| Split id | `dummy_pong_post_deep_seed_fix_heldout_v0` |
| Paired seat | `true` |
| Max env step | 1024 |
| MCTS simulations | 8 |
| Load state dict | `strict=true`, `ok=true`, `missing_keys=[]`, `unexpected_keys=[]` |
| Strict variant | `res_connection_in_dynamics_true` |

Baseline rows. Action histograms are raw action-id arrays as emitted by the
scorecard.

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
| `lightzero_iter16_vs_lagged_track_ball_1` | 64 | 27 | 33 | -0.09375 | -0.08897 | 78.031 | 4 | `[3140,1854,0]` |
| `lightzero_iter16_vs_random_uniform` | 64 | 27 | 37 | -0.15625 | -0.15240 | 12.641 | 0 | `[270,539,0]` |
| `lightzero_iter16_vs_track_ball` | 64 | 0 | 59 | -0.921875 | -0.91315 | 97.859 | 5 | `[3875,2388,0]` |

Aggregate learned MCTS action histogram across the three learned paired rows:
`[7285,4781,0]`.

## Player0-Only MCTS Control

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter16=ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar --episodes 32 --seed 1701 --split-id dummy_pong_post_deep_seed_fix_heldout_player0_only_v0 --eval-id mcts-scoreboard-post-deep-seed-fix-1024x16-iter16-player0-only --max-env-step 1024 --num-simulations 8 --run-id lz-dpong-20260509T154530Z-b049f29edb64 --attempt-id attempt-20260509T154530Z-ca60e4962603 --no-paired-seats
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-FriEt4BiggV7aYFoLqYgFq`

Artifacts:

```text
summary: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/eval/mcts-scoreboard-post-deep-seed-fix-1024x16-iter16-player0-only/summary.json
episodes: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/eval/mcts-scoreboard-post-deep-seed-fix-1024x16-iter16-player0-only/episodes.jsonl
```

Strict load again passed with `strict=true`, `ok=true`, no missing keys, and no
unexpected keys. Total episodes: 384. The baseline rows match the same heldout
sanity shape; learned rows are player0-only:

| Opponent | Episodes | LZ wins | Opp wins | LZ mean reward | Mean steps | Truncs | LZ actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lagged_track_ball_1` | 32 | 16 | 15 | 0.03125 | 44.562 | 1 | `[877,549,0]` |
| `random_uniform` | 32 | 14 | 18 | -0.12500 | 12.469 | 0 | `[138,261,0]` |
| `track_ball` | 32 | 0 | 29 | -0.90625 | 112.531 | 3 | `[2338,1263,0]` |

Aggregate learned player0-only action histogram: `[3353,2073,0]`.

## Strict Read

The deeper seed fix passes the modest train-side trust check: seed diversity is
real in this run, with top seed only 2/148 rows. The checkpoint-quality read is
still negative. Paired MCTS loses to random (`27-37`), loses slightly to
`lagged_track_ball_1` (`27-33` with 4 truncations), and gets crushed by
`track_ball` (`0-59`). The player0-only control does not rescue it: it is
roughly even against lag-1 (`16-15` plus one truncation), loses to random
(`14-18`), and still cannot beat `track_ball` (`0-29`).

The independent MCTS path is strict-loading the full LightZero checkpoint, so
this is no longer a load-adapter caveat. The action histograms are the live
policy-quality caveat: both paired and player0-only learned rows choose zero
action-index-2 moves. Seed plumbing looks fixed enough for this shape; policy
learning remains poor.

No pytest.
