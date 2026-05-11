# 2026-05-09 dummy pong lag-1 DAgger visual-policy smoke

## Question

Is off-trace closed-loop drift the likely reason exact-trace raster behavior
cloning stays weak against `lagged_track_ball_1`, and can the smallest
DAgger-style repair improve the visual lane?

## Setup

- Starting replay:
  `artifacts/local/dummy-pong-lag1-trace-replay-mirror-2026-05-09`.
- Starting visual policy:
  `artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-2026-05-09/checkpoint.npz`.
- New collector: `scripts/build_dummy_pong_lag1_dagger_replay.py`.
- Collector behavior: roll out the current visual checkpoint against
  `lagged_track_ball_1`, collect visited raster states for the learned ego
  seat, label each exact-scoreable state with the first action from an exact DP
  teacher using the current lagged-opponent memory, append rows to the existing
  supervised replay, retrain, and score.
- Eval gate: wins versus `lagged_track_ball_1`, random sanity, and default
  `track_ball` survival/tie diagnostics.

## Command

```sh
uv run python -m py_compile scripts/build_dummy_pong_lag1_dagger_replay.py
```

```sh
uv run python scripts/build_dummy_pong_lag1_dagger_replay.py \
  --source-replay-path artifacts/local/dummy-pong-lag1-trace-replay-mirror-2026-05-09 \
  --checkpoint-path artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-2026-05-09/checkpoint.npz \
  --episodes-per-seating 2 \
  --seed 10050921 \
  --max-rows 160 \
  --output-dir artifacts/local/dummy-pong-lag1-dagger-replay-smoke-2026-05-09
```

The first retrain used `--learning-rate 1.0`; it was unstable on this combined
replay, so the scored checkpoint used the smaller LR below.

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-dagger-replay-smoke-2026-05-09 \
  --epochs 300 \
  --learning-rate 0.05 \
  --validation-fraction 0.2 \
  --seed 10050923 \
  --output-dir artifacts/local/dummy-pong-lag1-dagger-visual-policy-lr005-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_dagger_visual_policy \
  --split-role smoke_baseline_seed \
  --checkpoint lag1_dagger_visual_lr005=artifacts/local/dummy-pong-lag1-dagger-visual-policy-lr005-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-dagger-visual-policy-scoreboard-lr005-smoke-2026-05-09
```

No pytest was run.

## Results

Compile passed.

Closed-loop collection:

| Source rows | DAgger rows | Total rows | DAgger positive terminal rows | DAgger skipped unlabelable states |
| ---: | ---: | ---: | ---: | ---: |
| 2,664 | 22 | 2,686 | 2 | 10 |

DAgger target labels:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 4 | 1 | 1 |
| `player_1` | 16 | 0 | 0 |

Combined replay target labels:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 668 | 5 | 665 |
| `player_1` | 680 | 4 | 664 |

Training, LR `0.05`:

| Split | Rows | Accuracy | Loss | Predicted up | Predicted stay | Predicted down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all rows | 2,686 | 0.9088 | 0.2544 | 1,353 | 0 | 1,333 |
| validation | 537 | 0.8827 | 0.2781 | 268 | 0 | 269 |

The LR `1.0` diagnostic run reached only 0.7074 all-row accuracy and was not
scored.

Scoreboard, seed `8050914`:

| Opponent | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lagged_track_ball_1` | 16 | 5 | 9 | 2 | 24.7500 | -0.2500 | -0.2227 |
| `random_uniform` | 16 | 11 | 5 | 0 | 18.3125 | 0.3750 | 0.3854 |
| `track_ball` | 16 | 0 | 12 | 4 | 39.4375 | -0.7500 | -0.7107 |

Previous same-seed visual rows:

| Checkpoint | `lagged_track_ball_1` wins | Random wins | `track_ball` truncations |
| --- | ---: | ---: | ---: |
| Original trace | 5/16 | 10/16 | 5/16 |
| Mirror-only | 6/16 | 11/16 | 4/16 |
| Mirror + oversample | 5/16 | 11/16 | 4/16 |
| DAgger tiny LR `0.05` | 5/16 | 11/16 | 4/16 |

## Interpretation

The DAgger shape is now implemented and usable: closed-loop visual-policy
states can be collected, exact-labeled with lag-memory-aware DP, appended to
replay, retrained, and scored through the unchanged checkpoint scoreboard.

This tiny first aggregation did not improve gameplay. It matched the
mirror-plus-oversample scoreboard row and stayed below mirror-only on the
primary lag-1 metric. The collection itself supports the off-trace-drift
hypothesis: two losing player-0 rollouts entered 10 states where no exact
teacher win was available from the current lagged-opponent memory, while the
scoreable collected labels were too few and too skewed to change the policy.

Closed-loop relabeling is still the next visual lane, but the next iteration
should collect more diverse scoreable visited states before retraining. Keep it
small: more rollout seeds, cap rows, include per-seat label histograms and
unlabelable-state counts, then compare against the same mirror-only baseline
and CEM-v2's 53/64 lag-1 row.

## Artifacts

- `artifacts/local/dummy-pong-lag1-dagger-replay-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-replay-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-lr005-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-lr005-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-scoreboard-lr005-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-scoreboard-lr005-smoke-2026-05-09/episodes.jsonl`

## Follow-ups

- Keep closed-loop relabeling as the next visual lane, but collect enough
  scoreable states to matter before retraining.
- Track skipped/unlabelable visited states as a drift diagnostic, not just a
  collector nuisance.
- Do not promote this first DAgger checkpoint; it does not beat mirror-only or
  approach CEM-v2.

## Broader closed-loop follow-up

The collector now supports repeated seed blocks, repeated rollouts per seed,
explicit ego-seat selection, epsilon exploration, multiple source checkpoints,
behavior-action histograms, and capped unlabelable-state examples in
`summary.json`. Under a row cap, checkpoint collection is interleaved so later
source checkpoints are not starved.

### Command

```sh
uv run python -m py_compile scripts/build_dummy_pong_lag1_dagger_replay.py
```

```sh
uv run python scripts/build_dummy_pong_lag1_dagger_replay.py \
  --source-replay-path artifacts/local/dummy-pong-lag1-trace-replay-mirror-2026-05-09 \
  --checkpoint-path artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-2026-05-09/checkpoint.npz \
  --checkpoint-path artifacts/local/dummy-pong-lag1-dagger-visual-policy-lr005-smoke-2026-05-09/checkpoint.npz \
  --episodes-per-seating 6 \
  --seed-count 10 \
  --rollouts-per-seed 2 \
  --exploration-epsilon 0.20 \
  --seed 10050931 \
  --max-rows 1200 \
  --max-unlabelable-examples 40 \
  --output-dir artifacts/local/dummy-pong-lag1-dagger-replay-broader-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-dagger-replay-broader-2026-05-09 \
  --epochs 300 \
  --learning-rate 0.05 \
  --validation-fraction 0.2 \
  --seed 10050933 \
  --output-dir artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-lr005-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-dagger-replay-broader-2026-05-09 \
  --epochs 300 \
  --learning-rate 0.05 \
  --validation-fraction 0.2 \
  --class-weighting balanced \
  --seed 10050934 \
  --output-dir artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-balanced-lr005-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_dagger_broader_visual_policy \
  --split-role smoke_baseline_seed \
  --checkpoint lag1_dagger_broader_lr005=artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-lr005-2026-05-09/checkpoint.npz \
  --checkpoint lag1_dagger_broader_balanced_lr005=artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-balanced-lr005-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-dagger-visual-policy-scoreboard-broader-2026-05-09
```

No pytest was run.

### Results

Compile passed.

Closed-loop collection:

| Source rows | Appended labeled rows | Total rows | Unlabelable visited states | Rollouts | Behavior truncations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,664 | 1,200 | 3,864 | 392 | 58 | 7 |

The 1,200 appended rows came from both behavior checkpoints under the cap:
467 rows from the mirror checkpoint and 733 rows from the first tiny DAgger
checkpoint.

DAgger teacher labels:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` | 514 | 3 | 7 |
| `player_1` | 655 | 11 | 10 |

Closed-loop behavior actions:

| Up | Stay | Down | Exploration up | Exploration stay | Exploration down |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 952 | 99 | 541 | 101 | 99 | 97 |

Training:

| Checkpoint | Class weighting | Rows | All accuracy | Validation accuracy | Predicted up | Predicted stay | Predicted down |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broader LR `0.05` | none | 3,864 | 0.8465 | 0.8383 | 2,312 | 0 | 1,552 |
| broader balanced LR `0.05` | balanced | 3,864 | 0.7733 | 0.7904 | 1,964 | 384 | 1,516 |

Scoreboard, seed `8050914`:

| Checkpoint | Opponent | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score | Learned shaped proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| broader LR `0.05` | `lagged_track_ball_1` | 16 | 5 | 7 | 4 | 38.0625 | -0.1250 | -0.1047 |
| broader LR `0.05` | `random_uniform` | 16 | 10 | 6 | 0 | 13.5000 | 0.2500 | 0.2711 |
| broader LR `0.05` | `track_ball` | 16 | 0 | 11 | 5 | 47.1250 | -0.6875 | -0.6474 |
| broader balanced LR `0.05` | `lagged_track_ball_1` | 16 | 6 | 9 | 1 | 18.4375 | -0.1875 | -0.1573 |
| broader balanced LR `0.05` | `random_uniform` | 16 | 6 | 10 | 0 | 14.8750 | -0.2500 | -0.2120 |
| broader balanced LR `0.05` | `track_ball` | 16 | 0 | 14 | 2 | 33.0000 | -0.8750 | -0.8000 |

Same-seed comparison:

| Checkpoint | `lagged_track_ball_1` wins | Random wins | `track_ball` truncations |
| --- | ---: | ---: | ---: |
| Mirror-only | 6/16 | 11/16 | 4/16 |
| DAgger tiny LR `0.05` | 5/16 | 11/16 | 4/16 |
| Broader DAgger LR `0.05` | 5/16 | 10/16 | 5/16 |
| Broader DAgger balanced LR `0.05` | 6/16 | 6/16 | 2/16 |

### Interpretation

The broader collector did its mechanical job: it appended 1,200 exact-DP
labels, used both source checkpoints, included both seats, exercised epsilon
behavior, and reported 392 unlabelable visited states. The label distribution,
however, became even more dominated by `up`. Unweighted training kept random
sanity but stayed at 5/16 versus lag-1; balanced training matched mirror-only
at 6/16 versus lag-1 but lost random sanity and default-`track_ball` survival.

This does not improve over the mirror-only 6/16 lag-1 result, and it remains
far below CEM-v2's 53/64 lag-1 scoreboard row. Closed-loop relabeling is useful
as a drift diagnostic, but this broader supervised append is not the missing
visual/raster-fed learner.

### Artifacts

- `artifacts/local/dummy-pong-lag1-dagger-replay-broader-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-replay-broader-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-lr005-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-lr005-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-balanced-lr005-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-broader-balanced-lr005-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-scoreboard-broader-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-dagger-visual-policy-scoreboard-broader-2026-05-09/episodes.jsonl`
