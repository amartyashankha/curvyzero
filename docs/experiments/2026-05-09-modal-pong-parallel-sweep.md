# 2026-05-09 Modal Pong Parallel Sweep

## Question

Do small parallel self-play variants improve the Modal Pong checkpoint
scoreboard against `track_ball`, or at least increase `track_ball` truncations
without collapsing against `random_uniform`?

## Setup

All runs used `src/curvyzero/infra/modal/dummy_pong_train_attempt.py` from
`random_uniform` and saved periodic checkpoints. Scoreboards used
`src/curvyzero/infra/modal/dummy_pong_scoreboard_attempt.py` with
`--episodes 32`, so learned-vs-baseline pair rows contain 64 seated games.

Prior reference: the 64-game Modal Pong repair run got 0 learned wins against
`track_ball`; its best pressure was ckpt25 with 20 truncations and 44
`track_ball` wins out of 64.

## Commands

Training command shape:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --run-id RUN_ID \
  --attempt-id train \
  --games 128 \
  --epochs EPOCHS \
  --seed SEED \
  --max-steps 120 \
  --policy random_uniform \
  --epsilon EPSILON \
  --policy-learning-rate POLICY_LR \
  --value-learning-rate 0.001 \
  --action-diversity-beta BETA \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs CHECKPOINT_EVERY
```

Scoreboard command shape:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --run-id SCOREBOARD_RUN_ID \
  --attempt-id score \
  --checkpoints LABEL=ref:CHECKPOINT_REF,... \
  --episodes 32 \
  --seed MONITOR_SEED \
  --split-id dummy_pong_parallel_sweep_monitor_v0 \
  --split-role monitor
```

## Results

Training runs:

| Variant | Run id | Modal app | Config | Periodic checkpoint refs |
| --- | --- | --- | --- | --- |
| More exploration | `pong-parallel-explore-eps020-20260509` | `ap-3q1pWQ4ubGbJgYNT54iazF` | games 128, epochs 100, seed 6050901, epsilon 0.2, LR 0.03, beta 0.05, every 25 | `training/dummy-pong/pong-parallel-explore-eps020-20260509/attempts/train/train/checkpoints/epoch-000025/checkpoint.npz`; `epoch-000050`; `epoch-000075`; `epoch-000100` |
| Lower LR / stability | `pong-parallel-stable-eps010-20260509` | `ap-WtoH8uykoDuDseisaRVdDW` | games 128, epochs 150, seed 6050902, epsilon 0.1, LR 0.01, beta 0.05, every 50 | `training/dummy-pong/pong-parallel-stable-eps010-20260509/attempts/train/train/checkpoints/epoch-000050/checkpoint.npz`; `epoch-000100`; `epoch-000150` |
| Higher diversity | `pong-parallel-diverse-beta010-20260509` | `ap-YCmnmmZGpeZ1cdbDr5wpwr` | games 128, epochs 100, seed 6050903, epsilon 0.15, LR 0.02, beta 0.1, every 25 | `training/dummy-pong/pong-parallel-diverse-beta010-20260509/attempts/train/train/checkpoints/epoch-000025/checkpoint.npz`; `epoch-000050`; `epoch-000075`; `epoch-000100` |

Scoreboards:

| Variant | Scoreboard run id | Modal app | Monitor seed | Checkpoint | Learned wins vs random | Learned wins vs `track_ball` | Truncations vs `track_ball` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| More exploration | `pong-parallel-explore-eps020-scoreboard-20260509` | `ap-xpAaGEyczvNnxzh7IVzG0n` | 7050901 | e25 | 29/64 | 0/64 | 16/64 |
| More exploration | `pong-parallel-explore-eps020-scoreboard-20260509` | `ap-xpAaGEyczvNnxzh7IVzG0n` | 7050901 | e50 | 29/64 | 0/64 | 10/64 |
| More exploration | `pong-parallel-explore-eps020-scoreboard-20260509` | `ap-xpAaGEyczvNnxzh7IVzG0n` | 7050901 | e75 | 30/64 | 0/64 | 14/64 |
| More exploration | `pong-parallel-explore-eps020-scoreboard-20260509` | `ap-xpAaGEyczvNnxzh7IVzG0n` | 7050901 | e100 | 31/64 | 0/64 | 10/64 |
| Lower LR / stability | `pong-parallel-stable-eps010-scoreboard-20260509` | `ap-3QoW0wGYWl53yVRiiAjpjL` | 7050902 | e50 | 29/64 | 0/64 | 16/64 |
| Lower LR / stability | `pong-parallel-stable-eps010-scoreboard-20260509` | `ap-3QoW0wGYWl53yVRiiAjpjL` | 7050902 | e100 | 32/64 | 0/64 | 11/64 |
| Lower LR / stability | `pong-parallel-stable-eps010-scoreboard-20260509` | `ap-3QoW0wGYWl53yVRiiAjpjL` | 7050902 | e150 | 33/64 | 0/64 | 7/64 |
| Higher diversity | `pong-parallel-diverse-beta010-scoreboard-20260509` | `ap-qCfoHIQDZtzzLY9Vm9cWUK` | 7050903 | e25 | 30/64 | 0/64 | 19/64 |
| Higher diversity | `pong-parallel-diverse-beta010-scoreboard-20260509` | `ap-qCfoHIQDZtzzLY9Vm9cWUK` | 7050903 | e50 | 30/64 | 0/64 | 6/64 |
| Higher diversity | `pong-parallel-diverse-beta010-scoreboard-20260509` | `ap-qCfoHIQDZtzzLY9Vm9cWUK` | 7050903 | e75 | 32/64 | 0/64 | 15/64 |
| Higher diversity | `pong-parallel-diverse-beta010-scoreboard-20260509` | `ap-qCfoHIQDZtzzLY9Vm9cWUK` | 7050903 | e100 | 29/64 | 0/64 | 14/64 |

Scoreboard summary refs:

- `training/dummy-pong/pong-parallel-explore-eps020-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/summary.json`
- `training/dummy-pong/pong-parallel-stable-eps010-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/summary.json`
- `training/dummy-pong/pong-parallel-diverse-beta010-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/summary.json`

## Survival / Loss-Delay Audit

The scoreboard `episodes.jsonl` rows were fetched from the Modal
`curvyzero-runs` Volume and aggregated for learned-vs-`track_ball` pair groups.
The rows include enough fields for loss-delay analysis: `steps`, `truncated`,
`winner`, and terminal rewards.

Fetched raw rows:

```sh
modal volume get curvyzero-runs training/dummy-pong/pong-parallel-explore-eps020-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/episodes.jsonl artifacts/local/pong-modal-scoreboard-survival-2026-05-09/explore_episodes.jsonl --force
modal volume get curvyzero-runs training/dummy-pong/pong-parallel-stable-eps010-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/episodes.jsonl artifacts/local/pong-modal-scoreboard-survival-2026-05-09/stable_episodes.jsonl --force
modal volume get curvyzero-runs training/dummy-pong/pong-parallel-diverse-beta010-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/episodes.jsonl artifacts/local/pong-modal-scoreboard-survival-2026-05-09/diverse_episodes.jsonl --force
```

Simple shaped eval proxy, from the learned policy's perspective:

```text
if learned wins: +1.0
if learned loses: -1.0 + 0.5 * episode_steps / 120
if truncated: 0.0
```

Less-negative is better. Since all learned-vs-`track_ball` rows still had 0
learned wins, the proxy mostly ranks delayed losses and truncations.

| Variant | Checkpoint | Episodes | Mean steps | Median steps | Truncations | `track_ball` wins | Learned wins | Mean loss steps | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| More exploration | e25 | 64 | 42.70 | 19.0 | 16/64 (25.0%) | 48/64 (75.0%) | 0/64 | 16.94 | -0.6971 |
| More exploration | e50 | 64 | 32.89 | 19.0 | 10/64 (15.6%) | 54/64 (84.4%) | 0/64 | 16.76 | -0.7848 |
| More exploration | e75 | 64 | 41.61 | 19.0 | 14/64 (21.9%) | 50/64 (78.1%) | 0/64 | 19.66 | -0.7173 |
| More exploration | e100 | 64 | 33.23 | 19.0 | 10/64 (15.6%) | 54/64 (84.4%) | 0/64 | 17.17 | -0.7834 |
| Lower LR / stability | e50 | 64 | 41.50 | 19.0 | 16/64 (25.0%) | 48/64 (75.0%) | 0/64 | 15.33 | -0.7021 |
| Lower LR / stability | e100 | 64 | 35.33 | 19.0 | 11/64 (17.2%) | 53/64 (82.8%) | 0/64 | 17.75 | -0.7669 |
| Lower LR / stability | e150 | 64 | 28.33 | 19.0 | 7/64 (10.9%) | 57/64 (89.1%) | 0/64 | 17.07 | -0.8273 |
| Higher diversity | e25 | 64 | 46.41 | 19.0 | 19/64 (29.7%) | 45/64 (70.3%) | 0/64 | 15.33 | -0.6582 |
| Higher diversity | e50 | 64 | 28.30 | 19.0 | 6/64 (9.4%) | 58/64 (90.6%) | 0/64 | 18.81 | -0.8352 |
| Higher diversity | e75 | 64 | 43.70 | 19.0 | 15/64 (23.4%) | 49/64 (76.6%) | 0/64 | 20.35 | -0.7007 |
| Higher diversity | e100 | 64 | 40.41 | 19.0 | 14/64 (21.9%) | 50/64 (78.1%) | 0/64 | 18.12 | -0.7223 |

Best rows by variant:

- More exploration: e25 by mean steps, truncations, and shaped proxy.
- Lower LR / stability: e50 by mean steps, truncations, and shaped proxy.
- Higher diversity: e25 by mean steps, truncations, and shaped proxy.
- Overall parallel sweep: higher-diversity e25 was best, but it still trailed
  the prior repair ckpt25 on mean steps, truncations, and shaped proxy.

## Interpretation

All variants still got 0 wins against `track_ball`, but the corrected diagnosis
must include survival and loss delay. The best pressure was the higher-diversity
e25 checkpoint: 46.41 mean steps, 19/64 truncations, 45/64 `track_ball` wins,
and a -0.6582 learned shaped proxy. That is still not better than the prior
repair run's ckpt25: 47.30 mean steps, 20/64 truncations, 44/64 `track_ball`
wins, and a -0.6467 shaped proxy.

Random wins are also not strong enough to rescue the result. No checkpoint both
improved `track_ball` pressure and beat `random_uniform`; the best random rows
were 33/64 or lower.

The earlier 0-win summary was incomplete. The corrected conclusion is not "all
0-win rows are identical"; it is "the sweep failed to beat the repair run even
after measuring survival length and shaped loss delay."

## Artifacts

Training summaries:

- `training/dummy-pong/pong-parallel-explore-eps020-20260509/attempts/train/train/summary.json`
- `training/dummy-pong/pong-parallel-stable-eps010-20260509/attempts/train/train/summary.json`
- `training/dummy-pong/pong-parallel-diverse-beta010-20260509/attempts/train/train/summary.json`

Scoreboard summaries are listed above.

## Follow-ups

Stop blind self-play scaling for this learner. Switch learner or curriculum,
but keep reporting learned-vs-`track_ball` survival length, truncation rate, and
loss-delay shaped proxy alongside wins. A curriculum may explicitly reward
angle control or loss delay against `track_ball`, or the current tiny policy
update can be replaced with a learner/search setup that optimizes beyond
random-vs-random score labels.
