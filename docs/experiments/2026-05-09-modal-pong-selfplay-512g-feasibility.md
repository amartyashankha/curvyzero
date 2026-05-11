# 2026-05-09 Modal Pong Self-Play 512-Game Feasibility

Correction note: this is a historical project-owned random-collection trainer
probe, not the current LightZero core plan and not final multiplayer self-play.
It tested whether more fresh random-policy environment interaction helped that
old trainer. The current plan is trusted LightZero whole-job Modal runs, honest
checkpoint curves with survival steps and shaped score, scaling actors/steps,
then frozen-checkpoint and later multiplayer self-play.

## Question

If the old Pong random-collection trainer gets more fresh random-policy games,
does learned-vs-`track_ball` survival or loss delay improve enough to beat the
repair ckpt25 survival baseline?

Prior baseline from the repair run: ckpt25 vs `track_ball` had 47.30 mean
steps, 19.0 median steps, 20/64 truncations, 44/64 `track_ball` wins, 0/64
learned wins, and a -0.6467 learned shaped proxy.

## Setup

This was a single undertraining/fresh-data probe, not a blind scaling sweep.

Training used `src/curvyzero/infra/modal/dummy_pong_train_attempt.py` with
512 fresh random-uniform games, conservative 75 epochs,
and checkpoints every 25 epochs.

Scoreboard used `src/curvyzero/infra/modal/dummy_pong_scoreboard_attempt.py`
with `--episodes 64`. The scoreboard evaluates both seatings for
learned-vs-baseline pairs, so each learned-vs-`track_ball` row below contains
128 episodes.

## Commands

Train:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --run-id pong-selfplay-512g-feasibility-20260509 \
  --attempt-id train \
  --games 512 \
  --max-steps 120 \
  --policy random_uniform \
  --epsilon 0.10 \
  --epochs 75 \
  --policy-learning-rate 0.03 \
  --value-learning-rate 0.001 \
  --action-diversity-beta 0.05 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 25 \
  --seed 8050901
```

Score e25/e50/e75:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --run-id pong-selfplay-512g-feasibility-scoreboard-20260509 \
  --attempt-id score \
  --checkpoints e25=ref:training/dummy-pong/pong-selfplay-512g-feasibility-20260509/attempts/train/train/checkpoints/epoch-000025/checkpoint.npz,e50=ref:training/dummy-pong/pong-selfplay-512g-feasibility-20260509/attempts/train/train/checkpoints/epoch-000050/checkpoint.npz,e75=ref:training/dummy-pong/pong-selfplay-512g-feasibility-20260509/attempts/train/train/checkpoints/epoch-000075/checkpoint.npz \
  --episodes 64 \
  --seed 9050901 \
  --split-id dummy_pong_selfplay_512g_feasibility_monitor_v0 \
  --split-role monitor
```

Fetch raw episodes:

```sh
modal volume get curvyzero-runs \
  training/dummy-pong/pong-selfplay-512g-feasibility-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/episodes.jsonl \
  artifacts/local/pong-modal-scoreboard-survival-2026-05-09/512g_feasibility_episodes.jsonl \
  --force
```

## Results

Training passed.

- Train Modal app run: `ap-cDuKcNbARkCJ7KFgsmM0eX`
- Train run id: `pong-selfplay-512g-feasibility-20260509`
- Train attempt id: `train`
- Remote train elapsed: 17.21 seconds; client elapsed: 20.59 seconds.
- Replay: 512 games, 14,814 rows, mean 14.47 steps, 0 truncations.
- Replay wins: `player_0=260`, `player_1=252`.
- Final all-row predicted action histogram: `down=1296`, `stay=0`,
  `up=13518`.

Scoreboard passed.

- Scoreboard Modal app run: `ap-fimxvb1eKARFiiQ5oYciLn`
- Scoreboard run id: `pong-selfplay-512g-feasibility-scoreboard-20260509`
- Scoreboard attempt id: `score`
- Remote eval elapsed: 5.93 seconds; client elapsed: 10.82 seconds.
- Summary ref:
  `training/dummy-pong/pong-selfplay-512g-feasibility-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/summary.json`
- Episodes ref:
  `training/dummy-pong/pong-selfplay-512g-feasibility-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/episodes.jsonl`

Main scoreboard rows:

| Checkpoint | vs random | vs `track_ball` |
| --- | ---: | ---: |
| e25 | 70/128 learned wins | 0/128 learned wins, 106/128 `track_ball` wins, 22 truncations |
| e50 | 64/128 learned wins | 0/128 learned wins, 116/128 `track_ball` wins, 12 truncations |
| e75 | 59/128 learned wins | 0/128 learned wins, 120/128 `track_ball` wins, 8 truncations |

Baseline sanity:

- `track_ball` beat `random_uniform` 128/128.
- `track_ball` vs `track_ball` truncated 64/64.

## Survival / Loss-Delay Audit

Simple shaped eval proxy, from the learned policy's perspective:

```text
if learned wins: +1.0
if learned loses: -1.0 + 0.5 * episode_steps / 120
if truncated: 0.0
```

Less-negative is better.

| Checkpoint | Episodes | Mean steps | Median steps | Truncations | `track_ball` wins | Learned wins | Mean loss steps | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e25 | 128 | 36.19 | 19.0 | 22/128 (17.2%) | 106/128 (82.8%) | 0/128 | 18.79 | -0.7633 |
| e50 | 128 | 27.70 | 19.0 | 12/128 (9.4%) | 116/128 (90.6%) | 0/128 | 18.15 | -0.8377 |
| e75 | 128 | 25.74 | 19.0 | 8/128 (6.2%) | 120/128 (93.8%) | 0/128 | 19.46 | -0.8615 |

## Interpretation

The undertraining/fresh-data hypothesis failed this one cheap test. The best
fresh-512 row was e25, but it trailed the repair ckpt25 survival baseline on
mean steps, truncation rate, `track_ball` loss rate, and shaped proxy.

Compared with repair ckpt25:

| Metric | Repair ckpt25 | Fresh-512 best e25 | Better? |
| --- | ---: | ---: | --- |
| Mean steps | 47.30 | 36.19 | No |
| Median steps | 19.0 | 19.0 | Tie |
| Truncation rate | 20/64 (31.2%) | 22/128 (17.2%) | No |
| `track_ball` win rate | 44/64 (68.8%) | 106/128 (82.8%) | No |
| Learned win rate | 0/64 | 0/128 | Tie |
| Learned shaped proxy | -0.6467 | -0.7633 | No |

The action histogram also looks collapsed toward `up`, with zero `stay`
predictions in the final train summary. Later checkpoints degraded further.

## Decision

Do not continue scaling this old project-owned trainer for Pong. More fresh
random-uniform environment data did not improve survival/loss-delay against
`track_ball`. Switch to a learner/curriculum change if continuing Pong.

## Artifacts

- Train summary:
  `training/dummy-pong/pong-selfplay-512g-feasibility-20260509/attempts/train/train/summary.json`
- Checkpoints:
  `training/dummy-pong/pong-selfplay-512g-feasibility-20260509/attempts/train/train/checkpoints/epoch-000025/checkpoint.npz`,
  `epoch-000050`, and `epoch-000075`
- Scoreboard summary:
  `training/dummy-pong/pong-selfplay-512g-feasibility-scoreboard-20260509/attempts/score/eval/checkpoint-scoreboard/summary.json`
- Local fetched episodes:
  `artifacts/local/pong-modal-scoreboard-survival-2026-05-09/512g_feasibility_episodes.jsonl`

## Follow-ups

Close the fresh-data probe. Preserve the survival/loss-delay audit rule, but
move the next Pong experiment to the LightZero-first whole-job plan rather than
more random data for the current tiny trainer.
