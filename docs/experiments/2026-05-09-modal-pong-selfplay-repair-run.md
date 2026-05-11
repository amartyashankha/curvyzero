# 2026-05-09 Modal Pong Self-Play Repair Run

## Question

Does the repaired tiny self-play trainer show any useful Pong progress when run
as a proper Modal train attempt?

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --games 64 \
  --max-steps 120 \
  --policy random_uniform \
  --epsilon 0.05 \
  --epochs 100 \
  --policy-learning-rate 0.05 \
  --value-learning-rate 0.001 \
  --action-diversity-beta 0.02 \
  --checkpoint-every-epochs 25 \
  --seed 101
```

Then scored checkpoints 25, 50, 75, and 100:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
  --checkpoints ckpt25=ref:training/dummy-pong/pong-train-20260509T040427Z-ab29b47a018c/attempts/attempt-20260509T040427Z-6f39b39085bc/train/checkpoints/epoch-000025/checkpoint.npz,ckpt50=ref:training/dummy-pong/pong-train-20260509T040427Z-ab29b47a018c/attempts/attempt-20260509T040427Z-6f39b39085bc/train/checkpoints/epoch-000050/checkpoint.npz,ckpt75=ref:training/dummy-pong/pong-train-20260509T040427Z-ab29b47a018c/attempts/attempt-20260509T040427Z-6f39b39085bc/train/checkpoints/epoch-000075/checkpoint.npz,ckpt100=ref:training/dummy-pong/pong-train-20260509T040427Z-ab29b47a018c/attempts/attempt-20260509T040427Z-6f39b39085bc/train/checkpoints/epoch-000100/checkpoint.npz \
  --episodes 32 \
  --seed 313 \
  --split-id dummy_pong_modal_selfplay_repair_monitor \
  --split-role monitor
```

## Results

Train passed and wrote four checkpoints.

- Train app run: `ap-5BTomnV4bEW3M3s7dWak5j`
- Train run id: `pong-train-20260509T040427Z-ab29b47a018c`
- Train attempt id: `attempt-20260509T040427Z-6f39b39085bc`
- Replay: 64 games, 1,992 rows, no truncations.
- Final predicted actions over all rows: `down=788`, `stay=996`, `up=208`.

Scoreboard passed.

- Scoreboard app run: `ap-1lxPivzHg9nAlLPLnNQGEf`
- Scoreboard run id: `pong-scoreboard-20260509T040553Z-86b1893e38ca`
- Scoreboard summary ref:
  `training/dummy-pong/pong-scoreboard-20260509T040553Z-86b1893e38ca/attempts/attempt-20260509T040553Z-dceafe306dde/eval/checkpoint-scoreboard/summary.json`

Main rows:

| Checkpoint | vs random | vs `track_ball` |
| --- | --- | --- |
| epoch 25 | 36/64 wins | 0/64 wins, `track_ball` 44/64, 20 truncations |
| epoch 50 | 39/64 wins | 0/64 wins, `track_ball` 47/64, 17 truncations |
| epoch 75 | 31/64 wins | 0/64 wins, `track_ball` 45/64, 19 truncations |
| epoch 100 | 31/64 wins | 0/64 wins, `track_ball` 51/64, 13 truncations |

Baseline sanity:

- `track_ball` beat `random_uniform` 64/64.
- `track_ball` vs `track_ball` truncated 32/32.

## Survival / Loss-Delay Audit

The first summary under-reported this run by reducing the learned-vs-`track_ball`
rows to learned wins only. The scoreboard `episodes.jsonl` rows also include
`steps`, `truncated`, `winner`, and terminal rewards, so they can report
survival and loss delay.

Fetched raw rows from the Modal Volume:

```sh
modal volume get curvyzero-runs \
  training/dummy-pong/pong-scoreboard-20260509T040553Z-86b1893e38ca/attempts/attempt-20260509T040553Z-dceafe306dde/eval/checkpoint-scoreboard/episodes.jsonl \
  artifacts/local/pong-modal-scoreboard-survival-2026-05-09/repair_episodes.jsonl \
  --force
```

Simple shaped eval proxy, from the learned policy's perspective:

```text
if learned wins: +1.0
if learned loses: -1.0 + 0.5 * episode_steps / 120
if truncated: 0.0
```

Less-negative is better. There were no learned wins, so this proxy mostly
measures delayed losses plus forced truncations.

| Checkpoint | Episodes | Mean steps | Median steps | Truncations | `track_ball` wins | Learned wins | Mean loss steps | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ckpt25 | 64 | 47.30 | 19.0 | 20/64 (31.2%) | 44/64 (68.8%) | 0/64 | 14.25 | -0.6467 |
| ckpt50 | 64 | 42.05 | 19.0 | 17/64 (26.6%) | 47/64 (73.4%) | 0/64 | 13.85 | -0.6920 |
| ckpt75 | 64 | 46.06 | 19.0 | 19/64 (29.7%) | 45/64 (70.3%) | 0/64 | 14.84 | -0.6596 |
| ckpt100 | 64 | 35.73 | 19.0 | 13/64 (20.3%) | 51/64 (79.7%) | 0/64 | 14.25 | -0.7495 |

## Interpretation

The Modal train-to-scoreboard path is now real and fast.

The learning result is still weak. Epoch 50 beat random best, but no checkpoint
won against `track_ball`. Epoch 25 had the best `track_ball` pressure because it
lost fewer games, forced more truncations, had the longest mean episode length,
and had the least-negative shaped loss-delay proxy.

The key correction is that 0/64 learned wins is not the whole eval result.
Survival length and loss delay show whether a checkpoint is at least pressuring
`track_ball`. This run still does not justify blind scaling, but future
decisions must report these metrics before declaring learning failed.

## Follow-ups

- Run a cheap parallel sweep with more exploration and lower learning rates.
- Score periodic checkpoints for every variant.
- If all variants stay at 0 wins against `track_ball`, switch to a simpler
  baseline/curriculum instead of doing another generation.
