# 2026-05-09 dummy pong imitation v0 periodic e1000 scoreboard

## Question

Does a longer supervised Pong imitation run produce periodic checkpoints that
improve on the scoreboard?

## Setup

- Replay:
  `artifacts/local/dummy-pong-imitation-replay-v0`
- Training output:
  `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09`
- Scoreboard output:
  `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-scoreboard-selection-2026-05-09`
- Training epochs: `1000`
- Periodic checkpoints every `250` epochs.
- Scoreboard split: `dummy_pong_imitation_v0_selection`
- Split role: `selection`
- Scoreboard episodes per seated matchup: `32`
- No pytest.

## Command

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-imitation-replay-v0 \
  --output-dir artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09 \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 1.0 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 250
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 11 \
  --split-id dummy_pong_imitation_v0_selection \
  --split-role selection \
  --checkpoint epoch250=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz \
  --checkpoint epoch500=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000500/checkpoint.npz \
  --checkpoint epoch750=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz \
  --checkpoint epoch1000=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-imitation-periodic-v0-e1000-scoreboard-selection-2026-05-09
```

```sh
uv run python scripts/select_dummy_pong_checkpoint.py \
  --summary artifacts/local/dummy-pong-imitation-periodic-v0-e1000-scoreboard-selection-2026-05-09/summary.json \
  --output-dir artifacts/local/dummy-pong-imitation-v0-e1000-selection-record-2026-05-09
```

## Results

- Training wrote four periodic checkpoints: epochs 250, 500, 750, and 1000.
- Final replay validation accuracy was `0.990234375`.
- Baseline sanity:
  - `track_ball` beat `random_uniform` 62/64, with 2 truncations.
  - `track_ball` versus `track_ball` truncated 32/32.
- Learned versus `random_uniform`:
  - epoch 250: 32/64.
  - epoch 500: 29/64.
  - epoch 750: 33/64.
  - epoch 1000: 42/64.
- Learned versus `track_ball`:
  - epoch 250: 0 learned wins, 52 `track_ball` wins, 12 truncations.
  - epoch 500: 0 learned wins, 29 `track_ball` wins, 35 truncations.
  - epoch 750: 0 learned wins, 46 `track_ball` wins, 18 truncations.
  - epoch 1000: 0 learned wins, 7 `track_ball` wins, 57 truncations.
- Learned versus learned:
  - epoch 1000 beat epoch 250 by 34/64 to 25/64.
  - epoch 1000 beat epoch 500 by 22/64 to 18/64, with 24 truncations.
  - epoch 1000 beat epoch 750 by 36/64 to 10/64.
- Selection record:
  - selected checkpoint: epoch 1000.
  - reason: all checkpoints had 0 wins against `track_ball`, then epoch 1000
    had the best `random_uniform` win rate at 42/64.
  - status at selection time: `selected_pending_heldout`.

## Interpretation

Longer imitation improved the final checkpoint against random and against most
older checkpoints. It still did not beat `track_ball` once.

The best current read is: the raster action-cloning path can train and preserve
checkpoint history, but copying `track_ball` from timeout-only replay does not
teach winning Pong. The next learner needs score-bearing data, a policy
improvement step, or a small search/target loop.

## Artifacts

- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000500/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-scoreboard-selection-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-imitation-periodic-v0-e1000-scoreboard-selection-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-imitation-v0-e1000-selection-record-2026-05-09/selection_record.json`

## Follow-ups

- Heldout has now been run separately:
  `docs/experiments/2026-05-09-dummy-pong-imitation-v0-heldout-scoreboard.md`.
- Do not move this learner to GPU. The current issue is the objective, not
  speed.
