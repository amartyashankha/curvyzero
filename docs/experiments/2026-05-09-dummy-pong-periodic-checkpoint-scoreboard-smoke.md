# 2026-05-09 dummy pong periodic checkpoint scoreboard smoke

## Question

Can the Pong checkpoint scoreboard compare real periodic policy checkpoints
from one training attempt against baselines and each other?

## Setup

- Training attempt:
  `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09`
- Checkpoints:
  - `epoch_1`: `checkpoints/epoch-000001/checkpoint.npz`
  - `epoch_3`: `checkpoints/epoch-000003/checkpoint.npz`
- Split: `dummy_pong_monitor_v0`
- Split role: `monitor`
- Episodes per seated matchup: `2`
- No pytest.

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  scripts/train_dummy_pong_imitation.py \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/run_dummy_pong_checkpoint_scoreboard.py
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor \
  --checkpoint epoch_1=artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000001/checkpoint.npz \
  --checkpoint epoch_3=artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000003/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09
```

## Results

- `py_compile` passed.
- The scoreboard loaded both periodic checkpoints.
- Baseline sanity:
  - `track_ball` beat `random_uniform` 4/4.
  - `track_ball` versus `track_ball` truncated 2/2.
- Learned versus `random_uniform`:
  - epoch 1: 1/4 wins.
  - epoch 3: 2/4 wins.
- Learned versus `track_ball`:
  - epoch 1: 0 learned wins, 2 `track_ball` wins, 2 truncations.
  - epoch 3: 0 learned wins, 2 `track_ball` wins, 2 truncations.
- Learned versus learned:
  - epoch 1 tied epoch 3 by 2/4 to 2/4.

## Interpretation

Periodic policy checkpoints are now scoreable. Epoch 3 looks slightly better
against random inside this tiny smoke, but both checkpoints score 0 wins against
`track_ball`. The input training run used only 64 imitation rows and three
epochs, so the quality result is intentionally weak.

The next useful run should produce meaningful periodic checkpoints, then use
the scoreboard labels `previous`, `latest`, and candidate `selected_best`.

## Artifacts

- `artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09/episodes.jsonl`
