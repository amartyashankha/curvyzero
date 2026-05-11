# 2026-05-09 dummy pong lookahead angle-tie g32 h32

## Question

Does a larger angle-tie short-lookahead replay improve Pong behavior beyond the
selected epoch-1000 imitation checkpoint?

## Setup

- Replay output:
  `artifacts/local/dummy-pong-lookahead-angle-tie-replay-g32-h32-2026-05-09`
- Training output:
  `artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09`
- Scoreboard output:
  `artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-scoreboard-monitor-2026-05-09`
- Selection output:
  `artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-selection-record-2026-05-09`
- Collector: random ego versus fixed `track_ball`.
- Labeling: try `up/stay/down`, roll out `track_ball` for both agents for 32
  steps, use score-delta return; when all actions tie, use `angle_control`.
- No pytest.

## Command

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 32 \
  --seed 7 \
  --max-steps 120 \
  --lookahead-steps 32 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy angle_control \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-replay-g32-h32-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lookahead-angle-tie-replay-g32-h32-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09 \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 250
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 313 \
  --split-id dummy_pong_lookahead_angle_tie_g32_h32_monitor \
  --split-role monitor \
  --checkpoint lookahead250=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz \
  --checkpoint lookahead500=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000500/checkpoint.npz \
  --checkpoint lookahead750=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz \
  --checkpoint lookahead1000=artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --checkpoint imitation1000=artifacts/local/dummy-pong-imitation-periodic-v0-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-scoreboard-monitor-2026-05-09
```

```sh
uv run python scripts/select_dummy_pong_checkpoint.py \
  --summary artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-scoreboard-monitor-2026-05-09/summary.json \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-selection-record-2026-05-09
```

## Results

Replay:

- Rows: 1,669.
- Targets different from `track_ball`: 442.
- Target source counts:
  - all actions tied, then `angle_control`: 1,532.
  - best-return tie broken by `angle_control`: 79.
  - unique best return: 58.
- Positive-return target rows: 0.
- Negative-return target rows: 225.

Training:

- Four periodic checkpoints were written: epochs 250, 500, 750, and 1000.
- Final validation accuracy was about 0.713. This is only label fit, not the
  success metric.

Scoreboard:

- Baseline sanity:
  - `track_ball` beat `random_uniform` 64/64.
  - `track_ball` versus `track_ball` truncated 32/32.
- Imitation epoch 1000:
  - beat random 44/64.
  - won 0/64 against `track_ball`.
  - `track_ball` won 10/64, with 54 truncations.
- Lookahead checkpoints:
  - epoch 250: beat random 41/64; won 0/64 against `track_ball`;
    `track_ball` won 37/64, with 27 truncations.
  - epoch 500: beat random 38/64; won 0/64 against `track_ball`;
    `track_ball` won 52/64, with 12 truncations.
  - epoch 750: beat random 39/64; won 0/64 against `track_ball`;
    `track_ball` won 48/64, with 16 truncations.
  - epoch 1000: beat random 40/64; won 0/64 against `track_ball`;
    `track_ball` won 32/64, with 32 truncations.
- Direct learned-vs-learned:
  - imitation beat lookahead 250 by 28/64 to 21/64.
  - imitation beat lookahead 500 by 46/64 to 14/64.
  - imitation beat lookahead 750 by 39/64 to 15/64.
  - imitation beat lookahead 1000 by 29/64 to 19/64.
- Selection record picked `imitation1000`, not a lookahead checkpoint.

## Interpretation

The larger angle-tie lookahead attempt produced many non-`track_ball` labels,
but those labels did not improve the scoreboard. No lookahead checkpoint scored
against `track_ball`, and all were worse than the selected imitation checkpoint
on the monitor split.

This is a useful negative result. Do not keep scaling one-step angle-tie labels
as the main path. The next policy-input step should either change the game
geometry so contact choices produce score signal faster, or use a deeper
ego-action-sequence lookahead.

## Artifacts

- `artifacts/local/dummy-pong-lookahead-angle-tie-replay-g32-h32-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-angle-tie-replay-g32-h32-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000250/checkpoint.npz`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000500/checkpoint.npz`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-000750/checkpoint.npz`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-g32-h32-e1000-2026-05-09/checkpoints/epoch-001000/checkpoint.npz`
- `artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-scoreboard-monitor-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-scoreboard-monitor-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-lookahead-angle-tie-g32-h32-selection-record-2026-05-09/selection_record.json`

## Follow-ups

- Stop scaling one-step angle-tie relabeling unless a bug is found.
- Next likely experiment: depth-2 ego action-sequence lookahead, or a smaller
  geometry where off-center returns can score within the lookahead horizon.
