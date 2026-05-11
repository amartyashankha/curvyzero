# 2026-05-09 dummy pong lag-1 trace visual-policy replay augmentation smoke

## Question

Can data-side balancing or augmentation fix the lag-1 visual-policy lane without
changing the imitation trainer?

## Setup

- Source builder: `scripts/build_dummy_pong_lag1_trace_replay.py`.
- Trainer: existing `scripts/train_dummy_pong_imitation.py`, unchanged for this
  run. The checkpoint metadata reports `class_weighting: none`.
- New replay options:
  - `--include-vertical-mirror`: add top/bottom mirrored raster rows and swap
    `up`/`down` actions.
  - `--balance-actions oversample`: after optional mirroring, oversample rare
    labels per ego agent to the largest agent/action bucket.
- Eval: existing checkpoint scoreboard, 8 episodes per seating, paired seats.

## Command

```sh
uv run python -m py_compile scripts/build_dummy_pong_lag1_trace_replay.py
```

Mirror-only augmentation:

```sh
uv run python scripts/build_dummy_pong_lag1_trace_replay.py \
  --max-steps 120 \
  --repeats 1 \
  --include-vertical-mirror \
  --balance-actions none \
  --balance-seed 9050915 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-replay-mirror-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-mirror-2026-05-09 \
  --epochs 300 \
  --learning-rate 1.0 \
  --validation-fraction 0.2 \
  --seed 9050918 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_trace_visual_policy_mirror_seed8050914 \
  --split-role smoke_baseline_seed \
  --checkpoint lag1_trace_visual_mirror=artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-mirror-seed8050914-2026-05-09
```

Mirror plus oversampling:

```sh
uv run python scripts/build_dummy_pong_lag1_trace_replay.py \
  --max-steps 120 \
  --repeats 1 \
  --include-vertical-mirror \
  --balance-actions oversample \
  --balance-seed 9050915 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-replay-mirror-balanced-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-mirror-balanced-2026-05-09 \
  --epochs 300 \
  --learning-rate 1.0 \
  --validation-fraction 0.2 \
  --seed 9050916 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-balanced-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_trace_visual_policy_mirror_balanced_seed8050914 \
  --split-role smoke_baseline_seed \
  --checkpoint lag1_trace_visual_mirror_balanced=artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-balanced-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-mirror-balanced-seed8050914-2026-05-09
```

No pytest was run.

## Results

Replay labels:

| Replay | Rows | `player_0` up | `player_0` stay | `player_0` down | `player_1` up | `player_1` stay | `player_1` down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original exact traces | 1,332 | 622 | 2 | 42 | 622 | 2 | 42 |
| Mirror-only | 2,664 | 664 | 4 | 664 | 664 | 4 | 664 |
| Mirror + oversample | 3,984 | 664 | 664 | 664 | 664 | 664 | 664 |

Checkpoint predicted action distribution on all replay rows:

| Replay | Rows | Accuracy | Predicted up | Predicted stay | Predicted down |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original exact traces | 1,332 | 0.9407 | 1,311 | 0 | 21 |
| Mirror-only | 2,664 | 0.8979 | 1,450 | 0 | 1,214 |
| Mirror + oversample | 3,984 | 0.9237 | 1,296 | 1,352 | 1,336 |

Scoreboard, seed `8050914`:

| Checkpoint | Opponent | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean score | Learned shaped proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original exact traces | `lagged_track_ball_1` | 16 | 5 | 7 | 4 | 38.0625 | -0.1250 | -0.1047 |
| Mirror-only | `lagged_track_ball_1` | 16 | 6 | 7 | 3 | 31.7500 | -0.0625 | -0.0422 |
| Mirror + oversample | `lagged_track_ball_1` | 16 | 5 | 9 | 2 | 24.7500 | -0.2500 | -0.2227 |
| Original exact traces | `random_uniform` | 16 | 10 | 6 | 0 | 13.5000 | 0.2500 | 0.2711 |
| Mirror-only | `random_uniform` | 16 | 11 | 5 | 0 | 14.1875 | 0.3750 | 0.3940 |
| Mirror + oversample | `random_uniform` | 16 | 11 | 5 | 0 | 18.3125 | 0.3750 | 0.3854 |
| Original exact traces | `track_ball` | 16 | 0 | 11 | 5 | 47.1250 | -0.6875 | -0.6474 |
| Mirror-only | `track_ball` | 16 | 0 | 12 | 4 | 40.1250 | -0.7500 | -0.7078 |
| Mirror + oversample | `track_ball` | 16 | 0 | 12 | 4 | 39.4375 | -0.7500 | -0.7107 |

The shaped proxy is the current Pong diagnostic rule: win `+1.0`, loss
`-1.0 + 0.5 * steps / 120`, truncation `0.0`.

## Interpretation

Vertical mirror augmentation is a small data-side improvement. It removes the
most damaging `up`/`down` replay asymmetry, improves the same-seed lag-1 row
from 5/16 to 6/16, and improves random sanity from 10/16 to 11/16.

Full oversampling is not a gameplay improvement. It produces an exactly
balanced target histogram and a balanced checkpoint prediction histogram, but
it only matches the old 5/16 lag-1 result and worsens loss-delay metrics. The
likely failure mode is over-amplifying the eight mirrored `stay` rows rather
than adding new decision coverage.

This does not pass the visual lane. CEM-v2 remains the positive score-pressure
baseline at 53/64 lag-1 scoreboard wins, and both data-side variants still lose
default `track_ball` survival relative to the original unweighted trace smoke.

## Artifacts

- `artifacts/local/dummy-pong-lag1-trace-replay-mirror-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-replay-mirror-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-mirror-seed8050914-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-mirror-seed8050914-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-replay-mirror-balanced-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-replay-mirror-balanced-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-balanced-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-mirror-balanced-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-mirror-balanced-seed8050914-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-mirror-balanced-seed8050914-2026-05-09/episodes.jsonl`

## Follow-ups

- Keep `--include-vertical-mirror`; it is a valid symmetry and helps a little.
- Do not promote pure oversampling of `stay` rows as the fix. The next data-side
  step should add genuinely new exact traces or a richer replay source instead
  of duplicating the same rare states.
