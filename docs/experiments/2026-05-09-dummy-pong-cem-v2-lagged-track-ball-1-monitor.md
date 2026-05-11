# 2026-05-09 dummy pong CEM-v2 lagged-track-ball-1 monitor

## Question

Can the tiny geometry-CEM learner convert the proven-scoreable
`lagged_track_ball_1` target into a loadable checkpoint with real score
pressure?

## Setup

- Geometry: `PongConfig(width=15,height=9,paddle_height=3,max_steps=120)`.
- Learner: `curvyzero.training.dummy_pong_cem_train`.
- Selection: score-primary CEM-v2,
  `dummy_pong_score_primary_loss_delay_selection_v1`.
- Opponent weights: `lagged_track_ball_1=1.0`, `random_uniform=0.10`,
  `track_ball=0.10`.
- Primary target: `lagged_track_ball_1`.
- Default `track_ball` remains a survival/tie diagnostic.

## Command

```sh
uv run python -m curvyzero.training.dummy_pong_cem_train \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --generations 8 \
  --population-size 32 \
  --elite-count 8 \
  --eval-games 16 \
  --seed 8050913 \
  --opponent-weight lagged_track_ball_1=1.0 \
  --opponent-weight random_uniform=0.10 \
  --opponent-weight track_ball=0.10 \
  --target-opponent-id lagged_track_ball_1 \
  --loss-delay-weight 0.5 \
  --truncation-value 0.0 \
  --output-dir artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_cem_v2_lagged_track_ball_1 \
  --split-role monitor \
  --checkpoint cem_v2=artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-scoreboard-2026-05-09
```

## Results

Monitor training summary:

| Eval slice | Episodes | Learner wins | Opponent wins/losses | Truncations | Mean steps | Mean score return | Mean shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best search candidate vs `lagged_track_ball_1` | 32 | 31 | 1 | 0 | 17.28125 | 0.9375 | 0.9385416667 |
| Final eval vs `lagged_track_ball_1` | 32 | 25 | 0 losses | 7 | 38.6875 | 0.78125 | 0.78125 |
| Final eval vs `random_uniform` | 32 | 30 | 2 | 0 | 27.9375 | 0.875 | 0.8770833333 |
| Final eval vs `track_ball` | 32 | 0 | 0 | 32 | 120.0 | 0.0 | 0.0 |

Checkpoint scoreboard:

| Row | Episodes | Learned wins | Opponent wins | Truncations | Mean steps | Learned mean reward | Learned shaped proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned_cem_v2_vs_lagged_track_ball_1` | 64 | 53 | 1 | 10 | 31.34375 | 0.8125 | 0.8130208333 |
| `learned_cem_v2_vs_random_uniform` | 64 | 60 | 4 | 0 | 22.4375 | 0.875 | 0.8770833333 |
| `learned_cem_v2_vs_track_ball` | 64 | 0 | 0 | 64 | 120.0 | 0.0 | 0.0 |

The scoreboard shaped proxy was computed from `episodes.jsonl` using the same
loss-delay rule: win `+1.0`, loss `-1.0 + 0.5 * steps / 120`, truncation `0.0`.

## Interpretation

CEM-v2 produces real score pressure against the proven-scoreable lag-1 target.
The monitor run found paired-seat target wins, the final eval stayed positive,
and the checkpoint scoreboard confirmed the signal on a fresh seed split with
53/64 wins versus `lagged_track_ball_1`.

Use this as the current score-pressure baseline for visual-policy work: a
visual checkpoint should either beat this row or imitate the capability with a
raster-only learner before promotion.

The old default `track_ball` row behaved as expected: 64/64 truncations, no
wins for either side, and shaped proxy `0.0`. That row is still a survival/tie
diagnostic, not a hard win gate for the default geometry.

## Artifacts

- `artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-scoreboard-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-scoreboard-2026-05-09/episodes.jsonl`

## Follow-ups

- Treat this checkpoint as the first positive project-owned score-pressure
  baseline for dummy Pong.
- Add a heldout split only if promoting this beyond the monitor claim; the
  current scoreboard split is the documented monitor run.
- The next learner question should compare against this CEM-v2 baseline or
  move the lag-1 target into a richer learner, not return to default
  `track_ball` as a hard win target.
- The visual pass gate is >50% wins versus `lagged_track_ball_1`, random
  sanity, and reported default-`track_ball` survival/tie diagnostics.
