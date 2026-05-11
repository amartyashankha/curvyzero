# 2026-05-09 dummy pong CEM-v2 score-pressure plan

## Question

How should the tiny geometry-CEM baseline adapt now that the target ladder found
`lag1_track_ball_normal`?

## Decision

Use CEM-v2 as the first learner against `lagged_track_ball_1`. The ladder has
proved that this target is scoreable in 40/40 normal reset/player cases while
keeping the default `width=15,height=9,paddle_height=3,max_steps=120` geometry.
Default `track_ball` remains a survival/tie baseline, not a hard win target.

The learner metric is score-primary:

```text
primary:   weighted mean score return
           win=+1.0, loss=-1.0, truncation=0.0
secondary: target opponent win rate
tertiary:  0.001 * weighted shaped proxy
           win=+1.0
           loss=-1.0 + loss_delay_weight * steps / max_steps
           truncation=truncation_value
```

This keeps survival/loss delay visible without letting full-length ties replace
wins on a scoreable target.

## Code Prepared

`src/curvyzero/training/dummy_pong_cem_train.py` now has the small knobs needed
for the next target:

- configurable `PongConfig`: `--width`, `--height`, `--paddle-height`,
  `--max-steps`;
- configurable CEM opponents through repeated `--opponent-weight POLICY=WEIGHT`;
- `--target-opponent-id` for target-specific win-rate and survival tie-breaks;
- built-in CEM opponents: `random_uniform`, `weak_track_ball`,
  `lagged_track_ball_1`, `track_ball`, `angle_control`;
- shared checkpoint scoreboard baseline `lagged_track_ball_1`;
- score-primary selection schema
  `dummy_pong_score_primary_loss_delay_selection_v1`.

## CEM-v2 Command Shape

First monitor run:

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

Then run the existing checkpoint scoreboard. It now includes
`lagged_track_ball_1` as a baseline alongside `random_uniform` and default
`track_ball`:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_cem_v2_lagged_track_ball_1 \
  --split-role monitor \
  --checkpoint cem_v2=artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-scoreboard-2026-05-09
```

## Pass Signals

- CEM final eval has at least one paired-seat learned win against
  `lagged_track_ball_1` on monitor seeds.
- Stronger pass: positive `selection_mean_score_return` against the target, not
  merely more truncations.
- Baseline sanity still holds: the checkpoint beats `random_uniform` more often
  than it loses, or any random regression is explicitly accepted for the target
  probe.
- Heldout scoreboard confirmation repeats the target-win signal on a fresh seed
  split.

## Stop Signals

- The target probe cannot reproduce a legal winning or loss-avoiding trace.
- CEM-v2 repeats CEM-v1: random wins plus survival/ties, with zero target wins
  and no positive target score return.
- The best candidate is just the `track_ball` prior on a setup where the target
  was supposed to create score pressure.
- Scoreboard learned-vs-`lagged_track_ball_1` remains 0 wins on monitor and
  heldout even when random wins are high and default `track_ball` survives.

## Verification

Compile passed:

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_cem_train.py \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/run_dummy_pong_checkpoint_scoreboard.py
```

Tiny CLI/config smoke passed:

```sh
uv run python -m curvyzero.training.dummy_pong_cem_train \
  --generations 1 \
  --population-size 2 \
  --elite-count 1 \
  --eval-games 1 \
  --seed 8050912 \
  --max-steps 20 \
  --opponent-weight lagged_track_ball_1=1.0 \
  --target-opponent-id lagged_track_ball_1 \
  --output-dir artifacts/local/dummy-pong-cem-v2-config-smoke-2026-05-09
```

The smoke only proves the new config path and score-primary summary fields; it
does not prove CEM-v2 can beat the lag-1 target.

Tiny scoreboard load smoke also passed and produced a
`learned_cem_v2_vs_lagged_track_ball_1` row:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 1 \
  --seed 9050912 \
  --split-id dummy_pong_cem_v2_config_smoke \
  --split-role smoke \
  --checkpoint cem_v2=artifacts/local/dummy-pong-cem-v2-config-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-cem-v2-config-scoreboard-smoke-2026-05-09
```
