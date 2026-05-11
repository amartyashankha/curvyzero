# 2026-05-09 dummy pong loss-delay lookahead smoke

## Question

Does the new loss-delay lookahead label improve the tiny Pong checkpoint against
`track_ball`, or at least create better survival pressure than plain random play?

## Setup

- Replay output:
  `artifacts/local/dummy-pong-loss-delay-lookahead-smoke-2026-05-09`
- Training output:
  `artifacts/local/dummy-pong-loss-delay-policy-smoke-2026-05-09`
- Scoreboard output:
  `artifacts/local/dummy-pong-loss-delay-scoreboard-smoke-2026-05-09`
- Collector: random ego versus fixed `track_ball`.
- Labeling: try `up/stay/down`, roll out `track_ball` for both agents for 32
  steps, use score-delta return plus `0.05 * steps_run / lookahead_steps` only
  for losing candidate rollouts.
- Ties: included, with `track_ball` as the tie-break policy.
- No pytest.

## Commands

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 4 \
  --seed 11 \
  --max-steps 120 \
  --lookahead-steps 32 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy track_ball \
  --loss-delay-alpha 0.05 \
  --output-dir artifacts/local/dummy-pong-loss-delay-lookahead-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-loss-delay-lookahead-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-loss-delay-policy-smoke-2026-05-09 \
  --seed 0 \
  --epochs 80 \
  --learning-rate 0.5 \
  --validation-fraction 0.2
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 16 \
  --seed 17 \
  --split-id dummy_pong_loss_delay_smoke \
  --split-role monitor \
  --checkpoint loss_delay=artifacts/local/dummy-pong-loss-delay-policy-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-loss-delay-scoreboard-smoke-2026-05-09
```

## Results

Replay:

- Rows: 251 from 251 sampled states.
- Targets different from collector action: 176.
- Targets different from `track_ball`: 0.
- Target source counts:
  - all actions tied, then `track_ball`: 231.
  - best-return tie broken by `track_ball`: 15.
  - unique best return: 5.
- Target return rows: 226 zero, 25 negative, 0 positive.
- Return spread rows: 231 zero spread, 20 positive spread.

Training:

- Final train accuracy: about 0.871.
- Final validation accuracy: 0.860.
- Final validation loss: about 0.435.

Scoreboard:

- Baseline sanity:
  - `track_ball` beat `random_uniform` 32/32.
  - `track_ball` versus `track_ball` truncated 16/16.
- `learned_loss_delay` versus `random_uniform`:
  - won 11/32.
  - `random_uniform` won 21/32.
  - mean reward was -0.3125 for `learned_loss_delay`.
- `learned_loss_delay` versus `track_ball`:
  - won 0/32.
  - `track_ball` won 27/32.
  - 5/32 episodes truncated.
  - mean reward was -0.84375 for `learned_loss_delay`.
  - mean steps were 34.4375.

## Interpretation

This smoke did not show a better checkpoint signal against `track_ball`. The
loss-delay target produced a learnable supervised dataset, but because the
tie-break policy was `track_ball`, every emitted target still matched
`track_ball`. The checkpoint also failed to beat random in the scoreboard.

There is a small survival hint against `track_ball`: 5 of 32 paired episodes
truncated, while `track_ball` beat `random_uniform` 32/32 with no truncations.
That is pressure, not progress. The honest result is still negative until the
checkpoint wins games or clearly improves the monitor scoreboard.

## Artifacts

- `artifacts/local/dummy-pong-loss-delay-lookahead-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-loss-delay-lookahead-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-loss-delay-policy-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-loss-delay-policy-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-loss-delay-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-loss-delay-scoreboard-smoke-2026-05-09/episodes.jsonl`
