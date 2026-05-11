# 2026-05-09 dummy pong lookahead relabel smoke

## Question

Can a small raster replay builder generate policy labels from short
score-delta lookahead against fixed `track_ball`, then train the existing tiny
policy on those labels?

## Setup

- New replay schema: `dummy_pong_lookahead_replay_row_v0`
- Collector: `random_uniform` ego versus fixed `track_ball`
- Lookahead: try ego actions `up/stay/down`, keep opponent on `track_ball`,
  then roll out both agents with `track_ball`
- Tiny smoke only; no pytest

## Command

```sh
uv run python scripts/build_dummy_pong_contact_outcomes.py \
  --states 8 \
  --seed 2 \
  --horizon 120 \
  --output-dir artifacts/local/dummy-pong-contact-outcomes-h120-diagnostic-2026-05-09
```

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_lookahead_replay.py \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  scripts/build_dummy_pong_lookahead_replay.py
```

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 2 \
  --seed 0 \
  --max-steps 60 \
  --lookahead-steps 16 \
  --collector-policy random_uniform \
  --output-dir artifacts/local/dummy-pong-lookahead-replay-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lookahead-replay-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lookahead-policy-train-smoke-2026-05-09 \
  --seed 0 \
  --epochs 50 \
  --learning-rate 0.5 \
  --validation-fraction 0.2
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 4 \
  --seed 5 \
  --split-id dummy_pong_lookahead_smoke \
  --split-role monitor \
  --checkpoint lookahead_smoke=artifacts/local/dummy-pong-lookahead-policy-train-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lookahead-policy-scoreboard-smoke-2026-05-09
```

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 2 \
  --seed 0 \
  --max-steps 60 \
  --lookahead-steps 16 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy angle_control \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-replay-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lookahead-angle-tie-replay-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-policy-train-smoke-2026-05-09 \
  --seed 0 \
  --epochs 80 \
  --learning-rate 0.5 \
  --validation-fraction 0.2
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 4 \
  --seed 5 \
  --split-id dummy_pong_lookahead_angle_tie_smoke \
  --split-role monitor \
  --checkpoint lookahead_angle_tie=artifacts/local/dummy-pong-lookahead-angle-tie-policy-train-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lookahead-angle-tie-scoreboard-smoke-2026-05-09
```

## Results

- Longer contact-outcome diagnostic still had flat score-delta returns:
  top/center/bottom contact choices all returned `0.0` over 120 steps.
- Strict score-separated lookahead replay emitted 9 rows from 131 sampled
  states. All 9 targets matched `track_ball`.
- The strict trained checkpoint was swept by `track_ball` 0/8, with no
  truncations.
- Angle-tie replay emitted all 131 sampled states: 122 rows were
  all-action-tied and used the `angle_control` tie-break, 9 rows had different
  candidate returns, and 41 targets differed from `track_ball`.
- The tiny trained checkpoint reached 0.73 validation accuracy on the
  angle-tie replay, but label accuracy is not the success metric.
- Scoreboard smoke: learned went 4/8 versus `random_uniform`, 0/8 versus
  `track_ball`, and `track_ball` won 6/8 with 2 truncations.

## Interpretation

This starts the next policy-input lane without adding MuZero machinery. The
strict score-return labels are too conservative so far: they mostly rediscover
`track_ball`. The angle-control tie-break creates non-`track_ball` targets only
when short score-delta returns are tied, and the first checkpoint showed mild
pressure but no wins.

Progress still means scoreboard wins or stronger pressure against
`track_ball`, not replay accuracy.

## Artifacts

- `artifacts/local/dummy-pong-contact-outcomes-h120-diagnostic-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-replay-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-replay-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lookahead-policy-train-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-policy-train-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lookahead-policy-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-policy-scoreboard-smoke-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-lookahead-angle-tie-replay-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-angle-tie-replay-smoke-2026-05-09/replay_rows.jsonl`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-train-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-angle-tie-policy-train-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-lookahead-angle-tie-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-lookahead-angle-tie-scoreboard-smoke-2026-05-09/episodes.jsonl`

## Follow-ups

- Run a larger angle-tie lookahead attempt with periodic checkpoints and score
  against `track_ball` plus the epoch-1000 imitation checkpoint.
- If it still gets 0 wins with weak truncation pressure, stop scaling one-step
  labels and change geometry or add depth-2 ego action-sequence lookahead.
