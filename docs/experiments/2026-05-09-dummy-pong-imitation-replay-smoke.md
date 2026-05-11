# 2026-05-09 dummy pong imitation replay smoke

## Question

Can Pong write a small learner-ready replay file from `track_ball` actions over
raster observations?

## Setup

- Environment: `dummy_pong_v0`
- Replay summary schema: `dummy_pong_imitation_replay_summary_v0`
- Replay row schema: `dummy_pong_imitation_replay_row_v0`
- Raster schema: `dummy_pong_raster_grid_v0`
- Target policy: `track_ball`
- Games: 2
- Max steps: 16
- Seed: 123

## Command

```sh
uv run python scripts/build_dummy_pong_imitation_replay.py \
  --games 2 \
  --seed 123 \
  --max-steps 16 \
  --output-dir artifacts/local/dummy-pong-imitation-replay-smoke
```

## Results

- The command completed.
- `summary.json` was written.
- `replay_rows.jsonl` was written.
- Total rows: 64
- Total environment steps: 32
- Both short games hit the 16-step cap.
- Action histogram was balanced enough for a smoke:
  - `player_0`: up 13, stay 5, down 14
  - `player_1`: up 13, stay 5, down 14

## Interpretation

Pong now has a simple replay artifact for the next learner. This does not prove
learning. It proves that raster frames and `track_ball` target actions can be
stored in one small supervised dataset.

## Artifacts

- `artifacts/local/dummy-pong-imitation-replay-smoke/summary.json`
- `artifacts/local/dummy-pong-imitation-replay-smoke/replay_rows.jsonl`

## Follow-ups

- Run the replay builder with more games.
- Add a tiny supervised raster policy learner.
- Add learned-checkpoint support to the Pong eval harness.
