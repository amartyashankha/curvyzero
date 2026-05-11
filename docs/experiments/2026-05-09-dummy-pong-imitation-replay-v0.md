# 2026-05-09 dummy pong imitation replay v0

## Question

Can the Pong replay builder create a larger learner input from `track_ball`
targets over raster observations?

## Setup

- Environment: `dummy_pong_v0`
- Replay summary schema: `dummy_pong_imitation_replay_summary_v0`
- Replay row schema: `dummy_pong_imitation_replay_row_v0`
- Raster schema: `dummy_pong_raster_grid_v0`
- Target policy: `track_ball`
- Games: 32
- Max steps: 120
- Seed: 0

## Command

```sh
uv run python scripts/build_dummy_pong_imitation_replay.py \
  --games 32 \
  --seed 0 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-imitation-replay-v0
```

## Results

- The command completed.
- `summary.json` was written.
- `replay_rows.jsonl` was written.
- Total rows: 7,680.
- Total environment steps: 3,840.
- Artifact size: about 11 MB.
- All 32 games hit the 120-step cap.
- All games had zero score reward because `track_ball` versus `track_ball`
  kept the point alive.
- Action histograms were the same for both players:
  - up: 520
  - stay: 2,807
  - down: 513

## Interpretation

This is a usable first supervised dataset for copying `track_ball` from raster
frames. It does not prove learning yet, and it is not a useful reward-learning
dataset because no scoring events happened.

For the next reward-learning smoke, use opponents or initial states that create
score changes. Keep rally length as a logged metric, not as the reward.

## Artifacts

- `artifacts/local/dummy-pong-imitation-replay-v0/summary.json`
- `artifacts/local/dummy-pong-imitation-replay-v0/replay_rows.jsonl`

## Follow-ups

- Train a tiny supervised raster policy on this replay.
- Add Pong eval support for learned checkpoints.
- Create a separate reward-learning replay or rollout smoke with score events.
