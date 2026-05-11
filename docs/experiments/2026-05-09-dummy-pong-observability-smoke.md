# 2026-05-09 dummy-pong-observability-smoke

## Question

Can the dummy Pong lane produce compact deterministic trace artifacts for
future debugging without one-off scripts?

## Setup

- Environment: `dummy_pong_v0`
- Observability summary schema: `dummy_pong_observability_summary_v0`
- Step row schema: `dummy_pong_observability_step_v0`
- Game row schema: `dummy_pong_observability_game_v0`
- Frame row schema: `dummy_pong_observability_frame_v0`
- Raster observation schema: `dummy_pong_raster_grid_v0`
- Raster source: `PongEnv.raster_observation()` exposes the same tiny visual
  grid on the environment; the observability harness writes it as JSON-friendly
  digit strings.
- Policies: `random_uniform`, `track_ball`
- Matchups: random-vs-track, track-vs-random, track-vs-track

## Command

```sh
PYTHONPATH=src python scripts/run_dummy_pong_observability.py \
  --games-per-match 1 \
  --seed 123 \
  --max-steps 16 \
  --output-dir artifacts/local/dummy-pong-observability-smoke
```

## Artifacts

- `artifacts/local/dummy-pong-observability-smoke/summary.json`
- `artifacts/local/dummy-pong-observability-smoke/games.jsonl`
- `artifacts/local/dummy-pong-observability-smoke/steps.jsonl`
- `artifacts/local/dummy-pong-observability-smoke/frames.jsonl`

## Results

- Games: 3
- Step rows: 48
- Raster frame rows: 51
- All three games truncated at the harness `--max-steps 16` cap.

## Trace Fields

Each step row includes the policy IDs by agent, match ID, game seed, step index,
joint action IDs and labels, ball position/velocity, paddle positions, rewards,
winner, last hit, terminal/truncation flags, terminal cause, and ego
observations by agent. It also includes a `raster_frame_id` that joins to a
frame row.

Each frame row is a tiny deterministic `height x width` raster encoded as
row-major digit strings. Legend: `0` empty, `1` player 0 paddle, `2` player 1
paddle, `3` ball, `4` ball-on-paddle overlap. Each game includes a reset frame
at `step_index: 0` plus one frame for every environment step.

## Caveats

This is observability only. It does not train, compute ratings, or emit image or
video files. The tabular ego observations remain debug/eval scaffolding; the
raster grid is the intended MuZero-facing Pong observation path. The smoke uses
a deliberately short `--max-steps` cap to keep JSONL artifacts small, so
truncation counts should be interpreted as harness behavior.
