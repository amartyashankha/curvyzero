# 2026-05-09 dummy-pong-scoring-data-smoke

## Question

What is the smallest next dummy Pong data smoke that produces actual score
events while keeping the v0 reward clean?

Reward stays score delta only:

```text
ego scores:       +1
opponent scores:  -1
no score event:    0
```

Rally length, paddle hits, and time survived are diagnostics only.

## Commands

```sh
uv run python scripts/run_dummy_pong_eval.py \
  --episodes 32 \
  --seed 0 \
  --output-dir artifacts/local/dummy-pong-scoring-data-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_observability.py \
  --games-per-match 8 \
  --seed 0 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-scoring-observability-smoke-2026-05-09
```

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-scoring-observability-smoke-2026-05-09 \
  --sample-frames 0
```

## Eval Results

- `random_uniform` vs `random_uniform`: 32/32 games scored, 0 truncations,
  mean 13.5 steps.
- `random_uniform` vs `track_ball`, both seats: 64/64 games scored, 0
  truncations, `track_ball` won 64/64, mean 25.875 steps.
- `track_ball` vs `track_ball`: 0/32 games scored, 32 truncations at 120 steps,
  mean reward 0.0.

## Observability Results

- Total trace: 24 games, 1,352 step rows, 1,376 raster frame rows.
- `random_uniform_p0__track_ball_p1`: 8/8 games scored, 0 truncations, player 1
  scored every game, rewards `player_0=-8`, `player_1=+8`, mean 30.0 steps.
- `track_ball_p0__random_uniform_p1`: 8/8 games scored, 0 truncations, player 0
  scored every game, rewards `player_0=+8`, `player_1=-8`, mean 19.0 steps.
- `track_ball_p0__track_ball_p1`: 0/8 games scored, 8 truncations, rewards 0.0,
  mean 120.0 steps.
- Inspector quality note: no obvious count or raster-shape problems detected;
  observed raster shape was `9x15` for all 1,376 frames.

The step trace had 16 nonzero reward steps, exactly matching the 16 scored
track-vs-random games. The 8 track-vs-track games contributed no reward events
and only max-step truncations.

## Conclusion

Use random opponents for the next reward-learning data smoke. The existing
observability harness already records raster frames, step rows, terminal score
events, and clean score-delta rewards for `track_ball` vs `random_uniform` in
both seats. Biased starts and shorter max steps are not needed for the first
reward-data smoke: shorter caps can create truncation artifacts, and random
opponents already produce dense enough score events at the normal 120-step cap.

Recommended next data action: add a narrow replay/export mode that records
`track_ball` vs `random_uniform` from both seats into learner-ready rows using
the same reward field semantics as the imitation replay. Keep `track_ball` as
the behavior target for now, store the random opponent action in the joint
action metadata, and do not add shaped rewards.

## Artifacts

- `artifacts/local/dummy-pong-scoring-data-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-data-smoke-2026-05-09/episodes.jsonl`
- `artifacts/local/dummy-pong-scoring-observability-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-observability-smoke-2026-05-09/games.jsonl`
- `artifacts/local/dummy-pong-scoring-observability-smoke-2026-05-09/steps.jsonl`
- `artifacts/local/dummy-pong-scoring-observability-smoke-2026-05-09/frames.jsonl`
