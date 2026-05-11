# 2026-05-09 dummy pong checkpoint scoreboard smoke

## Question

Can Pong checkpoint eval produce a small scoreboard artifact with learned
checkpoints versus `random_uniform`, `track_ball`, and learned checkpoint peers
without adding Elo, leagues, or dashboards?

## Setup

- Environment: `dummy_pong_v0`
- Eval seed: 0
- Episodes per seating: 2
- Split: `dummy_pong_monitor_v0`, role `monitor`
- Checkpoint path:
  `artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz`
- Checkpoint labels: `latest`, `previous`
- Important limitation: both labels point to the same checkpoint. The
  learned-vs-learned row is a plumbing smoke, not an old-vs-new policy result.

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_eval.py \
  scripts/run_dummy_pong_checkpoint_scoreboard.py
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 2 \
  --seed 0 \
  --split-id dummy_pong_monitor_v0 \
  --split-role monitor \
  --checkpoint latest=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --checkpoint previous=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-checkpoint-scoreboard-smoke-2026-05-09
```

## Results

- `py_compile` completed.
- Scoreboard completed and wrote `summary.json` plus `episodes.jsonl`.
- Total episodes: 28.
- Checkpoint specs in `summary.json` preserved both labels:
  `learned_latest` and `learned_previous`.
- Baseline sanity:
  - `random_uniform` vs `random_uniform`: random won 2/2.
  - `random_uniform` vs `track_ball`: `track_ball` won 4/4.
  - `track_ball` vs `track_ball`: 2/2 truncations.
- `learned_latest` vs `random_uniform`: learned won 1/4, random won 3/4.
- `learned_latest` vs `track_ball`: 4/4 truncations, no wins.
- `learned_previous` vs `random_uniform`: learned won 3/4, random won 1/4.
- `learned_previous` vs `track_ball`: 4/4 truncations, no wins.
- `learned_latest` vs `learned_previous`: 1 win each, 2/4 truncations.

## Interpretation

The command now writes the intended scoreboard shape: config, optional split
metadata, checkpoint specs, paired-seat group summaries, compact
`scoreboard_rows`, and exact artifact paths.

This smoke is only plumbing. Because `latest` and `previous` are the same
checkpoint, the learned-vs-learned row should not be read as policy progress.
The learned-vs-random differences are tiny-seed variance from paired seat
assignments, not a meaningful latest-versus-previous signal.

## Artifacts

- `artifacts/local/dummy-pong-checkpoint-scoreboard-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-checkpoint-scoreboard-smoke-2026-05-09/episodes.jsonl`

## Follow-ups

- Run the same command with real `latest`, `previous`, and `best` checkpoints
  once periodic Pong policy checkpoints exist.
- Keep angle-control and contact-outcome artifacts as diagnostics outside the
  main checkpoint scoreboard.
