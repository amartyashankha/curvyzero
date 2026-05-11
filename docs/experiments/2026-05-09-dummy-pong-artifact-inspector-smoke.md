# 2026-05-09 dummy Pong artifact inspector smoke

## Question

Can we inspect existing dummy Pong artifact directories quickly enough to judge
replay and trace quality without reading raw JSONL by hand?

## Tool

- Module: `curvyzero.training.dummy_pong_artifact_inspect`
- CLI: `scripts/inspect_dummy_pong_artifacts.py`

The inspector accepts a directory that may contain any of:

- `summary.json`
- `replay_rows.jsonl`
- `frames.jsonl`
- `games.jsonl`
- `steps.jsonl`

It prints compact JSON with detected files, row counts, schema IDs, raster shape
checks, replay action histograms, reward totals, terminal/truncated counts, and
sample frame strings when `frames.jsonl` exists.

## Commands

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_artifact_inspect.py \
  scripts/inspect_dummy_pong_artifacts.py
```

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-imitation-replay-smoke \
  --sample-frames 2
```

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-observability-smoke \
  --sample-frames 2
```

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-imitation-replay-v0 \
  --sample-frames 1
```

## Results

### `artifacts/local/dummy-pong-imitation-replay-smoke`

- Detected `summary.json` and `replay_rows.jsonl`.
- Replay rows: 64.
- Observed raster shape: `9x15` for all 64 rows.
- Action histogram:
  - `player_0`: up 13, stay 5, down 14
  - `player_1`: up 13, stay 5, down 14
- Reward totals: zero for both agents.
- Quality notes:
  - replay rewards are all zero;
  - replay terminal flags are truncations only;
  - summary reports every replay game truncated at the max-step cap.

### `artifacts/local/dummy-pong-observability-smoke`

- Detected `summary.json`, `games.jsonl`, `steps.jsonl`, and `frames.jsonl`.
- Games: 3.
- Steps: 48.
- Frames: 51.
- Observed raster shape: `9x15` for all 51 frames.
- Frame count matches the expected reset-frame shape: one reset frame per game
  plus one frame per step.
- Sample frame strings print correctly, so a future worker can inspect visual
  trace quality without opening the whole JSONL file.
- Quality note: no obvious count or raster-shape problems detected.

### `artifacts/local/dummy-pong-imitation-replay-v0`

- Detected `summary.json` and `replay_rows.jsonl`.
- Replay rows: 7,680.
- Observed raster shape: `9x15` for all 7,680 rows.
- Action histogram:
  - `player_0`: up 520, stay 2,807, down 513
  - `player_1`: up 520, stay 2,807, down 513
- Reward totals: zero for both agents.
- Quality notes:
  - replay rewards are all zero;
  - replay terminal flags are truncations only;
  - summary reports every replay game truncated at the max-step cap.

## Interpretation

The inspector proves that current Pong artifacts can be checked quickly for
basic replay and trace health. It is especially useful before training changes:
workers can confirm whether the artifact has visual rows, whether schema IDs
match, whether row counts line up with summaries, and whether the data has
reward signal or only imitation targets.

The `dummy-pong-imitation-replay-v0` artifact is useful for imitation learning
and action-distribution debugging. It is not useful as reward-learning evidence:
all rewards are zero and every game hits the max-step cap.
