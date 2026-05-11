# 2026-05-09 dummy pong scoring all-ego value train smoke

## Question

Can all-ego scoring replay rows be backed up into scalar value targets and fit
by a tiny raster value regressor?

This is value-target plumbing. It is not policy improvement, MuZero, self-play,
or action cloning.

## Setup

- Source replay:
  `artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09`
- Rows: 392
- Return groups: `(game_index, ego_agent)`, 16 total
- Reward values: `-1.0`, `0.0`, `1.0`
- Target backup:
  `target_return[t] = reward_after_step[t] + discount * target_return[t+1]`
- Discount: `1.0`
- Model: deterministic NumPy linear ridge regressor from `raster_grid` plus
  `ego_agent`
- Validation split: 0.2, seed 0

## Commands

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_value_train.py \
  scripts/train_dummy_pong_value.py
```

```sh
uv run python scripts/train_dummy_pong_value.py \
  --replay-path artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-scoring-all-ego-value-train-smoke-2026-05-09 \
  --seed 0 \
  --validation-fraction 0.2 \
  --discount 1.0 \
  --ridge-l2 0.000001
```

```sh
uv run python -c 'from pathlib import Path; import json; from curvyzero.training.dummy_pong_value_train import DummyPongValueRegressor; p=Path("artifacts/local/dummy-pong-scoring-all-ego-value-train-smoke-2026-05-09/checkpoint.npz"); m=DummyPongValueRegressor.load_checkpoint(p); rows=[json.loads(line) for line in Path("artifacts/local/dummy-pong-scoring-replay-all-ego-smoke-2026-05-09/replay_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]; print(json.dumps({"checkpoint": str(p), "metadata_kind": m.metadata["kind"], "prediction_first_row": m.predict_value(rows[0]["raster_grid"], rows[0]["ego_agent"])}))'
```

## Results

- Train rows: 314
- Validation rows: 78
- Target return values: 196 positive rows, 196 negative rows, 0 zero rows
- Train MSE: 0.8383901173375532
- Validation MSE: 1.6682499812536524
- All-row MSE: 1.0035152943412666
- Reload helper smoke succeeded:
  `metadata_kind=curvyzero_dummy_pong_value_regressor`
- First-row reload prediction: `-0.18705765024392113`

## Interpretation

The smoke proves that score-delta rewards can be backed up into `target_return`
labels, summarized, fit by a deterministic NumPy value model, saved, and
reloaded.

The fit is not a strategic Pong result. A single raster frame plus ego-agent id
does not expose all useful state for value prediction, especially ball velocity
and behavior-policy context. The next strategic Pong target is still learning
to control return angle through off-center paddle hits so a learned policy can
eventually beat `track_ball`; this smoke does not edit bounce physics or prove
that behavior.

## Artifacts

- `artifacts/local/dummy-pong-scoring-all-ego-value-train-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-scoring-all-ego-value-train-smoke-2026-05-09/checkpoint.npz`

## Follow-ups

- Keep this value target path as plumbing for later policy/search work.
- Add value inputs or temporal context before reading value MSE as a strong
  strategic signal.
- Keep the Pong North Star focused on angle control and off-center hits,
  eventually beating scripted `track_ball`.
