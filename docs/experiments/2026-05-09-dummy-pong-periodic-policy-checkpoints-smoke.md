# 2026-05-09 dummy pong periodic policy checkpoints smoke

## Question

Can the tiny Pong imitation trainer save periodic policy checkpoints in a
stable shape for future old-vs-new scoreboard evals while preserving the final
root `checkpoint.npz`?

## Setup

- Changed `src/curvyzero/training/dummy_pong_imitation_train.py`.
- Changed `scripts/train_dummy_pong_imitation.py`.
- Existing replay: `artifacts/local/dummy-pong-imitation-replay-smoke`.
- No pytest.

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  scripts/train_dummy_pong_imitation.py
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-imitation-replay-smoke \
  --output-dir artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09 \
  --seed 0 \
  --epochs 3 \
  --learning-rate 0.5 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 1
```

```sh
uv run python -c 'import json; from pathlib import Path; from curvyzero.training.dummy_pong_imitation_train import DummyPongImitationPolicy; root=Path("artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09"); summary=json.loads((root/"summary.json").read_text()); latest=Path(summary["checkpoints"]["latest"]["path"]); policy=DummyPongImitationPolicy.load_checkpoint(latest); print(json.dumps({"latest": str(latest), "count": summary["checkpoints"]["count"], "metadata_completed_epochs": policy.metadata.get("completed_epochs"), "weights_shape": list(policy.weights.shape)}))'
```

## Results

- `py_compile` passed.
- Training wrote `summary.json`, final root `checkpoint.npz`, and three
  periodic policy checkpoints.
- `summary.json` included:
  - `checkpoints.count`: `3`
  - `checkpoints.latest.path`:
    `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000003/checkpoint.npz`
  - `checkpoints.refs` for epochs 1, 2, and 3.
- Reloading the latest periodic checkpoint reported:
  `{"count": 3, "metadata_completed_epochs": 3, "weights_shape": [2, 681, 3]}`.

## Interpretation

Periodic Pong policy checkpoints are ready for scoreboard plumbing. This smoke
only proves artifact shape and reloadability; it does not prove better policy
quality or reward learning.

## Artifacts

- `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/summary.json`
- `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000001/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000002/checkpoint.npz`
- `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000003/checkpoint.npz`

## Follow-ups

- Run the Pong checkpoint scoreboard with periodic checkpoints from a meaningful
  longer training attempt.
- Select best checkpoints on a fixed selection split before making quality
  claims.
