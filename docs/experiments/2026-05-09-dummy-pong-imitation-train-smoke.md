# 2026-05-09 dummy pong imitation train smoke

## Question

Can the new Pong imitation replay train and save a tiny supervised raster policy
checkpoint?

## Setup

- Environment: `dummy_pong_v0`
- Source replay: `artifacts/local/dummy-pong-imitation-replay-v0/replay_rows.jsonl`
- Source replay rows: 7,680
- Source games: 32
- Source game result: all 32 games truncated at 120 steps
- Source score reward: zero for both players in every source game
- Target policy: `track_ball`
- Learner: per-ego-agent NumPy softmax linear classifier
- Input: raster grid decoded into one-hot cell features plus small geometry
  features derived only from the raster
- Seed: 0
- Epochs: 1,000

## Command

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_imitation_train.py \
  scripts/train_dummy_pong_imitation.py
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-imitation-replay-v0 \
  --output-dir artifacts/local/dummy-pong-imitation-train-smoke \
  --seed 0 \
  --epochs 1000 \
  --learning-rate 1.0 \
  --validation-fraction 0.2
```

```sh
uv run python -c 'import json; from pathlib import Path; from curvyzero.training.dummy_pong_imitation_train import DummyPongImitationPolicy; policy=DummyPongImitationPolicy.load_checkpoint(Path("artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz")); row=json.loads(Path("artifacts/local/dummy-pong-imitation-replay-v0/replay_rows.jsonl").read_text(encoding="utf-8").splitlines()[0]); print(json.dumps({"ego_agent": row["ego_agent"], "target_action_id": row["target_action_id"], "predicted_action_id": policy.predict_action_id(row["raster_grid"], row["ego_agent"]), "predicted_action_label": policy.predict_action_label(row["raster_grid"], row["ego_agent"])}))'
```

## Results

- `py_compile` completed.
- Training completed and wrote `summary.json` plus `checkpoint.npz`.
- Train rows: 6,144
- Validation rows: 1,536
- Train loss: 0.08276913487806879
- Train accuracy: 0.9954427083333334
- Validation loss: 0.09135482028546531
- Validation accuracy: 0.990234375
- All-rows accuracy: 0.9944010416666667
- The reload smoke predicted the first row correctly:
  `target_action_id=0`, `predicted_action_id=0`, `predicted_action_label=up`.

## Interpretation

This proves the visual Pong replay can feed a small supervised learner, save a
checkpoint, reload it, and choose an action from a raster grid for one ego
agent.

This does not prove reward learning. The replay came from `track_ball` versus
`track_ball`; all 32 source games timed out and produced zero score reward. Use
this checkpoint only as a behavioral cloning artifact that copies `track_ball`.

This also does not prove MuZero, planning, self-play improvement, or a learned
winning objective. The next honest step is learned-checkpoint eval support for
Pong, where this policy can be compared against `random_uniform` and
`track_ball` on heldout seeds.

## Artifacts

- `artifacts/local/dummy-pong-imitation-replay-v0/summary.json`
- `artifacts/local/dummy-pong-imitation-replay-v0/replay_rows.jsonl`
- `artifacts/local/dummy-pong-imitation-train-smoke/summary.json`
- `artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz`

## Follow-ups

- Add `learned:<checkpoint.npz>` support to the Pong eval harness.
- Compare learned checkpoint behavior against `random_uniform` and `track_ball`.
- Keep reward-learning claims blocked until there is a score-bearing rollout,
  train/eval split, and heldout learned-policy eval.
