# 2026-05-09 dummy pong selection record smoke

## Question

Can an existing Pong checkpoint scoreboard summary produce a small
`selection_record.json` without rerunning eval or touching the Modal wrapper?

## Setup

- Source scoreboard summary:
  `artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09/summary.json`
- Eval split in source summary: `dummy_pong_monitor_v0`, role `monitor`
- Candidates: `epoch_1` and `epoch_3`
- No pytest.

## Command

```sh
uv run python -m py_compile scripts/select_dummy_pong_checkpoint.py
```

```sh
uv run python scripts/select_dummy_pong_checkpoint.py \
  --summary artifacts/local/dummy-pong-periodic-checkpoint-scoreboard-smoke-2026-05-09/summary.json \
  --output-dir artifacts/local/dummy-pong-checkpoint-selection-record-smoke-2026-05-09
```

## Results

- `py_compile` passed.
- The selector wrote `selection_record.json`.
- Source summary SHA-256 recorded in the selection record:
  `bdde7832958f65e244b6a825dade88d538b8c92dcfcb013fedbd73547ca13419`.
- Selected label: `epoch_3`.
- Selected policy id: `learned_epoch_3`.
- Selected checkpoint path:
  `artifacts/local/dummy-pong-imitation-periodic-checkpoint-smoke-2026-05-09/checkpoints/epoch-000003/checkpoint.npz`.
- Both candidates had `0/4` wins against `track_ball`.
- `epoch_3` won `2/4` against `random_uniform`; `epoch_1` won `1/4`.
- Both candidates had `2/8` truncations across the required baseline rows.

## Interpretation

This proves the local selection-record lane only. The record selects a
checkpoint on one scoreboard split by the existing rule: win rate versus
`track_ball`, then win rate versus `random_uniform`, then lower truncation rate.

This is not heldout confirmation and does not prove final Pong checkpoint
quality. The source split was a monitor smoke, so the result should be read as
artifact plumbing plus auditable selection bookkeeping.

## Artifacts

- `artifacts/local/dummy-pong-checkpoint-selection-record-smoke-2026-05-09/selection_record.json`

## Follow-ups

- Run the selector on a real `selection` split once candidate Pong checkpoints
  are meaningful.
- Add a separate heldout confirmation lane before making quality claims.
