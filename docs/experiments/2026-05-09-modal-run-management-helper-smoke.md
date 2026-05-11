# 2026-05-09 Modal Run Management Helper Smoke

## Question

Can the Modal run-management design be represented as small reusable helper
functions before rewriting the training wrapper?

## Setup

- New helper module: `curvyzero.infra.modal.run_management`.
- Scope: run/attempt ids, relative Volume refs, stable JSON writing, SHA-256
  file summaries, and pointer manifest shapes for latest/best checkpoints.
- Existing `volume_dummy_survival.py` now reuses `file_summary(...)`; its
  deterministic `volume-smoke/...` behavior is unchanged.

## Command

```sh
python3 -m py_compile \
  src/curvyzero/infra/modal/run_management.py \
  src/curvyzero/infra/modal/volume_dummy_survival.py

PYTHONPATH=src python3 -c "from curvyzero.infra.modal.run_management import attempt_train_ref, best_checkpoint_pointer, checkpoint_file_ref, new_attempt_id, new_run_id; run_id=new_run_id('smoke'); attempt_id=new_attempt_id('attempt'); print(attempt_train_ref('dummy-survival', run_id, attempt_id)); print(checkpoint_file_ref('dummy-survival', run_id, 2)); print(best_checkpoint_pointer(task_id='dummy-survival', run_id=run_id, eval_id='eval-smoke', ranking_metric='survival_rate', metric_value=1.0, checkpoint_ref=checkpoint_file_ref('dummy-survival', run_id, 2).as_posix())['schema'])"
```

## Results

- Helper import/API smoke produced refs like:
  `training/dummy-survival/<run_id>/attempts/<attempt_id>/train`.
- Checkpoint refs use canonical
  `training/dummy-survival/<run_id>/checkpoints/iteration-000002/checkpoint.npz`.
- Best pointer schema is `curvyzero_modal_training_best_checkpoint/v1`.

## Interpretation

This is the first useful implementation step beyond a Modal Volume smoke. The
project now has shared path and manifest helpers, but no real run-attempt
training wrapper yet.

## Artifacts

No new run artifacts. This was a local import/compile smoke only.

## Follow-ups

- Add a Volume-backed dummy survival train-attempt wrapper using
  `training/dummy-survival/<run_id>/attempts/<attempt_id>/...`.
- Mirror periodic checkpoints into canonical checkpoint directories.
- Write `checkpoints/latest.json` after completed checkpoints.
- Add a Modal checkpoint sweep that writes `checkpoints/best.json`.
