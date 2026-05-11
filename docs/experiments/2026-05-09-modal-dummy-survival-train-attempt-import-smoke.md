# 2026-05-09 Modal Dummy Survival Train Attempt Import Smoke

## Question

Does the new Modal dummy survival train-attempt wrapper import cleanly and use
the intended Volume path shape?

## Setup

- New wrapper:
  `curvyzero.infra.modal.dummy_survival_train_attempt`.
- Task id: `dummy-survival`.
- Volume path shape:
  `training/dummy-survival/<run_id>/attempts/<attempt_id>/train`.
- No Modal training job was run.

## Commands

```sh
python3 -m py_compile \
  src/curvyzero/infra/modal/dummy_survival_train_attempt.py \
  src/curvyzero/infra/modal/run_management.py \
  src/curvyzero/infra/modal/volume_dummy_survival.py

uv run --extra modal python -c "from curvyzero.infra.modal import dummy_survival_train_attempt as m; from curvyzero.infra.modal.run_management import attempt_train_ref, latest_attempt_ref; print(m.APP_NAME); print(attempt_train_ref(m.TASK_ID, 'run-smoke', 'attempt-smoke')); print(latest_attempt_ref(m.TASK_ID, 'run-smoke'))"

uv run --extra modal python -c "from pathlib import Path; import tempfile; from curvyzero.infra.modal import dummy_survival_train_attempt as m; root=Path(tempfile.mkdtemp(prefix='curvyzero-attempt-smoke-')); m.RUNS_MOUNT=root; source=root/'training/dummy-survival/run-smoke/attempts/attempt-smoke/train/checkpoints/iteration-0001.npz'; source.parent.mkdir(parents=True); source.write_bytes(b'smoke'); summary={'periodic_checkpoints':[{'completed_iterations':1,'path':str(source),'eval':{'survival_rate':1.0}}]}; result=m._mirror_periodic_checkpoints(summary=summary, run_id='run-smoke', attempt_id='attempt-smoke'); print(result['files'][0]['checkpoint']['ref']); print(result['latest_pointer']['ref'])"
```

## Results

The import check printed:

```text
curvyzero-dummy-survival-train-attempt
training/dummy-survival/run-smoke/attempts/attempt-smoke/train
training/dummy-survival/run-smoke/latest_attempt.json
training/dummy-survival/run-smoke/checkpoints/iteration-000001/checkpoint.npz
training/dummy-survival/run-smoke/checkpoints/latest.json
```

## Interpretation

The wrapper compiles and imports locally. The helper path matches the required
run/attempt train directory. The periodic-checkpoint helper writes the expected
checkpoint copy and latest checkpoint pointer in a temporary local directory.
This check did not prove remote Modal execution or Volume writes.

## Follow-ups

- Run one tiny Modal job when remote write testing is desired.
- Add resume later, after the simple attempt path has been used.
