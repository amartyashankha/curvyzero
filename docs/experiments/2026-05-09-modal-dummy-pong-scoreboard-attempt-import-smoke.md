# 2026-05-09 Modal Dummy Pong Scoreboard Attempt Import Smoke

## Question

Can the new CPU Modal wrapper for the dummy Pong checkpoint scoreboard compile,
import, and point at the expected run/attempt eval path?

## Setup

- New wrapper:
  `curvyzero.infra.modal.dummy_pong_scoreboard_attempt`.
- Modal app: `curvyzero-dummy-pong-scoreboard-attempt`.
- Volume: `curvyzero-runs`.
- Task id: `dummy-pong`.
- Default eval id: `checkpoint-scoreboard`.
- Default output path shape:
  `training/dummy-pong/<run_id>/attempts/<attempt_id>/eval/checkpoint-scoreboard`.
- No remote Modal job was run for this smoke.

This wrapper is for remote eval and artifact plumbing. It does not prove Pong
policy quality.

## Commands

```sh
python3 -m py_compile \
  src/curvyzero/infra/modal/dummy_pong_scoreboard_attempt.py \
  src/curvyzero/infra/modal/run_management.py \
  scripts/run_dummy_pong_checkpoint_scoreboard.py

uv run --extra modal python -c "from curvyzero.infra.modal import dummy_pong_scoreboard_attempt as m; from curvyzero.infra.modal.run_management import attempt_eval_ref; print(m.APP_NAME); print(m.TASK_ID); print(attempt_eval_ref(m.TASK_ID, 'run-smoke', 'attempt-smoke', m.DEFAULT_EVAL_ID)); print(m._checkpoint_args_from_text('latest=ref:training/dummy-pong/run/checkpoint.npz, previous=/tmp/checkpoint.npz'))"

uv run --extra modal python -c "from curvyzero.infra.modal import dummy_pong_scoreboard_attempt as m; args, inputs = m._resolved_checkpoint_args(['latest=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz']); print(args[0]); print(inputs[0]['source_kind']); print(inputs[0]['file']['bytes'])"

uv run --extra modal python -c "from scripts.run_dummy_pong_checkpoint_scoreboard import _checkpoint_policy_arg; print(_checkpoint_policy_arg('latest=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz'))"
```

## Results

The compile command passed.

The import/path check printed:

```text
curvyzero-dummy-pong-scoreboard-attempt
dummy-pong
training/dummy-pong/run-smoke/attempts/attempt-smoke/eval/checkpoint-scoreboard
['latest=ref:training/dummy-pong/run/checkpoint.npz', 'previous=/tmp/checkpoint.npz']
```

The checkpoint resolver check printed:

```text
latest=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz
relative_path
13497
```

The scoreboard helper import printed:

```text
learned:latest=artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz
```

## Interpretation

The wrapper imports with Modal available, uses the run/attempt eval path helper,
can parse comma-separated checkpoint specs, and can resolve an existing local
checkpoint path for a tiny helper smoke.

The remote path of interest is still the Volume path:
`training/dummy-pong/<run_id>/attempts/<attempt_id>/eval/checkpoint-scoreboard`.
When run remotely, checkpoint refs should usually be passed as `ref:<volume_ref>`
so the Modal Function reads them from `curvyzero-runs`.

## Follow-ups

- Run one tiny remote Modal job after a Pong checkpoint exists in
  `curvyzero-runs`.
- Keep this as one whole eval job. Do not split scoreboard rows into per-step
  Modal calls.
