# Modal Training Run Management

Status: Draft

This design covers the next Modal shape for the toy MuZero training stack. It
uses Modal to run whole jobs and store saved outputs, not to run each training
step.

## Current State

CurvyZero already has three Modal training/output paths:

- Ephemeral dummy training jobs: `src/curvyzero/infra/modal/dummy_survival.py`
  and `src/curvyzero/infra/modal/dummy_line_duel.py` run one CPU Modal Function
  around the local dummy trainers and write saved outputs under
  `/tmp/artifacts/...`. These prove remote import/execution only; no Volume is
  attached. A Modal `Volume` is Modal's persistent shared file storage for
  remote jobs.
- Generic Volume smoke: `src/curvyzero/infra/modal/artifact_smoke.py`
  writes one JSON output file to the `curvyzero-runs` Volume at
  `experiments/<run_id>/attempts/<attempt_id>/artifact-smoke/artifact.json`,
  commits the Volume, and returns an exact ref.
- Volume dummy survival smoke:
  `src/curvyzero/infra/modal/volume_dummy_survival.py` runs dummy survival,
  writes normal training outputs to `curvyzero-runs`, commits, and returns file
  refs. Its current path is set by config:
  `training/dummy-survival/volume-smoke/seed-<seed>/iterations-<n>/episodes-per-iter-<n>/eval-episodes-<n>/`.

The local trainers already write useful files. Dummy survival writes
`summary.json`, `checkpoint.npz`, `iteration_metrics.jsonl`, and optional
periodic checkpoints under `checkpoints/iteration-0002.npz` style names. Dummy
line duel writes `summary.json`, `checkpoint.npz`, `iteration_metrics.jsonl`,
and `replay_rows.jsonl`. The eval modules can load explicit learned checkpoints
for comparison, and `scripts/run_dummy_survival_checkpoint_sweep.py` can rank
periodic survival checkpoints.

## Volume Layout

Use one active experiment Volume: `curvyzero-runs` mounted at `/runs`.

```text
/runs/training/<task_id>/<run_id>/
  run.json
  latest_attempt.json
  attempts/
    <attempt_id>/
      attempt.json
      train/
        summary.json
        iteration_metrics.jsonl
        replay_rows.jsonl
      eval/
        final/
          summary.json
          episodes.jsonl
      logs/
        stdout.txt
  checkpoints/
    iteration-000002/
      checkpoint.npz
      metadata.json
    iteration-000004/
      checkpoint.npz
      metadata.json
    latest.json
    best.json
  eval/
    <eval_id>/
      eval.json
      checkpoint_eval.jsonl
      episodes.jsonl
      best_checkpoint.json
      best_checkpoint_path.txt
```

`task_id` should be stable and plain, for example `dummy-survival` or
`dummy-line-duel`. `run_id` names the experiment intent and is reused across
retries/resumes. `attempt_id` names one Modal Function attempt and is always
immutable. Training attempts may write new checkpoint directories, but they must
not mutate old checkpoint payload files.

Pointer files are small JSON manifests. A manifest is a small file that lists
where the real output files are and how they were made.

- `latest_attempt.json`: most recent attempt id, Modal task id, status, start
  and end timestamps, and returned summary ref.
- `checkpoints/latest.json`: latest fully written checkpoint ref, attempt id,
  completed iteration/step, seed cursor, output hashes, and schema id.
- `checkpoints/best.json`: checkpoint selected by eval, eval id, ranking metric,
  metric value, and checkpoint ref.

Write payload files first, then pointer files, then `volume.commit()`. Any
separate reader or evaluator should call `volume.reload()` before reading refs
that another Function has committed.

## Resume Behavior

What can resume now:

- Eval can resume from policy checkpoints in the sense that both dummy evals can
  load `checkpoint.npz` files as learned policies.
- Dummy survival checkpoint sweeps can evaluate a directory of periodic
  checkpoints and select a best checkpoint.
- A Modal retry can safely discover a latest checkpoint pointer once that
  pointer exists, but the trainer does not yet know how to continue training
  from it.

What cannot resume yet:

- Training cannot faithfully continue from a checkpoint. Current checkpoints
  store tabular model state and metadata, but not replay contents for survival,
  RNG state/cursors, updater state beyond the tabular values, completed episode
  id for survival, or a canonical trainer state loader.
- Line duel writes replay rows, but still lacks a full run-state checkpoint that
  restores replay, next episode id, RNG, and loop counters.
- The current Volume dummy survival smoke has no `run_id`, `attempt_id`,
  `latest.json`, `best.json`, or idempotent retry behavior.

Required metadata before enabling Modal retries as real reentrant training:

- `schema_id`, `task_id`, `run_id`, `attempt_id`, and parent attempt/checkpoint.
- Full trainer config, seed, RNG state or deterministic cursor derivation, and
  completed iteration/step.
- Model checkpoint ref and hash.
- Replay checkpoint/chunk refs, replay capacity, and next episode id.
- Eval seed policy, eval metrics attached to the checkpoint, and ranking key.
- Code/dependency fingerprint: git commit when available, dirty flag when known,
  Python version, package pins, and Modal image/build marker.
- Terminal status: `running`, `completed`, `failed`, or `superseded`.

Until those fields exist, Modal Volume checkpoints are saved files for eval and
debugging, not full training-resume checkpoints.

## Job Split

Keep the first implementation as three CPU Functions that each run a whole job
and share the same image and Volume:

- `train_attempt(task_id, run_id, attempt_id, config, resume=True)`: owns one
  training loop inside one container, writes attempt outputs and checkpoints,
  commits after each checkpoint. It should initially support dummy survival only
  because survival already has periodic checkpoint output.
- `eval_checkpoint(task_id, run_id, eval_id, checkpoint_ref, config)`: reads one
  checkpoint from the Volume, evaluates it against fixed baselines/seeds, writes
  `eval/<eval_id>/...`, and returns compact metrics.
- `checkpoint_sweep(task_id, run_id, eval_id, checkpoint_glob, config)`: reads
  checkpoint pointers/manifests, evaluates candidates with `eval_checkpoint`
  style logic or local in-function looping, ranks them, writes
  `best_checkpoint.*`, and updates `checkpoints/best.json`.

Use Modal `.map`/`.starmap` for independent eval candidates only after the
single-function sweep works. Do not split environment ticks, MCTS expansion,
replay sampling, or gradient updates across Modal Functions.

## Scaling Path

Do not block toy runs on multi-GPU design. The near-term stack should run on CPU
or one cheap GPU and write the same files either way.

Next scale steps:

- Add optional `gpu=["L4", "T4"]` or similar only after CPU Volume run
  management is stable.
- Record device visibility, framework versions, memory, and throughput in
  `attempt.json`.
- Use seed/checkpoint/eval sweeps as parallel Modal jobs because each job can
  run independently.
- Keep replay chunks larger and fewer; if replay grows beyond active Volume
  comfort, move archival replay to a bucket while keeping manifests/pointers in
  `curvyzero-runs`.
- Defer multinode until one-container training is bottlenecked and checkpoint
  schemas are stable. The local Modal cluster example uses
  `@modal.experimental.clustered`, rank info, and `torch.distributed.run`, but
  that should remain a later launcher layer over the same run directory.

## Next Implementation Tasks

1. Add small run-id helpers for `run_id`, `attempt_id`, JSON writing, refs, and
   SHA-256 summaries, reusing the validation style in `artifact_smoke.py`.
2. Add a Volume-backed dummy survival train attempt with the proposed
   `training/dummy-survival/<run_id>/attempts/<attempt_id>/...` layout.
3. Teach the survival train wrapper to pass `checkpoint_every_iterations` and
   mirror periodic checkpoints into canonical `checkpoints/iteration-000NNN/`
   directories.
4. Write `checkpoints/latest.json` after each completed checkpoint and return
   it from the Modal Function.
5. Add a Modal checkpoint sweep Function for dummy survival that consumes the
   Volume checkpoint directory and writes `eval/<eval_id>/...` plus
   `checkpoints/best.json`.
6. Define the trainer-state metadata, then add actual training resume support
   locally before enabling `modal.Retries` for training attempts.
7. Port line duel to the same layout after survival is boring; include
   `replay_rows.jsonl` or replay chunks in attempt outputs from the start.

## Inspected Sources

- `src/curvyzero/infra/modal/artifact_smoke.py`
- `src/curvyzero/infra/modal/volume_dummy_survival.py`
- `src/curvyzero/infra/modal/dummy_survival.py`
- `src/curvyzero/infra/modal/dummy_line_duel.py`
- `src/curvyzero/training/dummy_survival.py`
- `src/curvyzero/training/dummy_survival_eval.py`
- `src/curvyzero/training/dummy_line_duel.py`
- `src/curvyzero/training/dummy_line_duel_eval.py`
- `scripts/run_dummy_survival_checkpoint_sweep.py`
- `docs/design/modal_architecture.md`
- `docs/design/training_architecture.md`
- `docs/runbooks/training_smokes.md`
- `docs/research/modal_training_patterns.md`
- `docs/research/modal_example_patterns.md`
- `docs/experiments/2026-05-08-modal-artifact-smoke.md`
- `docs/experiments/2026-05-08-modal-dummy-survival-smoke.md`
- `docs/experiments/2026-05-08-modal-dummy-line-duel-smoke.md`
- `docs/experiments/2026-05-09-modal-volume-dummy-survival-smoke.md`
- `docs/experiments/2026-05-09-dummy-survival-checkpoint-sweep-smoke.md`
- `/Users/shankha/modal-examples/06_gpu_and_ml/long-training.py`
- `/Users/shankha/modal-examples/03_scaling_out/basic_grid_search.py`
- `/Users/shankha/modal-examples/09_job_queues/dicts_and_queues.py`
- `/Users/shankha/modal-examples/14_clusters/simple_torch_cluster.py`
