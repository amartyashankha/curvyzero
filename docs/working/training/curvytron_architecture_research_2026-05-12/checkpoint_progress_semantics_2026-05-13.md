# Checkpoint Progress Semantics for Fresh Progress with Only `iteration_0`

Date: 2026-05-13

Scope: read-only code-path trace of
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
No source code was edited. This note distinguishes evidence that a checkpoint
file was actually saved from evidence that a hook or progress writer ran.

## Short Answer

Fresh `train/progress_latest.json` with increasing `learner_train_iter` does
not prove a new `iteration_N.pth.tar` file was saved. In the stock LightZero
path, CurvyZero wraps DI-engine's `SaveCkptHook.__call__` and writes progress
after the original hook returns. Upstream DI-engine's `SaveCkptHook` can return
without writing a file whenever its save predicate is false:

```text
engine.rank == 0 and engine.last_iter.val % self._freq == 0
```

CurvyZero then scans `exp_name/ckpt` for the highest visible
`iteration_*.pth.tar` and writes that checkpoint name into
`progress_latest.json`. If only `iteration_0.pth.tar` exists, the progress file
can honestly show:

```json
{
  "learner_train_iter": 178018,
  "iteration": 0,
  "checkpoint_name": "iteration_0.pth.tar",
  "source": "SaveCkptHook.__call__"
}
```

That means the wrapped hook path is still being reached with a learner object
whose `train_iter` is high. It does not mean DI-engine saved `iteration_178018`
or any modulo checkpoint after `iteration_0`.

## Local Stock LightZero Path

1. `main(...)` chooses the Modal train function. When background eval uses
   poller mode, it spawns
   `lightzero_curvytron_visual_survival_checkpoint_eval_poller` before spawning
   or running the trainer
   (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:9933-9975`).

2. `_run_lightzero_curvytron_visual_survival_train(...)` builds LightZero config
   and patches `policy.learn.learner.hook.save_ckpt_after_iter`
   (`...:4382-4491`). The default stock value is `100`
   (`...:348-350`), but many matrix launches override it to larger values such
   as `10000` or `15000`.

3. Before calling `lzero.entry.train_muzero`, the trainer installs three
   relevant wrappers:
   `restore_resume_state`, `restore_progress_writer`, and
   `restore_live_publisher` (`...:3556-3567`). Then it calls
   `train_muzero(...)` (`...:3596-3601`).

4. `_install_checkpoint_progress_writer(...)` patches
   `BaseLearner.save_checkpoint`. It calls the original method first, then
   writes `progress_latest.json` (`...:1868-1918`).

5. `_install_lightzero_full_resume_state_hooks(...)` separately patches
   upstream `ding.worker.learner.learner_hook.SaveCkptHook.__call__`. It calls
   the original hook first, then tries to save a CurvyZero resume sidecar, then
   writes `progress_latest.json` with
   `source="SaveCkptHook.__call__"` (`...:2039-2076`).

6. `_write_checkpoint_progress_latest(...)` scans only `exp_name / "ckpt"` for
   `iteration_*.pth.tar`, chooses the maximum numbered checkpoint, and records
   both that checkpoint iteration and the learner's current `train_iter`
   (`...:1804-1865`). If no checkpoint exists it falls back to
   `learner_train_iter`; if an old checkpoint exists it keeps reporting the old
   checkpoint.

7. `_save_lightzero_resume_sidecar_state(...)` computes the exact expected
   file `exp_name/ckpt/iteration_<learner.train_iter>.pth.tar`. If it is
   absent, it returns `{"saved": false, "reason":
   "matching_iteration_checkpoint_not_found"}` and the wrapper ignores that
   return value before writing progress (`...:2085-2097`).

8. After `train_muzero` returns, the stock trainer scans artifacts and mirrors
   checkpoints into the canonical run-level root
   `checkpoints/lightzero` (`...:3672-3676`, `...:5432-5510`). This is not a
   live mirror during training.

## Upstream Semantics Checked

Primary upstream references:

- DI-engine source/docs for `BaseLearner`: default
  `save_ckpt_after_iter=10000`, `train(...)` calls `after_iter` before
  incrementing `_last_iter`, and `save_checkpoint(...)` invokes the
  `save_ckpt_after_run` hook.
  <https://di-engine-docs.readthedocs.io/en/latest/_modules/ding/worker/learner/base_learner.html>
- DI-engine `SaveCkptHook` source: saves only when rank is 0 and
  `engine.last_iter.val % freq == 0`; filename defaults to
  `iteration_<engine.last_iter.val>.pth.tar`; payload includes `last_iter` and
  `last_step`.
  <https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/worker/learner/learner_hook.py>
- LightZero `train_muzero`: creates `BaseLearner`, calls
  `learner.call_hook('before_run')`, performs an initial evaluator call using
  `learner.save_checkpoint`, trains via `learner.train(...)`, stops at
  `max_env_step` or `max_train_iter`, then calls `learner.call_hook('after_run')`.
  <https://www.aidoczh.com/lightzero/_modules/lzero/entry/train_muzero.html>
- DI-engine hook overview: `save_ckpt_after_iter` is an `after_iter` hook with a
  frequency parameter; `save_ckpt_after_run` is an `after_run` hook.
  <https://di-engine-test.readthedocs.io/en/latest/feature/wrapper_hook_overview_en.html>

The key ordering is subtle: upstream `BaseLearner.train` calls `after_iter`
while `last_iter` is still the current value, then increments it. Therefore
`iteration_0.pth.tar` is expected on the first eligible `after_iter` hook call.
Later periodic checkpoints appear at `last_iter` values divisible by the
configured frequency.

## What Each Artifact Proves

Strong proof that a checkpoint was saved:

- A non-empty `lightzero_exp/ckpt/iteration_N.pth.tar` exists for `N > 0`.
- A `learner save ckpt in .../iteration_N.pth.tar` log line exists for `N > 0`.
- A matching `lightzero_exp/lightzero_resume_state/iteration_N.resume_state.pkl`
  exists. This is secondary proof because the sidecar writer refuses to write
  unless the matching checkpoint file exists.
- The final train summary's `lightzero_artifacts.checkpoint_files` includes
  `iteration_N.pth.tar`, or `checkpoint_mirror.copied_checkpoints` includes the
  canonical mirror for `iteration_N.pth.tar`. These are only final-summary
  proofs because mirroring happens after `train_muzero` returns.

Proof only that a hook/progress path ran:

- Fresh `progress_latest.timestamp` or `updated_at`.
- `progress_latest.source == "SaveCkptHook.__call__"`.
- Increasing `progress_latest.learner_train_iter`.
- A poller `seen_count`, `last_scan_count`, or eval/GIF status that never moves
  past `iteration_0`.

Proof only that downstream workers could see a checkpoint:

- Eval/inspection or GIF artifacts for `live_checkpoint_iteration_0`.
- A spawned eval/GIF function call id. The scheduler only spawns from visible
  checkpoint refs, but it may be the old `iteration_0` ref.

## Eval, GIF, and Poller Semantics

Hook launch mode:

- `_install_live_checkpoint_publisher(...)` wraps `BaseLearner.save_checkpoint`
  and calls `_spawn_checkpoint_eval_triggers(...)` after the original save
  returns (`...:1742-1794`).
- `_spawn_checkpoint_eval_triggers(...)` scans `exp_name`, filters names with
  `_live_eval_checkpoint_name`, and schedules eval/GIF for unseen checkpoint
  refs (`...:5693-5735`, `...:6078-6079`).
- The checkpoint ref passed to eval/GIF is the source attempt-local checkpoint,
  not the future canonical mirror (`...:5738-5743`).

Poller launch mode:

- The poller repeatedly reloads the Modal volume, scans the attempt-local
  `lightzero_exp`, filters live checkpoint names, waits for a stable
  size/mtime fingerprint, then spawns eval/GIF (`...:6193-6338`).
- Poller status writes use `runs.write_json`, but the poller does not commit
  the volume in its loop. Tests assert no `runs_volume.commit()` from the
  scheduling path.
- Eval/inspection and GIF workers do commit their own outputs using
  `_commit_runs_volume_with_backoff(...)` (`...:6487-6518`,
  `...:6527-6755`, `...:7726-7733`).

Therefore eval/GIF absence after `iteration_0` is usually downstream evidence
that no later source checkpoint was visible to the scheduler. It is not itself
proof of why saving stopped.

## Modal Volume Commit Semantics

For the stock trainer path, there is no unconditional final Modal volume commit
in train mode. The only explicit final commit in this function is gated to
`mode == "profile" and profile_volume_commit` (`...:3864-3877`).

This means:

- `progress_latest.json`, attempt manifests, source checkpoints, and final
  summaries are regular volume writes from the train container, not guaranteed
  by an explicit train-mode commit in this code path.
- Background eval/GIF workers do explicitly commit their own artifacts after
  they finish.
- A remote observer can see committed downstream eval/GIF artifacts more
  reliably than in-progress trainer-local JSON if Modal visibility is lagging.

Still, commit visibility lag alone does not explain a fresh, visible
`progress_latest.json` whose `learner_train_iter` advances while the same
visible checkpoint directory remains at `iteration_0`. That pattern points
inside the trainer's save predicate, save destination/name, or save completion
path.

## Ranked Likely Code-Path Failure Points

1. **CurvyZero writes progress after skipped `SaveCkptHook` calls.**
   This is the highest-confidence semantic issue. The wrapper writes progress
   after every `SaveCkptHook.__call__` return, but upstream only writes a file
   when `rank == 0` and `last_iter % freq == 0`. This fully explains fresh
   progress at non-cadence iterations. It does not fully explain staleness after
   many cadence boundaries unless another item below is also true.

2. **The hook's save predicate is not true at expected cadence boundaries.**
   If `engine.last_iter.val` is not the same value exposed by
   `engine.train_iter`, or if the wrapper sees a high `train_iter` but the
   upstream hook checks a different or stale `last_iter`, the progress writer
   can advance while the file-save condition never fires. This is especially
   plausible because the wrapper records `learner.train_iter`, while upstream
   checks `engine.last_iter.val`.

3. **The saving rank is not rank 0.**
   Upstream `SaveCkptHook` silently skips file saving unless `engine.rank == 0`.
   The CurvyZero wrapper still writes progress after the original hook returns.
   In multi-GPU or unexpected rank setup, this can produce exactly "hook ran,
   no file saved." The default command has `lightzero_multi_gpu=False`, so this
   is lower than item 2 for normal runs, but it is a direct code-path gate.

4. **A checkpoint is being saved under a different filename.**
   Upstream uses `engine.ckpt_name` when set, otherwise
   `iteration_<last_iter>.pth.tar`. CurvyZero scans only `iteration_*.pth.tar`
   in `exp_name/ckpt` for progress. If `engine.ckpt_name` remains set to a
   mutable name such as `ckpt_best.pth.tar`, progress can keep reporting the
   last numbered checkpoint. This is plausible for the initial evaluator path,
   but should not persist after `BaseLearner.save_checkpoint` resets
   `ckpt_name`; persistent evidence would be fresh `ckpt_best` mtimes with no
   fresh numbered checkpoints.

5. **A checkpoint is being saved outside `exp_name/ckpt`.**
   `_write_checkpoint_progress_latest` and the poller both scan the
   attempt-local `lightzero_exp/ckpt`. Mirroring scans under `exp_name`.
   If upstream sees a different `exp_name`, different working directory, or
   `engine.instance_name != "learner"` causing `ckpt_<instance_name>`, the save
   could be real but invisible to the progress/poller path.

6. **Checkpoint save fails or is incomplete without an outer failure signal.**
   If `save_file` does not complete but the original hook still returns, the
   wrapper will write progress. A thrown exception should normally bubble out
   before the progress write, so this requires a swallowed lower-level failure
   or abnormal filesystem behavior. Look for zero-byte or temp-like files,
   partial torch archives, and missing `learner save ckpt in ...` lines.

7. **Train-mode Modal volume commit visibility masks saved checkpoints.**
   The stock trainer does not explicitly commit during or after train mode.
   This can delay remote visibility of source checkpoints. It is a weaker
   explanation for the observed split when `progress_latest.json` itself is
   freshly visible from the same attempt root and the checkpoint directory still
   lists only `iteration_0`.

8. **Final mirror timing creates stale canonical checkpoint views.**
   The canonical run-level `checkpoints/lightzero` mirror is populated only
   after `train_muzero` returns. Any status reader that prioritizes the
   canonical mirror during a live run can show stale `iteration_0` even if a
   later attempt-local checkpoint exists. This affects status presentation more
   than the underlying source checkpoint path.

## Practical Read-Only Checks

For any suspicious run, the highest-signal checks are:

1. Compare `progress_latest.learner_train_iter` with
   `progress_latest.iteration` and configured `save_ckpt_after_iter`.
2. List attempt-local `train/lightzero_exp/ckpt`, not only the canonical
   `checkpoints/lightzero` mirror.
3. Check whether `ckpt_best.pth.tar` has a fresh mtime while numbered
   `iteration_*.pth.tar` files are stale.
4. Search trainer logs for `learner save ckpt in` lines and compare the listed
   filenames to the volume listing.
5. If `phase_profile.learner_save_checkpoint_calls` is high but numbered files
   are absent, remember that this profiles `BaseLearner.save_checkpoint`, not
   the periodic `SaveCkptHook.__call__` file write.

