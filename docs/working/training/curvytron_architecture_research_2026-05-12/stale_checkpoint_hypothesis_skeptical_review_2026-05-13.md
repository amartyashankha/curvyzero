# Stale Checkpoint Hypothesis Skeptical Review - 2026-05-13

Scope: read-only review of the current hypothesis:

> The learner is alive, but checkpoint publication or discovery is broken,
> possibly because of Modal Volume concurrency, visibility, or commit semantics.

I read the current working notes, especially:

- `stale_checkpoint_critique_agent_2026-05-13.md`
- `checkpoint_save_path_critique_agent_2026-05-13.md`
- `artifact_liveness_audit_agent_2026-05-13.md`
- nearby status, log, config, and volume notes in this directory

No source code was changed.

## Short Verdict

The first half is strong for the sampled high-iter stale rows:

Fresh `progress_latest.json` plus increasing `learner_train_iter` means a
training process was alive recently enough to run the CurvyZero
`SaveCkptHook.__call__` wrapper.

The second half is still too broad:

The evidence proves that hook-level progress can move while visible
`iteration_*.pth.tar` files do not move. It does not yet prove Modal Volume
publication is the cause. It also does not prove DI-engine actually attempted
to save at the missing checkpoint boundaries.

The safer wording is:

> Some resumed/preempted rows keep running the save hook wrapper and updating
> progress, but the expected durable checkpoint files and resume sidecars do not
> appear. The immediate unknown is whether DI-engine skipped saving, saved to a
> different place/name, failed inside save, or wrote files that Modal did not
> publish or expose to readers.

## Ranked Holes

### 1. No direct proof of a missing save attempt

Why this matters:

`progress_latest.source="SaveCkptHook.__call__"` only proves the wrapper ran.
The upstream save hook can return without writing a checkpoint if its own gate
is false.

What would weaken or disprove the current publication theory:

- A stale row shows `engine.last_iter.val` never hitting `10000`, `20000`, etc.,
  even while `learner.train_iter` is high.
- A stale row shows `engine.rank != 0`, so DI-engine correctly skipped saves.
- A stale row shows `SaveCkptHook._freq` is not the expected cadence.
- A stale row shows the hook is called after every iteration, but the real save
  condition is false for a local DI-engine reason.

Read-only check:

Search existing learner logs for DI-engine lines like
`learner save ckpt in ...` for stale and healthy comparator rows. If stale rows
never log a save after `iteration_0`, this is not yet a Modal publication bug.

### 2. `learner_train_iter` may not be the save-gate counter

Why this matters:

The progress file records `learner.train_iter`. DI-engine's periodic hook uses
`engine.last_iter.val`. The docs infer they should track each other, but they
have not been shown side by side for stale rows.

What would weaken the current theory:

- `learner.train_iter` is high, but `engine.last_iter.val` is stuck, reset, or
  drifting after auto-resume.
- `engine.last_iter.val` is always just off the modulo boundary because resume
  restores one counter but not the other.

Read-only check:

Look for `last_iter` in checkpoint payloads, resume sidecars, and learner logs.
Compare stale rows with a healthy resumed row. If logs do not expose it, this
becomes the top diagnostic field for the next run.

### 3. Modal Volume commit is plausible, not proven

Why this matters:

There is real volume pressure: cleanup hit layer limits, and app logs show
`DataLossError` from eval/GIF commit paths. But the documented `DataLossError`
examples are app-wide and mostly tied to eval/GIF workers, not the selected
train checkpoint writes.

What would weaken the Modal-volume explanation:

- No train-call logs contain volume write, commit, reload, `DataLossError`, or
  `ResourceExhaustedError` near missing save boundaries.
- Healthy rows on the same app, same volume, same cadence, and same time window
  keep writing large checkpoints normally.
- Stale rows also lack later resume sidecars, which are smaller than model
  checkpoints. That suggests the checkpoint file may never have existed at the
  expected path, not just that a large file failed to publish.

Read-only check:

Filter app logs by the exact stale train FunctionCall IDs and container IDs for
`DataLossError`, `ResourceExhaustedError`, `commit`, `save ckpt`, `save_file`,
`OSError`, and `PytorchStreamReader`. Do not use broad app-level hits as row
evidence.

### 4. Preemption is a cofactor, not a cause

Why this matters:

Stale rows were preempted, but healthy rows were also preempted and recovered.
Preemption only becomes explanatory if it changes resume state, counters,
paths, rank, or hook state.

What would weaken the current theory:

- A healthy preempted row resumes from the same path and continues checkpointing
  because `last_iter`, rank, and exp path are intact.
- A stale row starts failing before any preemption.
- A stale row has no preemption but still shows the same high-iter/stale-ckpt
  signature.

Read-only check:

For stale and healthy preempted pairs, align:

- container start times
- preemption times
- auto-resume selected checkpoint
- next `progress_latest.timestamp`
- next `iteration_*.pth.tar` mtime
- next resume sidecar mtime

The key question is what happens at the first save boundary after restart.

### 5. Path/name mismatch is only partly ruled out

Why this matters:

Sampled stale rows had no newer files in the expected attempt-local `ckpt`
directory, and canonical `checkpoints/lightzero` was absent. That is strong for
those rows. It does not prove all stale rows share that shape.

What would weaken the current theory:

- A stale row has fresh checkpoints under another attempt, another `exp_name`,
  a canonical mirror that status ignores, or a non-`iteration_*.pth.tar` name.
- A status row is stale only because the reader prefers a stale canonical
  directory over a fresher attempt-local directory.

Read-only check:

For each high-iter stale row, list the full run tree under:

- `attempts/*/train/lightzero_exp/ckpt`
- `attempts/*/train/lightzero_exp/lightzero_resume_state`
- `checkpoints/lightzero`
- `checkpoints/lightzero_resume_state`

Include all names, sizes, and mtimes, not just `iteration_*.pth.tar`.

### 6. The sidecar evidence cuts both ways

Why this matters:

Stale rows have fresh progress, but resume sidecars remain stuck at
`iteration_0` or `iteration_20000`. The current notes read this as support for
missing durable checkpoints. That is fair.

But it also weakens a pure Modal large-file theory:

If the problem were only that large `.pth.tar` files are hidden or not
published, we might still expect sidecar behavior to expose some later state or
error. Instead the sidecar code appears to skip because the exact checkpoint
file is absent.

What would strengthen the Modal-volume theory:

- Logs show DI-engine saved `iteration_10000.pth.tar`, but the sidecar then
  failed because the just-saved file was not visible from the same process or
  path.
- A partial or zero-size checkpoint file appears and then disappears or never
  stabilizes.

Read-only check:

Search for partial files, temp names, size-zero files, and recent directory
mtime changes in stale checkpoint dirs. Compare against healthy dirs during a
normal save.

### 7. The stale tail may be several bugs, not one

Why this matters:

The strongest evidence is for high-iter stale rows. Low-iter stale rows,
progress-absent rows, old dependency rows, and poller-stale rows may have
different causes.

What would weaken a single checkpoint-publication theory:

- Low-iter stale rows never crossed a save boundary.
- Progress-absent rows still have healthy checkpoint files.
- Poller-stale rows have fresh train checkpoints but no eval/GIF artifacts.
- Heavy/sim16 rows show long checkpoint gaps but eventually save.

Read-only check:

Split stale rows into separate buckets before explaining them:

- high learner iter, checkpoint stale
- low learner iter, checkpoint stale
- progress absent or stale
- checkpoint fresh, eval/GIF stale
- heartbeat stale, no fresh process signal

### 8. Logs are currently too sparse to clear the save path

Why this matters:

The absence of visible exceptions in Modal logs does not mean the save path is
clean. Healthy train logs are also sparse.

What would weaken the current theory:

- Learner logs show no attempted save lines at missing boundaries.
- Learner logs show warnings that explain skipped saves.
- Per-container logs show the process restarted before each save boundary,
  which could repeatedly train from an old checkpoint without ever reaching a
  durable boundary in that container.

Read-only check:

Use exact train FunctionCall IDs, container IDs, and learner log files. Avoid
broad app searches except to find timestamps worth narrowing.

## Alternative Causes Still Open

Ranked from most important to keep alive:

1. **Counter mismatch after resume.** `learner.train_iter` advances, but
   DI-engine's `last_iter` or hook state is not restored in a way that triggers
   periodic checkpoint saves.
2. **Rank-gated skip.** DI-engine thinks the process is not rank 0 after some
   restart or environment change, even though the manifest says multi-GPU is
   false.
3. **Hook state or `ckpt_name` mismatch.** The hook runs, but its frequency,
   checkpoint name, or save-hook registration is not what the run expects.
4. **Wrong path after resume.** Training writes or looks under a different
   `exp_name` than status/poller/progress expect.
5. **Actual save failure with poor logging.** `save_file` or filesystem write
   fails, but the retained logs do not show it clearly.
6. **Modal Volume visibility/commit issue.** Files are written but not visible
   to other readers, or large checkpoint writes are more fragile than JSON
   progress writes.
7. **Repeated restart before durable save.** Containers restart in a way that
   preserves or reports high progress but repeatedly resumes from the last old
   durable checkpoint.
8. **Mixed stale population.** Some stale rows are true checkpoint bugs, while
   others are slow configs, dead functions with stale status, poller lag, or
   status-reader path hazards.

## Ranked Next Read-Only Checks

1. **Find real save attempts.** For one stale k0, one stale k20, and one
   healthy comparator, search learner logs for `learner save ckpt in` and
   checkpoint paths. This is the cleanest discriminator.
2. **Compare counters.** Extract any available `last_iter`, `train_iter`,
   `last_step`, and resume-state iteration fields from existing checkpoint
   payload metadata, sidecars, logs, or formatted config.
3. **Align preemption with save boundaries.** For stale and healthy preempted
   rows, make a timeline of container starts, preemptions, resume checkpoint,
   progress writes, checkpoint mtimes, and sidecar mtimes.
4. **Do a full tree listing for high-iter stale rows.** Include all attempts,
   canonical dirs, sidecars, non-iteration names, partial files, mtimes, and
   sizes.
5. **Run exact per-call log searches.** Search by FunctionCall/container, not
   only app-wide, for checkpoint save lines and volume/storage errors.
6. **Stratify stale rows.** Separate high-iter checkpoint-missing rows from
   low-iter, progress-absent, heartbeat-stale, and eval/GIF-only stale rows.
7. **Check prune/cancel intersection.** Cross-check stale survivor train/poller
   FunctionCall IDs against cancellation manifests. This is unlikely but cheap
   and high impact if true.
8. **Compare same-recipe healthy rows.** For each high-iter stale recipe, find
   a healthy row with the same recipe/render/sim/collector/batch when possible.
   This tests whether recipe is a trigger or just a correlate.
9. **Check live volume visibility semantics only after the above.** A focused
   Modal Volume test should ask: can a process write a large file, see it from
   itself, update JSON, and still have other readers miss the large file? That
   test is useful, but it should not replace the DI-engine save-gate checks.
10. **If a diagnostic launch is allowed later, log the gate state.** The fields
    to log are `learner.train_iter`, `engine.last_iter.val`, `engine.rank`,
    `freq`, `ckpt_name`, `exp_name`, save path, and whether a matching file
    exists after the original hook returns.

## Bottom Line

The alive-learner part is solid for the sampled high-iter stale rows.

The Modal Volume part is still a suspect, not a conclusion. The strongest
missing evidence is whether DI-engine tried to save the missing checkpoints at
all. Until that is shown, the top suspects should stay broad: counter mismatch,
rank gating, hook state, path/name mismatch, save failure, and only then Modal
publication or discovery.
