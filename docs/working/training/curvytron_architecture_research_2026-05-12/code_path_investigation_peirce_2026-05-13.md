# CurvyTron Modal Code Path Investigation - Peirce - 2026-05-13

Scope: read-only investigation of how CurvyTron Modal training rows can report
`running` while checkpoints, evals, or GIFs are stale. Context source:
`run_health_check_2026-05-13.md`.

No code changes were made for this investigation.

## Bottom Line

A status row can look `running` while checkpoint progress is stale because
`train_status` is read from `status_heartbeat.json`, and that file is not a
continuous heartbeat. In the stock LightZero path it is written before
`train_muzero`, after auto-resume checking, and only once more after
`train_muzero` returns. If training hangs, slows dramatically, gets stuck inside
LightZero, or dies before the final writer runs, the last durable train status
can remain `running` indefinitely.

Checkpoint, eval, and GIF freshness are separate signals. Checkpoint progress is
written only when checkpoint-saving hooks fire. Eval/GIF artifacts are triggered
only from visible `iteration_*.pth.tar` checkpoint files. The status reader
does not derive liveness from checkpoint mtime, eval mtime, GIF mtime, or poller
heartbeat age; it simply rolls those artifacts up next to the train heartbeat.

The health-check conclusion is therefore consistent with the code: `running`
means "the last train heartbeat says running", not "the learner is currently
advancing checkpoints."

## Relevant Paths

- Stock LightZero launcher and background workers:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- Status reader:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`
- Eval harness:
  `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py`
- Run artifact helpers:
  `src/curvyzero/infra/modal/run_management.py`
- Two-seat local trainer path used by the same Modal launcher:
  `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- Tests with useful semantics:
  `tests/test_curvytron_run_status.py`
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py`

## Launch and Running Status Semantics

The stock LightZero train function writes attempt state, latest attempt, and a
train heartbeat as `running` before entering `train_muzero`. The heartbeat stage
is first `before_train_muzero`, then `auto_resume_checked`.

Code references:

- `_write_attempt_state(... status="running" ...)` and
  `_write_latest_attempt(... status="running" ...)` are called before training:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:3379-3397`.
- `_write_train_status_heartbeat(... status="running", stage="before_train_muzero" ...)`
  is called before importing/configuring training:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:3398-3407`.
- After auto-resume resolution, `_write_attempt_state(... status="running" ...)`
  and `_write_train_status_heartbeat(... stage="auto_resume_checked" ...)` are
  called again:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:3470-3489`.
- The final heartbeat is only written after `train_muzero` exits and summary
  writing begins:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:3832-3863`.

The heartbeat writer itself is a simple JSON overwrite. It records
`heartbeat_at`, `status`, `stage`, `summary_ref`, and `checkpoint_root_ref`, but
there is no background thread or loop that refreshes it during training:
`lightzero_curvyzero_stacked_debug_visual_survival_train.py:8098-8131`.

Run-management allows only four attempt statuses:
`running`, `completed`, `failed`, and `superseded`
(`run_management.py:27`, `run_management.py:310-365`). There is no status such
as `stale`, `hung`, or `heartbeat_expired`.

Consequence: if the process never reaches the final status writer, the row stays
`running` from the reader's point of view.

## `progress_latest.json` Semantics

In the stock LightZero path, `progress_latest.json` is checkpoint-driven. The
launcher installs a wrapper around `BaseLearner.save_checkpoint`. After the
original save returns, the wrapper calls `_write_checkpoint_progress_latest`.

Code references:

- `_install_checkpoint_progress_writer` patches `BaseLearner.save_checkpoint`:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:1868-1918`.
- `_write_checkpoint_progress_latest` scans `exp_name/ckpt` for the highest
  `iteration_*.pth.tar`, records `event: "checkpoint"`, `iteration`,
  `learner_train_iter`, `timestamp`, `updated_at`, `checkpoint_ref`, and
  `checkpoint_name`, then writes `progress_latest.json`:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804-1865`.
- The resume-state hook also updates `progress_latest.json` from
  `SaveCkptHook.__call__`; the test confirms the source field and checkpoint
  naming:
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py:90-146`.

This file is not an independent live training heartbeat. If no checkpoint is
saved, it does not advance. If checkpoint saving is slow because iterations are
slow, it is stale even when Python is still inside the training loop. If the hook
does not install or an exception happens inside the wrapper, the exception is
printed and training continues; no status field is flipped to failed
(`lightzero_curvyzero_stacked_debug_visual_survival_train.py:1895-1911`).

The JSON write helper is a direct file overwrite, not an atomic temp-file rename:
`run_management.py:264-276`. The status tests explicitly cover an empty or
partial `progress_latest.json` being treated as unreadable:
`tests/test_curvytron_run_status.py:280-302`.

For the experimental two-seat path, progress is more iteration-driven:
`_append_progress_line` appends to `progress.jsonl` and overwrites
`progress_latest.json` every configured progress interval, checkpoint iteration,
or problem:
`curvytron_two_seat_lightzero_train_smoke.py:3036-3054` and
`curvytron_two_seat_lightzero_train_smoke.py:966-990`. That is a different
semantics from the stock LightZero checkpoint hook.

## Checkpoint Saving and Mirroring

Stock LightZero source checkpoints live under the attempt-local LightZero
experiment directory: `attempts/<attempt_id>/train/lightzero_exp/ckpt`.
The progress writer discovers them by scanning that directory:
`lightzero_curvyzero_stacked_debug_visual_survival_train.py:1804-1826`.

The launcher also has a mirror step that copies checkpoint files into the
canonical run-level checkpoint root:
`training/<task>/<run>/checkpoints/lightzero`. However, that mirror is called
after `train_muzero` returns, not continuously during stock training:
`lightzero_curvyzero_stacked_debug_visual_survival_train.py:3672-3676`.

The background eval trigger intentionally uses source checkpoint refs, not
future mirrored refs. The test asserts this:
`tests/test_curvytron_live_checkpoint_eval_plumbing.py:2070-2142`.

Important status-reader detail: `_checkpoint_summary` looks for checkpoints in
the canonical run-level mirror first, then the attempt-local source directory:
`lightzero_curvytron_run_status.py:809-823`. It uses the first directory that
exists. Therefore, if a stale canonical checkpoint directory exists while a
running attempt has fresher source checkpoints, the displayed
`latest_checkpoint` can be stale because the reader never falls through to the
attempt-local directory. This is a plausible code-level explanation for a row
whose source training is progressing but whose status checkpoint rollup appears
stale.

For two-seat training, checkpoints are written directly to the canonical
checkpoint root. `_save_lightzero_policy_checkpoint` writes
`iteration_<n>.pth.tar`, then copies it to `ckpt_best.pth.tar` and
`latest.pth.tar`:
`curvytron_two_seat_lightzero_train_smoke.py:3553-3595`. The Modal wrapper passes
that canonical checkpoint directory when optimizer steps are allowed:
`lightzero_curvyzero_stacked_debug_visual_survival_train.py:8412-8417` and
`lightzero_curvyzero_stacked_debug_visual_survival_train.py:8604-8612`.

## Checkpoint Poller Semantics

The default background eval launch mode is poller
(`DEFAULT_BACKGROUND_EVAL_LAUNCH_KIND = "poller"`):
`lightzero_curvyzero_stacked_debug_visual_survival_train.py:439-449`.

When enabled, the local Modal launcher spawns
`lightzero_curvytron_visual_survival_checkpoint_eval_poller` before spawning the
training function:
`lightzero_curvyzero_stacked_debug_visual_survival_train.py:9933-9975`.

The poller:

- reloads the Modal volume if available:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6267-6272`;
- scans the attempt-local `exp_name` for checkpoint artifacts:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6274-6282`;
- filters to immutable live eval checkpoint names matching
  `iteration_<n>.pth.tar`, `.pth`, or `.pt`:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6078-6079`;
- waits for a stable size/mtime fingerprint for `stable_polls` polls before
  scheduling:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6291-6306`;
- schedules eval/inspection and optional GIF jobs from the source checkpoint ref:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6317-6333`;
- writes `checkpoint_eval_poller.json` as a status payload:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6089-6097` and
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6225-6265`;
- exits after train is done plus idle timeout, or after max runtime:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6267-6346`.

The poller status write also has no explicit volume commit in the poller loop.
The plumbing test asserts the poller itself does not call `runs_volume.commit`:
`tests/test_curvytron_live_checkpoint_eval_plumbing.py:2266-2437`.

Consequence: eval/GIF freshness depends on source checkpoint visibility to the
poller and successful worker scheduling. If source checkpoints stop advancing,
eval/GIF artifacts stop advancing. If the poller exits by max runtime while the
train heartbeat remains `running`, the row can remain `running` with no new
eval/GIFs.

## Eval and GIF Worker Semantics

The eval/inspection worker reloads the volume, waits up to 30 minutes for the
checkpoint to become visible, runs eval seeds, writes a manifest, writes
inspector reports, and commits the volume:

- checkpoint wait constants: `30 * 60` seconds with 10 second polling:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:437-438`;
- `_wait_for_visible_checkpoint`:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6458-6474`;
- eval/inspection manifest and report write:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6527-6778`;
- explicit commit after eval/inspection artifacts:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6750-6753`.

The GIF worker follows the same visibility pattern, loads the checkpoint, writes
GIF variants and a summary, and commits:

- worker waits/reloads and loads checkpoint:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:7520-7533`;
- summary write:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:7583-7728`;
- explicit commit:
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py:7729-7732`.

This means eval/GIF artifacts are more durable once produced than the poller
status itself, because workers explicitly commit their outputs.

## Status Reader Semantics

`curvytron_run_status` builds one row from independent artifact reads:

- `progress_latest.json` is loaded from the attempt train root:
  `lightzero_curvytron_run_status.py:863-876`.
- checkpoint summary is computed separately:
  `lightzero_curvytron_run_status.py:877`.
- eval manifests, poller status, train heartbeat, action observability, and GIF
  summaries are rolled up separately:
  `lightzero_curvytron_run_status.py:878-899`.

`train_status` and `train_stage` come only from `status_heartbeat.json`:
`lightzero_curvytron_run_status.py:773-806`. There is no age threshold on
`heartbeat_at`.

If progress is missing but the heartbeat exists, the status row reports
`progress_missing_reason: progress_latest_absent_after_train_heartbeat`; the
test confirms the row can simultaneously have:

- `progress_exists is False`;
- `train_status == "running"`;
- `background_poller_status == "running"`;
- GIF artifacts present.

Test reference: `tests/test_curvytron_run_status.py:205-278`.

If progress is unreadable, the row sets
`progress_missing_reason: progress_latest_unreadable` and `event: "unreadable"`:
`lightzero_curvytron_run_status.py:900-917` and
`tests/test_curvytron_run_status.py:280-302`.

Checkpoint mtimes are included in the checkpoint summary:
`lightzero_curvytron_run_status.py:836-859`. The test confirms
`latest_checkpoint_mtime` exists:
`tests/test_curvytron_run_status.py:305-327`.

Eval and GIF rollups sort artifacts by manifest timestamp/path, not by current
training liveness. Eval rollup keeps the latest manifest per checkpoint and then
reports the numerically highest checkpoint label as `latest_eval_checkpoint`:
`lightzero_curvytron_run_status.py:217-235` and
`lightzero_curvytron_run_status.py:561-645`. GIF rollup reports the newest GIF
summary by created timestamp/path:
`lightzero_curvytron_run_status.py:238-259` and
`lightzero_curvytron_run_status.py:697-770`.

## How A Row Can Look Running While Progress Is Stale

Observed row shape:

1. Training starts and writes `status_heartbeat.json` with `status: running`.
2. `train_muzero` enters the long-running training call.
3. No continuous heartbeat updates happen inside that call.
4. `progress_latest.json` advances only when LightZero saves a checkpoint.
5. Eval/GIF workers advance only after the checkpoint poller or hook sees a new
   source checkpoint and successfully schedules workers.
6. If checkpoint saving stops, slows, is not visible, or the poller exits/fails,
   progress/eval/GIF freshness stalls.
7. The status reader still sees the last train heartbeat as `running` because no
   final `completed` or `failed` heartbeat was written.

That is not a contradiction in the code. It is the current artifact model.

## Likely Code-Level Failure Modes

Most likely:

- **Stale train heartbeat:** The stock path does not refresh
  `status_heartbeat.json` during `train_muzero`. Any hang, very slow section, or
  pre-final termination leaves `train_status=running`.
- **Checkpoint cadence or loop slowness:** `progress_latest.json` is
  checkpoint-driven, so long iterations or expensive configs can make it old
  without necessarily proving the process is dead.
- **Checkpoint hook not installed or wrapper exception:** If `BaseLearner` is
  not found as expected, or the wrapper write fails, progress_latest can be
  absent/stale while training continues. The wrapper catches exceptions and
  prints instead of failing training.
- **Canonical checkpoint directory shadows fresher source checkpoints:** The
  status reader checks the run-level mirror before the attempt-local source
  directory. A stale mirror can make `latest_checkpoint` look stale even if
  `attempts/<attempt>/train/lightzero_exp/ckpt` has newer files.
- **Poller max runtime expires:** The poller defaults to 18 hours. If it exits
  but the train heartbeat remains `running`, no later checkpoints get eval/GIF
  work from the poller.
- **Poller visibility lag or missing commits:** The stock train path does not
  explicitly commit progress/checkpoint writes while running. The poller reloads,
  but its own status writes also are not explicitly committed. Worker artifacts
  are committed, so eval/GIF presence may be more reliable than poller status.
- **Spawn failures or partial scheduling:** `_spawn_one_checkpoint_background_eval`
  can record `spawn_failed`; if GIF scheduling succeeds but eval scheduling
  fails, the checkpoint can be marked scheduled/seen from the presence of at
  least one job. That can produce asymmetry between eval and GIF freshness.
- **Checkpoint visibility timeout in workers:** Eval/GIF workers wait up to 30
  minutes for the source checkpoint. If the ref never becomes visible in that
  worker, the worker fails or writes a failure summary rather than producing a
  useful fresh artifact.
- **Direct JSON overwrite partial read:** `progress_latest.json` is written by
  truncating/overwriting the file. The reader treats empty/partial JSON as
  unreadable, so a status row can lose progress even though other artifacts
  exist.
- **No status-reader freshness policy:** The reader includes checkpoint mtimes
  and artifact timestamps, but it does not classify stale rows. A stale row still
  prints `train_status=running` unless some external analysis compares mtimes or
  ages.

## Health Signal Ranking

For live health, the most trustworthy signals are:

1. New immutable checkpoint files by iteration and mtime, preferably from the
   attempt-local source directory as well as the canonical mirror.
2. New eval manifests for those checkpoint labels.
3. New GIF summaries for those checkpoint labels.
4. Poller status counts, with age caveats.
5. `progress_latest.json`, only as a checkpoint/progress convenience file.
6. `train_status=running`, only as "last train status writer reached a running
   stage."

The health-check document's recommendation to use checkpoint mtime and
eval/GIF freshness as primary health criteria matches the code paths inspected
here.
