# Stale Checkpoint Theory Critique - Agent - 2026-05-13

Scope: read-only critique of the current stale-checkpoint theory. I inspected
the local investigation notes, status artifacts, and relevant CurvyZero code
paths. No source code was changed.

Current theory under critique:

> `progress_latest` can update every `SaveCkptHook` call even when no new
> checkpoint file exists; several rows show high `learner_train_iter` but only
> `iteration_0` checkpoint; Modal preemption exists but is not sufficient
> because healthy rows also preempt.

## Short Read

The theory is plausible and now has real evidence, but it is still too broad.
The strongest fact is narrower:

Fresh `progress_latest.json` rows with `source="SaveCkptHook.__call__"` prove
the DI-engine save hook wrapper returned and CurvyZero wrote progress. They do
not prove DI-engine attempted a checkpoint save, wrote to the expected path, or
used the same iteration counter as `learner_train_iter`.

The stale rows with high `learner_train_iter` and only `iteration_0` are not
slow-start rows. They have crossed the configured visible checkpoint cadence by
multiple factors. But the missing discriminator is still: did DI-engine skip
saving because its own condition was false, save somewhere else or under another
name, fail silently in a lower write path, or restart/resume in a way that keeps
training from satisfying the expected save condition?

Preemption remains a likely cofactor, not an explanation. Healthy rows can
preempt and recover, so the useful question is not "was there preemption?" It
is "what changed in hook state, `last_iter`, `rank`, `ckpt_name`, exp path, or
visible volume files after the preemption?"

## Facts To Keep

- `train_status=running` is not a liveness signal. It comes from
  `status_heartbeat.json`, which is written before `train_muzero`, after
  auto-resume checking, and finally only after `train_muzero` returns. There is
  no continuous train heartbeat or expiry.
- The status snapshots show a real stale tail: `status_chunks_20260513e` has
  212 rows reporting `running`, 55 unchanged rows across the prior snapshot
  window, 36 rows older than 3 hours, and 10 rows older than 6 hours.
- `progress_latest.json` has mixed semantics. In the stock path,
  `_write_checkpoint_progress_latest` records `learner_train_iter` from
  `learner.train_iter`, but records `iteration` and `checkpoint_name` from the
  highest visible `iteration_*.pth.tar` under `exp_name/ckpt`.
- The sampled k0 rows are not merely waiting for the first cadence boundary.
  Local raw progress files in `status_chunks_20260513f` show:

| Run | `learner_train_iter` | Visible checkpoint | Source |
| --- | ---: | --- | --- |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | 175528 | `iteration_0.pth.tar` | `SaveCkptHook.__call__` |
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | 33998 | `iteration_0.pth.tar` | `SaveCkptHook.__call__` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | 65588 | `iteration_0.pth.tar` | `SaveCkptHook.__call__` |
| `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | 97749 | `iteration_0.pth.tar` | `SaveCkptHook.__call__` |
| `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011` | 100411 | `iteration_0.pth.tar` | `SaveCkptHook.__call__` |

- Cadence is not generally broken. Fresh mix2 canaries reached `iteration_10000`
  in about 21-23 minutes, and older k15000 survival rows reached first visible
  checkpoints in about 26-36 minutes. A row at k0 after 30 minutes is ambiguous;
  a row at k0 with `learner_train_iter=175528` is not.
- Healthy comparators use the same broad Modal app, GPU function family,
  progress source, and cadence, but their `iteration_*.pth.tar` files advance
  normally.

## Holes In The Current Theory

### 1. "Every SaveCkptHook call" needs a stricter definition

The local wrapper around `SaveCkptHook.__call__` writes progress after the
original hook returns. It does not inspect whether the original hook saved a
file. That supports the theory.

But "SaveCkptHook call" is not the same as "checkpoint-save attempt." DI-engine
gates actual saves. The investigation notes cite upstream behavior:

- save only when `engine.rank == 0`;
- save only when `engine.last_iter.val % freq == 0`;
- use `engine.ckpt_name` when set, otherwise `iteration_<engine.last_iter.val>.pth.tar`;
- log `learner save ckpt in ...` on actual save.

The stale progress payload uses `learner.train_iter`. The save gate may use
`engine.last_iter.val`. Until those counters are compared on the same rows, a
high `learner_train_iter` does not prove the DI-engine modulo condition fired.

### 2. `progress_latest` proves freshness, not durability

For stale rows, `progress_latest.timestamp` being fresh proves the wrapper is
alive enough to write JSON. It does not prove:

- a new `iteration_*.pth.tar` file was created;
- an attempted save reached the filesystem;
- a Modal volume commit made the file visible to other functions;
- status is reading the same directory the save path used;
- replay, collector, or model state is healthy.

Also, source matters. A payload from `BaseLearner.save_checkpoint` would be
stronger evidence of an actual save path returning, because that wrapper sits
around the learner save method. The sampled stale evidence is specifically
`source="SaveCkptHook.__call__"`, which is weaker.

### 3. Status UI can hide two different problems under one row

The UI/status row merges independent signals:

- heartbeat status from `status_heartbeat.json`;
- progress from `progress_latest.json`;
- checkpoint list from `_checkpoint_summary`;
- eval/GIF artifacts from poller and worker outputs.

Those signals can disagree. The status reader also checks canonical
`checkpoints/lightzero` before attempt-local `train/lightzero_exp/ckpt`. If a
stale canonical directory exists, a running attempt could have fresher
attempt-local checkpoints that status does not show. Existing sampled stale k0
rows reportedly had no canonical directory, so that is not their explanation,
but it remains a UI/status hazard for other rows.

### 4. Modal logs are weak negative evidence

Sparse train-call logs with only the transformer warning do not prove absence
of checkpoint errors. They only prove the queried retained streams did not show
an obvious Python exception. The log notes already show:

- `get_call_graph()` returned `InputStatus.PENDING` for both stale and healthy
  train calls;
- app-level logs had broad preemption;
- aggregate tracebacks mostly involved eval/GIF workers and were not tied to
  selected stale train FunctionCalls;
- train stdout/stderr is sparse for healthy rows too.

So "no error in Modal logs" should not reduce probability much. The stronger
log observation would be presence or absence of DI-engine learner logger lines
like `learner save ckpt in ...` around expected boundaries for the exact row.

### 5. Checkpoint cadence is a per-row threshold, not a wall-clock promise

The cadence facts refute simple slow-start for high-iter k0 rows, but they do
not make a universal wall-clock SLA. Rows with low `learner_train_iter` can
still be legitimately pre-cadence, slow, queued, or recovering. Keep two
cohorts separate:

- high-iter stale rows: checkpoint save/publication/naming/gating problem;
- low-iter stale rows: startup, collection, compute, queueing, or crash remains
  possible.

### 6. Preemption is necessary to inspect, not sufficient to blame

Healthy rows also preempt and later checkpoint. Therefore preemption should be
treated as a trigger that may expose a state bug, not as the bug itself.

Useful preemption hypotheses are more specific:

- after restart, auto-resume restores `learner.train_iter` but not
  `last_iter.val`;
- after restart, rank/worker state changes so `engine.rank != 0`;
- after restart, `ckpt_name` is stuck on a mutable/best name;
- after restart, the exp path or working directory changes;
- after restart, replay/collector state loss alters hook cadence;
- checkpoint file write completes locally but is not visible through the Modal
  volume path the poller/status reads.

The current docs do not yet distinguish these.

## Missing Checks

1. For high-iter stale rows, capture DI-engine's actual save gate state:
   `learner.train_iter`, `engine.last_iter.val`, `engine.rank`,
   `SaveCkptHook._freq`, `engine.ckpt_name`, and `engine.exp_name` at hook time.
2. Search exact learner logs, not only Modal app logs, for
   `learner save ckpt in` near expected boundaries. Compare stale rows to a
   healthy row with the same cadence.
3. Confirm actual configured cadence from each row's
   `formatted_total_config.py` or config artifact, not only from manifest intent.
4. For each stale sample, list both attempt-local `train/lightzero_exp/ckpt`
   and canonical `checkpoints/lightzero`, including non-iteration names, mtimes,
   sizes, and any partial/temp files.
5. Compare `progress_latest.timestamp`, file mtime, and checkpoint directory
   mtime. JSON freshness without directory mtime movement is the key signature.
6. Compare preemption timelines to checkpoint timelines: last checkpoint before
   preemption, first hook progress after restart, first successful checkpoint
   after restart if any.
7. Verify whether the progress writer's JSON write is visible without an
   explicit volume commit because it is small/metadata-only, while large
   checkpoint writes might require different visibility behavior.
8. Check whether all stale rows have `source="SaveCkptHook.__call__"`. Any stale
   row with `source="BaseLearner.save_checkpoint"` would shift the diagnosis.
9. Split stale rows into `high learner iter`, `low learner iter`,
   `progress absent`, and `progress stale` before applying one theory.
10. Inspect whether final summaries ever appear for stale rows. A row can remain
    `running` if `train_muzero` never reaches the final status writer, but a
    returned summary with failed mirror/copy problems would be a different case.

## Ranked Next Observations

1. Add or retrieve hook-gate telemetry for one stale k0 row and one healthy
   comparator: `train_iter`, `last_iter.val`, `rank`, `freq`, `ckpt_name`,
   `exp_name`, and whether the original hook called `save_file`.
2. Pull exact learner logger lines for the same pair and count actual
   `learner save ckpt in ...` messages by iteration. This is the cleanest
   no-code read if the logs exist.
3. Recheck one high-iter stale k0 row's attempt-local and canonical checkpoint
   directories at the same time as its progress file. Include non-iteration and
   partial files.
4. Build a stale-only matrix with `learner_train_iter - latest_checkpoint_iter`
   divided by `save_ckpt_after_iter`. Rank rows by missed checkpoint multiples,
   not just checkpoint age.
5. For preempted stale and preempted healthy rows, align container IDs,
   preemption timestamps, auto-resume selected checkpoint, and next successful
   or missing save boundary.
6. Inspect rows where stale status might be a canonical-first artifact. If
   canonical exists and attempt-local is fresher, status UI is partly wrong even
   if training is fine.
7. Confirm all row configs actually have `save_ckpt_after_iter=10000` or 15000
   in the formatted runtime config.
8. Sample low-iter stale rows separately. Do not force the high-iter
   checkpoint-publication theory onto rows that never crossed a save boundary.
9. Check Modal volume visibility behavior with an existing live row: can a fresh
   `progress_latest.json` appear while a newly written large checkpoint is
   hidden until commit or reload?
10. If a new diagnostic launch is allowed later, make a tiny run log
    SaveCkptHook gate state every call and explicitly log when no checkpoint
    file matching `last_iter` exists after the hook returns.

## Current Best Critique

The theory has the right center of gravity but should be worded more carefully:

> For sampled high-iter stale rows, CurvyZero's `SaveCkptHook.__call__` wrapper
> is refreshing `progress_latest.json` while the visible checkpoint directory
> remains stuck at an old `iteration_*.pth.tar`. This proves hook-level progress
> freshness can diverge from checkpoint durability. It does not yet prove
> DI-engine attempted and failed a save at each expected cadence boundary.

The next decisive observation is the DI-engine save gate state on stale versus
healthy rows. Without that, the current theory risks blaming Modal volume or
preemption too early when the immediate cause could be counter divergence,
rank gating, `ckpt_name`, exp path, or another hook-state mismatch after resume.
