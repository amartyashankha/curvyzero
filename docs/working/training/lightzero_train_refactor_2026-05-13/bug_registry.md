# Bug Registry

Purpose: keep known training-scaffolding bugs separate from refactor ideas.

## Bug 1: Fixed Checkpoint Path Misses Timestamped LightZero Dirs

Status: fixed in the training/status scaffolding and covered by focused
regression tests.

Plain description: some code looks only in:

```text
train/lightzero_exp/ckpt
```

but LightZero can write to:

```text
train/lightzero_exp_YYMMDD_HHMMSS/ckpt
```

Impact:

- latest checkpoint status can be stale;
- progress display can be stale;
- auto-resume can choose an old checkpoint;
- poller can miss checkpoints and fail to trigger eval/GIF;
- manifests can freeze stale checkpoint refs;
- tournaments can be fed stale refs if the caller bypasses broad discovery.

Candidate affected functions:

- `_latest_lightzero_iteration_checkpoint`
- `_write_checkpoint_progress_latest`
- `_prepare_lightzero_auto_resume`
- `_find_lightzero_resume_sidecar`
- `_scan_lightzero_artifacts`
- `_publish_live_lightzero_checkpoints`
- `_run_checkpoint_eval_poller`
- `lightzero_curvytron_run_status._checkpoint_summary`

Evidence:

- [../curvytron_architecture_research_2026-05-12/checkpoint_discovery_audit_2026-05-13.md](../curvytron_architecture_research_2026-05-12/checkpoint_discovery_audit_2026-05-13.md)
- [../checkpoint_tournament_checkpoint_discovery_handoff_2026-05-13.md](../checkpoint_tournament_checkpoint_discovery_handoff_2026-05-13.md)

Regression test requirement: create fake `lightzero_exp` and
`lightzero_exp_*` dirs and prove every trainer-side reader selects the broad
latest checkpoint.

Implemented regression coverage:

- `tests/test_lightzero_timestamped_checkpoint_discovery.py`
- progress latest selects timestamped checkpoint;
- auto-resume selects timestamped checkpoint;
- resume sidecar scans timestamped state dir;
- checkpoint eval poller scans timestamped checkpoint dirs;
- run-status checkpoint summary scans timestamped dirs;
- direct helper tests for timestamped sibling discovery and iteration-name
  parsing.
- resume-sidecar saving writes beside the timestamped checkpoint instead of
  silently failing because `lightzero_exp/ckpt` is empty.

Implementation note: pure path and parser helpers live in
`src/curvyzero/training/lightzero_checkpoints.py`. The Modal trainer keeps thin
wrappers and uses the helper module. Run status uses the same helper module.
The helper now also owns checkpoint candidate records and latest-selection
ordering.

## Bug 2: Artifact Workers Can See Unloadable Checkpoints

Status: observed in logs; not yet scoped for this refactor.

Plain description: eval/GIF workers sometimes fail with PyTorch checkpoint read
errors after a checkpoint path becomes visible.

Impact:

- GIF/eval artifacts may be missing or failed;
- this does not by itself prove training stopped.

Likely test direction:

- checkpoint stability check should consider size/mtime and possibly a safe
  load check before scheduling expensive readers.

Do not patch until Bug 1 tests and patch are clear.

## Bug 4: Active Scripts May Still Default To Fixed LightZero Exp Paths

Status: follow-up needed; not patched in this refactor slice.

Plain description: some scripts still build refs like:

```text
train/lightzero_exp/ckpt/iteration_N.pth.tar
```

instead of resolving broad `lightzero_exp*` checkpoint dirs.

Known candidates from read-only audit:

- `scripts/lightzero_live_eval_queue.py`
- `scripts/build_curvytron_opponent_mixture_manifest.py`

Impact:

- if these scripts are still active, they can freeze stale or nonexistent
  checkpoint refs;
- if they are legacy, they should be documented or deprecated so they do not
  confuse future agents.

Test direction: make a fake run with only timestamped checkpoints and prove the
script either resolves the broad checkpoint or clearly rejects the missing fixed
ref.

## Bug 5: Top-Level Frozen Opponent Ref May Allow Mutable Checkpoints

Status: fixed and covered by focused tests.

Plain description: mixture entries reject mutable frozen refs like `latest` and
`ckpt_best`, but the direct top-level frozen-opponent path should be audited and
made equally strict before opponent assignment snapshots are introduced.

Impact:

- a run could silently train against a moving opponent identity;
- run metadata would be less reproducible;
- tournament-fed assignment snapshots could be poisoned by mutable refs.

Regression coverage:

- `tests/test_opponent_mixture.py::test_top_level_frozen_opponent_resolution_rejects_mutable_or_non_iteration_ref`

Behavior: top-level `opponent_checkpoint_ref` for
`frozen_lightzero_checkpoint` now rejects `latest.pth.tar`,
`ckpt_best.pth.tar`, and non-`iteration_N.pth.tar` names before path
resolution.

## Bug 3: Volume Commit Failures In Eval/GIF Workers

Status: observed in logs; probably artifact publication, not learner failure.

Plain description: Modal volume commits can fail in background workers.

Impact:

- website artifacts can lag or fail.

This is lower priority than Bug 1 for the training refactor.

## Bug 6: Resume Sidecar Is Not Exact Stock Resume

Status: documented risk; do not claim exact equivalence.

Plain description: stock LightZero checkpoint resume restores the normal
checkpoint payload. CurvyZero adds a sidecar with collector/evaluator metadata,
RNG, and policy extras, but raw replay GameSegment objects are metadata only in
that save/load path.

Impact:

- fresh runs are still close to stock `train_muzero`;
- reused `run_id` runs can continue operationally, but may diverge from a truly
  uninterrupted stock run;
- summaries should describe resumed curves as continued runs, not fresh
  independent runs.

Test direction: if exact resume quality becomes important, compare a short
uninterrupted run against a stop/resume run with the same seed and explicit
checkpoint iteration. Until then, keep the caveat in the parity audit and
source of truth.

## Bug 7: Resume Sidecar Save Failure Could Fail Training After Stock Checkpoint

Status: fixed and covered by focused test.

Plain description: the resume sidecar hook runs after DI-engine's checkpoint
hook. Before the fix, a CurvyZero sidecar write failure could propagate and mark
the training call failed even though the stock checkpoint save had already
completed.

Impact:

- sidecar/artifact failure could shorten or fail a training run;
- this does not mutate learning batches, but it is not passive operational
  scaffolding if it can kill the run.

Regression coverage:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_resume_sidecar_save_failure_does_not_fail_stock_checkpoint_hook`

Behavior: sidecar save failures are logged with
`curvyzero resume sidecar save failed` and the stock checkpoint hook result is
returned unchanged.

## Bug 8: Train Smoke Can Return Before Final Artifacts Are Visible

Status: fixed and validated with a fresh waited CPU Modal smoke.

Plain description: `_run_visual_survival_train(mode="train")` wrote final
summary, action telemetry summary, artifact manifest, checkpoint mirrors, final
attempt state, and final heartbeat, but did not explicitly commit the Modal
Volume at the end of train mode. A waited Modal smoke could therefore return
`ok=true` while external `modal volume ls` still saw only earlier committed
files.

Evidence:

- Fresh smoke `curvytron-cadence-e2e-smoke-20260513-202837/train-smoke-001`
  returned `ok=true`, `called_train_muzero=true`, `problems=[]`, and telemetry
  `row_count=128`.
- Repeated Volume listing showed only `run.json`, `latest_attempt.json`,
  `attempt.json`, and `train/status_heartbeat.json`.
- Downloaded heartbeat still said `stage=before_train_muzero`; downloaded
  latest attempt still said `status=running`.

Impact:

- the training call can succeed, but the website/status tools may not see the
  final summary or mirrored checkpoints;
- this is trainer artifact scaffolding, not evidence that LightZero training or
  cadence stepping failed.

Fix:

- train mode now performs one final `_commit_runs_volume_with_backoff` after
  final artifacts are written and after `train_muzero` has returned.
- this is outside the hot training loop and does not add commits inside
  checkpoint hooks, collector, learner, or background eval triggers.

Regression coverage:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_calls_lightzero_train_muzero_entrypoint`
  now asserts train mode attempts a final commit and that the fake volume commit
  is called once.

Post-fix smoke:

- `curvytron-cadence-e2e-smoke-20260513-203914/train-smoke-001`
  returned `ok=true`, `called_train_muzero=true`, and `problems=[]`.
- Volume readback showed final train artifacts and mirrored checkpoints.
- Downloaded heartbeat/latest-attempt both said `completed`.

## Bug 9: Assignment Ref Was Not Accepted By The Real Checkpoint Poller

Status: fixed and covered by focused tests.

Plain description: the local launcher passed `opponent_assignment_ref` when it
spawned the checkpoint eval/GIF poller, but the real poller function did not
accept that argument and did not forward it into the poller command.

Impact:

- assignment-backed training runs with background eval/GIF could fail at poller
  launch time;
- even if launch was bypassed, eval/GIF could inspect the wrong opponent
  mixture rather than the assignment used by training.

Fix:

- `lightzero_curvytron_visual_survival_checkpoint_eval_poller` now accepts
  `opponent_assignment_ref`;
- it forwards the ref into `_checkpoint_eval_poller_command`, which resolves the
  assignment through the existing opponent-mixture contract.

Regression coverage:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_function_accepts_assignment_ref_and_resolves_command`
- existing command and launcher plumbing tests.

## Bug 10: Leaderboard Publish Could Advance An Unsafe Live Pointer

Status: fixed and covered by focused tests.

Plain description: the publisher could publish a trainer-facing public
leaderboard from a provisional rating snapshot, and it updated the live Dict
pointer before the Volume commit. If the commit failed, the pointer could name a
snapshot that was not durable.

Impact:

- training assignment selection could be based on provisional ratings without
  an explicit opt-in;
- readers could see a live pointer to missing or uncommitted Volume artifacts.

Fix:

- `curvytron_opponent_leaderboard_publish` refuses provisional rating snapshots
  unless `allow_live_provisional=True`;
- it writes snapshot/latest artifacts, commits the Volume, and only then
  updates the live Dict pointer;
- if the commit fails, it returns `pointer_published=false`.

Regression coverage:

- `tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_rejects_provisional_without_opt_in`
- `tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_commits_before_pointer_update`
- `tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_skips_pointer_update_on_commit_error`

Residual caveat: if the Dict write itself fails after a successful Volume
commit, the snapshot is durable but the live pointer is not advanced. Pointer
repair/fallback remains a next task.

## Bug 11: Leaderboard Assignment Selector Could Pick Bad Rows

Status: fixed and covered by focused tests.

Plain description: `allow_provisional=True` allowed any non-active row,
including retired rows, and champion selection assumed rows were already sorted.

Impact:

- a retired row could become a training opponent;
- malformed or API-shaped input order could choose the wrong champion.

Fix:

- provisional selection now allows only `active` and `provisional` rows;
- eligible rows are sorted deterministically before slot selection.

Regression coverage:

- `tests/test_opponent_leaderboard.py::test_select_opponent_assignment_sorts_unsorted_rows_before_selecting_champion`
- `tests/test_opponent_leaderboard.py::test_select_opponent_assignment_allow_provisional_excludes_retired_rows`
