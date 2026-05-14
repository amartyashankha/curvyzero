# Side Hook Refactor Critique

Date: 2026-05-13

Scope: trusted `--mode train` CurvyZero side hooks and checkpoint scaffolding.
No source refactor is recommended until the tests below exist.

## Priority Order

1. Extract exact resume-sidecar candidate selection.
2. Extract auto-resume checkpoint candidate selection.
3. Extract progress payload construction.
4. Only then consider shared hook patch/restore mechanics.

Do not start with the hook installers. They monkey-patch LightZero/DI-engine
classes and their safety depends on wrapper order, return-value pass-through,
and restore behavior.

## 1. Resume Sidecar Selection

Smallest safe extraction: move only the pure parts of
`_find_lightzero_resume_sidecar` into helper code:

- collect exact `iteration_N.resume_state.pkl` candidates from supplied dirs;
- ignore missing and zero-byte files;
- select by current policy: newest `mtime_ns`, then `size_bytes`, then
  ref/path string;
- leave run/attempt path construction and Modal ref formatting in the trainer.

Tests to add first:

- same-iteration sidecars across current timestamped exp dir, prior attempt,
  and mirror choose the current winner deterministically;
- missing dirs and zero-byte sidecars are reported but not selected;
- output preserves the existing `resume_state_lookup` shape used by
  `_prepare_lightzero_auto_resume`.

Why first: it is narrow, mostly pure, and reduces the most duplication risk
without touching stock LightZero execution.

## 2. Auto-Resume Checkpoint Selection

Next extraction: move only candidate collection/selection from
`_prepare_lightzero_auto_resume`.

Keep in trainer:

- `runs_volume.reload`;
- run root/current attempt/stable mirror path construction;
- Modal refs and final JSON envelope;
- `_set_load_ckpt_before_run` behavior.

Tests to add first:

- current timestamped `lightzero_exp_*` beats fixed `lightzero_exp` when it has
  the higher iteration;
- prior attempts and run checkpoint mirror remain candidates;
- empty checkpoints are ignored;
- selected checkpoint is paired with the exact-iteration sidecar lookup.

Why second: it depends on the sidecar lookup and affects whether fresh train
becomes a resumed train, so its behavior must be pinned before movement.

## 3. Progress Payload Construction

Third extraction: split `_write_checkpoint_progress_latest` into pure payload
building plus the existing trainer-side JSON write.

Tests to add first:

- payload uses latest timestamped checkpoint when present;
- payload falls back to learner `train_iter` when no checkpoint exists;
- `checkpoint_ref` can be absent for non-mounted paths without failing;
- `source` remains distinct for `BaseLearner.save_checkpoint` and
  `SaveCkptHook.__call__`.

Do not extract `_install_checkpoint_progress_writer` yet. The wrapper itself is
hook behavior, not payload logic.

## 4. Hook Patch/Restore Mechanics

Only after steps 1-3 are green, consider a tiny shared patch utility for:

- `_install_live_checkpoint_publisher`;
- `_install_checkpoint_progress_writer`;
- `_install_lightzero_full_resume_state_hooks`;
- `_install_lightzero_target_audit`.

Tests to add first:

- install progress writer and live checkpoint publisher on the same fake
  `BaseLearner.save_checkpoint`;
- assert original save runs exactly once before CurvyZero side effects;
- assert both side effects run and the original return value is preserved;
- restore in the same order used by `_run_visual_survival_train` and assert the
  class method is the original object;
- failure in eval spawn does not prevent progress writing or change the stock
  return value.

Do not move the full resume hook installer before adding resumed-run tests:

- `call_hook("before_run")` returns the original hook result;
- initial eval is skipped exactly once only after a sidecar load;
- `random_collect` is skipped only when replay buffer restoration reports true;
- without replay restoration, `random_collect` delegates to stock LightZero.

## Do Not Touch In This Cut

- `train_muzero` ownership of collector, search, replay, learner, and stock
  checkpoint creation.
- Environment semantics, reward variants, target ranges, or opponent mixture
  behavior.
- Tournament-fed opponent selection or live Modal Dict polling inside the
  trainer.
- Resume claims: sidecars are operational continuity, not exact uninterrupted
  replay equivalence.

## Stock Behavior Guardrail

Every hook must keep this contract:

- call the original LightZero/DI-engine method first where checkpoint saves are
  wrapped;
- return the original result;
- catch and log CurvyZero side-effect failures;
- keep CurvyZero writes outside learner data flow;
- restore patched methods exactly to their original objects.

Recommended next action: add the tests for step 1, then extract only exact
resume-sidecar candidate selection.
