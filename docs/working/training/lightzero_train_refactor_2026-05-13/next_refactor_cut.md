# Next Refactor Cut

Date: 2026-05-13

## Current Green State

Focused validation:

```text
uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q
102 passed, 1 skipped
```

Additional local locks now green:

- `test_stock_train_mode_calls_lightzero_train_muzero_entrypoint`
- `test_resume_sidecar_save_failure_does_not_fail_stock_checkpoint_hook`
- `test_target_audit_hooks_return_original_collect_and_replay_results`
- `test_live_checkpoint_publisher_calls_original_save_before_spawning`
- `test_fresh_resume_hooks_preserve_original_call_hook_eval_and_random_collect`
- `test_lightzero_resume_state_candidates_select_exact_iteration_by_mtime_and_size`
- `test_lightzero_resume_state_candidates_skip_empty_and_missing_dirs`
- `test_auto_resume_considers_prior_attempts_and_run_mirror`
- `test_auto_resume_ignores_empty_checkpoint_and_selects_matching_sidecar`
- `test_checkpoint_progress_payload_prefers_checkpoint_iteration`
- `test_checkpoint_progress_payload_falls_back_to_learner_iteration`
- `test_opponent_assignment_snapshot_requires_exact_schema_id`
- `test_canonical_assignment_json_sha256_is_stable_and_sensitive`

## Best Next Source Cut

Do not add registry behavior directly to the Modal trainer.

Best next code cut is one of these, in this order:

1. Done: pure assignment schema/hash helpers are in `opponent_registry.py`.
   Next, add trainer plumbing tests for consuming a frozen opponent assignment
   snapshot from an explicit ref/hash.
2. Make manifest builders consume assignment data instead of baked
   recent/mid/old refs.
3. Extract exact-iteration resume sidecar candidate lookup if resume code is
   touched again.
4. Extract poller candidate/stability logic after checkpoint/resume helpers are
   stable.

If the next edit is checkpoint/resume cleanup rather than assignment wiring,
follow this safe order:

1. Done: exact resume-sidecar candidate selection is extracted into
   `lightzero_checkpoints.py` and locally covered.
2. Done: auto-resume checkpoint scanning now uses the shared checkpoint
   candidate helper and is covered across current, prior-attempt, mirror, and
   empty-checkpoint cases.
3. Done: checkpoint progress payload construction is split from the JSON write
   and directly covered.
4. Next hook-cleanup step, if needed: add tests for installing progress writer
   and live checkpoint publisher together before extracting shared
   patch/restore mechanics.

Do not move resume semantics, eval skipping, random-collect skipping, or hook
installer behavior until resumed-run behavior has stronger tests.

For assignment wiring, do not add Modal Dict reads or tournament ranking logic
inside the trainer. The first launcher patch should accept one explicit
assignment ref/hash, verify it, convert it into the existing static opponent
mixture, and keep `train_muzero` receiving static config.

## Why

The largest live design risk is adding tournament-fed opponent selection into
the 10k-line trainer. The trainer should consume a frozen assignment snapshot.
Tournament ranking, Modal Dict layout, and refresh cadence should stay outside.

## Immediate Non-Goals

- no live Dict polling inside `train_muzero`;
- no environment redesign;
- no tournament ranking logic in trainer;
- no broad JSON payload extraction just for tidiness.

## Completed From Previous Cut

`src/curvyzero/training/opponent_registry.py` now parses pure assignment
snapshots into the existing opponent-mixture contract. It is not wired into the
trainer yet.
