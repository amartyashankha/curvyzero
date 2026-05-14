# Experiment Log

This refactor lane is not a training experiment lane. Use this file for local
validation commands and bug-reproduction notes only.

## 2026-05-13

- Created planning directory.
- Recorded checkpoint-discovery bug as first test-backed target.
- Launched first read-only/doc-only audit wave: test lockdown, architecture
  boundaries, checkpoint bug patch design, dirty-state risk, and local E2E
  contracts.
- Added main-thread trainer surface map.
- After a context transition, the first five audit handles were not reachable
  and no audit docs were present. Relaunched the core audits with new agents:
  Confucius, Aquinas, Turing, and Carver. Archimedes naming/test-cleanup
  critique remains running.
- Archimedes returned
  `naming_and_test_cleanup_critique_2026-05-13.md`. Main takeaway: use plain
  shared terms like `stock LightZero train path` and `lightzero_exp* discovery`;
  do not delete fixed-path tests until replacement coverage is green.
- Carver returned
  `dirty_state_and_e2e_contract_audit_2026-05-13.md`. Main takeaway: the safe
  first patch surface is trainer plumbing tests plus
  `lightzero_curvyzero_stacked_debug_visual_survival_train.py` and
  `lightzero_curvytron_run_status.py`.
- Added five red regression tests in
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py` for timestamped
  `lightzero_exp*` discovery across progress, auto-resume, resume sidecar,
  checkpoint poller, and run-status checkpoint summary.
- Focused red run:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py -q -k 'progress_latest_uses_timestamped or auto_resume_selects_timestamped or resume_sidecar_scans_timestamped or checkpoint_eval_poller_scans_timestamped or run_status_checkpoint_summary_scans_timestamped'`.
  Result: 5 failed for the intended fixed-path bug.
- Moved the new timestamped checkpoint-discovery regressions into a focused
  file:
  `tests/test_lightzero_timestamped_checkpoint_discovery.py`.
- Patched trainer/status discovery in place so broad `lightzero_exp*`
  checkpoint and resume-state dirs are scanned.
- Green focused bugfix runs:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py -q`
  -> 5 passed.
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_lightzero_timestamped_checkpoint_discovery.py -q`
  -> 52 passed, 1 skipped.
  `uv run pytest tests/test_curvytron_run_status.py tests/test_lightzero_timestamped_checkpoint_discovery.py -q`
  -> 11 passed.
- Extracted pure checkpoint path/parsing helpers into
  `src/curvyzero/training/lightzero_checkpoints.py`; trainer/status now import
  that helper module.
- Green focused extraction run:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py -q`
  -> 73 passed, 1 skipped.
- Style/syntax checks after extraction:
  `uv run ruff check tests/test_lightzero_timestamped_checkpoint_discovery.py src/curvyzero/training/lightzero_checkpoints.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`
  -> all checks passed.
  `uv run python -m py_compile tests/test_lightzero_timestamped_checkpoint_discovery.py src/curvyzero/training/lightzero_checkpoints.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`
  -> passed.
- Carson returned a read-only critique recommending the next useful extraction:
  checkpoint candidate records and latest-selection helpers, not generic JSON
  payload cleanup.
- Kant returned a read-only fixed-path audit. Main finding: resume-sidecar
  saving still looked only in `lightzero_exp/ckpt`, even though sidecar lookup
  already scanned timestamped dirs.
- Added helper-level checkpoint candidate tests and timestamped sidecar-save
  regression coverage.
- Fixed `_save_lightzero_resume_sidecar_state` to find the matching checkpoint
  across `lightzero_exp*` dirs and write the sidecar beside the actual
  checkpoint's exp dir.
- Extracted checkpoint candidate collection and latest-selection ordering into
  `src/curvyzero/training/lightzero_checkpoints.py`.
- Green focused run after the sidecar-save fix and candidate extraction:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py -q`
  -> 76 passed, 1 skipped.
- Added opponent assignment snapshot design note:
  `opponent_registry_design.md`.
- Chandrasekhar, Herschel, and McClintock returned read-only opponent-registry
  critiques. Main decisions folded into docs: trainer should consume frozen
  assignment snapshots, hard-coded recent/mid/old refs live mainly in manifest
  builders, and registry code must not bloat the Modal trainer.
- Added and fixed top-level frozen-opponent immutability regression:
  top-level `opponent_checkpoint_ref` now rejects `latest.pth.tar`,
  `ckpt_best.pth.tar`, and non-`iteration_N.pth.tar` names before path
  resolution.
- Green focused run after the opponent-ref guard:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py -q`
  -> 79 passed, 1 skipped.
- Added first pure opponent assignment parser:
  `src/curvyzero/training/opponent_registry.py`.
- Added assignment parser tests:
  `tests/test_opponent_registry.py`.
- Green assignment-focused run:
  `uv run pytest tests/test_opponent_registry.py tests/test_opponent_mixture.py -q`
  -> 22 passed.
- Green widened focused run:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 85 passed, 1 skipped.
- Modal dry check:
  `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode dry --output-detail compact`
  -> ok=true, called_train_muzero=false, no problems.
- Tiny Modal train smoke:
  `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode train --compute cpu --run-id curvytron-refactor-smoke-20260513 --attempt-id train-smoke-001 --max-train-iter 1 --max-env-step 64 --source-max-steps 64 --collector-env-num 1 --n-episode 1 --evaluator-env-num 1 --n-evaluator-episode 1 --num-simulations 1 --batch-size 4 --lightzero-eval-freq 0 --save-ckpt-after-iter 1 --env-telemetry-stride 1 --no-background-eval-enabled --no-background-gif-enabled --wait-for-train --output-detail compact`
  -> ok=true, called_train_muzero=true, no problems.
- Smoke artifact check:
  checkpoint dir contains `iteration_0.pth.tar`, `iteration_1.pth.tar`,
  `iteration_2.pth.tar`, and `ckpt_best.pth.tar`; resume sidecar dir contains
  matching sidecars for iterations 0, 1, and 2.
- Smoke progress/status check:
  `progress_latest.json` reports `iteration_2.pth.tar`; run-status reader
  reports `progress_exists=true`, `train_status=completed`,
  `latest_checkpoint=iteration_2`, `checkpoint_count=6` including attempt
  checkpoints and mirrored run checkpoints.
- Parallel stock-parity/boundary/validation audits returned. Main result:
  fresh `--mode train` still calls stock `train_muzero` and leaves collector,
  search, replay, learner, and stock checkpoint creation in LightZero. Main
  caveats: env/reward semantics are custom by design, auto-resume is not a
  fresh stock run, and resume sidecars are not exact replay-equivalent resume.
- Added `stock_lightzero_parity_audit.md`.
- Added boundary test:
  `test_stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action`.
- Focused boundary-test run:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action -q`
  -> 1 passed.
- Focused local gate after the boundary test:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 86 passed, 1 skipped.
- Style/check gate:
  `uv run ruff check tests/test_curvytron_live_checkpoint_eval_plumbing.py docs/working/training/lightzero_train_refactor_2026-05-13`
  -> all checks passed.
  `git diff --check`
  -> passed.
- First tiny Modal artifact smoke attempt:
  `curvytron-refactor-artifact-smoke-20260513-001`
  failed before remote training with a Modal source snapshot race:
  `vector_runtime.py was modified during build process`. The file had unrelated
  dirty environment changes and its mtime was stable on retry.
- Tiny Modal artifact smoke retry:
  `curvytron-refactor-artifact-smoke-20260513-002`,
  `artifact-smoke-001`, CPU, `--mode train`, `max_train_iter=1`,
  `save_ckpt_after_iter=1`, background eval/GIF enabled.
  Result: training completed, called stock `train_muzero`, and background
  artifacts completed.
- Artifact smoke status reader:
  `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids curvytron-refactor-artifact-smoke-20260513-002 --attempt-ids artifact-smoke-001 --output json`
  -> `train_status=completed`, `latest_checkpoint=iteration_2`,
  `background_poller_status=completed`, `background_poller_seen_count=3`,
  `background_poller_eval_completed_count=3`,
  `background_poller_gif_completed_count=3`, `gif_artifact_count=3`.
  Each GIF summary exposed both `gif_ref`/`raw.gif` and
  `collect_t1_gif_ref`/`collect_t1.gif`.
- Banach audit returned: fresh `--mode train` imports and calls stock
  `lzero.entry.train_muzero`; CurvyZero does not add a custom collector,
  search, replay, or learner loop in that lane.
- Bernoulli audit returned: progress writer, target audit, checkpoint helper,
  and background eval/GIF are passive for fresh runs. Non-passive edges are
  auto-resume, resume-sidecar loading, profile stop caps, and active reward
  target config patches.
- Helmholtz audit returned: strongest remaining local gap was a direct
  stock-entrypoint regression.
- Added and passed:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_calls_lightzero_train_muzero_entrypoint tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_resume_sidecar_save_failure_does_not_fail_stock_checkpoint_hook -q`
  -> 2 passed.
- Widened focused local gate after adding the entrypoint and sidecar-failure
  tests:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 88 passed, 1 skipped.
- Style/check gate after the same source edits:
  `uv run ruff check src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_curvytron_live_checkpoint_eval_plumbing.py docs/working/training/lightzero_train_refactor_2026-05-13`
  -> all checks passed.
  `git diff --check`
  -> passed.
- Tightened timestamped poller coverage so
  `test_checkpoint_eval_poller_scans_timestamped_lightzero_exp_dirs` now proves
  both eval and GIF workers receive the timestamped checkpoint ref.
- Added and passed hook-passivity tests:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_target_audit_hooks_return_original_collect_and_replay_results tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_live_checkpoint_publisher_calls_original_save_before_spawning -q`
  -> 2 passed.
- Added and passed fresh resume hook pass-through test:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_fresh_resume_hooks_preserve_original_call_hook_eval_and_random_collect -q`
  -> 1 passed.
- Widened focused gate after fresh hook-passivity tests:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 91 passed, 1 skipped.
- Style/check gate after fresh hook-passivity tests:
  `uv run ruff check src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_lightzero_timestamped_checkpoint_discovery.py docs/working/training/lightzero_train_refactor_2026-05-13`
  -> all checks passed.
  `git diff --check`
  -> passed.
- Dalton side-hook critique returned. Main decision: do not start by extracting
  hook installers. Extract pure resume-sidecar selection first, then
  auto-resume checkpoint selection, then progress payload construction.
- Euler opponent-assignment critique returned. Main decision: add pure
  assignment schema/hash helpers and trainer plumbing tests before any live
  Modal Dict wiring.
- Added resume-state candidate helpers in
  `src/curvyzero/training/lightzero_checkpoints.py` and wired
  `_find_lightzero_resume_sidecar` to use them while preserving trainer-owned
  path construction and output shape.
- Focused timestamped discovery run after resume-state helper extraction:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py -q`
  -> 12 passed.
- Widened focused gate after resume-state helper extraction:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 93 passed, 1 skipped.
- Style/check gate after resume-state helper extraction:
  `uv run ruff check src/curvyzero/training/lightzero_checkpoints.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_opponent_registry.py docs/working/training/lightzero_train_refactor_2026-05-13`
  -> all checks passed.
  `git diff --check`
  -> passed.
- Added auto-resume tests proving prior attempts and run checkpoint mirrors are
  candidates, empty checkpoints are ignored, and the selected checkpoint is
  paired with the matching exact-iteration resume sidecar.
- Rewired `_prepare_lightzero_auto_resume` to use
  `collect_lightzero_iteration_checkpoints` instead of hand-parsing checkpoint
  filenames inside the trainer.
- Focused timestamped discovery run after auto-resume helper use:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py -q`
  -> 14 passed.
- Widened focused gate after auto-resume helper use:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 95 passed, 1 skipped.
- Style/check gate after auto-resume helper use:
  `uv run ruff check src/curvyzero/training/lightzero_checkpoints.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_opponent_registry.py docs/working/training/lightzero_train_refactor_2026-05-13`
  -> all checks passed.
  `git diff --check`
  -> passed.
- Split checkpoint progress payload construction from JSON writing and added
  direct payload tests for checkpoint-preferred iteration and learner-iteration
  fallback.
- Focused timestamped discovery run after progress-payload split:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py -q`
  -> 16 passed.
- Widened focused gate after progress-payload split:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 97 passed, 1 skipped.
- Style/check gate after progress-payload split:
  `uv run ruff check src/curvyzero/training/lightzero_checkpoints.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_opponent_registry.py docs/working/training/lightzero_train_refactor_2026-05-13`
  -> all checks passed.
  `git diff --check`
  -> passed.
- Tightened opponent assignment snapshots so `schema_id` must exactly equal
  `curvyzero_opponent_assignment/v0`.
- Fresh post-cadence waited CPU Modal smoke:
  `curvytron-cadence-e2e-smoke-20260513-202837`, `train-smoke-001`, CPU,
  `--mode train`, `max_train_iter=1`, `max_env_step=64`,
  `source_max_steps=64`, `save_ckpt_after_iter=1`, background eval/GIF off.
  Result returned `ok=true`, `called_train_muzero=true`, `problems=[]`, and
  telemetry `row_count=128`.
- Artifact visibility check for that smoke found a trainer-scaffolding bug:
  repeated Volume listing showed only early files and downloaded
  `status_heartbeat.json` still said `stage=before_train_muzero`, while
  `latest_attempt.json` still said `status=running`. Root cause: train mode did
  not explicitly commit the Volume after final summary/checkpoint artifacts were
  written.
- Patched train mode to perform one final `_commit_runs_volume_with_backoff`
  after `train_muzero` has returned and after final artifacts are written. This
  is outside the hot training loop.
- Focused regression after final-commit patch:
  `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_calls_lightzero_train_muzero_entrypoint -q`
  -> 1 passed.
- Added `canonical_assignment_json_sha256` for future explicit assignment
  ref/hash verification.
- Pure opponent registry run:
  `uv run pytest tests/test_opponent_registry.py -q`
  -> 11 passed.
- Widened focused gate after assignment schema/hash helpers:
  `uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q`
  -> 102 passed, 1 skipped.
- Style/check gate after assignment schema/hash helpers:
  `uv run ruff check src/curvyzero/training/lightzero_checkpoints.py src/curvyzero/training/opponent_registry.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_opponent_registry.py docs/working/training/lightzero_train_refactor_2026-05-13`
  -> all checks passed.
  `git diff --check`
  -> passed.
