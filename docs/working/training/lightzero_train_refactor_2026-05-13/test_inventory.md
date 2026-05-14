# Test Inventory

Purpose: track existing tests, planned regression tests, and deletion candidates.

## Existing Tests To Preserve Unless Audit Says Otherwise

| File | Useful coverage |
| --- | --- |
| `tests/test_curvytron_live_checkpoint_eval_plumbing.py` | checkpoint progress writer, save hook, eval/GIF scheduling, launcher config wiring, registered-env boundary smoke, GIF marker behavior |
| `tests/test_lightzero_timestamped_checkpoint_discovery.py` | broad `lightzero_exp*` checkpoint/resume discovery across progress, resume, poller, status, and helper parsing |
| `tests/test_opponent_mixture.py` | opponent mixture parsing, immutable checkpoint refs, eval/GIF mixture metadata |
| `tests/test_opponent_registry.py` | opponent assignment snapshot parsing and immutable frozen-ref guardrails |
| `tests/test_opponent_leaderboard.py` | public leaderboard snapshot/pointer contracts and assignment slot selection |
| `tests/test_lightzero_checkpoint_opponent_provider.py` | checkpoint opponent provider support-head inference |
| `tests/test_curvytron_checkpoint_tournament.py` | tournament checkpoint discovery and artifact contracts |
| `tests/test_multiplayer_source_state_trainer_surface.py` | environment/trainer surface contract, mostly out of scope |

## Implemented Bug 1 Regression Tests

Current focused file:

- `test_progress_latest_uses_timestamped_lightzero_exp_checkpoint`
- `test_auto_resume_selects_timestamped_lightzero_exp_checkpoint`
- `test_resume_sidecar_scans_timestamped_lightzero_exp_state_dir`
- `test_checkpoint_eval_poller_scans_timestamped_lightzero_exp_dirs`
- `test_run_status_checkpoint_summary_scans_timestamped_lightzero_exp_dirs`
- `test_lightzero_checkpoint_helpers_list_timestamped_siblings`
- `test_lightzero_checkpoint_name_parsers_reject_invalid_names`

## Implemented Opponent Ref Guard Tests

- `test_modal_mixture_resolution_rejects_non_iteration_checkpoint_ref`
- `test_modal_mixture_resolution_rejects_mutable_frozen_checkpoint_ref`
- `test_top_level_frozen_opponent_resolution_rejects_mutable_or_non_iteration_ref`

## Implemented Opponent Assignment Tests

- `test_opponent_assignment_snapshot_parses_to_existing_mixture_contract`
- `test_opponent_assignment_snapshot_accepts_json_string`
- `test_opponent_assignment_snapshot_rejects_mutable_or_non_iteration_refs`
- `test_opponent_assignment_snapshot_requires_traceable_assignment_id`
- `test_opponent_assignment_ref_resolves_to_existing_mixture_contract`
- `test_opponent_assignment_artifact_writer_stores_assignment_and_audit`
- `test_checkpoint_eval_poller_command_resolves_assignment_ref`
- `test_checkpoint_eval_poller_function_accepts_assignment_ref_and_resolves_command`

## Implemented Leaderboard Selector Tests

- public leaderboard snapshot marks active/provisional rows;
- public leaderboard snapshot rejects mutable checkpoint refs;
- live pointer payload round-trips compact summary;
- assignment selector outputs parser-compatible assignment/audit;
- assignment selector is deterministic for the same snapshot;
- assignment selector sorts unsorted rows before selecting champion;
- assignment selector excludes retired rows even when provisional rows are allowed.

## Implemented Stock Boundary Tests

- `test_stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action`
  builds the trainer config, instantiates the registered LightZero-facing env,
  and proves one scalar ego action steps a mixed blank-canvas opponent without
  moving opponent logic into the trainer.
- `test_stock_train_mode_calls_lightzero_train_muzero_entrypoint` fakes
  `lzero.entry.train_muzero` and proves fresh local `--mode train` calls that
  entrypoint, records `called_train_muzero`, and stays on the fixed-opponent
  stock lane rather than the historical two-seat path.
- `test_resume_sidecar_save_failure_does_not_fail_stock_checkpoint_hook`
  proves a CurvyZero sidecar write failure is logged and does not fail the
  already-completed stock checkpoint hook.
- `test_target_audit_hooks_return_original_collect_and_replay_results` proves
  target-audit wrappers return original LightZero collect/replay results and
  restore patched methods.
- `test_live_checkpoint_publisher_calls_original_save_before_spawning` proves
  live checkpoint publication calls the stock checkpoint save before spawning
  background artifact work.
- `test_fresh_resume_hooks_preserve_original_call_hook_eval_and_random_collect`
  proves fresh runs still return the original LightZero `call_hook`, `eval`,
  and `random_collect` results when resume is inactive.

## Latest Focused Gates

- Local focused gate after assignment schema/hash helpers:
  `tests/test_lightzero_timestamped_checkpoint_discovery.py`,
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py`,
  `tests/test_curvytron_run_status.py`, `tests/test_opponent_mixture.py`, and
  `tests/test_opponent_registry.py` -> 102 passed, 1 skipped.
- Single focused hook-passivity check:
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_fresh_resume_hooks_preserve_original_call_hook_eval_and_random_collect`
  -> 1 passed.
- Timestamped checkpoint/resume helper focused check:
  `tests/test_lightzero_timestamped_checkpoint_discovery.py` -> 16 passed.
- Pure opponent registry focused check:
  `tests/test_opponent_registry.py` -> 11 passed.
- Tiny Modal CPU artifact smoke:
  `curvytron-refactor-artifact-smoke-20260513-002` completed training,
  checkpoint status, background inspection, and background GIF generation.

## Planned Regression Tests

Still planned:

- `test_manifest_checkpoint_selection_uses_broad_discovery`
- manifest builder does not embed baked historical recent/mid/old checkpoint
  refs.
- pointer repair/fallback command tests;
- safe assignment refresh boundary tests;
- online Elo continuation and queue/dedupe repair tests;
- larger bounded closed-loop smoke.

## Deletion Candidates

None yet. Do not delete tests until replacement coverage exists and a returned
audit identifies stale tests precisely.
