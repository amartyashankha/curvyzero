# Test And Migration Plan

Last updated: 2026-05-19

## Why This Exists

This refactor touches trainer setup code that is easy to break silently. Tests should be focused on contracts, not broad slow coverage.

## Test Strategy

- Unit-test extracted pure functions first.
- Smoke-test config construction without Modal.
- Keep env reset/step tests small and deterministic.
- Add regression tests for known footguns: hidden defaults, support bounds, observation perspective, opponent immortality flag handling, and invalid slot mixtures.
- Only run heavier end-to-end tests after pure extraction tests pass.

## Migration Strategy

1. Add tests around current behavior.
2. Extract one boundary.
3. Make the old call site delegate to the new module.
4. Run focused tests.
5. Remove stale fallback code in the touched path.
6. Record the result in `FINDINGS_LOG.md` and update `CURRENT_PHASE.md`.

## Test Boundary Migration

- Public-module tests come first.
- Trainer-private helper tests stay only as temporary parity during extraction.
- After the trainer delegates to the public module and integration smoke passes, shrink or delete the private-helper tests.
- The trainer integration suite should test orchestration seams, not pure reward/config/opponent/checkpoint logic.

## Reward Gate Status

Current passing local commands:

```bash
uv run pytest -q -p no:cacheprovider tests/test_reward_contracts.py
uv run pytest -q -p no:cacheprovider tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_reset_shape_and_metadata tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_step_and_terminal_telemetry tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_survival_plus_bonus_no_outcome_rewards_same_step_bonus tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_survival_plus_bonus_no_outcome_excludes_terminal_outcome tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_survival_plus_bonus_plus_outcome_scales_terminal_loss tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_registered_source_state_visual_survival_env_reuses_local_semantics
uv run pytest -q -p no:cacheprovider tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_calls_lightzero_train_muzero_entrypoint tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action
```

What this protects:

- shared reward normalization, alpha validation, support caps, `td_steps`, and reward-space bounds;
- source-state env reward metadata and reward behavior for bonus/outcome variants;
- trainer config plumbing for stock LightZero and source-state opponent mixture.

## Config Builder Gate Status

Current passing local command:

```bash
uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py
```

Result after compact experiment-surface cleanup: 25 passed.

What this protects:

- public target-support patching;
- JSON/plain normalization used by config audit surfaces;
- nested config path creation and checkpoint hook patch reporting;
- public same-signature visual-survival builder construction without importing Modal;
- typed `VisualSurvivalConfigSpec -> build_visual_survival_config(...) -> VisualSurvivalConfigResult` parity with the broad facade;
- signature drift between the broad facade and typed spec conversion;
- grouped placement of normalized run/training/timing/observation/behavior/reward/opponent knobs;
- unknown broad kwargs are rejected instead of silently surviving;
- compact `VisualSurvivalExperimentSpec` exposes only deliberate experiment fields, expands to current broad defaults, rejects unknown scale presets, and rejects internal launch knobs such as `batch_size`, render modes, backend, support caps, and source cadence;
- trainer `_build_visual_survival_configs(...)` facade matches the public builder across fixed, mixture, frozen, learner-seat, and observation-backend cases;
- policy-observation perspective contract is present in the builder surface;
- frozen opponent checkpoint/snapshot/state/use-cuda fields in env config and surface;
- eval does not re-import trainer-private `_build_visual_survival_configs`;
- `lightzero_config_builder.py` does not import Modal modules;
- visual-survival surface fields for learner seat, policy-observation backend, natural bonus spawn, and opponent assignment context;
- eval does not re-import trainer-private `_target_config_patches`.

Remaining after the typed builder cleanup:

- move protected trainer/eval/test callers from the broad facade to `VisualSurvivalConfigSpec` directly where doing so reduces code rather than churn;
- add a real eval builder smoke that does not monkeypatch the builder if eval gets touched again.

## Grouped Submit Gate Status

Current passing local command:

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvytron_shared_contracts.py tests/test_curvytron_survivaldiag_submitter.py tests/test_curvytron_tonight18_manifest.py::test_grouped_submit_accepts_compact_train_kwargs_with_current_defaults
```

Result: 19 passed.

What this protects:

- grouped submit's required training payload is only `mode`, `seed`, `run_id`, and `attempt_id`;
- missing policy-surface render fields use current defaults instead of making old flat defaults mandatory;
- minimal compact train kwargs are normalized so the poller receives the correct run identity before spawn;
- optional compact `experiment_spec` rows expand current reward/noise/current-scale fields before validation;
- train-only checkpoint fields are still rejected from poller kwargs;
- stale train-only/policy/config fields are rejected from compact poller kwargs;
- mutable initial checkpoint refs are still rejected.

Current `tonight18` compact migration gate:

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvytron_tonight18_manifest.py tests/test_curvytron_survivaldiag_submitter.py
```

Result: 32 passed.

What this additionally protects:

- `tonight18` emits compact-by-default `train_kwargs` but expands them locally for its own manifest validator;
- default-equal trainer fields are omitted from raw `tonight18` rows;
- explicit CLI overrides remain explicit in raw `tonight18` rows;
- assignment refs, refresh pointers, initial checkpoint refs, and non-default background eval/GIF fields stay launch-visible;
- compact submitter rows fail on mixed schemas, identity divergence, runtime/top-level ref conflicts, and action-noise bundle overrides.

Current broader compact/manifest audit gate:

```bash
uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py tests/test_curvytron_shared_contracts.py tests/test_curvytron_survivaldiag_submitter.py tests/test_curvytron_tonight18_manifest.py tests/test_curvytron_survivaldiag_manifest.py tests/test_curvytron_opponent_mixture_manifest.py
```

Result: 87 passed.

Current manifest-ref/feedback-lineage nearby gate:

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvytron_launch_manifest_ref_audit.py tests/test_curvytron_next_batch_manifest.py tests/test_feedback_loop_lineage.py tests/test_curvytron_training_candidate_controller_local.py
```

Result: 30 passed.

What this additionally protects:

- launch-manifest ref audit paths still see the refs they need after compacting default-equal train kwargs;
- next-batch manifest and feedback-lineage tests still pass;
- the local training-candidate controller fake now round-trips the required `opponent_split_*` reset metadata, matching the current trainer readiness proof.

Current real-script compact dry-run:

```bash
uv run python scripts/build_curvytron_tonight18_manifest.py --scratch-bootstrap --output-root /private/tmp/curvy_tonight18_compact_e2e --matrix-name curvy-compact-e2e-test
uv run python scripts/submit_curvytron_survivaldiag_manifest.py /private/tmp/curvy_tonight18_compact_e2e/curvy-compact-e2e-test/curvy-compact-e2e-test.json --output /private/tmp/curvy_tonight18_compact_e2e/curvy-compact-e2e-test/submission.dry_run.json
```

Result: builder wrote 18 rows; submitter dry-run accepted 18 selected rows, 3 assignment records, and 3 refresh-pointer records.

## Opponent Split Gate Status

Current passing local command:

```bash
uv run pytest -q -p no:cacheprovider tests/test_opponent_mixture.py tests/test_opponent_registry.py tests/test_opponent_leaderboard.py tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_opponent_mixture_selects_once_per_reset_not_per_step tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_opponent_mixture_refresh_applies_on_reset_and_records_assignment_context tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_context_is_passed_to_env_config tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_refresh_reset_param_uses_exact_collector_slot_split
```

Result: 63 passed.

What this protects:

- opponent mixture parsing and singleton split entries preserve refs/runtime flags;
- assignment context reaches env config and reset info;
- deterministic collector-env split uses exact slot metadata;
- ready-report proof fails if expected split metadata is absent or wrong.

## Compact Local Spine

Current passing local command:

```bash
uv run pytest -q -p no:cacheprovider tests/test_lightzero_config_builder.py tests/test_reward_contracts.py tests/test_opponent_mixture.py::test_singleton_mixture_preserves_entry_refs_and_only_reweights tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_refresh_ready_report_requires_all_envs tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_real_builder_config_reaches_fake_lightzero_entrypoint
```

Result: 35 passed.

Current no-Modal eval/trainer spine:

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvytron_live_checkpoint_eval_plumbing.py -k 'stock_train_mode_calls_lightzero_train_muzero_entrypoint or stock_train_mode_real_builder_config_reaches_fake_lightzero_entrypoint or stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action or survival_plus_bonus_no_outcome_uses_capped_separate_supports or survival_plus_bonus_plus_outcome_alpha_threads_to_env_policy_and_supports or fixed_opponent_target_support_and_td_steps_can_be_overridden or eval_episode_and_tables_preserve_reward_components or background_eval_inspection_and_gif_can_be_explicitly_enabled or test_live_checkpoint_trigger_spawns_eval_and_selfplay_gif_without_volume_commit or checkpoint_eval_poller_completes_eval_inspection_and_selfplay_gif_jobs or eval_infers_model_support_from_checkpoint_head_shapes or make_policy_and_env_applies_checkpoint_inferred_support_with_public_patches'
```

Result: 12 passed, 93 deselected.

Runtime reward/opponent reset slice:

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -k 'survival_plus_bonus_no_outcome or survival_plus_bonus_plus_outcome or opponent_mixture_refresh_applies_on_reset_and_records_assignment_context'
```

Result: 8 passed, 41 deselected.

## Remaining Gates To Build

- Pure reward/support contract tests: done for the active first cut.
  - reward variant normalization;
  - invalid alpha;
  - invalid support cap;
  - invalid `td_steps`;
  - support range for each active reward variant.
- Config-builder parity tests: active focused coverage now includes broad/typed parity, mixture, assignment context, frozen checkpoint fields, learner-seat mode, reward support, and public import boundaries. Add proactive hard-coded opponent only when that path is touched.
- Hook installation smoke test:
  - install all intended hooks together;
  - preserve original methods;
  - restore or avoid duplicate wrapping;
  - no Modal remote calls in local tests.
- Env helper parity test:
  - fixed seed;
  - short fixed action sequence;
  - same observation tensor, reward, done, and key info fields before/after extraction.
- Batch slot validation tests:
  - deterministic slot total is compatible with `collector_env_num`;
  - generated collector assignment reaches env reset;
  - incompatible split fails at config-build time.

## Existing Fast Commands

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_opponent_mixture.py tests/test_lightzero_checkpoint_opponent_provider.py tests/test_source_state_visual_survival_learner_seat_regression.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -k 'survival_plus_bonus or opponent_immortal or opponent_mixture or player_perspective or blank_canvas_noop'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_curvytron_live_checkpoint_eval_plumbing.py -k 'survival_plus_bonus or target_support or modal_config or checkpoint_progress_writer or save_ckpt_hook or live_checkpoint_publisher or fresh_resume_hooks or opponent_assignment_refresh or stock_source_state_mixture_config'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_trainer_contract.py -k 'reward or perspective or terminal'
```

## Staged Migration Order

1. Reward extraction.
2. Opponent assignment vocabulary extraction.
3. Config-builder extraction and typed spec/result boundary.
4. Checkpoint metadata payload extraction.
5. Status/progress payload extraction.
6. Env helper extraction where it naturally follows reward/telemetry work.
7. Hook-bundle extraction.
8. Local smoke with registered env and mocked `train_muzero`.

Current no-Modal real-builder smoke:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_real_builder_config_reaches_fake_lightzero_entrypoint`
- It uses the real `_build_visual_survival_configs(...)`, fakes only stock `train_muzero`, and asserts the fake receives config that can instantiate/step the registered source-state env.

## Done Criteria

- Each extraction has a named fast test command.
- No test asserts meaningless implementation details.
- Tests catch the behavior we actually care about before a full run is launched.
