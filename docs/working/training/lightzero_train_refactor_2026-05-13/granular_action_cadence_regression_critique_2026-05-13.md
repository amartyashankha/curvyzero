# Granular Action Cadence Regression Critique

Date: 2026-05-13

Scope: trusted stock LightZero `train_muzero` lane, centered on `CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv`, `_build_visual_survival_configs`, background checkpoint eval, and checkpoint GIF plumbing.

Status after patch: the main recommendations from this critique have been
implemented for the trusted `source_state_fixed_opponent` train lane. The
explicit `decision_ms` foot-gun is now blocked in trusted train/dry config, and
the stale audit doc is marked as historical.

## Findings

1. Default cadence is now pinned in two useful places, but not yet in telemetry rows.
   `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_visual_survival_default_is_one_source_frame_per_policy_action` asserts `decision_source_frames == 1`, `decision_ms == SOURCE_PHYSICS_STEP_MS`, one repeat, and one source substep. `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_modal_config_defaults_to_one_source_frame_per_policy_action` asserts `train_mod.DEFAULT_DECISION_MS == SOURCE_PHYSICS_STEP_MS` and no default repeat. Add the same cadence assertions to `test_source_state_visual_survival_step_and_terminal_telemetry`: after reading `rows[0]`, assert `decision_ms`, `decision_source_frames`, `max_ticks`, `max_source_ticks`, `physical_decision_ms_total`, and all `policy_action_repeat_*` fields. The telemetry writer already emits repeat fields and physical total, but the test currently never checks cadence in the persisted JSONL row.

2. `source_max_steps`/`max_ticks` needs a cap-semantics regression test, not only config presence.
   `_build_visual_survival_configs` writes both `source_max_steps` and `max_ticks` to the same configured value, and the env computes `_max_source_ticks = _max_ticks * decision_source_frames`. Under the trusted default this should mean `source_max_steps=N` permits exactly `N` granular source steps, not `N * 12`. Add `test_source_max_steps_caps_granular_source_ticks_with_default_cadence`: instantiate with `source_max_steps=3`, default cadence/repeat, step until done, and assert exactly 3 LightZero steps, `terminal_reason == "timeout"`, `max_source_ticks == 3`, and final `physical_step_index/source_tick_index == 3`. Also add the repeat edge case: with `source_max_steps=2` and explicit repeat 3, one LightZero transition should terminate with `policy_action_repeat_requested == 3` but `policy_action_repeat_executed == 2`.

3. The explicit action-repeat test should prove repeat is made out of granular source steps.
   `test_source_state_visual_survival_action_repeat_is_one_policy_transition` currently asserts wrapper-level counters and reward for repeat=3. Tighten it by recording the underlying source tick before/after and asserting delta 3, `physical_decision_ms_total == 3 * SOURCE_PHYSICS_STEP_MS`, and `env._last_batch.info["source_physics_substeps_executed"][0] == 1` for the final underlying vector step. This catches the risky regression where repeat accidentally becomes one larger `decision_ms` step.

4. Background eval/GIF config relies on defaults, but tests should make cadence observable there too.
   `_make_policy_and_env` in `lightzero_curvytron_visual_survival_eval.py` hardcodes `decision_ms=DEFAULT_DECISION_MS` when rebuilding eval/GIF envs. That is currently okay because `DEFAULT_DECISION_MS` imports the trusted one-frame source-state default, but the eval surface only reports `source_max_steps`. Add a plumbing test that monkeypatches `_build_visual_survival_configs` or inspects the fake call to assert checkpoint eval and GIF env construction use `decision_ms == SOURCE_PHYSICS_STEP_MS`, `policy_action_repeat_min/max == 1`, and `policy_action_repeat_extra_probability == 0.0`. Also extend `test_background_eval_inspection_and_gif_can_be_explicitly_enabled` to assert default `max_eval_steps == 65_536`, `selfplay_gif.max_steps is None`, `selfplay_gif.step_limit_kind == "until_environment_done"`, and `selfplay_gif.source_max_steps` follows the training `source_max_steps`.

5. GIF action traces can hide cadence regressions.
   `_capture_checkpoint_selfplay_gif_variant` records scalar actions and joint physical commits, but scalar trace rows omit `decision_ms`, `decision_source_frames`, `physical_step_index/source_tick_index`, and repeat metadata from `last_info`. Add a test around checkpoint GIF summary or variant capture that expects these fields in `action_trace.scalar_actions[*]` and `action_trace.joint_actions[*]`. Without that, a future GIF can show plausible actions while silently using a bundled cadence. If the trace schema should stay compact, at least require cadence fields in the variant `surface` and top-level summary.

6. Existing 300 ms / 12-frame assumptions are mostly outside the trusted lane, but two docs/tests can confuse this refactor.
   Active trusted stock tests use `train_mod.DEFAULT_DECISION_MS`, and `test_modal_config_defaults_to_one_source_frame_per_policy_action` now asserts the one-frame default. The explicit `decision_ms=300.0` entries in `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py` are profile/no-death stress tests, while the `decision_ms=300.0` payloads in `tests/test_curvytron_live_checkpoint_eval_plumbing.py` are two-seat launcher tests. `decision_source_frames=12` appears in product-fidelity and checkpoint-tournament docs/tests, not the stock fixed-opponent `train_muzero` lane. This critique originally flagged the cadence audit doc as stale after the patch; that doc is now marked historical.

## Exact Test Changes

- Adjust `test_source_state_visual_survival_step_and_terminal_telemetry` to assert persisted cadence/repeat telemetry.
- Add `test_source_state_visual_survival_source_max_steps_caps_granular_ticks`.
- Add `test_source_state_visual_survival_repeat_stops_at_source_max_steps_cap`.
- Tighten `test_source_state_visual_survival_action_repeat_is_one_policy_transition` with source tick deltas and per-vector-step substep assertions.
- Extend `test_modal_config_defaults_to_one_source_frame_per_policy_action` with `env_cfg["max_ticks"] == env_cfg["source_max_steps"]` and, if practical, a tiny env instantiation from that config.
- Extend `test_background_eval_inspection_and_gif_can_be_explicitly_enabled` for `source_max_steps`, `max_eval_steps`, GIF uncapped semantics, and cadence defaults.
- Add a focused eval/GIF plumbing test proving `_make_policy_and_env` receives/defaults to one-frame cadence and no policy repeat.
- Add a checkpoint GIF trace/surface assertion so action traces expose cadence, not just scalar action IDs.
