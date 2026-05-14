# Granular Action Cadence Footgun Critique - 2026-05-13

Target: trusted `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`.

Goal: one policy action per CurvyTron source physics step.

## Status After Patch

The main foot-gun from this critique is now fixed for the trusted
`source_state_fixed_opponent` stock train lane. Trusted train/dry config rejects
stale multi-frame `decision_ms` values, the trainer config passes
`decision_source_frames=1`, and the active launch manifest builders emit the
one-frame timing value. The rest of this note is preserved as the critique that
motivated the guard.

## Bottom Line

Before the guard, the default was fixed but the train surface was still too easy to misuse. A stale
`decision_ms=200.0` or `decision_ms=300.0` could still enter the trusted train path
and become 12 or 18 source physics substeps per policy action. That is the main
footgun this patch closed.

## Footguns

1. `decision_ms` is still the public train knob.

   Files/functions:
   - `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
   - `main`
   - `lightzero_curvytron_visual_survival_cpu`
   - `lightzero_curvytron_visual_survival_gpu`
   - `_run_visual_survival_train`
   - `_build_visual_survival_configs`
   - `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py::_source_frame_decision_config`

   The env default is now one frame:
   `DEFAULT_DECISION_SOURCE_FRAMES = 1` and
   `DEFAULT_DECISION_MS = SOURCE_PHYSICS_STEP_MS * DEFAULT_DECISION_SOURCE_FRAMES`.
   Good.

   But `_run_visual_survival_train` only checks `decision_ms > 0`. Then
   `_build_visual_survival_configs` writes only `"decision_ms": float(decision_ms)`
   into `main_config.env`. If an old launch command passes `--decision-ms 300`, the
   env accepts it, derives whole source frames, and holds the action for many
   source physics steps. No error. No warning.

   Recommended patch:
   - Add a trusted-train guard in `_run_visual_survival_train`: for
     `mode == "train"` and `env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT`,
     resolve cadence with the same logic as `_source_frame_decision_config`, then
     reject anything where `decision_source_frames != 1`.
   - Better: expose `decision_source_frames` as the primary knob, default it to `1`,
     pass it through `_build_visual_survival_configs`, and write both
     `decision_source_frames` and `decision_ms` into the env config and command.
   - Keep `decision_ms` only as a derived/display field or legacy override that
     must equal one source frame in trusted train mode.

   Recommended tests:
   - `tests/test_curvytron_live_checkpoint_eval_plumbing.py`: add a test that
     `_build_visual_survival_configs(..., decision_ms=300.0,
     env_variant=ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT, ...)` is rejected for
     trusted train config, or that the caller-level guard rejects it before config
     build.
   - Add a CLI/default payload test around `main(..., mode="train",
     decision_ms=300.0, wait_for_train=False)` with a fake spawn, asserting it
     raises and does not schedule a run.

2. `source_max_steps` hides stale cadence.

   Files/functions:
   - `curvyzero_source_state_visual_survival_lightzero_env.py::CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.__init__`
   - `curvyzero_source_state_visual_survival_lightzero_env.py::_new_env`

   The wrapper reads `source_max_steps` into `_max_ticks`, then computes
   `_max_source_ticks = _max_ticks * _decision_source_frames`, and passes that as
   `VectorMultiplayerEnv(max_ticks=...)`.

   With the new one-frame default, this is fine. With stale `decision_ms=200/300`,
   `source_max_steps=256` quietly becomes a physical cap of 3,072 or 4,608 source
   physics ticks. The name says "source steps", but the code treats it as policy
   decisions when `decision_source_frames > 1`.

   Recommended patch:
   - In trusted train mode, assert `max_source_ticks == source_max_steps`.
   - Add `source_max_steps_semantics: "source_physics_steps"` to the command/env
     metadata once the guard is in place.

   Recommended tests:
   - Add a local env test that default config with `source_max_steps=5` reports
     `decision_source_frames == 1`, `max_ticks == 5`, and `max_source_ticks == 5`.
   - Add a negative trusted-train test for `source_max_steps=5, decision_ms=300.0`
     so it cannot silently become 90 source physics ticks.

3. Background eval and GIF do not carry cadence explicitly.

   Files/functions:
   - `lightzero_curvyzero_stacked_debug_visual_survival_train.py::_background_eval_config_from_command`
   - `lightzero_curvyzero_stacked_debug_visual_survival_train.py::_background_gif_config_from_command`
   - `lightzero_curvyzero_stacked_debug_visual_survival_train.py::_spawn_one_checkpoint_background_eval`
   - `lightzero_curvyzero_stacked_debug_visual_survival_train.py::_spawn_one_checkpoint_background_gif`
   - `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py::_make_policy_and_env`

   These configs thread `source_max_steps`, but not `decision_ms` or
   `decision_source_frames`. Today eval rebuilds with `DEFAULT_DECISION_MS`, which
   is now one source frame. That means stale train cadence and background eval/GIF
   cadence can diverge silently: training may hold actions for many source ticks
   while eval/GIF looks granular.

   Recommended patch:
   - Include `decision_source_frames`, `decision_ms`, and
     `source_physics_step_ms` in background eval and GIF config.
   - Reject background eval/GIF if the training command cadence is not trusted
     one-frame cadence.
   - Record the effective cadence in eval/GIF manifests.

   Recommended tests:
   - Extend `test_background_eval_inspection_and_gif_can_be_explicitly_enabled`
     to assert `config["decision_source_frames"] == 1`,
     `config["decision_ms"] == SOURCE_PHYSICS_STEP_MS`, and the same under
     `config["selfplay_gif"]`.
   - Extend `test_live_checkpoint_trigger_spawns_eval_and_selfplay_gif_without_volume_commit`
     to assert the spawned eval and GIF calls receive the cadence fields.

4. Telemetry rows do not expose the core cadence.

   Files/functions:
   - `curvyzero_source_state_visual_survival_lightzero_env.py::_write_telemetry_row`
   - `lightzero_curvyzero_stacked_debug_visual_survival_train.py::_summarize_env_step_telemetry`
   - `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py::_row_from_result`

   `timestep.info` has `decision_ms`, `decision_source_frames`, and
   `source_physics_step_ms`, but `_write_telemetry_row` does not write those fields.
   It writes `physical_decision_ms_total`, which is useful but not enough. The
   training summary therefore cannot easily say, "this run used one action per
   source frame."

   Eval rows include `decision_ms` but not `decision_source_frames`.

   Recommended patch:
   - Add `decision_ms`, `decision_source_frames`, `source_physics_step_ms`,
     `source_frame_decision`, and `max_source_ticks` to env telemetry rows.
   - Add cadence aggregation to `_summarize_env_step_telemetry`, including a
     `cadence_ok` boolean.
   - Add `decision_source_frames` to eval rows and manifest headers.

   Recommended tests:
   - Extend `test_source_state_visual_survival_step_and_terminal_telemetry` to
     assert telemetry row cadence fields.
   - Extend `test_env_step_telemetry_summary_reports_action_and_death_observability`
     to assert cadence summary fields.
   - Extend eval row tests to assert `decision_source_frames == 1`.

5. GIF capture has a nearby old-default trap.

   Files/functions:
   - `lightzero_curvyzero_stacked_debug_visual_survival_train.py::_capture_checkpoint_selfplay_gif_variant`
   - `src/curvyzero/training/curvyzero_source_state_visual_turn_commit_lightzero_env.py::_source_frame_decision_config`

   `_capture_checkpoint_selfplay_gif_variant` normally uses
   `ENV_VARIANT_SOURCE_STATE_TURN_COMMIT` when there is no opponent mixture. The
   turn-commit env still defaults to `DEFAULT_DECISION_SOURCE_FRAMES` imported from
   `vector_multiplayer_env`, which is `12`. The current GIF path is probably saved
   because eval `_make_policy_and_env` passes `DEFAULT_DECISION_MS`, now one frame,
   through `_build_visual_survival_configs`. But this is fragile. If a future GIF
   helper instantiates the turn-commit env without explicit cadence, it goes back
   to 12-frame action holds.

   Recommended patch:
   - Make the source-state turn-commit env default match the one-frame source-state
     survival env, or require explicit `decision_source_frames`.
   - Add a one-frame assertion to GIF capture summaries.

   Recommended tests:
   - Add a direct test for `CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv`
     default cadence, or explicitly document that its default is not trusted and
     assert GIF capture never relies on that default.
   - Extend GIF tests to inspect variant `surface` or telemetry and assert
     `decision_source_frames == 1`.

## Tests That Already Help

- `test_source_state_visual_survival_default_is_one_source_frame_per_policy_action`
  checks the local survival env default.
- `test_modal_config_defaults_to_one_source_frame_per_policy_action` checks
  `train_mod.DEFAULT_DECISION_MS` and the generated env config default.

Those are good, but they only protect the happy path. They do not protect stale
CLI/manifests that explicitly pass `200ms` or `300ms`.

## Hard Recommendation

For trusted `--mode train`, stop accepting cadence by silence. Either:

1. remove `decision_ms` from the trusted user-facing path and use
   `decision_source_frames=1`, or
2. keep `decision_ms` but fail fast unless it resolves to exactly one source
   physics frame.

Anything else lets old 200ms/300ms launch commands create a run that looks
trusted but is not granular.
