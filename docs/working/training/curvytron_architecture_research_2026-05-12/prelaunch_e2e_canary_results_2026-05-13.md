# Prelaunch E2E Canary Results - 2026-05-13

## curvytron-prelaunch-e2e-blank-20260513b

Validation target:

- `run_id`: `curvytron-prelaunch-e2e-blank-20260513b`
- `attempt_id`: `blank-survbonus-ckptgif-smoke-sharedsupport`
- Training lane: stock LightZero `train_muzero`
- Reward variant: `survival_plus_bonus_no_outcome`
- Opponent runtime mode: `blank_canvas_noop`
- Source-state trail render mode: `body_circles_fast`

Verdict: checkpoint/eval/GIF plumbing landed on the Modal volume.

Evidence from `curvyzero-runs`:

- Train summary exists at `training/lightzero-curvytron-visual-survival/curvytron-prelaunch-e2e-blank-20260513b/attempts/blank-survbonus-ckptgif-smoke-sharedsupport/train/summary.json`.
- `summary.json` reports `ok=true`, `trainer_entrypoint=lzero.entry.train_muzero`, `called_train_muzero=true`, `reward_schema_id=curvyzero_survival_plus_bonus_no_outcome/v0`, and `train_result.ok=true`.
- Checkpoint mirror exists under `training/lightzero-curvytron-visual-survival/curvytron-prelaunch-e2e-blank-20260513b/checkpoints/lightzero` with `iteration_0.pth.tar` through `iteration_12.pth.tar` plus `ckpt_best.pth.tar`. The summary mirror block reports 14 copied checkpoint files.
- `checkpoint_eval_poller.json` exists and reports `status=completed`, `seen_count=13`, `scheduled_count=13`, `eval_completed_count=13`, `gif_scheduled_count=13`, `gif_completed_count=13`, `outstanding_count=0`, and `train_done=true`.
- The attempt eval root contains 13 eval folders: `live_checkpoint_iteration_0` through `live_checkpoint_iteration_12`.
- Example eval artifact for `iteration_12` exists:
  - Manifest: `eval/live_checkpoint_iteration_12/manifest_steps64_seedsn1_ce0ff9ef6785_20260513T050816Z.json`
  - Summary: `eval/live_checkpoint_iteration_12/iteration_12_steps64_seed1654615998/curvytron_visual_survival_eval_iteration_12_steps64_seed1654615998_20260513T050806Z.json`
  - Manifest reports `ok=true`, `result_count=1`, `job_kind=lightzero_curvytron_visual_survival_live_checkpoint_eval`.
  - Summary reports `ok=true`, strict model load, `steps_survived=64`, `steps_run=64`, action histogram `{0:18, 1:17, 2:29}`, terminal `round_survivor_win`, and `death_cause_name=own_trail`.
- Example GIF folder for `iteration_12` exists at `eval/live_checkpoint_iteration_12/selfplay` and contains `summary.json`, `raw.gif`, `collect_t1.gif`, `raw_frames.npz`, `collect_t1_frames.npz`, and both turn-commit telemetry JSONLs.
- The `iteration_12` GIF summary reports `ok=true`; `raw.gif` is present with 15 frames and `collect_t1.gif` is present with 17 frames. Both variants include non-collapsed action summaries.

Telemetry/tooling notes:

- Training action observability is usable: `action_observability.json` reports `status=ok`, `row_count=75`, `scalar_step_count=75`, action histogram `{0:28, 1:22, 2:25}`, physical action histogram `{0:28, 1:22, 2:25}`, opponent action histogram `{1:75}`, reward sum `72.0`, reward mean `0.96`, and terminal reasons `{none:72, round_survivor_win:3}`.
- Required fields are present for downstream tooling: `requested_ego_action`, `executed_ego_action`, `joint_action`, fixed opponent action, `terminal_reason`, reward/trainer reward fields, `survival_reward_for_ego`, `sparse_outcome_reward_for_ego`, death fields, and `trail_render_mode=body_circles_fast` in train telemetry.
- No persisted zero compact row-count problem was observed in the inspected JSON artifacts: the action observability compact surface reports 75 rows. One minor compact/log anomaly remains: `train_result.log_signals.checkpoint_iterations` is empty and `max_checkpoint_iteration=null` even though stderr and volume listings prove checkpoints `iteration_0` through `iteration_12` were saved and mirrored.

Conclusion: this canary validates the shared-support prelaunch plumbing path for stock LightZero training, checkpoint mirroring, background checkpoint eval, and both GIF variants on volume.
