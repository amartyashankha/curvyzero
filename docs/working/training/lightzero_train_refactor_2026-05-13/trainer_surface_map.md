# Trainer Surface Map

Date: 2026-05-13

This is the main-thread first pass. Subagent audits may revise it.

## File Size Reality

The active trainer entrypoint is currently very large:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

It is about 10k lines. The cleanup should be small tested cuts, not a rewrite.

## Major Responsibility Groups

| Area | Current examples | Refactor stance |
| --- | --- | --- |
| Reward/config normalization | `_normalize_reward_variant_for_env`, `_reward_policy_for_variant`, `_lightzero_target_config_for_reward`, `_build_visual_survival_configs` | Leave behavior stable. Extract only after tests pin launcher config outputs. |
| LightZero profiling/hooks | `_install_lightzero_phase_profile`, `_install_live_checkpoint_publisher`, `_install_checkpoint_progress_writer`, `_install_lightzero_full_resume_state_hooks` | Modal/trainer-adjacent. Keep careful; hooks are brittle. |
| Checkpoint discovery/progress | `_latest_lightzero_iteration_checkpoint`, `_write_checkpoint_progress_latest`, `_scan_lightzero_artifacts`, `_publish_live_lightzero_checkpoints` | First bugfix target. Needs a central broad-discovery helper. |
| Resume state | `_prepare_lightzero_auto_resume`, `_find_lightzero_resume_sidecar`, resume-state save/load helpers | Same bug surface as checkpoint discovery. Tests before patch. |
| Background eval/GIF | `_background_eval_config_from_command`, `_background_gif_config_from_command`, `_spawn_one_checkpoint_background_eval`, `_spawn_one_checkpoint_background_gif`, `_run_checkpoint_eval_poller`, `_run_checkpoint_selfplay_gif` | Keep external to trainer hot path. Refactor payload builders only after tests. |
| Status/attempt metadata | `_write_attempt_state`, `_write_latest_attempt`, `_write_train_status_heartbeat` | Thin artifact IO; should be easy to keep pure payload builders separate from writes. |
| Modal functions | `lightzero_curvytron_visual_survival_*`, checkpoint eval/GIF/poller functions, `main` | Should become thin wrappers around pure helpers. |
| Historical two-seat path | `_run_two_seat_selfplay_payload`, `lightzero_curvytron_two_seat_selfplay_*` | Historical/untrusted for learning claims. Do not let this drive the stock trainer refactor. |

## Known Bug Surface

These functions currently look suspicious for fixed-path assumptions:

- `_latest_lightzero_iteration_checkpoint(exp_name)` scans only `exp_name / "ckpt"`.
- `_write_checkpoint_progress_latest(...)` depends on `_latest_lightzero_iteration_checkpoint`.
- `_prepare_lightzero_auto_resume(...)` scans current/prior attempts at
  `train/lightzero_exp/ckpt`, plus the stable mirror.
- `_find_lightzero_resume_sidecar(...)` scans current/prior attempts at
  `train/lightzero_exp/<resume_state_dir>`, plus the stable mirror.
- `_scan_lightzero_artifacts(exp_name)` recursively scans only one `exp_name`.
- `_run_checkpoint_eval_poller(...)` calls `_scan_lightzero_artifacts(str(exp_name))`.
- `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py::_checkpoint_summary`
  chooses the first existing checkpoint dir from stable mirror or fixed
  `train/lightzero_exp/ckpt`.

These should all use one tested broad-discovery contract or an explicitly
documented fallback.

## Existing Test Coverage Observed

Useful current tests:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`
  - checkpoint progress writer;
  - save-hook trigger source checkpoint ref;
  - poller scheduling eval/GIF jobs;
  - local launcher GIF config wiring.
- `tests/test_opponent_mixture.py`
  - mixture spec parsing and threading to eval/GIF;
  - immutable frozen checkpoint refs.
- `tests/test_curvytron_checkpoint_tournament.py`
  - tournament broad checkpoint discovery already has tests for
    `lightzero_exp*`.

Coverage gap:

- trainer-side progress/status/resume/poller code does not yet have broad
  `lightzero_exp*` regression tests.
- status CLI/web summary does not yet have broad `lightzero_exp*` regression
  tests.

## First Extraction Candidate

After tests and bugfix:

```text
src/curvyzero/training/lightzero_checkpoints.py
```

Potential pure functions:

- parse iteration checkpoint name;
- parse resume sidecar name;
- list LightZero checkpoint dirs under an attempt train root;
- list run checkpoint candidates across current attempt, prior attempts, and
  stable mirror;
- select latest checkpoint deterministically.

Do not create this module until the test plan is final.

