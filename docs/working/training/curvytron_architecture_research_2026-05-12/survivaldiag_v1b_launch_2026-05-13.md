# Survivaldiag V1B Launch

Date: 2026-05-13.

Status: launched and early health checked.

## What Launched

- Matrix prefix: `survivaldiag-v1b-20260513h`.
- Row count: 50.
- Launcher: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train`.
- Training path: stock LightZero `train_muzero`.
- Reward: `survival_plus_bonus_no_outcome`.
- Main opponent lane: `blank_canvas_noop`.
- Episode cap: `source_max_steps=65536`.
- Train caps: `max_train_iter=300000`, `max_env_step=30000000`.
- Checkpoint cadence: `save_ckpt_after_iter=5000`.
- Background checkpoint eval/GIF: enabled through the poller path.
- Stock LightZero in-loop eval: disabled with `lightzero_eval_freq=0`.

## Matrix Shape

- Render pairing: 25 `body_circles_fast`, 25 `browser_lines`.
- Row kinds:
  - 4 exact preflight rows.
  - 32 blank-canvas core rows.
  - 8 blank-canvas extra repeat rows.
  - 4 passive-immortal dirty-control rows.
  - 2 sim16 compute sentinel rows.
- Stochasticity profiles:
  - 8 none.
  - 8 low.
  - 22 medium.
  - 12 high.

## Launch Artifacts

- Manifest: `artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.json`.
- Rows: `artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.rows.jsonl`.
- Commands: `artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.review_commands.txt`.

## Fixes Before Launch

- Attempt IDs were shortened to stay under the run-management 96-character limit.
- Stock train Modal functions now use a 16-hour timeout.
- Checkpoint eval poller now uses an 18-hour runtime setting and a 20-hour Modal timeout.
- Regenerated launch commands use `background_eval_poller_max_runtime_sec=64800`.

## Validation Before Launch

Focused local checks passed:

```text
uv run pytest tests/test_curvytron_survivaldiag_manifest.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_lightzero_phase_profiler.py -q
56 passed, 1 skipped

uv run ruff check scripts/build_curvytron_survivaldiag_manifest.py tests/test_curvytron_survivaldiag_manifest.py src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_lightzero_phase_profiler.py
All checks passed
```

## Early Health

First status sweeps after launch:

- All 50 rows returned spawned metadata during dispatch.
- `modal app list` showed fresh detached CurvyZero apps with live tasks.
- `curvytron_run_status --output eval-summary` showed all 50 rows with
  `train_status=running`.
- Early rows already produced `iteration_0` checkpoints, poller status, and eval
  manifests. Later rows were still warming up or entering `train_muzero`.
- The latest sweep showed rows 1 and 2 at `iteration_5000`, many other rows at
  `iteration_0`, and a few rows still warming up or missing eval output.
- By 2026-05-13 03:13 EDT, curve-summary showed all 50 rows still running.
  Eighteen rows had already written an `iteration_5000` checkpoint, and at
  least twelve rows had an `iteration_5000` eval manifest.
- Row 1 live artifact spot-check:
  `eval_reward_variant=survival_plus_bonus_no_outcome`,
  `model_reward_variant=survival_plus_bonus_no_outcome`,
  `env_reward_variant=survival_plus_bonus_no_outcome`. Its eval reward is
  survival-style reward, not sparse outcome reward: mean steps moved from
  `10.125` at `iteration_0` to `32.5` at `iteration_5000`, and
  `mean_training_reward` moved from `9.125` to `31.5`.
- Row 1 greedy action summary also changed from fully collapsed at
  `iteration_0` to non-collapsed at `iteration_5000`
  (`top_action_fraction=0.415`).
- By 2026-05-13 03:22 EDT, the final early sweep still showed all 50 rows
  running. Forty-eight rows had reached at least `iteration_5000`, twelve rows
  had reached `iteration_10000`, and checkpoint pollers were still running.
  Several rows already show higher eval mean steps than their initial eval, but
  the curves only have one to three points each.
- By 2026-05-13 03:24 EDT, a fresh sweep still showed all 50 rows running.
  Forty-nine rows had reached at least `iteration_5000`, nineteen rows had
  reached `iteration_10000`, and checkpoint pollers were still running.
  Some eval manifests lag current checkpoints, which is expected while the
  background pollers are still active.
- By 2026-05-13 03:26 EDT, another fresh sweep still showed all rows running,
  checkpointing continuing, and checkpoint pollers active. No duplicate
  training batch was launched.

This is an early liveness check only. It is not a learning claim.

## Cleanup Note

This launch still used one detached Modal app per row. That is known dashboard
clutter. A separate cleanup lane is investigating one deployed Modal app plus
many spawned function calls. Do not retrofit that into this running batch.

## GIF Browser

Redeployed after the final early health sweeps:

```text
uv run --extra modal modal deploy -m curvyzero.infra.modal.curvytron_gif_browser
```

URL:
`https://modal-labs-shankha-dev--curvyzero-curvytron-gif-browser--bada8e.modal.run`

Latest deploy time: 2026-05-13 03:26 EDT.
