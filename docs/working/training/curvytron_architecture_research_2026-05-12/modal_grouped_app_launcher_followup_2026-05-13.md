# Modal Grouped App Launcher Follow-Up

Date: 2026-05-13

Purpose: record the dashboard-cleanliness question without blocking the current
overnight survivaldiag launch.

## Current Decision

Do not switch the overnight batch to a new grouped-app submitter cold.

The existing path launches one `modal run --detach` per row. That makes the
Modal dashboard messy, but it is the path we already canaried for training,
checkpoint eval, GIF generation, and status readout.

2026-05-13 follow-up: Lovelace independently confirmed the deployed-app pattern
and ran a safe CPU toy proof with `curvyzero-background-spawn-probe`: one
deployed app, two `Function.from_name(...).spawn(...)` calls with different
arguments, and separate volume refs written successfully. No GPU jobs were run.

## Better Pattern

A cleaner future launcher should:

- deploy one named Modal app for
  `curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train`;
- submit many calls to the already-defined training functions with different
  kwargs, using `modal.Function.from_name(...).spawn(...)`;
- preserve the current local-entrypoint behavior that starts the checkpoint
  poller before starting the train function.

Concrete sketch:

```python
poller_fn = modal.Function.from_name(
    "curvyzero-lightzero-curvytron-visual-survival-train",
    "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
)
train_fn = modal.Function.from_name(
    "curvyzero-lightzero-curvytron-visual-survival-train",
    "lightzero_curvytron_visual_survival_gpu_cpu40",
)

poller_call = poller_fn.spawn(**poller_kwargs)
train_kwargs["background_eval_launch_kind"] = "poller"
train_call = train_fn.spawn(**train_kwargs)
```

Relevant current functions:

- `lightzero_curvytron_visual_survival_gpu`;
- `lightzero_curvytron_visual_survival_gpu_cpu40`;
- `lightzero_curvytron_visual_survival_h100_cpu40`;
- `lightzero_curvytron_visual_survival_h100x2_cpu40`;
- `lightzero_curvytron_visual_survival_cpu64`;
- `lightzero_curvytron_visual_survival_checkpoint_eval_poller`.

Current code refs:

- app name: `APP_NAME` in
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`;
- app object: `app = modal.App(APP_NAME)`;
- checkpoint poller function:
  `lightzero_curvytron_visual_survival_checkpoint_eval_poller`;
- local entrypoint poller spawn happens before the train spawn in `main`;
- eval/GIF child jobs are spawned by the checkpoint handling path.

## Risk

The current local entrypoint does more than train. For
`background_eval_launch_kind=poller`, it computes the train artifact path,
spawns the checkpoint poller, then spawns the train function. A grouped submitter
must do the same thing or it can accidentally launch training without the live
eval/GIF poller.

## Next Step

Build a small deployed-app submitter and test it on a 1-2 row canary after the
current overnight batch is launched or deliberately paused. If that canary
writes train status, checkpoint, poller status, eval, GIF, and run-status fields
under one Modal app, promote it for future large matrices.

Useful Modal docs:

- Apps: https://modal.com/docs/guide/apps
- Managing deployments: https://modal.com/docs/guide/managing-deployments
- Invoking deployed functions: https://modal.com/docs/guide/trigger-deployed-functions
- `modal.Function`: https://modal.com/docs/reference/modal.Function
