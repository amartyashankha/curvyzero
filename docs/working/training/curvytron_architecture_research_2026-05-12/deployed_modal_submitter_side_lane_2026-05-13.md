# Deployed Modal Submitter Side Lane

Date: 2026-05-13

Question: can future CurvyTron launchers avoid one ephemeral Modal app per
training row by deploying one app and spawning many calls with different args?

## Answer

Yes. This is a clean orchestration change, not a trainer refactor, as long as
the submitter preserves the current local-entrypoint orchestration:

1. spawn the checkpoint eval poller for the run;
2. then spawn the train function with the same kwargs;
3. record both call IDs and the same Volume refs the current launcher prints.

Do not switch the current overnight batch cold. The existing
`modal run --detach` path is already canaried for training, checkpoint eval,
GIF generation, and status readout. Move this after the batch, behind a small
1-2 row canary.

## Current Code Shape

Current app:

- `APP_NAME = "curvyzero-lightzero-curvytron-visual-survival-train"` at
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:309`.
- one `modal.App(APP_NAME)` at line 775.

Current callable functions already split the work:

- checkpoint eval: `lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect`
  at line 8390.
- checkpoint GIF: `lightzero_curvytron_visual_survival_checkpoint_selfplay_gif`
  at line 8445.
- checkpoint poller: `lightzero_curvytron_visual_survival_checkpoint_eval_poller`
  at line 8500.
- stock train functions include `lightzero_curvytron_visual_survival_gpu` at
  line 8749, plus `cpu64`, `gpu_cpu40`, `h100_cpu40`, and `h100x2_cpu40`.

The local entrypoint is not a thin train call. For stock `mode=train`, it
normalizes arguments, chooses the compute-specific train function, computes
`exp_name_ref`, spawns `lightzero_curvytron_visual_survival_checkpoint_eval_poller`
when `background_eval_launch_kind == "poller"`, then spawns train and prints
the train call id plus poller/status refs. That path is lines 9491-9687.

Two-seat mode has the same important shape: build payload, derive background
eval config, spawn poller via `_spawn_two_seat_checkpoint_poller`, then spawn
the two-seat train function. That path is lines 9291-9469.

## Recommended Pattern

Deploy once:

```sh
uv run --extra modal modal deploy -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train
```

Then use a small local Python submitter, not `modal run` per row:

```python
import modal

APP = "curvyzero-lightzero-curvytron-visual-survival-train"

train_fn = modal.Function.from_name(APP, "lightzero_curvytron_visual_survival_gpu")
poller_fn = modal.Function.from_name(
    APP, "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
)

poller_call = poller_fn.spawn(**poller_kwargs)
train_kwargs["background_eval_launch_kind"] = "poller"
train_call = train_fn.spawn(**train_kwargs)
```

For two-seat rows, look up `lightzero_curvytron_two_seat_selfplay_gpu` and
`lightzero_curvytron_visual_survival_checkpoint_eval_poller`, then mirror the
current payload and poller kwargs from the local entrypoint.

Do not call the train function directly without the poller spawn when live
eval/GIF behavior is expected. The poller is the thing that later discovers
checkpoints and spawns eval/GIF jobs.

## Tiny Proof

No GPU or LightZero work was launched. I used the existing CPU sleep-marker
probe.

Commands run:

```sh
uv run --extra modal modal deploy -m curvyzero.infra.modal.background_spawn_probe

uv run --extra modal python -c 'import json, modal; f = modal.Function.from_name("curvyzero-background-spawn-probe", "background_sleep_and_mark"); calls = []
for idx in range(2):
    call = f.spawn(run_id="deployed-spawn-probe-20260513", attempt_id=f"from-name-{idx}", steps=1, sleep_sec=0.1)
    calls.append(call)
print(json.dumps({"calls": [getattr(c, "object_id", None) or getattr(c, "id", None) for c in calls]}, sort_keys=True))
print(json.dumps([c.get() for c in calls], sort_keys=True))'

uv run --extra modal modal volume ls curvyzero-runs training/modal-background-spawn-probe/deployed-spawn-probe-20260513/attempts
```

Result:

- deployed app: `curvyzero-background-spawn-probe`;
- spawned calls:
  `fc-01KRG2C5FNEJSZTJ0M27J72J17` and
  `fc-01KRG2C5HJXFRSM5QEVKDGS4F7`;
- both returned `ok: true`;
- Volume refs appeared under separate attempts:
  `from-name-0` and `from-name-1`.

This proves one deployed app can host multiple spawned calls with different
parameters and durable Volume outputs. It does not prove long GPU trainer
health, checkpoint cadence, or policy quality.

## Caveats

- Modal docs say deployments group repeated executions under one app and avoid
  clutter from programmatically triggering many ephemeral app runs:
  https://modal.com/docs/guide/managing-deployments
- Modal docs say `Function.from_name(app_name, name)` references a function
  from a deployed app, and `.spawn(...)` returns a `FunctionCall` handle:
  https://modal.com/docs/reference/modal.Function
- The deployed submitter must reimplement only the launcher/orchestration
  surface, not LightZero training internals.
- The current entrypoint has a lot of CLI normalization. The least risky
  implementation is to factor that argument-to-kwargs logic into small helper
  functions after the overnight batch, then have both the CLI entrypoint and
  deployed submitter call the same helpers.
- Keep run IDs and attempt IDs unique. A grouped app does not protect shared
  Volume paths from concurrent writers.
- Redeploying the app updates future calls. Existing running calls continue on
  the version that accepted them, per Modal deployment semantics.

