# LightZero Modal Background Launch Patterns

## Root Cause

The frequent-checkpoint trainer failure path is a Modal launch lifecycle problem,
not a policy-quality result.

The wrapper entrypoint in
`src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py` used
`train_fn.remote(...)` from an `@app.local_entrypoint()`. That means the local
entrypoint submits the remote trainer and waits for the return value. Modal's
CLI warning for detached apps says detached apps that use `.remote()` or
`.map()` may still be canceled when the local caller disconnects, and recommends
`.spawn()` for detached/background work.

The killed frequent-checkpoint attempt
`train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath` matches that
failure mode. It reached training, wrote `iteration_0` and `ckpt_best`, then
Modal stopped the app after local client disconnect before `iteration_1000`.

The current detached attempt
`train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath` was
also launched before the background-launch patch, so it was still submitted
through the same `.remote()` call path. `modal run --detach` improved the app
lifecycle, but the CLI warning means it was not the robust background pattern
for this local entrypoint.

## Patch

Future `--mode train` launches now use:

```python
train_fn.spawn(...)
```

The local entrypoint prints a small JSON launch record with the Modal function
call id and the expected progress ref, then returns. The training function still
writes progress, summaries, manifests, and checkpoints to the Modal Volume.

Dry/smoke mode still uses `.remote()` so short validation commands keep printing
the full result directly.

## Claim

Use `.spawn()` for long LightZero training launched from this wrapper, and
launch the wrapper with `modal run --detach`. A toy validation in
`docs/working/modal_background_spawn_probe_2026-05-10.md` found that plain
`modal run` plus `.spawn()` returned a function call id but did not produce
Volume markers; `modal run --detach` plus `.spawn()` did.

The training result is stored in the Volume and should be monitored through
progress JSON and checkpoints, not by keeping the local `modal run` process
alive until the trainer returns.

The first frequent-checkpoint replacement failure does not show that the
checkpoint cadence override or learner is broken. It shows the training call was
coupled to the local caller lifecycle.

## Non-claim

This does not say the Pong policy is improving.

This does not reinterpret the earlier weak final-checkpoint signal.

This does not prove the already-running detached app will complete, because it
was launched before the `.spawn()` patch. Treat it as a live attempt to poll,
not as a clean validation of the new launch pattern.

No pytest was run for this lifecycle patch.
