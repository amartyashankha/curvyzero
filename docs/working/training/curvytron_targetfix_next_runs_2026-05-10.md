# CurvyTron targetfix next runs - 2026-05-10

Purpose: record the next target-fixed two-seat training launch attempt after the
target/batch fix, and keep the status clear after the accumulated-replay pivot.

No pytest was run.

## Updated read

Do not treat larger `current_iteration` replay runs as the main scale answer.
The corrected target-fix scale eval came back flat:

| Checkpoint | Mean survival over 64 seeds |
| --- | ---: |
| `iteration_0` | 196.156 |
| `iteration_16` | 197.453 |

That means the old bounded loop is mechanically useful as a baseline only. It
updates from rows collected in the current outer iteration unless
`--replay-scope accumulated` is set. The accumulated replay patch is now the
right path for the next main run. Another worker is launching
`curvytron-two-seat-accum-b4-s21-32x16-u4-sim8` or a nearby accumulated-replay
shape, so no duplicate accumulated run was launched here.

## Baseline A - submitted before the pivot

This command was launched before the follow-up direction arrived. It did not
set `--replay-scope accumulated`, so it is an old-scope
`current_iteration` replay baseline.

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 10 \
  --batch-size 16 \
  --steps 64 \
  --outer-iterations 32 \
  --collect-steps-per-iteration 64 \
  --updates-per-iteration 1 \
  --num-simulations 2 \
  --allow-optimizer-step \
  --run-id curvytron-two-seat-targetfix-next-a-b16-s10-32x64-u1-sim2 \
  --attempt-id targetfix-next-a-b16-s10-32x64-u1-sim2-20260510 \
  --output summary
```

Modal app run:

```text
ap-cESW5vHcmtWIkj57QIdD3l
```

Expected full-run checkpoint count:

- `iteration_0.pth.tar` through `iteration_32.pth.tar`: 33 iteration
  checkpoints.
- `latest.pth.tar` and `ckpt_best.pth.tar`: 2 rolling aliases.

Observed status:

- The Modal function hit its 600 second timeout before returning a summary.
- No attempt `summary.json` was visible under the run's `attempts/` path.
- Volume listing showed checkpoints through `iteration_20.pth.tar`, plus
  `latest.pth.tar` and `ckpt_best.pth.tar`.
- Treat the run as incomplete and old-scope. It should not be used as the main
  scale signal.

Visible checkpoint root:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-next-a-b16-s10-32x64-u1-sim2/checkpoints/lightzero
```

Visible iteration checkpoints at timeout:

```text
iteration_0.pth.tar ... iteration_20.pth.tar
latest.pth.tar
ckpt_best.pth.tar
```

## Baseline B - intentionally skipped

The planned second old-scope run was not launched after the follow-up context.
This avoids duplicating work that is unlikely to be useful now that the
`current_iteration` loop has a flat 64-seed scale eval and accumulated replay is
available.

Original candidate, not launched:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 11 \
  --batch-size 16 \
  --steps 64 \
  --outer-iterations 32 \
  --collect-steps-per-iteration 64 \
  --updates-per-iteration 2 \
  --num-simulations 2 \
  --allow-optimizer-step \
  --run-id curvytron-two-seat-targetfix-next-b-b16-s11-32x64-u2-sim2 \
  --attempt-id targetfix-next-b-b16-s11-32x64-u2-sim2-20260510 \
  --output summary
```

## Next eval points

For the already-launched old-scope baseline, only eval it if a sanity comparison
is useful. Suggested sparse points from the visible partial run:
`0,4,8,12,16,20`.

For the accumulated-replay main run being handled elsewhere, use the main curve
points instead: `0,1,2,4,8,16,24,32`, with the same fixed target/batch eval
protocol used for the target-fix checks.
