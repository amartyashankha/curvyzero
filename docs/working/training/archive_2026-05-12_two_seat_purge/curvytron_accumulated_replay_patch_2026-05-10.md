# CurvyTron accumulated replay patch - 2026-05-10

## Scope

Files changed:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py`

I did not run pytest for this patch.

## Design

Motivation update: a corrected scale run using `current_iteration` replay stayed
effectively flat over 64-seed eval, from 196.156 at `iteration_0` to 197.453 at
`iteration_16`. Accumulated replay is therefore the next small lever to test
before chasing larger architecture changes.

The two-seat bounded trainer now has two opt-in replay controls:

- `--replay-scope current_iteration|accumulated`
- `--learner-sample-size N`

`current_iteration` is the default and preserves the previous behavior: every learner update after an outer iteration samples from only that iteration's collected replay rows.

`accumulated` makes each learner update sample from all `replay_rows` collected so far in the run. This is still a small local adapter, not LightZero's full upstream GameBuffer, but it now reuses prior rows across outer iterations.

`--learner-sample-size` is optional. When it is unset, the learner receives all rows available under the selected scope. When it is set and smaller than the available rows, the code samples without replacement. The RNG is deterministic and derives from the run seed, outer iteration, and cumulative learner update index. Selected row indices are sorted after sampling so the learner adapter sees rows in collection order.

The sampled batch still carries:

- `iteration_batch`
- `env_row_id_batch`
- `player_id_batch`
- `decision_index_batch`

That keeps the shared learner adapter on the metadata-bearing survival target path instead of the legacy immediate-reward fallback.

Summary/result payloads now expose:

- `replay_scope`
- `replay_rows_available`
- `learner_batch_size`
- learner sample metadata including `sampled_without_replacement` and `sample_indices`

## Local command examples

Preserve old behavior:

```bash
uv run python -m curvyzero.training.curvytron_two_seat_lightzero_train_smoke \
  --seed 0 --batch-size 1 --outer-iterations 2 \
  --collect-steps-per-iteration 4 --updates-per-iteration 1 \
  --num-simulations 2 --replay-scope current_iteration
```

Use accumulated replay with a bounded learner batch:

```bash
uv run python -m curvyzero.training.curvytron_two_seat_lightzero_train_smoke \
  --seed 0 --batch-size 1 --outer-iterations 4 \
  --collect-steps-per-iteration 8 --updates-per-iteration 2 \
  --num-simulations 2 --replay-scope accumulated \
  --learner-sample-size 32 --allow-optimizer-step
```

## Modal command example

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 0 --batch-size 4 --outer-iterations 32 \
  --collect-steps-per-iteration 16 --updates-per-iteration 4 \
  --num-simulations 8 --replay-scope accumulated \
  --learner-sample-size 128 --allow-optimizer-step \
  --run-id curvytron-two-seat-accum-s0 \
  --attempt-id accum-replay-b4-i32-u4-sim8 --output summary
```

## Risks

This is accumulated replay for the local smoke trainer only. It does not add LightZero's distributed collector, replay priorities, target network cadence, or upstream GameBuffer target builder.

Bounded random samples can truncate the available future rows for a trajectory. The metadata survival target path still runs, but returns are computed from the sampled rows, not from the full replay table. Larger `learner_sample_size` values reduce that distortion.

The accumulated replay table is in memory for the duration of the smoke run. This is fine for the intended bounded runs, but it is not a long-horizon replay storage design.

Both seats still share one live policy object and are evaluated as one-row policy calls. This patch changes replay reuse, not the collector architecture.
