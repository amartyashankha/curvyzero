# CurvyTron accumulated replay run - 2026-05-10

## Scope

Launched the first accumulated-replay CurvyTron two-seat bounded LightZero
trainer run using the patched smoke trainer with:

- `replay_scope=accumulated`
- `learner_sample_size=128`
- optimizer steps enabled
- seed `21`
- `batch_size=4`
- `outer_iterations=32`
- `collect_steps_per_iteration=16`
- `updates_per_iteration=4`
- train/eval `num_simulations=8`

I did not run pytest.

## Training

The direct global `modal run ...` form could not import the local `curvyzero`
package, so the actual launch used the repo's documented `uv run --extra modal`
wrapper.

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 21 \
  --batch-size 4 \
  --outer-iterations 32 \
  --collect-steps-per-iteration 16 \
  --updates-per-iteration 4 \
  --num-simulations 8 \
  --replay-scope accumulated \
  --learner-sample-size 128 \
  --allow-optimizer-step \
  --run-id curvytron-two-seat-accum-b4-s21-32x16-u4-sim8-20260510a \
  --attempt-id accum-b4-s21-32x16-u4-sim8-20260510a \
  --output summary
```

Modal app run:

```text
ap-NVEt7vLApH2f1mor4tMdgy
```

Run ids:

```text
run_id: curvytron-two-seat-accum-b4-s21-32x16-u4-sim8-20260510a
attempt_id: accum-b4-s21-32x16-u4-sim8-20260510a
```

Training summary ref:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-accum-b4-s21-32x16-u4-sim8-20260510a/attempts/accum-b4-s21-32x16-u4-sim8-20260510a/train/summary.json
```

Checkpoint root:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-accum-b4-s21-32x16-u4-sim8-20260510a/checkpoints/lightzero
```

Training completed with `ok: true`. The run wrote 33 checkpoints
(`iteration_0` through `iteration_32`), ended with `steps_survived: 512`, and
reported accumulated replay active with `4096` replay rows available. The final
learner forward reported `optimizer_step: allowed`, `status: updated`, and
`model_parameters_changed: true`.

## Eval

The full requested 64-seed eval completed with `num_simulations=8`, so no
32-seed fallback was needed.

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --run-id curvytron-two-seat-accum-b4-s21-32x16-u4-sim8-20260510a \
  --attempt-id accum-b4-s21-32x16-u4-sim8-20260510a \
  --eval-id accum_curve_n64_steps512_sim8_20260510a \
  --selected-iterations 0,1,2,4,8,16,24,32 \
  --max-eval-steps 512 \
  --eval-seed-count 64 \
  --num-simulations 8 \
  --batch-size 64 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Modal app run:

```text
ap-JJBqd49g9xOs8Q8xEh2fv8
```

Manifest:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-accum-b4-s21-32x16-u4-sim8-20260510a/attempts/accum-b4-s21-32x16-u4-sim8-20260510a/eval/accum_curve_n64_steps512_sim8_20260510a/manifest_steps512_seedsn64_de770778a1c3_20260510T215622Z.json
```

Eval seed sampler:

```text
6532985159430981418
```

## Survival Curve

All 512 checkpoint/seed eval jobs returned `ok: true` and strict checkpoint
loads were true. No episodes capped at 512 steps and no failures were reported.

| Checkpoint | Seeds | Mean steps | Median | Min | Max | Capped | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iteration_0 | 64 | 191.688 | 180 | 40 | 368 | 0 | 0 |
| iteration_1 | 64 | 201.844 | 198 | 35 | 368 | 0 | 0 |
| iteration_2 | 64 | 201.844 | 198 | 35 | 368 | 0 | 0 |
| iteration_4 | 64 | 201.844 | 198 | 35 | 368 | 0 | 0 |
| iteration_8 | 64 | 201.844 | 198 | 35 | 368 | 0 | 0 |
| iteration_16 | 64 | 201.844 | 198 | 35 | 368 | 0 | 0 |
| iteration_24 | 64 | 201.844 | 198 | 35 | 368 | 0 | 0 |
| iteration_32 | 64 | 201.844 | 198 | 35 | 368 | 0 | 0 |

## Verdict

Mechanically, the accumulated-replay run and eval passed: training completed,
checkpoints loaded strictly, the 64-seed sim8 curve completed, and no eval jobs
failed.

Policy-quality read: weak/flat. The curve improves by only `+10.156` mean steps
from `iteration_0` to `iteration_1`, then stays exactly flat through
`iteration_32`. The per-seed rows also show trained checkpoints using only
action `0` in the sampled output lines, while `iteration_0` still had mixed
actions on those same seeds. Treat this as another mechanically valid run with
no convincing accumulated-replay learning signal yet.

## Strategy Note

This run is a bounded custom-scaffold experiment, not the main future path.
Do not launch more custom-loop scale from it. The next mainline direction is
native `train_muzero` single-ego training with fixed or frozen opponents, plus a
small joint-action collector experiment.
