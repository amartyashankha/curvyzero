# CurvyTron Target-Fix Eval And Scale - 2026-05-10

Purpose: run the immediate corrected-target eval, then launch one larger
target-fixed two-seat CurvyTron run if the eval loads and returns sane numbers.

No pytest was run.

## Eval Command

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --run-id curvytron-two-seat-targetfix-smoke-b4-s8-2x16-u1b \
  --attempt-id targetfix-smoke-b4-2x16-u1b-20260510 \
  --selected-iterations 0,1,2 \
  --max-eval-steps 512 \
  --eval-seed-count 32 \
  --num-simulations 2 \
  --batch-size 64 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Modal app run:

```text
ap-YahZBzvw3AfdZ7Moya9JSO
```

Manifest:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-smoke-b4-s8-2x16-u1b/attempts/targetfix-smoke-b4-2x16-u1b-20260510/eval/checkpoint_curve/manifest_steps512_seedsn32_425d6f2a6ad9_20260510T213150Z.json
```

Eval seed sampler:

```text
8705867297756004882
```

Eval seeds:

```text
1634749711,41841131,327526143,1189983218,636552240,1894179878,1124894785,326147422,94000708,1515338075,1339751207,1718075609,86353217,77575868,1863190728,1215652041,1546163767,531714624,1024274507,895433363,2039911502,1185382238,1724273616,1871304118,36313183,1753235498,1509067672,1156634298,1050053376,1239227745,1236723154,761828330
```

## Eval Result

All 96 checkpoint/seed jobs returned `ok: true`.
All checkpoint loads were strict.

| Checkpoint | Seeds | Mean steps | Median | Min | Max | OK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iteration_0 | 32 | 192.969 | 183.0 | 45 | 354 | 32 |
| iteration_1 | 32 | 155.188 | 141.5 | 64 | 312 | 32 |
| iteration_2 | 32 | 151.406 | 126.5 | 55 | 354 | 32 |

Plain read: the corrected-target smoke checkpoints load and eval cleanly. The
numbers are sane, but the trained checkpoints are lower than `iteration_0` on
this 32-seed panel. Treat this as a mechanical pass, not a learning win.

## Scale Run Command

This used the smaller allowed scale point to stay inside the wrapper timeout:
batch 8, 16 outer iterations, 64 collect steps per iteration, 1 update per
iteration, `num_simulations=2`, and real optimizer steps enabled.

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
  --seed 9 \
  --batch-size 8 \
  --steps 64 \
  --outer-iterations 16 \
  --collect-steps-per-iteration 64 \
  --updates-per-iteration 1 \
  --num-simulations 2 \
  --allow-optimizer-step \
  --run-id curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2 \
  --attempt-id targetfix-scale-b8-s9-16x64-u1-sim2-20260510 \
  --output summary
```

Modal app run:

```text
ap-VMx6JHQTGYtgGYoJDX0ZB8
```

Summary:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2/attempts/targetfix-scale-b8-s9-16x64-u1-sim2-20260510/train/summary.json
```

Checkpoint root:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2/checkpoints/lightzero
```

## Scale Run Result

| Field | Value |
| --- | --- |
| ok | true |
| problems | none |
| mode | bounded_two_seat_lightzero_collect_replay_real_train_smoke |
| batch_size | 8 |
| outer_iterations | 16 |
| collect_steps_per_iteration | 64 |
| updates_per_iteration | 1 |
| num_simulations | 2 |
| replay rows | 16384 |
| learner batch size | 16384 |
| replay reward sum | 15638.0 |
| checkpoints | 17, `iteration_0` through `iteration_16` |
| latest checkpoint | `latest.pth.tar` |
| best checkpoint | `ckpt_best.pth.tar` |
| final learner status | updated |
| final optimizer step | allowed |
| final model changed | true |
| final model hash | `d30831e2adf86259` |

Checkpoint examples:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2/checkpoints/lightzero/iteration_16.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2/checkpoints/lightzero/latest.pth.tar
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2/checkpoints/lightzero/ckpt_best.pth.tar
```

## Next Action

Eval the scale run before launching a larger lane. Use selected iterations
`0,1,2,4,8,12,16` with the same 512-step cap, 32 eval seeds,
`num_simulations=2`, and `batch_size=64`.

Do not claim learning yet. The immediate target-fix eval passed mechanically,
but it showed lower mean survival after training.

## Scale Run Survival Curve Eval

Ran the corrected scale-run curve with the Modal CurvyTron visual survival eval
wrapper. The requested 64-seed panel completed, so no 32-seed fallback was
needed.

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --run-id curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2 \
  --attempt-id targetfix-scale-b8-s9-16x64-u1-sim2-20260510 \
  --eval-id scale_curve_n64_steps512 \
  --selected-iterations 0,1,2,4,8,12,16 \
  --max-eval-steps 512 \
  --eval-seed-count 64 \
  --num-simulations 2 \
  --batch-size 64 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Modal app run:

```text
ap-ptarVvX4FGIJqHX7hlY2dD
```

Manifest:

```text
training/lightzero-curvytron-visual-survival/curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2/attempts/targetfix-scale-b8-s9-16x64-u1-sim2-20260510/eval/scale_curve_n64_steps512/manifest_steps512_seedsn64_d02637462447_20260510T214456Z.json
```

Eval seed sampler:

```text
8339752774872641715
```

All 448 checkpoint/seed jobs returned `ok: true`; all strict checkpoint loads
reported true. There were no capped episodes and no failures.

| Checkpoint | Seeds | Mean steps survived | Capped count | Failure count |
| --- | ---: | ---: | ---: | ---: |
| iteration_0 | 64 | 196.156 | 0 | 0 |
| iteration_1 | 64 | 195.922 | 0 | 0 |
| iteration_2 | 64 | 193.547 | 0 | 0 |
| iteration_4 | 64 | 192.703 | 0 | 0 |
| iteration_8 | 64 | 197.422 | 0 | 0 |
| iteration_12 | 64 | 196.297 | 0 | 0 |
| iteration_16 | 64 | 197.453 | 0 | 0 |

Plain read: stayed flat. The final checkpoint is only +1.297 mean steps over
`iteration_0`, and the selected curve moves within a narrow 192.703-197.453
band rather than showing a convincing upward survival trend.
