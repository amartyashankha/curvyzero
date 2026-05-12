# CurvyTron Next Native Experiment Decision - 2026-05-10

No pytest was run.

## Decision

Run one larger-seed eval panel before launching more native training:

```text
s92 matched frozen-opponent confirmation eval
checkpoints: iteration_0, iteration_384, iteration_434
eval seeds: 32 random starts from sampler seed 20260510
opponent: the same frozen s47 iteration_200 checkpoint used to train s92
metric: mean and distribution of steps survived, cap 1024
```

Do not launch another native fixed-straight or frozen-refresh train until this
eval tells us whether the best native frozen-opponent bump is real.

## Why This One

Current evidence:

- Native fixed-straight `train_muzero` is mechanically working, including
  player-aware observations, strict checkpoint loads, and random-seed eval
  panels. It does not show reliable learning. s100 was lower than its initial
  checkpoint, s101 only had a small 64-seed lift that disappeared on a fresh
  8-seed spread, and s102 was flat.
- Native frozen-checkpoint opponents are mechanically feasible and have run.
  The first s44-s47 runs produced high but unstable matched-opponent survival.
- The best staged frozen-refresh signal is s92 matched against its training
  opponent: `503.125 -> 589.000 -> 541.750` mean steps around
  `iteration_0/384/434` on only 8 eval seeds. Fixed-straight eval for the same
  run stayed flat.
- s93, trained from s92 `iteration_384`, did not rescue the lane. That makes a
  new frozen-refresh train lower value until we know whether the s92 bump was
  robust or just seed-panel noise.
- The custom two-seat lane stayed flat and has known target/batch-size issues,
  so it should not be the next native LightZero experiment.

This eval is the cheapest useful discriminator:

```text
If s92 iteration_384 still beats iteration_0 and iteration_434 on 32 seeds:
  the frozen-opponent signal is real enough to justify pooled-opponent eval or
  a carefully instrumented next refresh.

If the advantage collapses:
  stop refreshing frozen opponents for now and move to telemetry/profile work:
  opponent perspective, action collapse, value/reward scale, and true two-seat
  target/batch fixes.
```

## Exact Command

Matched frozen-opponent confirmation eval:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute gpu-l4-t4-cpu40 \
  --run-id curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536 \
  --attempt-id train-gpu-l4t4-refresh-s47iter200-65536x256-s92-wait-20260510 \
  --eval-id s92_matched_s47iter200_confirm32_iter0_384_434_20260510 \
  --selected-iterations 0,384,434 \
  --eval-seed-count 32 \
  --eval-seed-rng-seed 20260510 \
  --max-eval-steps 1024 \
  --source-max-steps 1024 \
  --num-simulations 4 \
  --batch-size 64 \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-frozen-s42-iter293-s47-65536/checkpoints/lightzero/iteration_200.pth.tar \
  --opponent-snapshot-ref curvytron_visual_survival_s47_iteration_200 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

Pass bar: `iteration_384` should beat `iteration_0` by a visible margin on the
32-seed aggregate, with no strict-load failures and no single-seed-only story.
Otherwise treat the s92 matched bump as unconfirmed and switch away from new
native training launches.

## Result

The 32-seed matched frozen-opponent confirmation eval completed.

```text
Modal app: ap-36CHuCQOLopsoMQ7hSgrn8
manifest:
training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536/attempts/train-gpu-l4t4-refresh-s47iter200-65536x256-s92-wait-20260510/eval/s92_matched_s47iter200_confirm32_iter0_384_434_20260510/manifest_steps1024_seedsn32_b39991066252_20260510T220709Z.json
```

| checkpoint | seeds | mean steps | median | min | max | capped | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `iteration_0` | 32 | 151.781 | 138.0 | 43 | 334 | 0 | 0 |
| `iteration_384` | 32 | 417.031 | 170.0 | 32 | 1024 | 9 | 0 |
| `iteration_434` | 32 | 500.438 | 482.5 | 32 | 1024 | 10 | 0 |

Read: this confirms a real matched frozen-opponent survival lift. It is noisy:
some seeds still die quickly, and the median at `iteration_384` is much lower
than the mean because many capped runs pull the average up. But this is still
the strongest CurvyTron learning signal so far, and it is from the native
LightZero `train_muzero` lane, not the custom two-seat trainer.

Next useful checks:

- Evaluate against a broader opponent panel, not only the training opponent.
- Watch for action collapse: many high-survival rows use mostly one turn
  direction, so this may be exploiting the matched opponent rather than learning
  general survival.
- Let the background native fixed/frozen jobs finish, then evaluate them with
  the same survival panels.
