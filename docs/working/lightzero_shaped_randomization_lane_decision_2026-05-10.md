# LightZero Shaped Randomization Lane Decision - 2026-05-10

Last updated: `2026-05-10 09:41 EDT`.

Purpose: decide whether to launch more survival-shaped or random-start Pong
training now, while keeping the normal proof lane clean.

## Decision

Do not launch another training wave right now.

The current survival-shaped lane is enough for the next decision point. There
are already survival-shaped training runs across seeds `30`-`37`, `60`-`61`,
and `80`-`82`. The newest wave also added a lower shaping value
(`0.0005`) and a longer shaped run, so the missing evidence is eval coverage
over existing checkpoints, not more launch volume.

Plain read: launch more only after the already-running shaped and normal runs
have stock-evaluator survival curves at `0,1000,5000`, and later checkpoints
for any run that improves over its own `iteration_0`.

## Why This Is The Safe Choice

- Survival-shaped runs are side-lane telemetry. They cannot establish the
  normal stock-reward proof lane.
- The wrapper keeps shaped runs separate by switching to
  `atari_lightzero_survival_shaped` and requiring `survival-shaped` in both
  run and attempt ids.
- The current launch wrapper exposes training seed, but not a separate
  random-start distribution knob. In this lane, varied/random starts mean fresh
  training seeds and fresh run/attempt ids unless the run is explicitly paired.
- Wave10 and wave11 already have enough pending work to answer whether shaping
  helps stock survival at early checkpoints.
- More immediate launches would increase artifact/eval bookkeeping before the
  existing runs have answered the survival question.

## Existing Coverage To Consume First

| group | lane | seeds | shaping | status to consume next |
| --- | --- | --- | --- | --- |
| CPU40 shaped eval wave | shaped | `30`-`37` | `0.001` | already has early stock-eval reads; continue only selected later rows |
| wave10 | shaped | `60`-`61` | `0.001` | evaluate/finish `0,1000,5000` and later if improving |
| wave11 | shaped | `80`-`82` | `0.0005`, `0.001` | first eval target is `0,1000,5000` |
| wave10/wave11 | normal | `50`-`57`, `70`-`76` | `0` | normal proof lane; keep separate from shaped telemetry |

## Hold Gate

Before any new shaped/randomization launch, finish this minimum readout:

- Existing shaped seeds `60`, `61`, `80`, `81`, and `82`:
  `iteration_0`, `iteration_1000`, `iteration_5000` under strict stock eval.
- Existing normal seeds `50`-`57` and `70`-`76`:
  same first pass, then later checkpoints only for same-run survival
  improvements.
- Any promising checkpoint gets robustness eval on a fresh pseudo-random eval
  seed set, with the generator seed/list recorded, and the same `2048` cap.

Report stock steps survived versus same-run `iteration_0` first. Stock return
and reward counts remain secondary.

## Exact Launch Plan If The Hold Gate Still Leaves A Gap

Launch this as `wave12-micro`, not as normal proof-lane evidence. Use L4/T4
CPU16 only, `65536` max env steps, checkpoint cadence `1000`, and fresh ids.

| seed | lane | compute | max env step | shaping | run id | attempt id | purpose |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 90 | normal | `gpu-l4-t4-cpu16` | 65536 | 0 | `lz-visual-pong-exact-installed-0.2.0-s90-wave12micro-l4cpu16` | `train-normal-wave12micro-s90-65536-ckpt1000-l4cpu16-relpath` | one contemporaneous stock-control sentinel |
| 91 | shaped | `gpu-l4-t4-cpu16` | 65536 | 0.00025 | `lz-visual-pong-survival-shaped-step0p00025-s91-wave12micro-l4cpu16` | `train-survival-shaped-step0p00025-wave12micro-s91-65536-ckpt1000-l4cpu16-relpath` | lower shaping check |
| 92 | shaped | `gpu-l4-t4-cpu16` | 65536 | 0.0005 | `lz-visual-pong-survival-shaped-step0p0005-s92-wave12micro-l4cpu16` | `train-survival-shaped-step0p0005-wave12micro-s92-65536-ckpt1000-l4cpu16-relpath` | repeat lower shaping with fresh seed |
| 93 | shaped | `gpu-l4-t4-cpu16` | 65536 | 0.001 | `lz-visual-pong-survival-shaped-step0p001-s93-wave12micro-l4cpu16` | `train-survival-shaped-step0p001-wave12micro-s93-65536-ckpt1000-l4cpu16-relpath` | compare against existing shaped setting |

Do not add H100 or CPU40 training in this micro wave. CPU40 remains useful for
eval fan-out, not for adding more training variety here.

## Launch Commands For Later Use

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu16 --seed 90 --run-id lz-visual-pong-exact-installed-0.2.0-s90-wave12micro-l4cpu16 --attempt-id train-normal-wave12micro-s90-65536-ckpt1000-l4cpu16-relpath --max-env-step-override 65536 --save-ckpt-after-iter-override 1000
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu16 --seed 91 --run-id lz-visual-pong-survival-shaped-step0p00025-s91-wave12micro-l4cpu16 --attempt-id train-survival-shaped-step0p00025-wave12micro-s91-65536-ckpt1000-l4cpu16-relpath --max-env-step-override 65536 --save-ckpt-after-iter-override 1000 --survival-reward-per-step 0.00025
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu16 --seed 92 --run-id lz-visual-pong-survival-shaped-step0p0005-s92-wave12micro-l4cpu16 --attempt-id train-survival-shaped-step0p0005-wave12micro-s92-65536-ckpt1000-l4cpu16-relpath --max-env-step-override 65536 --save-ckpt-after-iter-override 1000 --survival-reward-per-step 0.0005
```

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu16 --seed 93 --run-id lz-visual-pong-survival-shaped-step0p001-s93-wave12micro-l4cpu16 --attempt-id train-survival-shaped-step0p001-wave12micro-s93-65536-ckpt1000-l4cpu16-relpath --max-env-step-override 65536 --save-ckpt-after-iter-override 1000 --survival-reward-per-step 0.001
```

## Eval Contract For This Plan

- First pass: `iteration_0`, `iteration_1000`, `iteration_5000`.
- Compute: `gpu-l4-t4-cpu40`.
- Strict no-fallback checkpoint load.
- `max_eval_steps=2048`, `max_episode_steps=2048`.
- `--group-size 1 --max-parallel-launches 64`.
- Add `--eval-seeds <fresh-eval-seed-list>` only for checkpoints with a
  stock-survival improvement over their own `iteration_0`; record the generator
  seed and exact list.

## Non-Claims

This hold and optional micro matrix do not claim solved Pong, CurvyTron
readiness, or that shaped reward is the final objective. Shaped rows remain
side-lane telemetry. Normal proof still requires sustained stock-evaluator
survival improvement from normal stock-reward runs.
