# 2026-05-09 Modal LightZero Pong Checkpoint State Diff

Scope: official installed `LightZero==0.2.0` Atari Pong run
`lz-visual-pong-8192-sim25-s0`, attempt
`train-8192-sim25-b64-env4-auto`. No training and no pytest.

## Question

Why did `ckpt_best` behave differently from periodic checkpoints in the
8192/sim25 eval curve?

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_checkpoint_diff \
  --run-id lz-visual-pong-8192-sim25-s0 \
  --attempt-id train-8192-sim25-b64-env4-auto \
  --left-label iteration_932 \
  --right-label ckpt_best
```

Modal app: `ap-yIGfkon1zNYV11hsjyxWO6`.

Artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_state_diff/iteration_932_vs_ckpt_best_20260509T223817Z.json
sha256 8bfe73bcacf4f4fa72f0cb96dc5838f098b75c40ad574e790582e184371a2fbf
```

## Result

`ckpt_best` is not credible quality evidence.

The two checkpoints have the same model key set and tensor shapes:

| Field | `iteration_932` | `ckpt_best` |
| --- | ---: | ---: |
| file size | `96,211,827` bytes | `64,190,491` bytes |
| state path | `model` | `model` |
| tensor count | `175` | `175` |
| common keys | `175` | `175` |
| shape mismatches | `0` | `0` |
| saved `last_iter` | `932` | `0` |
| saved `last_step` | `3728` | `0` |
| optimizer state count | `97` | `0` |
| first norm batch counter | `5592` | `0` |
| first norm running mean | nonzero | all `0` |
| first norm running var | trained values | all `1` |
| first norm weight | trained values | all `1` |

The state tensors differ strongly:

```text
global_mean_abs_diff: 0.07748383971601208
global_max_abs_diff: 15526.0068359375
largest differences: batch-norm counters and running variances
```

## Plain Read

`ckpt_best` looks like an initial or reset-style checkpoint, not the best learned
policy from the 8192/sim25 run. That explains why it was smaller, why its logits
looked default-like, and why manual/stock eval disagreed.

Do not treat the manual `ckpt_best` return `0` as learning. The useful eval
curve is the periodic checkpoints, and those all collapsed to one action and
returned capped `-6`.

## Next

Keep `ckpt_best` out of quality summaries for this run unless we can explain
LightZero's best-checkpoint save path. Use periodic checkpoints plus explicit
parallel eval manifests for learning curves.
