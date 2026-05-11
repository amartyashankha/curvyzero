# 2026-05-09 Modal LightZero Pong GPU512 Control

## Question

Can the official Atari Pong cold-start control use a cheap GPU and produce a
true later checkpoint without becoming a large run?

## Setup

Official Atari Pong only:

- env: `PongNoFrameskip-v4`
- source config: `zoo.atari.config.atari_muzero_config`
- entrypoint: `lzero.entry.train_muzero`
- wrapper: `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- eval wrapper: `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
- model: stock conv MuZero, observation shape `[4, 64, 64]`, action space `6`

Wrapper update:

- named the cheap GPU list as `CHEAP_GPU_RESOURCE = ["L4", "T4"]`;
- kept the GPU Modal function on that cheap list only;
- added runtime CUDA telemetry to the trainer summary.

No pytest was run. Compile check passed:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py
```

## Commands

Train:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train --max-env-step 512 --max-train-iter 4 --batch-size 8 --max-episode-steps 128 --game-segment-length 16
```

Eval final checkpoint:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T180945Z-29b83d6ee638/checkpoints/lightzero/iteration_4.pth.tar --run-id lz-visual-pong-20260509T180945Z-29b83d6ee638 --attempt-id attempt-20260509T180945Z-dc971b1ec0ff --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T180945Z-29b83d6ee638/attempts/attempt-20260509T180945Z-dc971b1ec0ff/eval/iteration_4_gpu512/lightzero_visual_pong_eval_gpu512_iteration4_20260509T180945Z.json --max-env-step 512 --max-train-iter 4 --batch-size 8 --max-episode-steps 128 --game-segment-length 16 --max-eval-steps 128 --step-detail-limit 8 --no-allow-model-fallback
```

## Results

Training Modal app:

```text
ap-NcECoDQrcIfrbpqmBRODbP
run_id: lz-visual-pong-20260509T180945Z-29b83d6ee638
attempt_id: attempt-20260509T180945Z-dc971b1ec0ff
ok: true
compute: gpu-l4-t4
use_cuda: true
remote_elapsed_sec: 33.142496
train_result.elapsed_sec: 21.577609
runtime GPU: NVIDIA L4
CUDA available: true
final_rewards: [-2.0, -2.0, -3.0, -2.0]
```

Mirrored checkpoints:

```text
ckpt_best.pth.tar sha256 71bc4988adfca1546f4ea70bb77d3f148c04d46ffa08198aa8b1aed5de722e04
iteration_0.pth.tar sha256 7adab1a93e771bb0fda25ba9b6f8b2742e3d212475c63c674e235862398779b9
iteration_1.pth.tar sha256 a7215f42dbcad79ad51fc8657d82083a45a8e57f523798663fe8eda7bf529f44
iteration_2.pth.tar sha256 c8e8cb75d762804bf186384da03ab3f9ec07bbd10ca6b27d371760147bc5f54e
iteration_3.pth.tar sha256 ec0a80ee19a642ea0680f5f30751cacc5614f1a4e2eae72f7f9e1075f9b56e57
iteration_4.pth.tar sha256 3b05d64b3089875aa06ddf2632caaf397f199a4e1190cf8bca6ac5ccba722bd9
```

Eval Modal app:

```text
ap-AUNehXPKKdkXbPOCW5WM7B
checkpoint: iteration_4.pth.tar
checkpoint sha256: 3b05d64b3089875aa06ddf2632caaf397f199a4e1190cf8bca6ac5ccba722bd9
ok: true
remote_elapsed_sec: 39.992162
steps_run: 128
policy_eval_step_count: 128
fallback_step_count: 0
action_histogram: {0:21, 1:24, 2:22, 3:25, 4:22, 5:14}
reward_histogram: {-1.0:2, 0.0:126}
total_reward: -2.0
nonzero rewards: step 60 -> -1.0, step 95 -> -1.0
terminal: step 127, TimeLimit.truncated true, eval_episode_return -2.0
```

Eval artifact:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T180945Z-29b83d6ee638/attempts/attempt-20260509T180945Z-dc971b1ec0ff/eval/iteration_4_gpu512/lightzero_visual_pong_eval_gpu512_iteration4_20260509T180945Z.json
sha256: f8ff3804818ab36cba6a39dc2be5236120611b7f683cbb9ee134a665c943a21b
```

## Comparison

| Checkpoint | Eval cap | Actions | Return | Nonzero rewards |
| --- | ---: | --- | ---: | --- |
| old tiny `iteration_1` | 128 | `{0:128}` | `-2.0` | `60:-1`, `95:-1` |
| scale128 `iteration_1` | 128 | `{0:128}` | `-2.0` | `60:-1`, `95:-1` |
| GPU512 `iteration_4` | 128 | `{0:21,1:24,2:22,3:25,4:22,5:14}` | `-2.0` | `60:-1`, `95:-1` |

## Interpretation

This rung clears the wrapper/GPU/cap blocker. A cheap Modal L4 can run the
official Atari Pong trainer with `policy.cuda=true`, produce later checkpoints
through `iteration_4`, and evaluate the final checkpoint through
`MuZeroPolicy.eval_mode.forward` with no fallback.

It is still not a policy-quality win. The final checkpoint is no longer
all-action-0, but under the same 128-step official Atari eval cap it gets the
same `-2.0` return and same losing reward timing as the action-0 baselines.

Next official Atari decision: do not rerun GPU512 as-is for quality. A next
run must either evaluate more seeds/checkpoints to understand whether the
diverse action distribution is noise, or explicitly raise the official Atari
budget with a higher eval bar. Keep this lane separate from dummy Pong and
CurvyTron.
