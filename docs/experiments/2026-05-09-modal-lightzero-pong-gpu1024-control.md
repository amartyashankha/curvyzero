# 2026-05-09 Modal LightZero Pong GPU1024 Control

## Question

Can the official Atari Pong visual-control run use the same cheap Modal GPU
path at a slightly higher cap and show any real signal?

## Setup

Official Atari Pong only:

- env: `PongNoFrameskip-v4`
- source config: `zoo.atari.config.atari_muzero_config`
- entrypoint: `lzero.entry.train_muzero`
- train wrapper: `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- eval wrapper: `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
- model: stock LightZero visual MuZero, `[4, 64, 64]` frames, 6 Atari actions

This is not custom dummy Pong.

Tiny wrapper fix:

- raised the trainer validation ceiling for `max_env_step` from `512` to `1024`;
- left the cheap GPU resource list as `["L4", "T4"]`;
- left the official Atari config path and ROM image path unchanged.

No pytest was run. Compile check passed:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py
```

## Train Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train --max-env-step 1024 --max-train-iter 8 --batch-size 8 --max-episode-steps 256 --game-segment-length 16
```

## Train Result

Modal app:

```text
ap-UzmZV3BpAGweSxPLznIxyh
```

Run:

```text
run_id: lz-visual-pong-20260509T182028Z-e353ce21f85c
attempt_id: attempt-20260509T182028Z-da8887a05e46
ok: true
compute: gpu-l4-t4
runtime GPU: NVIDIA L4
CUDA available: true
remote_elapsed_sec: 41.101580
train_elapsed_sec: 30.614016
final_rewards_seen_in_trainer_logs: [-6.0, -6.0, -6.0, -6.0]
```

The command asked for `max_train_iter=8`, but the 1024 environment-step cap
stopped the run after checkpoints through `iteration_4`. That is expected for
this capped control.

Mirrored checkpoints:

```text
ckpt_best.pth.tar sha256 71bc4988adfca1546f4ea70bb77d3f148c04d46ffa08198aa8b1aed5de722e04
iteration_0.pth.tar sha256 9f2cbc2a8845d6d0165b01a98e31740817c273eed63efffcf0fccf56e2ab73d2
iteration_1.pth.tar sha256 9680ab2f366d0dc300484678335b980e3676cce877df46db3a6c998a9809cee4
iteration_2.pth.tar sha256 2f2b48a1235ee2170059e291a0b308332f648d230a4db64ccc1686fd5e057958
iteration_3.pth.tar sha256 37fce63728cea65d3b0e1aad9cd517a94abd0c834bd30313611b9a9b01cbaff7
iteration_4.pth.tar sha256 ebde4be6369d82ac05b3ddfb1f1f9b74d6a14fe4b900a24f86446b535091f0c7
```

## Eval Commands

Each eval used `--max-eval-steps 256`, `--max-episode-steps 256`, and
`--no-allow-model-fallback`.

Current run, first checkpoint:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/checkpoints/lightzero/iteration_0.pth.tar --run-id lz-visual-pong-20260509T182028Z-e353ce21f85c --attempt-id attempt-20260509T182028Z-da8887a05e46 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/attempts/attempt-20260509T182028Z-da8887a05e46/eval/iteration_0_gpu1024_eval256/lightzero_visual_pong_eval_gpu1024_iteration0_20260509T182028Z.json --max-env-step 1024 --max-train-iter 8 --batch-size 8 --max-episode-steps 256 --game-segment-length 16 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

Current run, midpoint:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/checkpoints/lightzero/iteration_2.pth.tar --run-id lz-visual-pong-20260509T182028Z-e353ce21f85c --attempt-id attempt-20260509T182028Z-da8887a05e46 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/attempts/attempt-20260509T182028Z-da8887a05e46/eval/iteration_2_gpu1024_eval256/lightzero_visual_pong_eval_gpu1024_iteration2_20260509T182028Z.json --max-env-step 1024 --max-train-iter 8 --batch-size 8 --max-episode-steps 256 --game-segment-length 16 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

Current run, final checkpoint:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/checkpoints/lightzero/iteration_4.pth.tar --run-id lz-visual-pong-20260509T182028Z-e353ce21f85c --attempt-id attempt-20260509T182028Z-da8887a05e46 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/attempts/attempt-20260509T182028Z-da8887a05e46/eval/iteration_4_gpu1024_eval256/lightzero_visual_pong_eval_gpu1024_iteration4_20260509T182028Z.json --max-env-step 1024 --max-train-iter 8 --batch-size 8 --max-episode-steps 256 --game-segment-length 16 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

Cheap old baseline, same 256-step eval cap:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T180945Z-29b83d6ee638/checkpoints/lightzero/iteration_4.pth.tar --run-id lz-visual-pong-20260509T180945Z-29b83d6ee638 --attempt-id attempt-20260509T180945Z-dc971b1ec0ff --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T180945Z-29b83d6ee638/attempts/attempt-20260509T180945Z-dc971b1ec0ff/eval/iteration_4_gpu512_eval256/lightzero_visual_pong_eval_gpu512_iteration4_eval256_20260509T180945Z.json --max-env-step 512 --max-train-iter 4 --batch-size 8 --max-episode-steps 256 --game-segment-length 16 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

Eval Modal apps:

```text
iteration_0 GPU1024: ap-jQVlznsOZFTrUJJXvgqjqM
iteration_2 GPU1024: ap-dhvmTlJkagvgfGYsfb8Z7r
iteration_4 GPU1024: ap-5rayW77BqmdlN3u9WmS4lc
iteration_4 GPU512 baseline: ap-XzZAR4HrpoXGxvfzfqJTpH
```

## Eval Results

| Checkpoint | Train cap | Eval cap | Fallback | Actions | Return | Nonzero reward steps | Terminal/truncation |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| GPU1024 `iteration_0` | 1024 | 256 | 0 | `{0:38,1:88,2:130}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | done at 255, `TimeLimit.truncated=true` |
| GPU1024 `iteration_2` | 1024 | 256 | 0 | `{0:37,1:64,2:155}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | done at 255, `TimeLimit.truncated=true` |
| GPU1024 `iteration_4` | 1024 | 256 | 0 | `{0:57,1:36,2:37,3:41,4:36,5:49}` | `-3.0` | `60:-1, 95:-1, 143:+1, 208:-1, 243:-1` | done at 255, `TimeLimit.truncated=true` |
| GPU512 `iteration_4` baseline | 512 | 256 | 0 | `{0:44,1:43,2:43,3:44,4:33,5:49}` | `-5.0` | `60:-1, 95:-1, 130:-1, 165:-1, 219:-1` | done at 255, `TimeLimit.truncated=true` |

Eval artifact refs:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/attempts/attempt-20260509T182028Z-da8887a05e46/eval/iteration_0_gpu1024_eval256/lightzero_visual_pong_eval_gpu1024_iteration0_20260509T182028Z.json
training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/attempts/attempt-20260509T182028Z-da8887a05e46/eval/iteration_2_gpu1024_eval256/lightzero_visual_pong_eval_gpu1024_iteration2_20260509T182028Z.json
training/lightzero-official-visual-pong/lz-visual-pong-20260509T182028Z-e353ce21f85c/attempts/attempt-20260509T182028Z-da8887a05e46/eval/iteration_4_gpu1024_eval256/lightzero_visual_pong_eval_gpu1024_iteration4_20260509T182028Z.json
training/lightzero-official-visual-pong/lz-visual-pong-20260509T180945Z-29b83d6ee638/attempts/attempt-20260509T180945Z-dc971b1ec0ff/eval/iteration_4_gpu512_eval256/lightzero_visual_pong_eval_gpu512_iteration4_eval256_20260509T180945Z.json
```

## Interpretation

There is a small real signal, but not a solved-policy signal.

The final GPU1024 checkpoint used all six Atari actions, got one positive Pong
reward at eval step 143, and improved the 256-step return from the old GPU512
baseline `-5.0` to `-3.0`. Earlier checkpoints in the same run still lost every
point on the fixed 256-step eval window.

This means the official visual Atari path is alive and changing under the
cheap GPU control. It does not yet prove stable learning. The next official
Atari step should keep the same no-fallback eval rule and raise the training
budget only if we are willing to pay for a true quality probe.
