# 2026-05-09 Modal LightZero Pong GPU2048 Control

## Question

Does the official Atari Pong visual-control signal strengthen if the same cheap
Modal GPU lane is doubled from `1024` to `2048` LightZero env steps?

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

- raised the trainer validation ceiling for `max_env_step` from `1024` to
  `2048`;
- raised the trainer validation ceiling for `max_train_iter` from `8` to `16`;
- left the cheap GPU resource list as `["L4", "T4"]`;
- left the official Atari config path and ROM image path unchanged.

No pytest was run. Compile check passed:

```sh
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py
```

## Train Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train --max-env-step 2048 --max-train-iter 16 --batch-size 8 --max-episode-steps 256 --game-segment-length 16
```

## Train Result

Modal app:

```text
ap-EWYNfBssH9EJsvIQICt95r
```

Run:

```text
run_id: lz-visual-pong-20260509T184505Z-300f9451a48c
attempt_id: attempt-20260509T184505Z-cc766e1d7c36
ok: true
compute: gpu-l4-t4
runtime GPU: NVIDIA L4
CUDA available: true
remote_elapsed_sec: 84.367742
train_elapsed_sec: 68.872471
final_rewards_seen_in_trainer_logs: [-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]
```

The command asked for `max_train_iter=16`, but the `2048` environment-step cap
stopped the run after checkpoints through `iteration_8`. That matches the
earlier capped-control pattern.

Mirrored checkpoints:

```text
ckpt_best.pth.tar sha256 71bc4988adfca1546f4ea70bb77d3f148c04d46ffa08198aa8b1aed5de722e04
iteration_0.pth.tar sha256 cf449f63e3dee1b5006377821410476ae8499625b20dfdb1361b8838ec938e13
iteration_1.pth.tar sha256 6c8fc7f52272ccfc747c3a4fd3e9bf2468a6e9eec7aa29cf07d1bb0d28073322
iteration_2.pth.tar sha256 82b1202a507d6f2679c6b5b64a6359dce846c5610d290d16828a72d4986842ce
iteration_3.pth.tar sha256 18b3af1d0ab2a0a85045c05f66d86f9a863bb0ab1a00012b92e1867f8004de92
iteration_4.pth.tar sha256 bd582182ddcfedbe0b2af6c1b5f2237b1bde1d8f7093acfd7bb932534e2fd794
iteration_5.pth.tar sha256 f5a6732855f242d4648001c67bf8b0d2b35fbd74a38e562852b17500291b2cce
iteration_6.pth.tar sha256 c73341aa903a2a628c30f4105db55ae8a2a46e6fab17fae5ef0d695a04fb0e72
iteration_7.pth.tar sha256 dacf084ac3e61225201fbae9846d23c7fb10716863dffe225883107e58261aee
iteration_8.pth.tar sha256 c8bd0857ae455a796dcc5d9fce9bbc457c3169a2698a9b28ec3d4722aa5f8066
```

## Eval Commands

Each eval used `--max-eval-steps 256`, `--max-episode-steps 256`, and
`--no-allow-model-fallback`.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/checkpoints/lightzero/iteration_0.pth.tar --run-id lz-visual-pong-20260509T184505Z-300f9451a48c --attempt-id attempt-20260509T184505Z-cc766e1d7c36 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/attempts/attempt-20260509T184505Z-cc766e1d7c36/eval/iteration_0_gpu2048_eval256/lightzero_visual_pong_eval_gpu2048_iteration0_20260509T184505Z.json --max-env-step 2048 --max-train-iter 16 --batch-size 8 --max-episode-steps 256 --game-segment-length 16 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/checkpoints/lightzero/iteration_4.pth.tar --run-id lz-visual-pong-20260509T184505Z-300f9451a48c --attempt-id attempt-20260509T184505Z-cc766e1d7c36 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/attempts/attempt-20260509T184505Z-cc766e1d7c36/eval/iteration_4_gpu2048_eval256/lightzero_visual_pong_eval_gpu2048_iteration4_20260509T184505Z.json --max-env-step 2048 --max-train-iter 16 --batch-size 8 --max-episode-steps 256 --game-segment-length 16 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/checkpoints/lightzero/iteration_8.pth.tar --run-id lz-visual-pong-20260509T184505Z-300f9451a48c --attempt-id attempt-20260509T184505Z-cc766e1d7c36 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/attempts/attempt-20260509T184505Z-cc766e1d7c36/eval/iteration_8_gpu2048_eval256/lightzero_visual_pong_eval_gpu2048_iteration8_20260509T184505Z.json --max-env-step 2048 --max-train-iter 16 --batch-size 8 --max-episode-steps 256 --game-segment-length 16 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

Eval Modal apps:

```text
iteration_0 GPU2048: ap-AEzp2ehrRKH6yF0Y98oaF1
iteration_4 GPU2048: ap-5FI0urkP96erEagTMDR30W
iteration_8 GPU2048: ap-4sFSadQFf2k23fNEgN24W4
```

## Eval Results

| Checkpoint | Train cap | Eval cap | Fallback | Actions | Return | Nonzero reward steps | Terminal/truncation |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| GPU2048 `iteration_0` | 2048 | 256 | 0 | `{0:35,1:37,2:184}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | done at 255, `TimeLimit.truncated=true` |
| GPU2048 `iteration_4` | 2048 | 256 | 0 | `{0:35,1:51,2:170}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | done at 255, `TimeLimit.truncated=true` |
| GPU2048 `iteration_8` | 2048 | 256 | 0 | `{0:48,1:37,2:171}` | `-6.0` | `60:-1, 95:-1, 130:-1, 165:-1, 200:-1, 235:-1` | done at 255, `TimeLimit.truncated=true` |
| GPU1024 `iteration_4` baseline | 1024 | 256 | 0 | `{0:57,1:36,2:37,3:41,4:36,5:49}` | `-3.0` | `60:-1, 95:-1, 143:+1, 208:-1, 243:-1` | done at 255, `TimeLimit.truncated=true` |
| GPU512 `iteration_4` baseline | 512 | 256 | 0 | `{0:44,1:43,2:43,3:44,4:33,5:49}` | `-5.0` | `60:-1, 95:-1, 130:-1, 165:-1, 219:-1` | done at 255, `TimeLimit.truncated=true` |

Eval artifact refs:

```text
training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/attempts/attempt-20260509T184505Z-cc766e1d7c36/eval/iteration_0_gpu2048_eval256/lightzero_visual_pong_eval_gpu2048_iteration0_20260509T184505Z.json
training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/attempts/attempt-20260509T184505Z-cc766e1d7c36/eval/iteration_4_gpu2048_eval256/lightzero_visual_pong_eval_gpu2048_iteration4_20260509T184505Z.json
training/lightzero-official-visual-pong/lz-visual-pong-20260509T184505Z-300f9451a48c/attempts/attempt-20260509T184505Z-cc766e1d7c36/eval/iteration_8_gpu2048_eval256/lightzero_visual_pong_eval_gpu2048_iteration8_20260509T184505Z.json
```

## Interpretation

The 2048 rung did not strengthen the GPU1024 signal.

All sampled GPU2048 checkpoints acted through `MuZeroPolicy.eval_mode.forward`
with no fallback, but they used only actions `0`, `1`, and `2`, got no positive
Pong rewards, and finished the fixed 256-step eval window at `-6.0`. This is
worse than the prior GPU1024 final checkpoint under the same eval cap, which
got one positive point and returned `-3.0`.

This is not a solved-policy result. It is also not evidence that the official
Atari lane is broken: the wrapper still trains, mirrors checkpoints, and
evaluates real ALE Pong without fallback. But the next useful move is not a
blind same-shape `4096` spend. If continuing, either add seed/checkpoint
replication at this scale or change one controlled training knob while keeping
the no-fallback eval rule.
